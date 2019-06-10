#coding:utf-8
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import vgg16

from utils import *
from target import *
from loss import *


def decompose_vgg16(ckpt_path):
    vgg16_net = vgg16()
    vgg16_net.load_state_dict(torch.load(ckpt_path))

    use_drop = True

    features = list(vgg16_net.features)[:30]
    # freeze top four conv
    for layer in features[:10]:
        for p in layer.parameters():
            p.requires_grad = False

    classifier = vgg16_net.classifier
    classifier = nn.Sequential(*classifier)
    del classifier[6]
    if (not use_drop):
        del classifier[5]
        del classifier[2]

    features = nn.Sequential(*features)

    classifier = nn.Sequential(*classifier)
    return features, classifier
    # return features


def normal_init(m, mean, stddev, truncated=False):
    """ weight initalizer: truncated normal and random normal """
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()


class RegionProposalNetwork(nn.Module):

    def __init__(self, 
            in_channels=512, 
            mid_channels=512, 
            feature_stride=16,
            anchor_ratios=[0.5, 1, 2],
            anchor_scales=[8, 16, 32],
            n_anchor = 9
            ):
        super(RegionProposalNetwork, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.score = nn.Conv2d(mid_channels,  n_anchor*2, 1, 1, 0)
        self.locate = nn.Conv2d(mid_channels, n_anchor*4, 1, 1, 0)

        self.n_anchor = n_anchor

        normal_init(self.conv1,  0, 0.01)
        normal_init(self.score,  0, 0.01)
        normal_init(self.locate, 0, 0.01)


    def forward(self, x):
        n, _, hh, ww = x.shape

        x = F.relu(self.conv1(x))
        
        rpn_locs = self.locate(x)
        
        rpn_scores = self.score(x)
        rpn_scores_reshape = rpn_scores.permute(0, 2, 3, 1).contiguous()
        rpn_softmax_scores = F.softmax(rpn_scores_reshape.view(n, hh, ww, self.n_anchor, 2), dim=4)

        rpn_softmax_scores = rpn_softmax_scores.contiguous().view(n, hh, ww, self.n_anchor*2).permute(0,3,1,2)

        return rpn_locs, rpn_scores, rpn_softmax_scores


class ProposalLayer(object):

    def __init__(self):
        self.feat_stride = 16
        self.anchor_scales = [8, 16, 32]
        self.anchor_ratios = [0.5, 1., 2.]
        self.anchors = generate_anchors(self.feat_stride, self.anchor_scales, self.anchor_ratios)
        self.n_anchors = self.anchors.shape[0]

    def __call__(self, STATE, bbox_deltas, scores, im_info):
        
        if ('Train' == STATE):
            pre_nms_topN = 12000
            post_nms_topN = 2000
            nms_thresh = 0.7
            min_size = 16
        else:
            pre_nms_topN = 6000
            post_nms_topN = 300
            nms_thresh = 0.7
            min_size = 16

        height, width = scores.shape[-2:]

        shift_x = np.arange(0, width) * self.feat_stride
        shift_y = np.arange(0, height) * self.feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), 
                            shift_x.ravel(), shift_y.ravel())).transpose()

        # add shift to anchor
        A = self.n_anchors
        K = shifts.shape[0]
        anchors = self.anchors.reshape((1, A, 4)) + \
                                        shifts.reshape((1, K, 4)).transpose((1, 0, 2))
        anchors = anchors.reshape((K * A, 4))
        print('anchors', anchors.shape)

        # tensor to numpy
        bbox_deltas = bbox_deltas.detach().numpy()
        scores = scores.detach().numpy()
        # Note: should be carefully think
        scores = scores[:, 1::2, :, :]

        bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1)).reshape((-1, 4))
        scores = scores.transpose((0, 2, 3, 1)).reshape((-1, 1))

        print('bbox_deltas', bbox_deltas.shape)
        print('scores', scores.shape)

        # 1. convert anchor to proposals, via bbox transformation
        proposals = bbox_transform_inv(anchors, bbox_deltas)
        print('proposals', proposals.shape)

        # 2. clip to image
        proposals = clip_boxes(proposals, im_info[:2])
        print('proposals', proposals.shape)


        # 3. remove small predicted boxes
        keep = filter_boxes(proposals, min_size * im_info[2])
        print('keep', keep.shape)
        proposals = proposals[keep, :]
        scores = scores[keep]

        # NMS
        # 4.1 sort
        order = scores.ravel().argsort()[::-1]
        print('order', order.shape)

        if pre_nms_topN > 0:
            order = order[:pre_nms_topN]
        proposals = proposals[order, :]
        scores = scores[order]

        print('proposals', proposals.shape)
        print('nms_thresh', nms_thresh)
        print('scores', scores.shape)

        # 4.2 nms
        keep = my_non_maximum_suppression(proposals, nms_thresh, scores)
        if (post_nms_topN > 0):
            keep = keep[:post_nms_topN]
        print('nms keep', keep)

        proposals = proposals[keep, :]
        scores = scores[keep]

        # 5. output rois blob
        # only support a single input image, batch_indexes are 0
        batch_indexes = np.zeros((proposals.shape[0], 1), dtype=np.float32)
        roi = np.hstack((batch_indexes, proposals.astype(np.float32, copy=False)))
        
        return roi


class RoIPoolingLayer(nn.Module):
    def __init__(self, pooled_h=7, pooled_w=7, spatial_scale=0.0625, pool_type='MAX'):

        super(RoIPoolingLayer, self).__init__()

        self.pooled_h = pooled_h
        self.pooled_w = pooled_w
        self.spatial_scale = spatial_scale
        self.pool_type = pool_type

    def forward(self, features, rois):
        output = []
        rois = torch.Tensor(rois)
        num_rois = rois.contiguous().view(-1, 5).shape[0]

        if (num_rois == 0):
            return torch.Tensor([])

        h, w = features.shape[-2:]

        rois[:, 1:].mul_(self.spatial_scale)
        rois = rois.long()

        size = (self.pooled_h, self.pooled_w)

        for i in range(num_rois):
            roi = rois[i]
            batch_idx = roi[0]

            print('roi', roi)
            im = features[batch_idx, :, roi[2]:(roi[4]+1), roi[1]:(roi[3]+1)]
            print('im', im.shape)
            print('size', size)

            if ('MAX' == self.pool_type):                
                output.append(F.adaptive_max_pool2d(im, size))
     
        output = torch.stack(output, 0)

        return output


class ODetector(nn.Module):

    def __init__(self, classifier, n_class):
        # n_class includes the background
        super(ODetector, self).__init__()

        self.classifier = classifier
        self.cls_loc = nn.Linear(4096, n_class * 4)
        self.score = nn.Linear(4096, n_class)
        self.n_class = n_class

        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)

    def forward(self, rois):
        n_rois = rois.shape[0]
        if (0 == n_rois):
            return torch.Tensor([]), torch.Tensor([])
        rois = rois.view(n_rois,-1)
        fc7 = self.classifier(rois)
        roi_cls_locs = self.cls_loc(fc7)
        roi_scores = self.score(fc7)
        return roi_cls_locs, roi_scores


class FasterRCNN_VGG16(nn.Module):

    def __init__(self,
                 vgg16_ckpt,
                 n_class=20, 
                 anchor_ratios=[0.5, 1, 2],
                 anchor_scales=[8, 16, 32]):

        super(FasterRCNN_VGG16, self).__init__()

        self.Extractor, self.Classifier = decompose_vgg16(vgg16_ckpt)

        self.RPN = RegionProposalNetwork(512, 512, 
                                    anchor_ratios=anchor_ratios,
                                    anchor_scales=anchor_scales)

        self.Proposal = ProposalLayer()

        self.RoIPooling = RoIPoolingLayer()

        self.Detector = ODetector(self.Classifier, n_class+1)

        self.AnchorTarget = AnchorTargetLayer()

        self.ProposalTarget = ProposalTargetLayer()


    # im_info: [h, w, scale]
    def inference(self, x, im_info, gt_bboxes):
        features = self.Extractor(x)
        print('features', features.shape)

        rpn_locs, rpn_scores, rpn_softmax_scores = self.RPN(features)
        print('rpn_locs', rpn_locs.shape)
        print('rpn_scores', rpn_scores.shape)
        print('rpn_softmax_scores', rpn_softmax_scores.shape)

        roi = self.Proposal('Train', rpn_locs, rpn_softmax_scores, im_info)
        print('roi', roi.shape)

        rois_p = self.RoIPooling(features, roi)
        print('rois_p', rois_p.shape)

        roi_locs, roi_scores = self.Detector(rois_p)
        print('roi_locs', roi_locs.shape)
        print('roi_scores', roi_scores.shape)

        print("# AnchorTarget!")
        rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, \
            rpn_bbox_outside_weights = self.AnchorTarget(rpn_scores, gt_bboxes, x, im_info)

        print('rpn_labels', rpn_labels.shape)
        print('rpn_bbox_targets', rpn_bbox_targets.shape)
        print('rpn_bbox_inside_weights', rpn_bbox_inside_weights.shape)
        print('rpn_bbox_outside_weights', rpn_bbox_outside_weights.shape)

        print('# ProposalTarget')
        rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights \
            =  self.ProposalTarget(roi, gt_bboxes)

        print('rois', rois.shape)
        print('labels', labels.shape)
        print('bbox_targets', bbox_targets.shape)
        print('bbox_inside_weights', bbox_inside_weights.shape)
        print('bbox_outside_weights', bbox_outside_weights.shape)



    def train_step(self):
        print("train_step")



    def create_rois(self, config):
     
        rois = torch.rand((config[2], 5))
        rois[:, 0] = rois[:, 0] * config[0]
        rois[:, 1:] = rois[:, 1:] * config[1]
        for j in range(config[2]):
            max_, min_ = max(rois[j, 1], rois[j, 3]), min(rois[j, 1], rois[j, 3])
            rois[j, 1], rois[j, 3] = min_, max_
            max_, min_ = max(rois[j, 2], rois[j, 4]), min(rois[j, 2], rois[j, 4])
            rois[j, 2], rois[j, 4] = min_, max_
        rois = torch.floor(rois)
        return rois


    def testing(self):

        config = [1, 50, 3]
        T = 5
        has_backward = True

        roi_pooling = RoIPoolingLayer()

        x = torch.rand((config[0], 512, config[1], config[1]))
        rois = self.create_rois(config)
    
        for t in range(T):
            output = roi_pooling(x,rois)
            print('roi', output.shape)
            roi_locs, roi_scores = self.Detector(output)
            print('roi_locs', roi_locs.shape)
            print('roi_scores', roi_scores.shape)