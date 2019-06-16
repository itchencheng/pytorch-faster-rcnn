#coding:utf-8
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import vgg16

from utils import *
from target import *
from loss import *

from ProposalLayer import *
from RoIPoolingLayer import *
from RPN import *


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

        # loss
        self.CELoss_1 = nn.CrossEntropyLoss(ignore_index=-1)
        self.SmoothL1_1 = WeightedSmoothL1Loss()

        self.CELoss_2 = nn.CrossEntropyLoss()
        self.SmoothL1_2 = WeightedSmoothL1Loss()


    # im_info: [h, w, scale]
    def inference(self, x, im_info):
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


    def train_step(self, x, im_info, gt_bboxes):
        features = self.Extractor(x)
        print('features', features.shape)

        rpn_locs, rpn_scores, rpn_softmax_scores = self.RPN(features)
        print('rpn_locs', rpn_locs.shape)
        print('rpn_scores', rpn_scores.shape)
        print('rpn_softmax_scores', rpn_softmax_scores.shape)

        rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, \
            rpn_bbox_outside_weights = self.AnchorTarget(features.shape[-2:], rpn_scores, gt_bboxes, x, im_info)
        print('rpn_labels', rpn_labels.shape)
        print('rpn_bbox_targets', rpn_bbox_targets.shape, rpn_bbox_targets.dtype)
        print('rpn_bbox_inside_weights', rpn_bbox_inside_weights.shape, rpn_bbox_inside_weights.dtype)
        print('rpn_bbox_outside_weights', rpn_bbox_outside_weights.shape, rpn_bbox_outside_weights.dtype)

        roi = self.Proposal('Train', rpn_locs, rpn_softmax_scores, im_info)
        print('roi', roi.shape)
        print('roi', roi)

        shape = rpn_scores.shape
        rpn_scores = rpn_scores.contiguous().view(-1, 2)
        print(rpn_scores.shape, rpn_labels.shape)
        print(type(rpn_scores), type(rpn_labels))
        print(rpn_scores.dtype, rpn_labels.dtype)

        rpn_cls_loss = self.CELoss_1(rpn_scores, rpn_labels)
        print(rpn_cls_loss)
        rpn_loc_loss = self.SmoothL1_1(rpn_locs, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights)



        rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights \
            =  self.ProposalTarget(roi, gt_bboxes)
        print('rois', rois.shape)
        print('labels', labels.shape)
        print('bbox_targets', bbox_targets.shape)
        print('bbox_inside_weights', bbox_inside_weights.shape)
        print('bbox_outside_weights', bbox_outside_weights.shape)

        # features_in = features.detach()
        features_in = features

        rois_p = self.RoIPooling(features_in, rois)
        print('rois_p', rois_p.shape)

        roi_locs, roi_scores = self.Detector(rois_p)
        print('roi_locs', roi_locs.shape)
        print('roi_scores', roi_scores.shape)

        cls_loss = self.CELoss_2(roi_scores,
                                 labels)

        # cls_loss = self.CELoss_2(roi_scores, labels)
        loc_loss = self.SmoothL1_2(roi_locs, bbox_targets, bbox_inside_weights, bbox_outside_weights)

        total_loss = rpn_loc_loss + rpn_cls_loss + loc_loss + cls_loss
        print("==rpn_loc_loss", rpn_loc_loss) 
        print("==rpn_cls_loss", rpn_cls_loss)
        print("==cls_loss", cls_loss)
        print("==loc_loss", loc_loss) 
        print("==total_loss", total_loss)

        # print(self)

        return total_loss
        # return rpn_loc_loss+rpn_cls_loss
        # return loc_loss 
        #return cls_loss
