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


use_drop = False

def decompose_vgg16(ckpt_path=None):
    vgg16_net = vgg16()
    
    if (ckpt_path != None):
        vgg16_net.load_state_dict(torch.load(ckpt_path))

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
                 vgg16_ckpt=None,
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
        self.CELoss_2 = nn.CrossEntropyLoss()

        # param
        self.rpn_sigma = 3.0
        self.roi_sigma = 1.0
        self.loc_normalize_mean = (0.,  0.,  0.,  0. )
        self.loc_normalize_std  = (0.1, 0.1, 0.2, 0.2)

    # im_info: [h, w, scale]
    def inference(self, x, im_info):

        features = self.Extractor(x)

        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.RPN(features, im_info, "TEST")

        roi_indices = roi_indices.reshape(-1,1)

        debug('rois', rois)
        debug('roi_indices', roi_indices)
        debug('features', features)

        indices_and_rois = np.concatenate((roi_indices, rois), axis=1)
        indices_and_rois = change_to_tensor(indices_and_rois)

        debug('indices_and_rois', indices_and_rois)

        rois_p = self.RoIPooling(features, indices_and_rois)

        roi_cls_locs, roi_scores = self.Detector(rois_p)

        return roi_cls_locs, roi_scores, rois, roi_indices


    def train_step(self, x, gt_bboxes, labels, im_info):

        features = self.Extractor(x)
        
        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.RPN(features, im_info, "TRAIN")

        # Since batch size is one, convert variables to singular form
        rpn_scores = rpn_scores[0]

        gt_rpn_loc, gt_rpn_label = self.AnchorTarget(gt_bboxes, anchor, im_info[:2])
        gt_rpn_label = change_to_tensor(gt_rpn_label).long()
        gt_rpn_loc = change_to_tensor(gt_rpn_loc)

        # ----------- RPN loss ---------#
        rpn_loc_loss = x_fast_rcnn_loc_loss(rpn_locs,
                                            gt_rpn_loc,
                                            gt_rpn_label.data,
                                            self.rpn_sigma)
        print(rpn_scores.shape, gt_rpn_label.shape)
        rpn_cls_loss = self.CELoss_1(rpn_scores, gt_rpn_label)

        print('rpn_loc_loss', rpn_loc_loss)
        print('rpn_cls_loss', rpn_cls_loss)




        # ------------------ Detector losses -------------------#
        sample_roi, gt_roi_loc, gt_roi_label = self.ProposalTarget(rois, 
                                                                    gt_bboxes,
                                                                    labels,
                                                                    self.loc_normalize_mean,
                                                                    self.loc_normalize_std)
        # print('sample_roi', sample_roi.shape, sample_roi.dtype)
        # print(sample_roi)
        # print('gt_roi_loc', gt_roi_loc.shape, gt_rpn_loc.dtype)
        # print(gt_roi_loc)
        # print('gt_rpn_label', gt_rpn_label.shape, gt_rpn_label.dtype)
        # print(gt_rpn_label)

        '''
        features = change_to_tensor(np.fromfile('/home/chen/docs/temp0', dtype=np.float32)).reshape((features.shape))
        sample_roi = np.fromfile('/home/chen/docs/temp1', dtype=np.float32).reshape((-1,4))
        sample_roi_indices = np.fromfile('/home/chen/docs/temp2', dtype=np.int32).reshape((-1,1))
        '''

        sample_roi_indices = torch.zeros(len(sample_roi), 1)

        print(sample_roi_indices.shape, sample_roi.shape)

        indices_and_rois = np.concatenate((sample_roi_indices, sample_roi), axis=1)
        #print('indices_and_rois', indices_and_rois.shape, indices_and_rois.dtype)
        #print(indices_and_rois)

        indices_and_rois = change_to_tensor(indices_and_rois)

        rois_p = self.RoIPooling(features, indices_and_rois)
        #print('rois_p', rois_p.shape)
        #print(rois_p)

        roi_cls_locs, roi_scores = self.Detector(rois_p)

        #print("old, gt_roi_label", gt_roi_label.shape, gt_roi_label.dtype)
        
        '''
        roi_cls_locs = change_to_tensor(np.fromfile('/home/chen/docs/temp0', dtype=np.float32)).reshape((-1,84))
        roi_scores = change_to_tensor(np.fromfile('/home/chen/docs/temp1', dtype=np.float32)).reshape((-1,21))
        gt_roi_label = change_to_tensor(np.fromfile('/home/chen/docs/temp2', dtype=np.int32)).reshape(-1,).long()
        gt_roi_loc = change_to_tensor(np.fromfile('/home/chen/docs/temp3', dtype=np.float32)).reshape(-1,4)
        '''

        '''
        print('roi_cls_locs', roi_cls_locs.shape, roi_cls_locs.dtype)
        print(roi_cls_locs)
        print('roi_scores', roi_scores.shape, roi_scores.dtype)
        print(roi_scores)
        print('gt_roi_label', gt_roi_label.shape, gt_roi_label.dtype)
        print(gt_roi_label)
        print('gt_roi_loc', gt_roi_loc.shape, gt_roi_loc.dtype)
        print(gt_roi_loc)
        '''


        # detector loss
        gt_roi_label = change_to_tensor(gt_roi_label).long()
        roi_cls_loss = self.CELoss_2(roi_scores, gt_roi_label)
        
        n_sample = roi_cls_locs.shape[0]
        roi_locs = roi_cls_locs.view(n_sample, -1, 4)[np.arange(0, n_sample), gt_roi_label]
        gt_roi_loc = change_to_tensor(gt_roi_loc)
        gt_roi_label = change_to_tensor(gt_roi_label).long()
        print("roi_locs", roi_locs.shape, roi_locs.dtype)
        

        #print("===========================")
        #debug('roi_locs', roi_locs)
        #debug('gt_roi_loc', gt_roi_loc)
        #debug('gt_roi_label', gt_roi_label)
        roi_loc_loss = x_fast_rcnn_loc_loss(roi_locs.contiguous(), 
                                             gt_roi_loc, 
                                             gt_roi_label, 
                                             self.roi_sigma)

        total_loss = rpn_loc_loss + rpn_cls_loss + roi_loc_loss + roi_cls_loss
        print("==rpn_loc_loss", rpn_loc_loss) 
        print("==rpn_cls_loss", rpn_cls_loss)
        print("==loc_loss", roi_loc_loss) 
        print("==cls_loss", roi_cls_loss)
        print("==total_loss", total_loss)

        return total_loss

