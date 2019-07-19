#coding:utf-8
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import vgg16

from utils import *
from loss import *

from ProposalLayer import *
from RoIPoolingLayer import *
from RPN import *
from AnchorTargetLayer import *
from ProposalTargetLayer import *


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
        self.loc_normalize_mean_np = np.array((0.,  0.,  0.,  0.), np.float32)
        self.loc_normalize_std_np  = np.array((0.1, 0.1, 0.2, 0.2), np.float32)

    # im_info: [h, w, scale]
    def inference(self, x, im_info):

        features = self.Extractor(x)

        rpn_locs, rpn_scores, rois_np, roi_indices_np, anchor_np = self.RPN(features, im_info, "TEST")

        indices_and_rois_np = np.concatenate((roi_indices_np, rois_np), axis=1)
        indices_and_rois = torch.from_numpy(indices_and_rois_np)

        rois_p = self.RoIPooling(features, indices_and_rois)

        roi_cls_locs, roi_scores = self.Detector(rois_p)

        return roi_cls_locs, roi_scores, rois_np, roi_indices_np


    def train_step(self, x, gt_bboxes_np, labels_np, im_info_np):

        features = self.Extractor(x)
        
        rpn_locs, rpn_scores, rois_np, roi_indices_np, anchor_np = self.RPN(features, im_info_np, "TRAIN")

        gt_rpn_loc_np, gt_rpn_label_np = self.AnchorTarget(gt_bboxes_np, anchor_np, im_info_np[:2])
        gt_rpn_loc = torch.from_numpy(gt_rpn_loc_np).to(device)
        gt_rpn_label = torch.from_numpy(gt_rpn_label_np).long().to(device)

        # ----------- RPN loss ---------#
        rpn_loc_loss = x_fast_rcnn_loc_loss(rpn_locs,
                                            gt_rpn_loc,
                                            gt_rpn_label.data,
                                            self.rpn_sigma)

        rpn_cls_loss = self.CELoss_1(rpn_scores[0], gt_rpn_label)


        # ------------------ Detector losses -------------------#
        sample_roi_np, gt_roi_loc_np, gt_roi_label_np = self.ProposalTarget(rois_np, 
                                                                            gt_bboxes_np,
                                                                            labels_np,
                                                                            self.loc_normalize_mean_np,
                                                                            self.loc_normalize_std_np)

        sample_roi_indices_np = np.zeros((len(sample_roi_np), 1))
        indices_and_rois_np = np.concatenate((sample_roi_indices_np, sample_roi_np), axis=1)
        indices_and_rois = torch.from_numpy(indices_and_rois_np).to(device)

        rois_p = self.RoIPooling(features, indices_and_rois)
        roi_cls_locs, roi_scores = self.Detector(rois_p)


        # detector loss
        gt_roi_label = torch.from_numpy(gt_roi_label_np).long().to(device)
        roi_cls_loss = self.CELoss_2(roi_scores, gt_roi_label)

        n_sample = roi_cls_locs.shape[0]
        roi_locs = roi_cls_locs.view(n_sample, -1, 4)[np.arange(0, n_sample), gt_roi_label]
        
        gt_roi_loc = torch.from_numpy(gt_roi_loc_np).to(device)
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


    '''
    def get_optimizer(self, lr, weight_decay, use_adam):
        """
        return optimizer, It could be overwriten if you want to specify 
        special optimizer
        """
        params = []
        for key, value in dict(self.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    params += [{'params': [value], 'lr': lr * 2, 'weight_decay': 0}]
                else:
                    params += [{'params': [value], 'lr': lr, 'weight_decay': weight_decay}]
        if use_adam:
            self.optimizer = t.optim.Adam(params)
        else:
            self.optimizer = t.optim.SGD(params, momentum=0.9)
        return self.optimizer
    '''