#coding:utf-8
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import vgg16

from utils import *

class ProposalLayer(object):

    def __init__(self):
        self.nms_thresh = 0.7
        self.min_size = 16

    def __call__(self, STATE, bbox_deltas, scores, anchors, im_info):
        
        if ('TRAIN' == STATE):
            pre_nms_topN = 12000
            post_nms_topN = 2000
        else:
            pre_nms_topN = 6000
            post_nms_topN = 300
        
        # 1. convert anchor to proposals, via bbox transformation
        proposals = bbox_transform_inv(anchors, bbox_deltas)
        #print('proposals', proposals.shape)

        # 2. clip to image
        proposals = clip_boxes(proposals, im_info[:2])
        #print('proposals', proposals.shape)

        # 3. remove small predicted boxes
        keep = filter_boxes(proposals, self.min_size * im_info[2])
        proposals = proposals[keep, :]
        scores = scores[keep]

        # NMS
        # 4.1 sort
        order = scores.ravel().argsort()[::-1]
        if pre_nms_topN > 0:
            order = order[:pre_nms_topN]
        proposals = proposals[order, :]

        # 4.2 nms
        keep = my_non_maximum_suppression(proposals, self.nms_thresh)
        if (post_nms_topN > 0):
            keep = keep[:post_nms_topN]
        proposals = proposals[keep, :]

        return proposals
