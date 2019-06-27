#coding:utf-8
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import vgg16

from utils import *
from ProposalLayer import *

class RegionProposalNetwork(nn.Module):

    def __init__(self, 
            in_channels=512, 
            mid_channels=512, 
            feature_stride=16,
            anchor_ratios=[0.5, 1, 2],
            anchor_scales=[8, 16, 32],
            ):
        super(RegionProposalNetwork, self).__init__()

        self.anchor_base = generate_anchors(feature_stride, anchor_scales, anchor_ratios)
        n_anchor = self.anchor_base.shape[0]

        self.feature_stride = feature_stride

        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.score = nn.Conv2d(mid_channels,  n_anchor*2, 1, 1, 0)
        self.locate = nn.Conv2d(mid_channels, n_anchor*4, 1, 1, 0)

        normal_init(self.conv1,  0, 0.01)
        normal_init(self.score,  0, 0.01)
        normal_init(self.locate, 0, 0.01)

        self.proposal_layer = ProposalLayer()

    def forward(self, x, img_info, PHASE):
        n, _, hh, ww = x.shape

        anchors = enumerate_shifted_anchor(self.anchor_base, self.feature_stride, hh, ww)
        n_anchor = anchors.shape[0] // (hh * ww)

        ''' conv '''
        x = F.relu(self.conv1(x))
        
        ''' loc '''
        rpn_locs = self.locate(x)
        rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)

        ''' cls '''
        rpn_scores = self.score(x)
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous().view(n, hh, ww, n_anchor, 2)
        rpn_softmax_scores = F.softmax(rpn_scores, dim=4)
        rpn_scores = rpn_scores.view(n, -1, 2)

        rpn_fg_scores = rpn_softmax_scores[:, :, :, :, 1].contiguous()
        rpn_fg_scores = rpn_fg_scores.view(n, -1)

        ''' create roi proposal '''
        rois = list()
        roi_indices = list()
        for i in range(n):

            roi = self.proposal_layer(PHASE,
                                      rpn_locs[i].cpu().data.numpy(),
                                      rpn_fg_scores[i].cpu().data.numpy(),
                                      anchors,
                                      img_info)

            rois.append(roi)

            batch_index = i * np.ones((len(roi),), dtype=np.int32)
            roi_indices.append(batch_index)

        rois = np.concatenate(rois, axis=0)
        roi_indices = np.concatenate(roi_indices, axis=0)
        
        return rpn_locs, rpn_scores, rois, roi_indices, anchors
