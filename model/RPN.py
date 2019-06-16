#coding:utf-8
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import vgg16

from utils import *


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
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous().view(n, hh, ww, self.n_anchor, 2)

        rpn_softmax_scores = F.softmax(rpn_scores, dim=4)

        rpn_softmax_scores = rpn_softmax_scores.contiguous().view(n, hh, ww, self.n_anchor*2).permute(0,3,1,2)

        return rpn_locs, rpn_scores, rpn_softmax_scores
