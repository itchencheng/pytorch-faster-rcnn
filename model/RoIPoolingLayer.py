#coding:utf-8
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import vgg16

from utils import *

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class RoIPoolingLayer(nn.Module):
    def __init__(self, pooled_h=7, pooled_w=7, spatial_scale=0.0625, pool_type='MAX'):

        super(RoIPoolingLayer, self).__init__()

        self.pooled_h = pooled_h
        self.pooled_w = pooled_w
        self.spatial_scale = spatial_scale
        self.pool_type = pool_type

    def forward(self, features, rois):
        # roi format "batch_idx, x, y, x, y"
        
        output = []

        num_rois = rois.contiguous().view(-1, 5).shape[0]

        if (num_rois == 0):
            return torch.Tensor([])

        h, w = features.shape[-2:]

        rois[:, 1:].mul_(self.spatial_scale)
        rois = rois.long()
        
        size = (self.pooled_h, self.pooled_w)

        for i in range(num_rois):
            roi = rois[i]
            #print("-roi", roi)
            batch_idx = roi[0]

            #print('roi', roi)
            im = features[batch_idx, :, roi[2]:(roi[4]+1), roi[1]:(roi[3]+1)]
            #print('im', im.shape)
            #print('size', size)

            if ('MAX' == self.pool_type):                
                output.append(F.adaptive_max_pool2d(im, size))
     
        output = torch.stack(output, 0)

        return output