#coding:utf-8
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import vgg16

from utils import *

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
