
import torch
import numpy as np
from utils import *

#np.random.seed(23)

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class AnchorTargetLayer(object):
    def __init__(self):
        self.POSITIVE_OVERLAP = 0.7
        self.NEGATIVE_OVERLAP = 0.3

        self.RPN_BATCHSIZE = 256
        self.RPN_POSITIVE_RATIO = 0.5

    def __call__(self, gt_bboxes, anchors, img_size):
        
        img_H, img_W = img_size
        n_anchors = len(anchors)

        ### As paper said:
        # during training, we ignore all cross-boudary anchors,
        #  so, they don't contribute to the loss. Or, they will
        #  introduce larget error terms, training will not converge.
        inds_inside = np.where(
            (anchors[:, 0] >= 0) &
            (anchors[:, 1] >= 0) &
            (anchors[:, 2] <= img_H) &  # width
            (anchors[:, 3] <= img_W)    # height
        )[0]        
        anchors = anchors[inds_inside]

        ### create labels
        # label: 1 is positive, 0 is negative, -1 is dont care
        labels = np.empty((len(inds_inside), ), dtype=np.int32)
        labels.fill(-1)

        # calculate iou between anchors/gt_bboxes
        iou_values = calculate_iou(anchors, gt_bboxes)

        # Now, begin to set labels
        # 1. for each anchor, find bboxes with highest iou
        argmax_overlaps = iou_values.argmax(axis=1)
        max_overlaps = iou_values[np.arange(len(inds_inside)), argmax_overlaps]
        # 2. for each bbox, find anchor with highest iou
        gt_argmax_overlaps = iou_values.argmax(axis=0)
        gt_max_overlaps = iou_values[gt_argmax_overlaps, np.arange(iou_values.shape[1])]

        ################################################################################
        # gt_argmax_overlaps = np.where(iou_values == gt_max_overlaps)[0]

        # Negative label
        # (iii) non-positive anchors, if IoU ratio < 0.3 for all ground-truth boxes 
        labels[max_overlaps < self.NEGATIVE_OVERLAP] = 0

        # Positive label
        # (i) the anchor with hightest IOU overlap for a groundtruth bbox
        labels[gt_argmax_overlaps] = 1
        # (ii) the anchor has a IOU > 0.7, for any groundtruth bbox
        labels[max_overlaps >= self.POSITIVE_OVERLAP] = 1

        # choose Positvie & Negative
        # subsample positive labels if we have too many
        n_pos = int(self.RPN_POSITIVE_RATIO * self.RPN_BATCHSIZE)
        pos_index = np.where(labels == 1)[0]
        if len(pos_index) > n_pos:
            disable_index = np.random.choice(
                pos_index, size=(len(pos_index) - n_pos), replace=False)
            labels[disable_index] = -1

        # subsample negative labels if we have too many
        n_neg = self.RPN_BATCHSIZE - np.sum(labels == 1)
        neg_index = np.where(labels == 0)[0]
        if len(neg_index) > n_neg:
            disable_index = np.random.choice(
                neg_index, size=(len(neg_index) - n_neg), replace=False)
            labels[disable_index] = -1

        # Note:
        #   the target is to make each anchor close to its highest bbox
        bbox_targets = bbox_transform(anchors, gt_bboxes[argmax_overlaps])

        # map up to original set of anchors
        labels = unmap(labels, n_anchors, inds_inside, fill=-1)
        bbox_targets = unmap(bbox_targets, n_anchors, inds_inside, fill=0)

        return bbox_targets, labels

