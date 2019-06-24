
import torch
import numpy as np
from utils import *


# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class AnchorTargetLayer(object):
    def __init__(self):
        self.POSITIVE_OVERLAP = 0.7
        self.NEGATIVE_OVERLAP = 0.3

        self.RPN_BATCHSIZE = 256
        self.RPN_POSITIVE_RATIO = 0.5

        print("init")

    def __call__(self, gt_bboxes, anchors, img_size):
        
        img_H, img_W = img_size
        n_anchors = len(anchors)

        ### only keep anchors inside image
        inds_inside = np.where(
            (anchors[:, 0] >= 0) &
            (anchors[:, 1] >= 0) &
            (anchors[:, 2] <= img_H) &  # width
            (anchors[:, 3] <= img_W)    # height
        )[0]        
        anchors = anchors[inds_inside, :]

        ### create labels
        # label: 1 is positive, 0 is negative, -1 is dont care
        labels = np.empty((len(inds_inside), ), dtype=np.long)
        labels.fill(-1)

        # calculate iou between anchors/gt_bboxes
        print('anchors', anchors.shape)
        print('gt_bboxes', gt_bboxes.shape)
        iou_values = calculate_iou(anchors, gt_bboxes)


        # Now, begin to set labels
        # 1. find bboxes with high iou for each anchor
        argmax_overlaps = iou_values.argmax(axis=1)
        max_overlaps = iou_values[np.arange(len(inds_inside)), argmax_overlaps]
        # 2. find anchor with high iou for each bbox
        gt_argmax_overlaps = iou_values.argmax(axis=0)
        gt_max_overlaps = iou_values[gt_argmax_overlaps, np.arange(iou_values.shape[1])]
        ################################################################################
        gt_argmax_overlaps = np.where(iou_values == gt_max_overlaps)[0]


        # Positive label
        # (i) the anchor with hightest IOU overlap for a groundtruth bbox
        labels[gt_argmax_overlaps] = 1
        # (ii) the anchor has a IOU > 0.7, for any groundtruth bbox
        labels[max_overlaps > self.POSITIVE_OVERLAP] = 1
        # Negative label
        # (iii) non-positive anchors, if IoU ratio < 0.3 for all ground-truth boxes 
        labels[max_overlaps < self.NEGATIVE_OVERLAP] = 0

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
        bbox_targets = bbox_transform(anchors, gt_bboxes[argmax_overlaps, :])

        # map up to original set of anchors
        labels = unmap(labels, n_anchors, inds_inside, fill=-1)
        bbox_targets = unmap(bbox_targets, n_anchors, inds_inside, fill=0)

        return bbox_targets, labels



class ProposalTargetLayer(object):
    def __init__(self):
        # Minibatch size (number of regions of interest [ROIs])
        self.N_SAMPLE = 128

        # Fraction of minibatch that is labeled foreground (i.e. class > 0)
        self.POS_RATIO = 0.25

        self.POS_IOU_THRESH = 0.5

        self.NEG_IOU_THRESH_HI = 0.5
        self.NEG_IOU_THRESH_LO = 0.0

        self.num_classes = 21

    def __call__(self, all_rois, gt_boxes, labels, loc_normalize_mean, loc_normalize_std):
        # Proposal ROIs (batchIdx, y1, x2, y2, label), coming from RPN
        # GT boxes (x1, y1, x2, y2, label)

        print('all_rois', all_rois.shape, all_rois.dtype)
        print(all_rois)
        print('gt_boxes', gt_boxes.shape, gt_boxes.dtype)
        print(gt_boxes)
        print('labels', labels.shape, labels.dtype)
        print(labels)
        print('loc_normalize_mean', loc_normalize_mean)
        print('loc_normalize_std', loc_normalize_std)


        # Include ground-truth boxes in the set of candidate rois
        all_rois = np.concatenate((all_rois, gt_boxes), axis=0)

        n_pos_rois_per_image = np.round(self.POS_RATIO * self.N_SAMPLE)

        # calculate iou between roi/gt_boxes
        iou_values = calculate_iou(all_rois, gt_boxes)

        # find its gt_bbox index, for each roi
        gt_assignment = iou_values.argmax(axis=1)
        # get the biggest gt_bbox IoU value, for each roi
        max_overlaps = iou_values.max(axis=1)
        
        # the label, for the gt_boxes
        # Note: 0 means background, 1-20 for 20 classes
        gt_roi_labels = labels[gt_assignment] + 1


        # Select foreground RoIs as those with >= FG_THRESH overlap
        #    if a roi has a RoI >= 0.5, it is fore ground.
        fg_inds = np.where(max_overlaps >= self.POS_IOU_THRESH)[0]
        fg_rois_of_this_image = int(min(n_pos_rois_per_image, fg_inds.size))
        if fg_inds.size > 0:
            fg_inds = np.random.choice(fg_inds, size=fg_rois_of_this_image, replace=False)

        # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
        bg_inds = np.where((max_overlaps >= self.NEG_IOU_THRESH_LO) & 
                            (max_overlaps < self.NEG_IOU_THRESH_HI))[0]
        bg_rois_of_this_image = self.N_SAMPLE - fg_rois_of_this_image
        bg_rois_of_this_image = int(min(bg_rois_of_this_image, bg_inds.size))
        if bg_inds.size > 0:
            bg_inds = np.random.choice(bg_inds, size=bg_rois_of_this_image, replace=False)

        # The indices that we're selecting (both fg and bg)
        keep_inds = np.append(fg_inds, bg_inds)

        # Select sampled values from various arrays:
        gt_roi_labels = gt_roi_labels[keep_inds]

        # Clamp labels for the background RoIs to 0
        gt_roi_labels[fg_rois_of_this_image:] = 0
        sample_rois = all_rois[keep_inds]

        # transform bbox
        gt_roi_locs = bbox_transform(sample_rois, gt_boxes[gt_assignment[keep_inds]])
        # normalize gt_roi_locs
        gt_roi_locs = (gt_roi_locs - np.array(loc_normalize_mean, np.float32)) / \
                        np.array(loc_normalize_std, np.float32)

        return sample_rois, gt_roi_locs, gt_roi_labels
