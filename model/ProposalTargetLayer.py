

import torch
import numpy as np
from utils import *

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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

        # Include ground-truth boxes in the set of candidate rois
        all_rois = np.concatenate((all_rois, gt_boxes), axis=0)

        # calculate iou between roi/gt_boxes
        iou_values = calculate_iou(all_rois, gt_boxes)

        # find its gt_bbox index, for each roi
        gt_assignment = iou_values.argmax(axis=1)
        # get the biggest gt_bbox IoU value, for each roi
        max_overlaps = iou_values.max(axis=1)
        
        # the label, for the gt_boxes
        # Note: 0 means background, 1-20 for 20 classes
        gt_roi_labels = labels[gt_assignment] + 1


        n_pos_rois_per_image = np.round(self.POS_RATIO * self.N_SAMPLE)

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
        gt_roi_locs = (gt_roi_locs - loc_normalize_mean) / loc_normalize_std

        return sample_rois, gt_roi_locs, gt_roi_labels
