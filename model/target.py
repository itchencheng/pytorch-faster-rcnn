
import numpy as np
from utils import *


class AnchorTargetLayer(object):
    def __init__(self):
        self.feat_stride = 16
        self.anchor_scales = [8, 16, 32]
        self.anchor_ratios = [0.5, 1., 2.]
        self.anchors = generate_anchors(self.feat_stride, self.anchor_scales, self.anchor_ratios)
        self.n_anchors = self.anchors.shape[0]

        self.allowed_border = 0

        self.POSITIVE_OVERLAP = 0.7
        self.NEGATIVE_OVERLAP = 0.3

        self.RPN_BATCHSIZE = 256
        self.RPN_POSITIVE_RATIO = 0.5

        print("init")

    def __call__(self, rpn_cls_score, gt_bboxes, img, im_info):

        # input
        # 0 rpn_cls_score 
        # 1 gt_bboxes 
        # 2 im_infp 
        # 3 img
        
        height, width = rpn_cls_score.shape[-2:]

        # Enumerate all shifts
        shift_x = np.arange(0, width) * self.feat_stride
        shift_y = np.arange(0, height) * self.feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                            shift_x.ravel(), shift_y.ravel())).transpose()

        # Enumerate all shifted anchors:
        #
        # add A anchors (1, A, 4) to
        # cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)
        # reshape to (K*A, 4) shifted anchors
        A = self.n_anchors
        K = shifts.shape[0]
        anchors = self.anchors.reshape((1, A, 4)) + \
                  shifts.reshape((1, K, 4)).transpose((1, 0, 2))
        all_anchors = anchors.reshape((K * A, 4))

        n_all_anchors = int(K * A)

        print('all_anchors', all_anchors.shape)

        # only keep anchors inside image

        inds_inside = np.where(
            (all_anchors[:, 0] >= -self.allowed_border) &
            (all_anchors[:, 1] >= -self.allowed_border) &
            (all_anchors[:, 2] < im_info[1] + self.allowed_border) &  # width
            (all_anchors[:, 3] < im_info[0] + self.allowed_border)    # height
        )[0]
        
        anchors = all_anchors[inds_inside, :]

        print('anchors', anchors.shape)

        # label: 1 is positive, 0 is negative, -1 is dont care
        labels = np.empty((len(inds_inside), ), dtype=np.int32)
        labels.fill(-1)

        # calculate iou between anchors/gt_bboxes
        iou_values = calculate_iou(anchors, gt_bboxes[:,:4])

        # Now, begin to set labels
        # 1. find bboxes with high iou for each anchor
        argmax_overlaps = iou_values.argmax(axis=1)
        max_overlaps = iou_values[np.arange(len(inds_inside)), argmax_overlaps]
        # 2. find anchor with high iou for each bbox
        gt_argmax_overlaps = iou_values.argmax(axis=0)
        gt_max_overlaps = iou_values[gt_argmax_overlaps, np.arange(iou_values.shape[1])]


        
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


        bbox_targets = np.zeros((len(inds_inside), 4), dtype=np.float32)
        bbox_targets = bbox_transform(anchors, gt_bboxes[argmax_overlaps, :])

        # inside weight for smooth_l1_loss
        #  positive sample is [1, 1, 1, 1]
        #  neigtive sample is [0, 0, 0, 0] 
        bbox_inside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
        bbox_inside_weights[labels == 1, :] = np.array([1, 1, 1, 1])

        # outside weigth for smooth_l1_loss
        bbox_outside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
        num_examples = np.sum(labels >= 0)
        positive_weights = np.ones((1, 4)) * 1.0 / num_examples
        negative_weights = np.ones((1, 4)) * 1.0 / num_examples
        bbox_outside_weights[labels == 1, :] = positive_weights
        bbox_outside_weights[labels == 0, :] = negative_weights


        # map up to original set of anchors
        labels = unmap(labels, n_all_anchors, inds_inside, fill=-1)
        bbox_targets = unmap(bbox_targets, n_all_anchors, inds_inside, fill=0)
        bbox_inside_weights = unmap(bbox_inside_weights, n_all_anchors, inds_inside, fill=0)
        bbox_outside_weights = unmap(bbox_outside_weights, n_all_anchors, inds_inside, fill=0)


        # labels
        labels = labels.reshape((1, height, width, A)).transpose(0, 3, 1, 2)
        rpn_labels = labels.reshape((1, 1, A * height, width))


        # bbox_targets
        rpn_bbox_targets = bbox_targets \
            .reshape((1, height, width, A * 4)).transpose(0, 3, 1, 2)

        # bbox_inside_weights
        rpn_bbox_inside_weights = bbox_inside_weights \
            .reshape((1, height, width, A * 4)).transpose(0, 3, 1, 2)
        assert rpn_bbox_inside_weights.shape[2] == height
        assert rpn_bbox_inside_weights.shape[3] == width

        # bbox_outside_weights
        rpn_bbox_outside_weights = bbox_outside_weights \
            .reshape((1, height, width, A * 4)).transpose(0, 3, 1, 2)
        assert rpn_bbox_outside_weights.shape[2] == height
        assert rpn_bbox_outside_weights.shape[3] == width

        return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights




class ProposalTargetLayer(object):
    def __init__(self):
        # Minibatch size (number of regions of interest [ROIs])
        self.BATCH_SIZE = 128

        # Fraction of minibatch that is labeled foreground (i.e. class > 0)
        self.FG_FRACTION = 0.25

        self.FG_THRESH = 0.5

        self.BG_THRESH_HI = 0.5
        self.BG_THRESH_LO = 0.1

        self.num_classes = 21

    def __call__(self, all_rois, gt_boxes):
        # Proposal ROIs (0, x1, y1, x2, y2) coming from RPN
        # GT boxes (x1, y1, x2, y2, label)
        
        # Note: chencheng, not understand
        '''
        # Include ground-truth boxes in the set of candidate rois
        zeros = np.zeros((gt_boxes.shape[0], 1), dtype=gt_boxes.dtype)
        all_rois = np.vstack(
            (all_rois, np.hstack((zeros, gt_boxes[:, :-1])))
        )
        '''

        print(gt_boxes)

        num_images = 1
        rois_per_image = self.BATCH_SIZE / num_images
        fg_rois_per_image = np.round(self.FG_FRACTION * rois_per_image)


        # calculate iou between roi/gt_bboxes
        iou_values = calculate_iou(all_rois, gt_boxes)
        # find its gt_bbox, for each roi
        gt_assignment = iou_values.argmax(axis=1)
        # get the biggest gt_bbox IoU, for each roi
        max_overlaps = iou_values.max(axis=1)
        # the label, for the gt_bboxes
        labels = gt_boxes[gt_assignment, 4]
        print(labels)

        # Select foreground RoIs as those with >= FG_THRESH overlap
        #    if a roi has a RoI >= 0.5, it is fore ground.
        fg_inds = np.where(max_overlaps >= self.FG_THRESH)[0]
        fg_rois_of_this_image = min(fg_rois_per_image, fg_inds.size)
        if fg_inds.size > 0:
            fg_inds = np.random.choice(fg_inds, size=fg_rois_of_this_image, replace=False)


        # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
        bg_inds = np.where((max_overlaps >= self.BG_THRESH_LO) & 
                            (max_overlaps < self.BG_THRESH_HI))[0]
        bg_rois_of_this_image = rois_per_image - fg_rois_of_this_image
        bg_rois_of_this_image = min(bg_rois_of_this_image, bg_inds.size)
        if bg_inds.size > 0:
            bg_inds = npr.choice(bg_inds, size=bg_rois_of_this_image, replace=False)


        # The indices that we're selecting (both fg and bg)
        keep_inds = np.append(fg_inds, bg_inds)

        # Select sampled values from various arrays:
        labels = labels[keep_inds]

        # Clamp labels for the background RoIs to 0
        labels[fg_rois_of_this_image:] = 0
        rois = all_rois[keep_inds]


        # transform bbox
        bbox_target_data = bbox_transform(rois[:, 1:5], gt_boxes[gt_assignment[keep_inds], :4])
        bbox_target_data = np.hstack((labels[:, np.newaxis],
                                      bbox_target_data)).astype(np.float32, copy=False)

        bbox_targets, bbox_inside_weights = \
            get_bbox_regression_labels(bbox_target_data, self.num_classes)


        bbox_outside_weights = np.array(bbox_inside_weights > 0)

        return rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights
