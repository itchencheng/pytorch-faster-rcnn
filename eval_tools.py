from __future__ import division

from collections import defaultdict
import itertools
import numpy as np
import six

from model import *
from data import *

import sys

import torch

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def calc_detection_voc_ap(prec, rec, use_07_metric=False):

    n_fg_class = len(prec)

    #print(prec)
    #print(rec)

    ap = np.empty(n_fg_class)
    for lbl in range(n_fg_class):
        if (prec[lbl] is None) or (rec[lbl] is None):
            ap[lbl] = np.nan
            continue

        if use_07_metric:
            # 11 point metric
            ap[l] = 0
            for t in np.arange(0., 1.1, 0.1):
                if np.sum(rec[l] >= t) == 0:
                    p = 0
                else:
                    p = np.max(np.nan_to_num(prec[l])[rec[l] >= t])
                ap[l] += p / 11
        else:
            # correct AP calculation
            # first append sentinel values at the end
            mpre = np.concatenate(([0], np.nan_to_num(prec[l]), [0]))
            mrec = np.concatenate(([0], rec[l], [1]))

            mpre = np.maximum.accumulate(mpre[::-1])[::-1]

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
            ap[l] = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap




def calc_detection_voc_prec_rec(
        pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels,
        gt_difficults=None,
        iou_thresh=0.5):

    pred_bboxes = iter(pred_bboxes)
    pred_labels = iter(pred_labels)
    pred_scores = iter(pred_scores)
    gt_bboxes = iter(gt_bboxes)
    gt_labels = iter(gt_labels)
    if gt_difficults is None:
        gt_difficults = itertools.repeat(None)
    else:
        gt_difficults = iter(gt_difficults)

    n_pos = defaultdict(int)
    score = defaultdict(list)
    match = defaultdict(list)

    # for each input image
    for pred_bbox, pred_label, pred_score, gt_bbox, gt_label, gt_difficult in \
                                                six.moves.zip(
                                                    pred_bboxes, pred_labels, pred_scores,
                                                    gt_bboxes, gt_labels, gt_difficults):

        if gt_difficult is None:
            gt_difficult = np.zeros(gt_bbox.shape[0], dtype=bool)

        # For the label "lbl"
        for lbl in np.unique(np.concatenate((pred_label, gt_label)).astype(int)):
            # 筛选当前类别lbl的预测结果pred_xx
            pred_mask_l = (pred_label == lbl)
            pred_bbox_l = pred_bbox[pred_mask_l]
            pred_score_l = pred_score[pred_mask_l]
            # 并按照(置信度)排序
            order = pred_score_l.argsort()[::-1]
            pred_bbox_l = pred_bbox_l[order]
            pred_score_l = pred_score_l[order]

            # 筛选当前类别lbl的ground truth
            gt_mask_l = (gt_label == lbl)
            gt_bbox_l = gt_bbox[gt_mask_l]
            gt_difficult_l = gt_difficult[gt_mask_l]
            
            # n_pos[lbl]: 记录lbl类别的ground truth个数
            n_pos[lbl] += np.logical_not(gt_difficult_l).sum() # the number of non-difficult
            # score[lbl]: 记录lbl类别的预测置信度
            score[lbl].extend(pred_score_l)

            # 如果没有预测为lbl类别的，则跳过
            if len(pred_bbox_l) == 0:
                continue
            # 如果有lbl的ground truth，对于预测为lbl的判断，match都为0
            if len(gt_bbox_l) == 0:
                match[lbl].extend((0,) * pred_bbox_l.shape[0])
                continue

            # VOC evaluation follows integer typed bounding boxes.
            pred_bbox_l = pred_bbox_l.copy()
            pred_bbox_l[:, 2:] += 1
            gt_bbox_l = gt_bbox_l.copy()
            gt_bbox_l[:, 2:] += 1
            # 计算IoU
            iou = calculate_iou(pred_bbox_l, gt_bbox_l)

            gt_index = iou.argmax(axis=1)
            # 如果IoU小于threshold则认为无效，记为-1
            gt_index[iou.max(axis=1) < iou_thresh] = -1
            del iou

            # 将selec初始化为全false
            selec = np.zeros(gt_bbox_l.shape[0], dtype=bool)
            for gt_idx in gt_index:
                # 对于IoU大于threshold的pred bbox
                if gt_idx >= 0:
                    # 如果是difficult，则match为-1
                    if gt_difficult_l[gt_idx]:
                        match[lbl].append(-1)
                    else:
                        # ground truth的个数固定，先到先得。
                        if not selec[gt_idx]:
                            match[lbl].append(1)
                        else:
                            match[lbl].append(0)
                    selec[gt_idx] = True
                else:
                    match[lbl].append(0)

    for iter_ in (pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels, gt_difficults):
        if next(iter_, None) is not None:
            raise ValueError('Length of input iterables need to be same.')

    n_fg_class = max(n_pos.keys()) + 1
    prec = [None] * n_fg_class
    rec = [None] * n_fg_class

    for lbl in n_pos.keys():
        score_l = np.array(score[lbl])
        match_l = np.array(match[lbl], dtype=np.int8)

        # 感觉本身那就拍好序的
        order = score_l.argsort()[::-1]
        match_l = match_l[order]

        # match为所有的预测数，排序后记录所有的tp和fp。
        tp = np.cumsum(match_l == 1) # 预测对的总bbox数
        fp = np.cumsum(match_l == 0) # 预测错的总bbox数

        # If an element of fp + tp is 0,
        # the corresponding element of prec[l] is nan.
        prec[lbl] = tp / (fp + tp)
        # If n_pos[lbl] is 0, rec[lbl] is None.
        if n_pos[lbl] > 0:
            rec[lbl] = tp / n_pos[lbl]

    return prec, rec


def eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels,
        gt_difficults=None,
        iou_thresh=0.5, use_07_metric=False):

    prec, rec = calc_detection_voc_prec_rec(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults,
        iou_thresh=iou_thresh)

    ap = calc_detection_voc_ap(prec, rec, use_07_metric=use_07_metric)

    return {'ap': ap, 'map': np.nanmean(ap)}


def evalx(test_dataloader, faster_rcnn, test_num=10000):
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()

    for ii, (imgs_, gt_bboxes_, gt_labels_, gt_difficults_, img_info_) in enumerate(test_dataloader):

        imgs_ = imgs_.to(device)

        '''
        It seems that:
        the dataloader would add a batch axis, and change to tensor
        '''
        gt_bboxes_ = gt_bboxes_[0].numpy()
        gt_labels_ = gt_labels_[0].numpy()
        gt_difficults_ = gt_difficults_[0].numpy()
        img_info_ = img_info_[0].numpy()

        # run
        roi_cls_locs, roi_scores, rois, roi_indices = faster_rcnn.inference(imgs_, img_info_)

        # postprodess
        pred_bboxes_, pred_labels_, pred_scores_  = postprocess(imgs_, \
                                                roi_cls_locs, roi_scores, rois, roi_indices)

        gt_bboxes.append(gt_bboxes_)
        gt_labels.append(gt_labels_)
        gt_difficults.append(gt_difficults_)

        pred_bboxes.append(pred_bboxes_)
        pred_labels.append(pred_labels_)
        pred_scores.append(pred_scores_)

        if ii == test_num: 
            break

    result = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults,
        use_07_metric=True)

    return result