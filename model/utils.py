
from torchvision import transforms as transforms
from skimage import transform as sktsf
import numpy as np
import torch
import torch.nn.functional as F


# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def debug(name, x):
    print(name, x.shape, x.dtype)
    print(x)


def normal_init(m, mean, stddev, truncated=False):
    """ weight initalizer: truncated normal and random normal """

    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()
    '''
    m.weight.data.fill_(0.02)
    m.bias.data.zero_()
    '''    
        

# ======================= pre-post process ===========================
def pytorch_normalze(img):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    img = normalize(torch.from_numpy(img).float())
    return img.numpy()


def preprocess(img, g_bbox):
    min_size = 600
    max_size = 1000

    # image
    C, H, W = img.shape

    scale_min = float(min_size) / np.min((H, W))
    scale_max = float(max_size) / np.max((H, W))
    '''
    scale_min = min_size / min(H, W)
    scale_max = max_size / max(H, W)
    '''
    scale = min(scale_min, scale_max)

    img = img / 255.
    img = sktsf.resize(img, (C, H*scale, W*scale), mode='reflect', anti_aliasing=False)
    img = pytorch_normalze(img)

    _, o_H, o_W = img.shape

    # bbox
    g_bbox[:,0::2] = float(o_H) / H * g_bbox[:,0::2]
    g_bbox[:,1::2] = float(o_W) / W * g_bbox[:,1::2]

    img_info = np.array((img.shape[-2], img.shape[-1], scale)).astype(np.float32)

    # to torch.Tensor
    img = torch.from_numpy(img)

    return img, g_bbox, img_info





def suppress(n_class, raw_cls_bbox, raw_prob, nms_thresh, score_thresh):
    bbox = list()
    label = list()
    score = list()

    raw_cls_bbox = raw_cls_bbox.reshape((-1, n_class, 4))

    print('raw_cls_bbox', raw_cls_bbox.shape)
    print('raw_prob', raw_prob.shape)

    # skip cls_id = 0 because it is the background class
    for l in range(1, n_class):
        cls_bbox_l = raw_cls_bbox[:, l, :]
        prob_l = raw_prob[:, l]

        mask = prob_l > score_thresh
        cls_bbox_l = cls_bbox_l[mask]
        prob_l = prob_l[mask]

        if (len(cls_bbox_l) == 0):
            continue

        sorted_idx = np.argsort(prob_l)[::-1]
        sorted_bbox = cls_bbox_l[sorted_idx]
        sorted_prob = prob_l[sorted_idx]

        keep = my_non_maximum_suppression(sorted_bbox, nms_thresh)

        # The labels are in [0, n_class - 2].
        bbox.append(sorted_bbox[keep])
        label.append((l - 1) * np.ones((len(keep),)))
        score.append(sorted_prob[keep])

    if (len(bbox) == 0):
        print('no prediction get!')
        return np.array([[0,0,0,0]], dtype=np.float32), \
                np.array([0], dtype=np.int32), \
                np.array([0], dtype=np.float32) 

    bbox = np.concatenate(bbox, axis=0).astype(np.float32)
    label = np.concatenate(label, axis=0).astype(np.int32)
    score = np.concatenate(score, axis=0).astype(np.float32)

    print('bbox', bbox)
    print('label', label)
    print('score', score)

    return bbox, label, score


def xsuppress(n_class, raw_cls_bbox, raw_prob, nms_thresh, score_thresh):
    bbox = list()
    label = list()
    score = list()

    # dont care idx=0
    raw_prob[:,0] = 0

    raw_cls_bbox = raw_cls_bbox.reshape((-1, n_class, 4))
    raw_prob_idx = np.argmax(raw_prob, axis=1)

    # select max
    raw_cls_bbox = raw_cls_bbox[np.arange(len(raw_prob_idx)), raw_prob_idx]
    raw_prob = raw_prob[np.arange(len(raw_prob_idx)), raw_prob_idx]

    # threshold
    keep = raw_prob > score_thresh
    cls_bbox = raw_cls_bbox[keep]
    cls_prob = raw_prob[keep]
    cls_pred = raw_prob_idx[keep]

    if (len(keep) == 0):
        return

    # nms
    sorted_idx = np.argsort(cls_prob)[::-1]
    sorted_bbox = cls_bbox[sorted_idx]
    sorted_prob = cls_prob[sorted_idx]
    sorted_pred = cls_pred[sorted_idx]

    keep = my_non_maximum_suppression(sorted_bbox, nms_thresh)

    if (len(keep) == 0):
        print('no prediction get!')
        return np.array([[0,0,0,0]], dtype=np.float32), \
                np.array([0], dtype=np.int32), \
                np.array([0], dtype=np.float32) 

    bbox = sorted_bbox[keep]
    label = sorted_pred[keep]-1
    score = sorted_prob[keep]

    return bbox, label, score


def postprocess(data_in, roi_cls_locs, roi_scores, rois, roi_indices):
    n_class = 21
    loc_normalize_mean = (0., 0., 0., 0.)
    loc_normalize_std = (0.1, 0.1, 0.2, 0.2)

    nms_thresh = 0.3
    score_thresh = 0.05

    size = data_in.shape[-2:]
    print(size)
    mean = torch.Tensor(loc_normalize_mean).to(device).repeat(n_class)
    std  = torch.Tensor(loc_normalize_std).to(device).repeat(n_class)

    roi = torch.Tensor(rois)
    print(type(rois))
    print(type(roi))
    roi = roi.view(-1, 1, 4)
    roi_cls_loc = roi_cls_locs * std + mean
    roi_cls_loc = roi_cls_loc.view(-1, n_class, 4)
    roi = roi.view(-1, 1, 4).expand_as(roi_cls_loc)

    cls_bbox = bbox_transform_inv(roi.detach().cpu().numpy().reshape((-1, 4)), roi_cls_loc.detach().cpu().numpy().reshape(-1, 4))
    cls_bbox = torch.Tensor(cls_bbox).view(-1, n_class*4)

    cls_bbox[:, 0::2] = (cls_bbox[:, 0::2]).clamp(min=0, max=size[0])
    cls_bbox[:, 1::2] = (cls_bbox[:, 1::2]).clamp(min=0, max=size[1])

    prob = F.softmax(roi_scores, dim=1)

    raw_cls_bbox = cls_bbox.detach().numpy()
    raw_prob = prob.detach().cpu().numpy()

    bbox, label, score = suppress(n_class, raw_cls_bbox, raw_prob, nms_thresh, score_thresh)

    return bbox, label, score


#==================================================================

# Note:
# ratio = h / w
# generate anchors
# [[ -37.254834    -82.50966799   53.254834     98.50966799]
#  [ -82.50966799 -173.01933598   98.50966799  189.01933598]
#  [-173.01933598 -354.03867197  189.01933598  370.03867197]
#  [ -56.          -56.           72.           72.        ]
#  [-120.         -120.          136.          136.        ]
#  [-248.         -248.          264.          264.        ]
#  [ -82.50966799  -37.254834     98.50966799   53.254834  ]
#  [-173.01933598  -82.50966799  189.01933598   98.50966799]
#  [-354.03867197 -173.01933598  370.03867197  189.01933598]]

def generate_anchors(base_size=16, anchor_scales=[8, 16, 32], anchor_ratios=[0.5, 1., 2.]):
    base_anchor = np.array([0, 0, base_size, base_size])
    w = base_anchor[2] - base_anchor[0]
    h = base_anchor[3] - base_anchor[1]

    ctr_x = base_anchor[0] + 0.5 * w
    ctr_y = base_anchor[1] + 0.5 * h

    size = w * h
    size_ratios = size / anchor_ratios

    ws = np.sqrt(size_ratios)
    hs = ws * anchor_ratios

    anchor_scales = np.asarray(anchor_scales)

    ws = np.hstack([ws[i]*anchor_scales  for i in range(len(ws))])
    hs = np.hstack([hs[i]*anchor_scales  for i in range(len(hs))])

    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]

    anchors = np.hstack(((ctr_y - 0.5 * hs),
                         (ctr_x - 0.5 * ws),
                         (ctr_y + 0.5 * hs),
                         (ctr_x + 0.5 * ws))).astype(np.float32)

    return anchors


def enumerate_shifted_anchor(anchors, feat_stride, height, width):
    # Enumerate all shifted anchors:
    #
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors

    # Enumerate all shifts
    shift_x = np.arange(0, width) * feat_stride
    shift_y = np.arange(0, height) * feat_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shift_y.ravel(), shift_x.ravel(),
                        shift_y.ravel(), shift_x.ravel())).transpose()

    A = len(anchors)
    K = shifts.shape[0]
    anchors = anchors.reshape((1, A, 4)) + \
              shifts.reshape((1, K, 4)).transpose((1, 0, 2))
    all_anchors = anchors.reshape((K * A, 4)).astype(np.float32)

    return all_anchors


def bbox_transform(ex_rois, gt_rois):
    ex_heights = ex_rois[:, 2] - ex_rois[:, 0]
    ex_widths = ex_rois[:, 3] - ex_rois[:, 1]
    ex_ctr_y = ex_rois[:, 0] + 0.5 * ex_heights
    ex_ctr_x = ex_rois[:, 1] + 0.5 * ex_widths

    gt_heights = gt_rois[:, 2] - gt_rois[:, 0]
    gt_widths = gt_rois[:, 3] - gt_rois[:, 1]

    gt_ctr_y = gt_rois[:, 0] + 0.5 * gt_heights
    gt_ctr_x = gt_rois[:, 1] + 0.5 * gt_widths

    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)

    targets = np.vstack(
        (targets_dy, targets_dx, targets_dh, targets_dw)).transpose()
    return targets


def bbox_transform_inv(anchors, deltas):
    if anchors.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

    anchors = anchors.astype(deltas.dtype, copy=False)

    heights = anchors[:, 2] - anchors[:, 0]
    widths = anchors[:, 3] - anchors[:, 1]
    ctr_y = anchors[:, 0] + 0.5 * heights
    ctr_x = anchors[:, 1] + 0.5 * widths

    dy = deltas[:, 0::4]
    dx = deltas[:, 1::4]
    dh = deltas[:, 2::4]
    dw = deltas[:, 3::4]

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    # y1
    pred_boxes[:, 0::4] = pred_ctr_y - 0.5 * pred_h
    # x1
    pred_boxes[:, 1::4] = pred_ctr_x - 0.5 * pred_w
    # y2
    pred_boxes[:, 2::4] = pred_ctr_y + 0.5 * pred_h
    # x2
    pred_boxes[:, 3::4] = pred_ctr_x + 0.5 * pred_w

    return pred_boxes


def clip_boxes(boxes, im_shape):
    h, w = im_shape[:2]
    # y1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], h), 0)
    # x1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], w), 0)
    # y2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], h), 0)
    # x2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], w), 0)
    return boxes


def filter_boxes(boxes, min_size):
    """Remove all boxes with any side smaller than min_size."""
    ws = boxes[:, 2] - boxes[:, 0]
    hs = boxes[:, 3] - boxes[:, 1]
    keep = np.where((ws >= min_size) & (hs >= min_size))[0]
    return keep


def my_non_maximum_suppression(bboxes, nms_thresh):

    if (bboxes.shape[0] == 0):
        return []

    ordered_bboxes = bboxes

    keep = np.array([], np.int64)

    valid_flag = np.ones(len(bboxes), dtype=np.int32)

    i = 0
    for i in range(len(ordered_bboxes)):
        if (valid_flag[i] == 0): 
            continue

        max_bbox = ordered_bboxes[i]
        ordered_bbox = ordered_bboxes

        two_area = (ordered_bbox[:,2]-ordered_bbox[:,0]) * (ordered_bbox[:,3]-ordered_bbox[:,1]) + \
                   (max_bbox[2]-max_bbox[0]) * (max_bbox[3]-max_bbox[1])

        left = np.maximum(ordered_bbox[:,0], max_bbox[0]*np.ones(ordered_bbox[:,0].shape))
        right = np.minimum(ordered_bbox[:,2], max_bbox[2]*np.ones(ordered_bbox[:,0].shape))
        top = np.maximum(ordered_bbox[:,1], max_bbox[1]*np.ones(ordered_bbox[:,0].shape))
        bottom = np.minimum(ordered_bbox[:,3], max_bbox[3]*np.ones(ordered_bbox[:,0].shape))
        height = np.maximum(bottom-top, np.zeros(ordered_bbox[:,0].shape))
        width = np.maximum(right-left, np.zeros(ordered_bbox[:,0].shape))
        
        area_i = height * width
        iou_value = area_i / (two_area - area_i)

        thresh_idx = iou_value > nms_thresh
        valid_flag[thresh_idx] = 0

        keep = np.append(keep, i)

    return keep


def calculate_iou(bboxes, refer_bboxes):
    # anchors: Kx4
    # bboxes:  Nx4
    # return: overlap, KxN
    n_bboxes = bboxes.shape[0]
    n_refer_bboxes  = refer_bboxes.shape[0]

    iou_value_list = []

    for i in range(n_refer_bboxes):

        refer_bbox = refer_bboxes[i]

        two_area = (bboxes[:,2]-bboxes[:,0]) * (bboxes[:,3]-bboxes[:,1]) + \
                    (refer_bbox[2]-refer_bbox[0]) * (refer_bbox[3]-refer_bbox[1])

        left = np.maximum(bboxes[:,0], refer_bbox[0]*np.ones(bboxes[:,0].shape))
        right = np.minimum(bboxes[:,2], refer_bbox[2]*np.ones(bboxes[:,0].shape))
        top = np.maximum(bboxes[:,1], refer_bbox[1]*np.ones(bboxes[:,0].shape))
        bottom = np.minimum(bboxes[:,3], refer_bbox[3]*np.ones(bboxes[:,0].shape))
        height = np.maximum(bottom-top, np.zeros(bboxes[:,0].shape))
        width = np.maximum(right-left, np.zeros(bboxes[:,0].shape))
        
        area_i = height * width
        iou_value = area_i / (two_area - area_i)

        iou_value_list.append(iou_value)

    iou_values = np.vstack(iou_value_list).transpose()

    return iou_values

"""
def bbox_iou(bbox_a, bbox_b):

    if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
        raise IndexError

    # top left
    tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
    # bottom right
    br = np.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:])

    area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)
    area_a = np.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)
    area_b = np.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)
    return area_i / (area_a[:, None] + area_b - area_i)
"""

def unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if len(data.shape) == 1:
        ret = np.empty((count, ), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count, ) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data
    return ret


