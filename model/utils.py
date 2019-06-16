
import numpy as np



def normal_init(m, mean, stddev, truncated=False):
    """ weight initalizer: truncated normal and random normal """
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()
        

# Note:
# ratio = h / w
# generate anchors
# [[ -84.  -40.  100.   56.]
#  [-176.  -88.  192.  104.]
#  [-360. -184.  376.  200.]
#  [ -56.  -56.   72.   72.]
#  [-120. -120.  136.  136.]
#  [-248. -248.  264.  264.]
#  [ -36.  -80.   52.   96.]
#  [ -80. -168.   96.  184.]
#  [-168. -344.  184.  360.]]

def generate_anchors(base_size=16, anchor_scales=[8, 16, 32], anchor_ratios=[0.5, 1., 2.]):
    base_anchor = np.array([0, 0, base_size, base_size])
    w = base_anchor[2] - base_anchor[0]
    h = base_anchor[3] - base_anchor[1]

    ctr_x = base_anchor[0] + 0.5 * w
    ctr_y = base_anchor[1] + 0.5 * h

    size = w * h
    size_ratios = size / anchor_ratios

    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * anchor_ratios)

    anchor_scales = np.asarray(anchor_scales)

    ws = np.hstack([ws[i]*anchor_scales  for i in range(len(ws))])
    hs = np.hstack([hs[i]*anchor_scales  for i in range(len(hs))])

    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]

    anchors = np.hstack(((ctr_x - 0.5 * ws),
                         (ctr_y - 0.5 * hs),
                         (ctr_x + 0.5 * ws),
                         (ctr_y + 0.5 * hs)))

    return anchors


def bbox_transform(ex_rois, gt_rois):
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0]
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1]
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0]
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1]
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)

    targets = np.vstack(
        (targets_dx, targets_dy, targets_dw, targets_dh)).transpose()
    return targets


def bbox_transform_inv(anchors, deltas):
    if anchors.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

    anchors = anchors.astype(deltas.dtype, copy=False)

    widths = anchors[:, 2] - anchors[:, 0]
    heights = anchors[:, 3] - anchors[:, 1]
    ctr_x = anchors[:, 0] + 0.5 * widths
    ctr_y = anchors[:, 1] + 0.5 * heights

    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes


def clip_boxes(boxes, im_shape):
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
    return boxes


def filter_boxes(boxes, min_size):
    """Remove all boxes with any side smaller than min_size."""
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    keep = np.where((ws >= min_size) & (hs >= min_size))[0]
    return keep


def my_non_maximum_suppression(bboxes, nms_thresh, probs):

    if (bboxes.shape[0] == 0):
        return []

    max_min_order = np.argsort(probs)[::-1].ravel()
    print('max_min_order', max_min_order.shape)
    ordered_bboxes = bboxes[max_min_order]
    ordered_probs = probs[max_min_order]

    print('ordered_bboxes', ordered_bboxes.shape)
    print('ordered_probs', ordered_probs.shape)

    max_bbox = ordered_bboxes[0]

    keep = np.array([], np.int64)

    valid_flag = np.ones(len(max_min_order), dtype=np.int32)

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

        keep = np.append(keep, max_min_order[i])

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



def get_bbox_regression_labels(bbox_target_data, num_classes):
    '''
    Input: box regression targets (bbox_target_data) are in a format:
        N x (class, tx, ty, tw, th)
    The function is to change the target format, to confront with CNN output.
        Make the bbox_target N x (K x 4).
        K is the num_of_classes, equal to 21.

    '''
    class_info = bbox_target_data[:, 4]

    bbox_targets = np.zeros((class_info.size, 4*num_classes), dtype=np.float32)
    
    bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    
    inds = np.where(class_info > 0)[0]
    print('inds', inds)

    for ind in inds:
        clas = int(class_info[ind])
        start = 4 * clas
        end = start + 4
        bbox_targets[ind, start:end] = bbox_target_data[ind, :4]
        bbox_inside_weights[ind, start:end] = (1., 1., 1., 1.)

    return bbox_targets, bbox_inside_weights