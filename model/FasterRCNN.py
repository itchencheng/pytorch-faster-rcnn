#coding:utf-8


from torch import nn
from torchvision.models import vgg16



def decompose_vgg16(ckpt_path):
    vgg16_net = vgg16()
    vgg16_net.load_state_dict(torch.load(ckpt_path))

    features = list(vgg16_net.features)[:30]
    # freeze top four conv
    for layer in features[:10]:
        for p in layer.paramters():
            p.requires_grad = False

    classifier = vgg16_net.classifier
    classifier = nn.Sequential(*classifier)
    del classifier[6]
    if (not use_drop):
        del classifier[5]
        del classifier[2]

    features = nn.Sequential(*features)
    classifier = nn.Sequential(*classifier)

    return features, classifier



#==================================================================================================
def loc2bbox(src_bbox, loc):
    if (src_bbox.shape[0] == 0):
        return np.zeos((0, 4), dtype=loc.dtype)

    src_bbox = src_bbox.astype(src_bbox.dtype, copy=False)
    src_height = src_bbox[:, 2] - src_bbox[:, 0]
    src_width  = src_bbox[:, 3] - src_bbox[:, 1]
    src_ctr_y  = src_bbox[:, 0] + 0.5 * src_height
    src_ctr_x  = src_bbox[:, 1] + 0.5 * src_width

    dy = loc[:, 0::4]
    dx = loc[:, 1::4]
    dh = loc[:, 2::4]
    dw = loc[:, 3::4]

    ctr_y = dy * src_height[:, np.newaxis] + src_ctr_y[:, np.newaxis]
    ctr_x = dx * src_width[:, np.newaxis] + src_ctr_x[:, np.newaxis]
    h = np.exp(dh) * src_height[:, np.newaxis]
    w = np.exp(dw) * src_width[:, np.newaxis]

    dst_bbox = np.zeros(loc.shape, dtype=loc.dtype)
    dst_bbox[:, 0::4] = ctr_y - 0.5 * h
    dst_bbox[:, 1::4] = ctr_x - 0.5 * w
    dst_bbox[:, 2::4] = ctr_y + 0.5 * h
    dst_bbox[:, 3::4] = ctr_x + 0.5 * w

    return dst_bbox


def non_maximum_suppression(bbox, thresh, score=None, limit=None):
    if (len(bbox) == 0):
        return np.zeros((0,), dtype=np.int32)

    n_bbox = bbox.shape[0]

    if (score is not None):
        order = score.argsort()[::-1].astype(np.int32)
    else:
        order = np.arange(n_bbox, dtype=np.int32)

    sorted_bbox = bbox[order, :]
    selec, n_selec = 




class ProposalCreator:
    def __init__(self,
        parent_model,
        nms_thresh=0.7,
        n_train_pre_nms=12000,
        n_train_post_nms=2000,
        n_test_pre_nms=6000,
        n_test_post_nms=300,
        min_size=16
        ):
        self.parent_model = parent_model
        self.nms_thresh = nms_thresh
        self.n_train_pre_nms = n_train_pre_nms
        self.n_train_post_nms = n_train_post_nms
        self.n_test_pre_nms = n_test_pre_nms
        self.n_test_post_nms = n_test_post_nms

    # define __call__ function, make the class a runnable class
    def __call__(self, loc, score, anchor, img_size, scale=1.):
        if (self.parent_model.training):
            n_pre_nms = self.n_train_pre_nms
            n_post_nms = self.n_train_post_nms
        else:
            n_pre_nms = self.n_test_pre_nms
            n_post_nms = self.n_test_post_nms

        roi = loc2bbox(anchor, loc)

        roi[:, slice(0, 4, 2)] = np.clip(roi[:, slice(0, 4, 2)], 0, img_size[0])
        roi[:, slice(1, 4, 2)] = np.clip(roi[:, slice(1, 4, 2)], 0, img_size[1])

        min_size = self.min_size * scale
        hs = roi[:, 2] - roi[:, 0]
        ws = roi[:, 3] - roi[:, 1]
        keep = np.where((hs >= min_size) & (ws >= min_size))[0]
        roi = roi[keep, :]
        score = score[keep]

        order = score.ravel().argsort()[::-1]
        if (n_pre_nms > 0):
            order = order[:n_pre_nms]
        roi = roi[order, :]

        keep = non_maximum_suppression()

        if (n_post_nms > 0):
            keep = keep[:n_post_nms]
        roi = roi[keep]

        return roi



#==================================================================================================
""" Note:
    Generate approximiately (-h/2, -w/2, h/2, w/2) anchor box.
"""

def normal_init(m, mean, stddev, truncated=False):
    """
    weight initalizer: truncated normal and random normal.
    """
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()

class RegionProposalNetwork(nn.Module):

    def __init__(self, 
        in_channels=512, 
        mid_channels=512, 
        feature_stride=16,
        anchor_ratios=[0.5, 1, 2],
        anchor_scales=[8, 16, 32]
        ):
        super(RegionProposalNetwork, self).__init__()
        
        self.anchor_base = self.generate_base_anchors(feature_stride, anchor_ratios, anchor_scales)

        self.feature_stride=feature_stride
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.score = nn.Conv2d(mid_channels,  n_anchor*2, 1, 1, 0)
        self.locate = nn.Conv2d(mid_channels, n_anchor*4, 1, 1, 0)

        normal_init(self.conv1,  0, 0.01)
        normal_init(self.score,  0, 0.01)
        normal_init(self.locate, 0, 0.01)


    def forward(self, x, img_size, scale=1.):
        n, _, hh, ww = x.shape
        anchor = emuerate_shifted_anchor(np.array(self.anchor_base), self.feature_stride, hh, ww)
        n_anchor = anchor.shape[0] // (hh * ww) # candidate anchors per location

        x = F.relu(self.conv1(x))
        
        rpn_locs = self.locate(x)
        rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)
        
        rpn_scores = self.score(x)
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous().view(n, -1, 2)
        rpn_softmax_scores = F.softmax(rpn_scores.view(n, hh, ww, n_anchor, 2), dim=4)

        rpn_fg_scores = rpn_softmax_scores[:,:,:,:,1].contiguous()
        rpn_fg_scores = rpn_fg_scores.view(n, -1)

        rois = list()
        roi_indices = list()

        for i in range(n):
            roi = self.proposal_layer()
            batch_index = i * np.ones((len(roi),), dtype=np.int32)
            rois.append(roi)
            roi_indices.append(batch_index)

        rois = np.concatenation(rois, axis=0)
        roi_indices = np.concatenation(roi_indices, axis=0)

        return rpn_locs, rpn_scores, rois, roi_indices, anchor


    def generate_base_anchors(self,
        base_size=16,
        anchor_ratios=[0.5, 1, 2],
        anchor_scales=[8, 16, 32]
        ):
        px = base_size / 2
        py = base_size / 2
        anchor_base = np.zeros(((len(anchor_ratios) * len(anchor_scales)), 4), dtype=np.float32)
        for i in range(len(anchor_ratios)):
            for j in range(len(anchor_scales)):
                h = base_size * anchor_scales[j] * np.sqrt(anchor_ratios[i])
                w = base_size * anchor_scales[j] * np.sqrt(1. / anchor_ratios[i])

                index = i * len(anchor_scales) + j
                anchor_base[index, 0] = py - h / 2.
                anchor_base[index, 1] = px - w / 2.
                anchor_base[index, 2] = py + h / 2.
                anchor_base[index, 3] = px + w / 2.

        return anchor_base

    ''' Note:
        The anchor_base, only has the (-h/2, w/2, h/2, w/2).
        To be used as real anchors, the anchor_base should be shifted,
        that is, to add offset(x, y)
    '''
    def emuerate_shifted_anchor(anchor_base, feature_stride, height, width):
        shift_y = np.arange(0, height*feature_stride, feature_stride)
        shift_x = np.arange(0, width*fetaure_stride, fetaure_stride)
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)

        shift_xy = np.stack((shift_y.ravel(), shift_x.ravel(), shift_y.ravel(), shift_x.ravel()), axis=1)

        A = anchor_base.shape[0]
        K = shift.shape[0]

        anchor = anchor_base.reshape((1, A, 4)) + shift.reshape((1, K, 4)).transpose(1, 0, 2)
        anchor = anchor.reshape((K*A, 4)).astype(np.float32)
        return anchor




class FasterRCNN_VGG16(nn.Module):

    def __init__(self,
                 n_class=20, 
                 anchor_ratios=[0.5, 1, 2],
                 anchor_scales=[8, 16, 32]):
        extractor, classifier = decompose_vgg16()

        rpn = RegionProposalNetwork(512, 512, 
                                    anchor_ratios=anchor_ratios,
                                    anchor_scales=anchor_scales)