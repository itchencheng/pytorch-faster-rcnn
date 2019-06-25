#coding:utf-8

from data import *
import sys

import PIL.Image as Image
import numpy as np

from model import *

import torch
from torchvision import transforms as transforms
from skimage import transform as sktsf

import cv2

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


n_class = 21
loc_normalize_mean = (0., 0., 0., 0.)
loc_normalize_std = (0.1, 0.1, 0.2, 0.2)

nms_thresh = 0.3
score_thresh = 0.7


VOC_BBOX_LABEL_NAMES = (
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor')



def pytorch_normalze(img):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    img = normalize(torch.from_numpy(img).float())
    return img.numpy()


def preprocess(img, g_bbox):
    min_size = 600
    max_size = 1000

    # image
    img = np.transpose(img, (2,0,1))

    C, H, W = img.shape
    scale_min = float(min_size) / np.min((H, W)) # Note: whether should change to float
    scale_max = float(max_size) / np.max((H, W))

    scale = min(scale_min, scale_max)

    img = img / 255.
    img = sktsf.resize(img, (C, H*scale, W*scale), mode='reflect', anti_aliasing=False)
    img = pytorch_normalze(img)

    # bbox
    g_bbox = g_bbox*scale

    img_info = (img.shape[-2], img.shape[-1], scale)

    return img, g_bbox, img_info




def save_model(cnn_model, ckpt_name):
    # save models
    state_dict = {"state": cnn_model.state_dict()}
    torch.save(state_dict, ckpt_name)
    print("Model saved! %s" %(ckpt_name))


def do_training():
    # create model
    cnn_model = FasterRCNN_VGG16('./model/vgg16/vgg16-397923af.pth').to(device)

    learning_rate = 1e-4
    optimizer = torch.optim.SGD(cnn_model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0001)


    voc2007_dir = '/home/chen/dataset/voc2007/VOC2007'
    dataset = Voc2007Dataset(voc2007_dir, 'trainval', None)

    dataset_size = len(dataset)

    for epoch in range(15):
        for img_idx in range(dataset_size):

            '''-------------------------------------------'''
            one_example = dataset[img_idx]

            img, gt_bboxes, labels, difficult = one_example

            gt_bboxes = np.array(gt_bboxes).astype(np.float32)
            img = np.array(img).astype(np.float32)
            
            img, gt_bboxes, img_info = preprocess(img, gt_bboxes)

            img = img[np.newaxis, :]
            img = torch.Tensor(img).to(device)
            '''-------------------------------------------'''

            print('img', img.shape, img_info[-3:-1])
            print(img)

            print("img_bbox:")
            print(gt_bboxes)

            print("label_:")
            print(labels)

            print("scale:")
            print(img_info[-1])

            optimizer.zero_grad()

            loss = cnn_model.train_step(img, gt_bboxes, labels, img_info)
                    
            loss.backward()
            optimizer.step()

            if (9 == img_idx%100):
                save_model(cnn_model, "model_save2.pth")

    print("# complete!!!!!")        


def xmy_non_maximum_suppression(bboxes, nms_thresh, probs):

    max_min_order = np.argsort(probs)[::-1]
    ordered_bboxes = bboxes[max_min_order]
    ordered_probs = probs[max_min_order]

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

        top = np.maximum(ordered_bbox[:,0], max_bbox[0]*np.ones(ordered_bbox[:,0].shape))
        bottom = np.minimum(ordered_bbox[:,2], max_bbox[2]*np.ones(ordered_bbox[:,0].shape))
        left = np.maximum(ordered_bbox[:,1], max_bbox[1]*np.ones(ordered_bbox[:,0].shape))
        right = np.minimum(ordered_bbox[:,3], max_bbox[3]*np.ones(ordered_bbox[:,0].shape))
        height = np.maximum(bottom-top, np.zeros(ordered_bbox[:,0].shape))
        width = np.maximum(right-left, np.zeros(ordered_bbox[:,0].shape))
        
        area_i = height * width
        iou_value = area_i / (two_area - area_i)

        thresh_idx = iou_value > nms_thresh

        valid_flag[thresh_idx] = 0

        keep = np.append(keep, max_min_order[i])

    return keep


def suppress(n_class, raw_cls_bbox, raw_prob):
    bbox = list()
    label = list()
    score = list()
    # skip cls_id = 0 because it is the background class
    for l in range(1, n_class):
        cls_bbox_l = raw_cls_bbox.reshape((-1, n_class, 4))[:, l, :]
        prob_l = raw_prob[:, l]

        print(cls_bbox_l.shape)

        mask = prob_l > score_thresh
        cls_bbox_l = cls_bbox_l[mask]
        prob_l = prob_l[mask]

        if (len(cls_bbox_l) == 0):
            continue

        print('cls_bbox_l: ', cls_bbox_l.shape)
        print('prob_l: ', prob_l.shape)

        keep = xmy_non_maximum_suppression(cls_bbox_l, nms_thresh, prob_l)

        # The labels are in [0, n_class - 2].
        bbox.append(cls_bbox_l[keep])
        label.append((l - 1) * np.ones((len(keep),)))
        score.append(prob_l[keep])

    bbox = np.concatenate(bbox, axis=0).astype(np.float32)
    label = np.concatenate(label, axis=0).astype(np.int32)
    score = np.concatenate(score, axis=0).astype(np.float32)
    return bbox, label, score



def postprocess(data_in, scale, roi_cls_locs, roi_scores, rois, roi_indices):
    size = data_in.shape[-2:]
    print(size)
    mean = torch.Tensor(loc_normalize_mean).to(device).repeat(n_class)
    std  = torch.Tensor(loc_normalize_std).to(device).repeat(n_class)

    roi = torch.Tensor(rois) / scale
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

    bbox, label, score = suppress(n_class, raw_cls_bbox, raw_prob)

    return bbox, label, score


def plot_box(img_file, bboxes, labels, scores):
    img = cv2.imread(img_file)

    n_box = len(labels)

    for i in range(n_box):
        bbox = bboxes[i]
        label = labels[i]
        score = scores[i]

        x1, y1 = bbox[0], bbox[1]
        x2, y2 = bbox[2], bbox[3]

        cv2.rectangle(img, (y1, x1), (y2, x2), (0,255,0), 4)

        font = cv2.FONT_HERSHEY_SIMPLEX
        text = "%s, %f" %(VOC_BBOX_LABEL_NAMES[label], score)
        cv2.putText(img, text, (y1, x1), font, 0.4, (0,0,0), 1)

        print(text)

    new_jpg = '%s.new.jpg' %(img_file)
    cv2.imwrite(new_jpg, img)
    print(new_jpg)



def do_inference():
    img_file = sys.argv[1]

    # get data_in
    data_in = Image.open(img_file)

    gt_bboxes = np.array([[0,0,0,0]]).astype(np.float32)
    data_in = np.array(data_in).astype(np.float32)

    # preprocess
    data_in, _, img_info = preprocess(data_in, gt_bboxes)
    print(data_in.shape)

    data_in = data_in[np.newaxis, :]
    data_in = torch.from_numpy(data_in)

    # to gpu
    data_in = data_in.to(device)
    faster_rcnn = FasterRCNN_VGG16('./model/vgg16/vgg16-397923af.pth').to(device)
    ckpt = torch.load("model_faster.pth")
    faster_rcnn.load_state_dict(ckpt['state'])

    # run
    roi_cls_locs, roi_scores, rois, roi_indices = faster_rcnn.inference(data_in, img_info)

    print('roi_cls_locs: ', roi_cls_locs.shape)
    print('roi_scores: ', roi_scores.shape)
    print('rois: ', rois.shape)
    print('roi_indices: ', roi_indices.shape)

    # postprodess
    print("postprodess!")
    bboxes, labels, scores = postprocess(data_in, img_info[-1], roi_cls_locs, roi_scores, rois, roi_indices)
    
    plot_box(img_file, bboxes, labels, scores)

    print("complete!")
    

def main():
    #do_training()
    do_inference()



if __name__ == "__main__":
    main()