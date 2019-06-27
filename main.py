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

    img_info = np.array((img.shape[-2], img.shape[-1], scale))

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



def suppress(n_class, raw_cls_bbox, raw_prob):
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

    bbox = np.concatenate(bbox, axis=0).astype(np.float32)
    label = np.concatenate(label, axis=0).astype(np.int32)
    score = np.concatenate(score, axis=0).astype(np.float32)

    debug('bbox', bbox)
    debug('label', label)
    debug('score', score)

    return bbox, label, score


def xsuppress(n_class, raw_cls_bbox, raw_prob):
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
        return

    bbox = sorted_bbox[keep]
    label = sorted_pred[keep]-1
    score = sorted_prob[keep]

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
    faster_rcnn = FasterRCNN_VGG16().to(device)
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
    do_training()
    #do_inference()



if __name__ == "__main__":
    main()