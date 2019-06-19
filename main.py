#coding:utf-8

from data import *
import sys

import PIL.Image as Image
import numpy as np

from model import *

import torch
from torchvision import transforms as transforms
from skimage import transform as sktsf

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
    scale_min = min_size / np.min((H, W)) # Note: whether should change to float
    scale_max = max_size / np.max((H, W))

    scale = min(scale_min, scale_max)

    img = img / 255.
    img = sktsf.resize(img, (C, H*scale, W*scale), mode='reflect', anti_aliasing=False)
    img = pytorch_normalze(img)

    # bbox
    g_bbox[:,:4] = g_bbox[:,:4]*scale

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

    cnn_model.load_state_dict(torch.load("model_save.pth")["state"])

    learning_rate = 1e-3
    optimizer = torch.optim.SGD(cnn_model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0001)


    voc2007_dir = '/home/chen/dataset/voc2007/VOC2007'
    dataset = Voc2007Dataset(voc2007_dir, 'train', None)

    dataset_size = len(dataset)

    for epoch in range(15):
        for img_idx in range(dataset_size):

            '''-------------------------------------------'''
            one_example = dataset[img_idx]

            img, gt_bboxes, difficult = one_example
            #print('img', img)
            #print('gt_bboxes', gt_bboxes)

            gt_bboxes = np.array(gt_bboxes).astype(np.float32)
            img = np.array(img).astype(np.float32)
            
            img, gt_bboxes, img_info = preprocess(img, gt_bboxes)
            #print('img', np.sum(img>0))
            #print('gt_bboxes', gt_bboxes)
            #print('img_info', img_info)


            img = img[np.newaxis, :]
            img = torch.Tensor(img).to(device)
            '''-------------------------------------------'''

            optimizer.zero_grad()

            loss = cnn_model.train_step(img, img_info, gt_bboxes)
                    
            loss.backward()
            optimizer.step()

            if (9 == img_idx%100):
                save_model(cnn_model, "model_save1.pth")

    print("# complete!!!!!")        


nms_thresh = 0.7
score_thresh = 0.05


def suppress(n_class, raw_cls_bbox, raw_prob):
    bbox = list()
    label = list()
    score = list()
    # skip cls_id = 0 because it is the background class
    for l in range(1, n_class):
        cls_bbox_l = raw_cls_bbox.reshape((-1, n_class, 4))[:, l, :]
        prob_l = raw_prob[:, l]

        mask = prob_l > score_thresh
        cls_bbox_l = cls_bbox_l[mask]
        prob_l = prob_l[mask]

        if (len(cls_bbox_l) == 0):
            continue

        keep = my_non_maximum_suppression(cls_bbox_l, nms_thresh, prob_l)

        # The labels are in [0, n_class - 2].
        bbox.append(cls_bbox_l[keep])
        label.append((l - 1) * np.ones((len(keep),)))
        score.append(prob_l[keep])

    if (0 < len(bbox)):
        bbox = np.concatenate(bbox, axis=0).astype(np.float32)
    if (0 < len(label)):    
        label = np.concatenate(label, axis=0).astype(np.int32)
    if (0 < len(score)):
        score = np.concatenate(score, axis=0).astype(np.float32)

    return bbox, label, score


def do_inference():
    # create model
    cnn_model = FasterRCNN_VGG16('./model/vgg16/vgg16-397923af.pth').to(device)

    cnn_model.load_state_dict(torch.load("model_save.pth")["state"])

    voc2007_dir = '/home/chen/dataset/voc2007/VOC2007'
    dataset = Voc2007Dataset(voc2007_dir, 'train', None)

    for img_idx in range(5):

        '''-------------------------------------------'''
        one_example = dataset[img_idx]

        img, gt_bboxes, difficult = one_example
        #print('img', img)
        #print('gt_bboxes', gt_bboxes)

        gt_bboxes = np.array(gt_bboxes).astype(np.float32)
        img = np.array(img).astype(np.float32)
        
        img, gt_bboxes, img_info = preprocess(img, gt_bboxes)
        #print('img', np.sum(img>0))
        #print('gt_bboxes', gt_bboxes)
        print('img_info', img_info)

        img = img[np.newaxis, :]
        img = torch.Tensor(img).to(device)
        '''-------------------------------------------'''
        print(gt_bboxes)

        roi_locs, roi_scores, roi = cnn_model.inference(img, img_info)

        roi = roi.detach().cpu().numpy()
        roi_locs = roi_locs.detach().cpu().numpy()

        cls_bbox = bbox_transform_inv(roi, roi_locs)

        n_classes = 21
        cls_bbox = torch.Tensor(cls_bbox).view(-1, n_classes*4)
        cls_bbox[:, 0::2] = (cls_bbox[:, 0::2]).clamp(min=0, max=img_info[0])
        cls_bbox[:, 1::2] = (cls_bbox[:, 1::2]).clamp(min=0, max=img_info[1])

        print('roi_scores', roi_scores.shape)
        prob = F.softmax(roi_scores, dim=1)
        print('prob', prob)

        raw_cls_bbox = cls_bbox.detach().numpy()
        raw_prob = prob.detach().cpu().numpy()

        bbox, label, score = suppress(n_classes, raw_cls_bbox, raw_prob)

        print(bbox)
        print(label)
        print(score)

    print("# complete!!!!!")        

def main():
    do_training()
    #do_inference()



if __name__ == "__main__":
    main()