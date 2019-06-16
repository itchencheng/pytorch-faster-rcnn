#coding:utf-8

from data import *
import sys

import PIL.Image as Image
import numpy as np

from model import *

import torch


def get_one_sample(img_idx):
    # for a deep learning task
    # 1. dataset
    # 2. model
    # 3. loss and optimizer
    # 4. lr policy

    # VOC 2007 dataset
    voc2007_dir = '/home/chen/dataset/voc2007/VOC2007'
    dataset = Voc2007Dataset(voc2007_dir, 'train', None)
    
    return dataset[img_idx]


def save_model(cnn_model, ckpt_name):
    # save models
    state_dict = {"state": cnn_model.state_dict()}
    torch.save(state_dict, ckpt_name)
    print("Model saved! %s" %(ckpt_name))



def test_one_image():

    SCALES = 600
    MAX_SIZE = 1000

    # create model
    cnn_model = FasterRCNN_VGG16('./model/vgg16/vgg16-397923af.pth')

    learning_rate = 1e-3
    optimizer = torch.optim.SGD(cnn_model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0001)


    for img_idx in range(10000):

        '''-------------------------------------------'''
        one_example = get_one_sample(img_idx)
        print(one_example)

        # train()
        img, gt_bboxes, difficult = one_example
        gt_bboxes = np.array(gt_bboxes)
        img = np.array(img).astype(np.float32)
        img = np.transpose(img, (2,0,1))
        print(img.shape)

        # numpy to Tensor
        img = img[np.newaxis, :]
        img = torch.Tensor(img)
        im_shape = img.shape[-2:]
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])

        target_size = SCALES
        im_scale = float(target_size) / float(im_size_min)

        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > MAX_SIZE:
            im_scale = float(MAX_SIZE) / float(im_size_max)

        print('im_scale', im_scale)

        img_info = (img.shape[-2], img.shape[-1], im_scale)
        print('img_info', img_info)

        '''-------------------------------------------'''

        optimizer.zero_grad()

        loss = cnn_model.train_step(img, img_info, gt_bboxes)
                
        loss.backward()
        optimizer.step()

        if (9 == img_idx%10):
            save_model(cnn_model, "model_save.pth")


    print("# complete!!!!!")        


def main():
    test_one_image()



if __name__ == "__main__":
    main()