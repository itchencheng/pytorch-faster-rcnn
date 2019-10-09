#coding:utf-8

from data import *
import sys

import PIL.Image as Image
import numpy as np

from model import *

import torch

import cv2

import eval_tools

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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


def save_model(cnn_model, ckpt_name):
    # save models
    state_dict = {"state": cnn_model.state_dict()}
    torch.save(state_dict, ckpt_name)
    print("Model saved! %s" %(ckpt_name))


def do_training():
    # create model
    faster_rcnn = FasterRCNN_VGG16('./model/vgg16/vgg16-397923af.pth').to(device)

    # create dataset and dataloader
    voc2007_dir = '/home/chen/dataset/voc2007/VOC2007'
    dataset = Voc2007Dataset(voc2007_dir, 'trainval', preprocess)
    dataloader = torch.utils.data.DataLoader(dataset, \
                                                  batch_size=1, \
                                                  shuffle=True, \
                                                  num_workers=0)

    test_dataset = Voc2007Dataset(voc2007_dir, 'test', preprocess)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, \
                                                  batch_size=1, \
                                                  shuffle=False, \
                                                  num_workers=0, \
                                                  pin_memory=True)

    # to train
    train_flag = True
    epoch = 0
    iter_count = 0

    while(train_flag):
  
        for ii, (imgs_, gt_bboxes_, gt_labels_, gt_difficults_, img_info_) in enumerate(dataloader):

            # change data type
            imgs_ = imgs_.to(device)
            gt_bboxes_ = gt_bboxes_[0].numpy()
            gt_labels_ = gt_labels_[0].numpy()
            gt_difficults_ = gt_difficults_[0].numpy()
            img_info_ = img_info_[0].numpy()

            # update parameters
            #optimizer.zero_grad()

            loss = faster_rcnn.train_step(imgs_, gt_bboxes_, gt_labels_, img_info_)
                    
            #loss.backward()
            #optimizer.step()

            if (0 == iter_count%200):
                save_model(faster_rcnn, "checkpoints/fasterrcnn_20190718.pth")

            '''
            if (iter_count == 60000):
                learning_rate = 0.0001
                optimizer = torch.optim.SGD(faster_rcnn.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0005)

            if (iter_count == 80000):
                train_flag = False
                break
            '''

            print('epoch %d: iter %d' %(epoch, iter_count))
            iter_count += 1


            #if (iter_count == 10):
            #    exit(0)

        ''' eval '''
        eval_result = eval_tools.evalx(test_dataloader, faster_rcnn, test_num=1000)
        eval_name = "checkpoints/fasterrcnn_20190724_epoch%d_%.4f" %(epoch, eval_result['map'])
        print(eval_name)
        save_model(faster_rcnn, eval_name)

        if (epoch == 9):
            break
            learning_rate = learning_rate * 0.1
            optimizer = torch.optim.SGD(faster_rcnn.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0005)


        if (epoch == 13):
            break
            
        epoch += 1


    print("# complete!!!!!")        



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
    data_in = data_in.convert('RGB')
    data_in = np.array(data_in).astype(np.float32)
    data_in = np.transpose(data_in, (2,0,1))

    gt_bboxes = np.array([[0,0,0,0]]).astype(np.float32)
    data_in = np.array(data_in).astype(np.float32)

    # preprocess
    data_in, _, img_info = preprocess(data_in, gt_bboxes)
    print(data_in.shape)
    data_in = data_in[np.newaxis, :]

    # to gpu
    data_in = data_in.to(device)
    faster_rcnn = FasterRCNN_VGG16().to(device)
    #ckpt = torch.load("checkpoints/fasterrcnn_20190712.pth")
    #faster_rcnn.load_state_dict(ckpt['state'])

    # run
    debug('data_in', data_in)
    debug('img_info', img_info)
    roi_cls_locs, roi_scores, rois, roi_indices = faster_rcnn.inference(data_in, img_info)

    print('roi_cls_locs: ', roi_cls_locs.shape)
    print('roi_scores: ', roi_scores.shape)
    print('rois: ', rois.shape)
    print('roi_indices: ', roi_indices.shape)

    # postprodess
    print("postprodess!")
    bboxes, labels, scores = postprocess(data_in, roi_cls_locs, roi_scores, rois, roi_indices)
    
    bboxes = bboxes / img_info[-1]

    plot_box(img_file, bboxes, labels, scores)

    print("complete!")
    

def main():
    do_training()
    #do_inference()



if __name__ == "__main__":
    main()