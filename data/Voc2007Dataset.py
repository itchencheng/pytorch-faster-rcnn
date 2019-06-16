#coding:utf-8

from torch.utils.data import Dataset
import os
import PIL.Image as Image
import re

import numpy as np
import xml.etree.ElementTree as ET

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


class Voc2007Dataset(Dataset):
    def __init__(self, dir_path, tag='train', transform=None):
        super(Voc2007Dataset, self).__init__()
        # tags: train, trainval, val, test
        id_list_file = os.path.join(dir_path, 'ImageSets', 'Main', '%s.txt' %(tag))
        
        fi = open(id_list_file)
        self.ids = []
        lines = fi.readlines()
        for item in lines:
            item = item.strip()
            self.ids.append(item)
        fi.close()

        self.dir_path = dir_path
        self.transform = transform
        self.use_difficult = False

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id_ = self.ids[idx]
        print('img_id', id_)

        anno = ET.parse(os.path.join(self.dir_path, 'Annotations', id_ + '.xml'))
        bbox = list()
        label = list()
        difficult = list()
        for obj in anno.findall('object'):
            # when in not using difficult split, and the object is
            # difficult, skipt it.
            if not self.use_difficult and int(obj.find('difficult').text) == 1:
                continue

            difficult.append(int(obj.find('difficult').text))
            bndbox_anno = obj.find('bndbox')
            # subtract 1 to make pixel indexes 0-based
            bbox.append([
                int(bndbox_anno.find(tag).text) - 1
                for tag in ('xmin', 'ymin', 'xmax', 'ymax')])
            name = obj.find('name').text.lower().strip()
            label.append(VOC_BBOX_LABEL_NAMES.index(name))
        #bbox = np.stack(bbox).astype(np.float32)
        #label = np.stack(label).astype(np.int32)
        # When `use_difficult==False`, all elements in `difficult` are False.
        difficult = np.array(difficult, dtype=np.bool).astype(np.uint8)  # PyTorch don't support np.bool

        # Load a image
        img_file = os.path.join(self.dir_path, 'JPEGImages', id_ + '.jpg')
        img = Image.open(img_file)
        if (self.transform != None):
            img = self.transform(img)

        # if self.return_difficult:
        #     return img, bbox, label, difficult

        assert(len(bbox) == len(label))

        for idx in range(len(label)):
            bbox[idx].append(label[idx])
            print("###", bbox[idx])

        return img, bbox, difficult
