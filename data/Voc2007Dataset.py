#coding:utf-8

from torch.utils.data import Dataset
import os
import PIL.Image as Image
import re

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

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        # image
        img_file = os.path.join(self.dir_path, 'JPEGImages', img_id+'.jpg')
        img = Image.open(img_file)
        if (self.transform != None):
            img = self.transform(img)
        # label
        annotation_file = os.path.join(self.dir_path, 'Annotations', '%s.xml' %(img_id))
        name = []
        label = []
        pose = []
        truncate = []
        difficult = []
        bbox = []

        fi = open(annotation_file)
        lines = fi.readlines()
        for line in lines:
            line = line.strip()
            # name
            pat = '<name>([a-zA-Z]+)</name>'
            tmp = re.findall(pat, line)
            if (len(tmp)>=1):
                tmp = tmp[0]
                name.append(tmp)
                label.append(VOC_BBOX_LABEL_NAMES.index(tmp))

            # bbox
            if (line=='<bndbox>'):
                bbox.append([])
                bbox[-1].append(0)
                bbox[-1].append(0)
                bbox[-1].append(0)
                bbox[-1].append(0)

            pat = '<xmin>([0-9]+)</xmin>'
            tmp = re.findall(pat, line)
            if (len(tmp)>=1):
                tmp = tmp[0]
                bbox[-1][0] = int(tmp)
            pat = '<ymin>([0-9]+)</ymin>'
            tmp = re.findall(pat, line)
            if (len(tmp)>=1):
                tmp = tmp[0]
                bbox[-1][1] = int(tmp)
            pat = '<xmax>([0-9]+)</xmax>'
            tmp = re.findall(pat, line)
            if (len(tmp)>=1):
                tmp = tmp[0]
                bbox[-1][2] = int(tmp)
            pat = '<ymax>([0-9]+)</ymax>'
            tmp = re.findall(pat, line)
            if (len(tmp)>=1):
                tmp = tmp[0]
                bbox[-1][3] = int(tmp)

            # difficult
            pat = '<difficult>([01])</difficult>'
            tmp = re.findall(pat, line)
            if (len(tmp) > 0):
                difficult.append(int(tmp[0]))


        fi.close()

        return img, bbox, label, difficult

