#coding:utf-8

from data import *

def train():
    # for a deep learning task
    # 1. dataset
    # 2. model
    # 3. loss and optimizer
    # 4. lr policy

    # VOC 2007 dataset
    voc2007_dir = '/home/chen/dataset/voc2007/VOC2007'
    dataset = Voc2007Dataset(voc2007_dir, 'train', None)
    
    for i in range(5):
        print(dataset[i])

    print('# train finished!')


def main():
    train()


if __name__ == "__main__":
    main()