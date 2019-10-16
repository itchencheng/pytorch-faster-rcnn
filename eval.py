
from model import *
from data import *


import eval_tools

import sys
import os

'''
 python eval.py model.pth
'''
def main():
    pth_file = sys.argv[1]
    print("load:", pth_file)

    faster_rcnn = FasterRCNN_VGG16().to(device)
    ckpt = torch.load(pth_file)
    faster_rcnn.load_state_dict(ckpt['state'])

    voc2007_dir = '/home/chen/dataset/voc2007/VOC2007'
    test_dataset = Voc2007Dataset(voc2007_dir, 'test', preprocess)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, \
                                                  batch_size=1, \
                                                  shuffle=False, \
                                                  num_workers=0)
    test_num = 1000

    eval_result = eval_tools.evalx(test_dataloader, faster_rcnn, test_num=test_num)

    print(eval_result)


if __name__ == "__main__":
    main()