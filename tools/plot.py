#coding:utf-8

import matplotlib.pyplot as plt 
import sys
import re

# ('img_id', '007963')
# (16650, 1)
# ('==rpn_loc_loss', tensor(0.0045, device='cuda:0', grad_fn=<DivBackward0>))
# ('==rpn_cls_loss', tensor(0.7054, device='cuda:0', grad_fn=<NllLossBackward>))
# ('==loc_loss', tensor(0.3688, device='cuda:0', grad_fn=<DivBackward0>))
# ('==cls_loss', tensor(2.9332, device='cuda:0', grad_fn=<NllLossBackward>))
# ('==total_loss', tensor(4.0119, device='cuda:0', grad_fn=<AddBackward0>))


def parse_logs(log_file):
    f = open(log_file)

    lines = f.readlines()

    rpn_loc_loss_list = []
    rpn_cls_loss_list = []
    loc_loss_list = []
    cls_loss_list = []
    total_loss_list = []

    rpn_loc_loss_re = r"==rpn_loc_loss\', tensor\(([.0-9]+),"
    rpn_cls_loss_re = r"==rpn_cls_loss\', tensor\(([.0-9]+),"
    loc_loss_re = r"==loc_loss\', tensor\(([.0-9]+),"
    cls_loss_re = r"==cls_loss\', tensor\(([.0-9]+),"
    total_loss_re = r"==total_loss\', tensor\(([.0-9]+),"


    for line in lines:

        float_number = re.findall(rpn_loc_loss_re, line)
        if (len(float_number) > 0):
            rpn_loc_loss_list.append(float(float_number[0]))

        float_number = re.findall(rpn_cls_loss_re, line)
        if (len(float_number) > 0):
            rpn_cls_loss_list.append(float(float_number[0]))

        float_number = re.findall(loc_loss_re, line)
        if (len(float_number) > 0):
            loc_loss_list.append(float(float_number[0]))

        float_number = re.findall(cls_loss_re, line)
        if (len(float_number) > 0):
            cls_loss_list.append(float(float_number[0]))

        float_number = re.findall(total_loss_re, line)
        if (len(float_number) > 0):
            total_loss_list.append(float(float_number[0]))

    f.close()

    # --------- plot -----------
    plt.title('Result Analysis')

    print(len(total_loss_list))

    train_iter_list = range(len(total_loss_list))

    plt.plot(train_iter_list, total_loss_list[:len(train_iter_list)], color='green', label='total-loss')
    #plt.scatter(train_iter_list, total_loss_list[:len(train_iter_list)], color='green', label='total-loss')
    
    plt.legend() # 显示图例

    plt.xlabel('iteration times')
    plt.ylabel('loss')
    plt.show()



def main():
    log_file = sys.argv[1]
    print(log_file)

    parse_logs(log_file)

    print("complete!")



if __name__ == "__main__":
    main()