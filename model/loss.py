
import torch
import torch.nn as nn


class WeightedSmoothL1Loss(nn.Module):
    # loss = sum(w_out * smoothL1Loss(w_in * (pred-target)))
    
    def __init__(self):
        super(WeightedSmoothL1Loss, self).__init__()
        self.smooth_l1_loss = nn.SmoothL1Loss(reduction='none')

    def forward(self, pred, target, inside_weight, outside_weight):
        diff = pred - target
        diff = inside_weight * diff
        loss = self.smooth_l1_loss(diff, torch.zeros(diff.shape))
        loss = outside_weight * loss
        loss = torch.sum(loss)
        return loss
