import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class MultiScaleLoss(nn.Module):

    def __init__(self, weights=None):
        super(MultiScaleLoss,self).__init__()

    def forward(self, input, target, mask):
        if type(input) is tuple:
            out1, out2, out3 = input[0][mask], input[1][mask], input[2][mask]
            target = target[mask]
            loss = 0.5*F.smooth_l1_loss(out1, target, size_average=True) + 0.7*F.smooth_l1_loss(out2, target, size_average=True) + F.smooth_l1_loss(out3, target, size_average=True)
            return loss
        else:
            out = input[mask]
            target = target[mask]
            loss = torch.mean(torch.abs(out - target))
            return loss


class MultiScaleLossSparse(MultiScaleLoss):

    def __init__(self):
        super(MultiScaleLoss,self).__init__()

    def forward(self, input, target, mask):
        if type(input) is tuple:
            # input[i].data[target_.data == 0] = 0
            out1, out2, out3 = input[0][mask], input[1][mask], input[2][mask]
            target = target[mask]
            loss = 0.5*F.smooth_l1_loss(out1, target, size_average=True) + 0.7*F.smooth_l1_loss(out2, target, size_average=True) + F.smooth_l1_loss(out3, target, size_average=True)
            return loss
        else:
            out = input[mask]
            target = target[mask]
            loss = torch.mean(torch.abs(out - target))
            return loss
