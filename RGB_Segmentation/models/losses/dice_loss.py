"""
Dice Loss Function:

Written by someone at Kaggle (https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch)

Summary:
This loss function computes the Dice Loss for Pytorch.

Dice loss is a metric based on the Dice similarity score.
Dice_score = (2xIoU)
"""
import torch
from torch import nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    @staticmethod
    def forward(inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice
