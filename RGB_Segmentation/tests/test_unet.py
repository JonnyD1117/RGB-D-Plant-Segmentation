"""
Author: Jonathan Dorsey
Date: 10/9/2021
Summary:

This file contains the implementation for previewing the predicted masks from the trail UNET model
"""

import os
from argparse import ArgumentParser
from tqdm import tqdm

from Carvana_Dataset.CarvanaDS import CarvanaData
from Carvana_Dataset.ValDS import ValidationData

import torch
import torchvision
from torch import optim
from torch.utils.data import DataLoader, random_split

# from RGB_Segmentation.unet import UNet
from RGB_Segmentation.off_the_shelf_unet import UNet
# from RGB_Segmentation.off_the_shelf_unet_w_sigmoid import UNet

model = UNet()
checkpoint = torch.load(r"C:\Users\Indy-Windows\Documents\RGB-D-Plant-Segmentation\RGB_Segmentation\logs\my_model\version_15\checkpoints\epoch=35-step=8531.ckpt")
model.load_state_dict(checkpoint['state_dict'], strict=False)

validation_dataloader = DataLoader(ValidationData(), shuffle=True, drop_last=True, batch_size=1)

EPOCHS = 1

#
batch = next(iter(validation_dataloader))
image, mask = batch

y_hat = model(image)

print(f"Prediction Shape = {y_hat.shape}, Min = {torch.min(y_hat)}, Max = {torch.max(y_hat)}")