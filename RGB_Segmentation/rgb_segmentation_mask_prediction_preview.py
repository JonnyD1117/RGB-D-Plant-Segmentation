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

import pytorch_lightning as pl
from pytorch_lightning.core import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.profiler import SimpleProfiler, AdvancedProfiler

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.model_checkpoint import Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from torchmetrics.functional import dice_score
from pl_bolts.losses.object_detection import iou_loss

# from RGB_Segmentation.unet import UNet
from RGB_Segmentation.off_the_shelf_unet import UNet

model = UNet()
# print(model.state_dict())
checkpoint = torch.load(r"C:\Users\Indy-Windows\Documents\RGB-D-Plant-Segmentation\RGB_Segmentation\logs\my_model\version_5\checkpoints\epoch=77-step=36971.ckpt")
# print(checkpoint['state_dict'])
model.load_state_dict(checkpoint['state_dict'], strict=False)
# print(model.state_dict())

validation_dataloader = DataLoader(ValidationData(), shuffle=True, drop_last=True, batch_size=1)


EPOCHS = 1

for epoch in range(EPOCHS):

    for batch in validation_dataloader:
        image, mask = batch

        y_hat = model(image)

        print(f"Pred Mask Shape = {y_hat.shape}")
        print(f"Pred Mask Type = {type(y_hat)}")
        import matplotlib.pyplot as plt

        # pred_mask = torch.squeeze(y_hat)
        pred_mask = y_hat[0][0][:][:]
        pred_mask = torch.sigmoid_(pred_mask) > .5
        # print(f"Pred Mask Shape Post Squeeze = {pred_mask.shape}")

        pred_mask = pred_mask.detach().numpy()

        # print(pred_mask)
        plt.imshow(pred_mask)
        plt.show()


        # print(type(image))
        #
        # import matplotlib.pyplot as plt
        # real_image = image[0][:][:][:]
        # real_image = real_image.permute(1, 2, 0)
        # plt.imshow(real_image)
        # plt.show()
        #
        # real_mask = mask[0][:][:][:]
        # real_mask = real_mask.permute(1, 2, 0)
        # # plt.imshow(real_mask)
        # # plt.show()

        break