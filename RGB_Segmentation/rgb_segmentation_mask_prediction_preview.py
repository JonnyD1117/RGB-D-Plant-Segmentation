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
import matplotlib.pyplot as plt

from torchmetrics.functional import dice_score
from pl_bolts.losses.object_detection import iou_loss

# from RGB_Segmentation.unet import UNet
# from RGB_Segmentation.off_the_shelf_unet import UNet
from RGB_Segmentation.off_the_shelf_unet_w_sigmoid import UNet
from RGB_Segmentation.bce_dice_loss import DiceBCELoss


def plot_img_mask_pred(image, mask, untrained_pred, trained_pred, no_train_loss, train_loss):
    # Transform Tensors to Images
    untrained_pred_mask = untrained_pred[0][0][:][:]
    untrained_pred_mask = untrained_pred_mask.detach().numpy()

    trained_pred_mask = trained_pred[0][0][:][:]
    trained_pred_mask = trained_pred_mask.detach().numpy()

    real_image = image[0][:][:][:]
    real_image = real_image.permute(1, 2, 0)

    real_mask = mask[0][:][:][:]
    real_mask = real_mask.permute(1, 2, 0)

    # create figure
    fig = plt.figure(figsize=(15, 5))
    fig.suptitle(f"Untrained Loss = {round(float(no_train_loss.detach().numpy()), 2)} & Trained Loss {round(float(trained_loss.detach().numpy()), 2)}")

    # setting values to rows and column variables
    rows = 1
    columns = 4

    # Adds a subplot at the 1st position
    fig.add_subplot(rows, columns, 1)

    # showing image
    plt.imshow(real_image)
    plt.axis('off')
    plt.title("Image")

    # Adds a subplot at the 2nd position
    fig.add_subplot(rows, columns, 2)

    # showing image
    plt.imshow(real_mask)
    plt.axis('off')
    plt.title("Mask")

    # Adds a subplot at the 3rd position
    fig.add_subplot(rows, columns, 3)

    # showing image
    plt.imshow(untrained_pred_mask)
    plt.axis('off')
    plt.xlabel(f"UT Loss = {no_train_loss}")
    plt.title("UT Mask")

    # Adds a subplot at the 3rd position
    fig.add_subplot(rows, columns, 4)

    # showing image
    plt.imshow(trained_pred_mask)
    plt.axis('off')
    plt.xlabel(f"T Loss = {train_loss}")
    plt.title("T Mask")

    print(f"T Loss = {train_loss} & UT Loss = {no_train_loss}")
    plt.show()


NUM_SAMPLES = 10
VERSION = 15
EPOCH = 44
STEP = 10664

model = UNet()
train_model = UNet()
rootdir = r"C:\Users\Indy-Windows\Documents\RGB-D-Plant-Segmentation\RGB_Segmentation\logs\my_model"
path = rootdir + f"\\version_{VERSION}\\checkpoints\\epoch={EPOCH}-step={STEP}.ckpt"
checkpoint = torch.load(path)
train_model.load_state_dict(checkpoint['state_dict'], strict=False)

validation_dataloader = DataLoader(ValidationData(), shuffle=True, drop_last=True, batch_size=1)
loss_criterion = DiceBCELoss()


for ind, batch in enumerate(validation_dataloader):
    image, mask = batch

    if ind < NUM_SAMPLES:
        # Untrained Model Loss
        y_hat_untrained = model(image)
        untrained_loss = loss_criterion(y_hat_untrained, mask)

        # Trained Model
        y_hat_trained = train_model(image)
        trained_loss = loss_criterion(y_hat_trained, mask)

        plot_img_mask_pred(image, mask, y_hat_untrained, y_hat_trained, untrained_loss, trained_loss)
    break


# Train Model Loss






