"""
Author: Jonathan Dorsey
Date: 10/9/2021
Summary:

This file contains the implementation for previewing the predicted masks from the trail UNET model
"""

import os
import re
from argparse import ArgumentParser
from pathlib import Path
import numpy as np

from RGB_Segmentation.data.Carvana_Dataset.ValDS import ValidationData

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

# from RGB_Segmentation.unet import UNet
from RGB_Segmentation.models.unets.off_the_shelf_unet import UNet
# from RGB_Segmentation.models.unets.off_the_shelf_unet_w_sigmoid import UNet
from RGB_Segmentation.models.losses.bce_dice_loss import DiceBCELoss


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



def get_current_model_checkpoint(root_dir=os.path.join(Path(__file__).parents[1], "logs\\pytorch_logs"), epoch_num=None):
    print(f"Root directory = {root_dir}")
    # Get Version # from logs directory
    version_list = os.listdir(root_dir)
    version_list = [int(re.findall(r'\d{1,3}', vs)[-1]) for vs in version_list]
    version = max(version_list)
    # Check Epoch from Version subdirectory
    version_path = os.path.join(root_dir, f"version{version}\\checkpoints")

    if epoch_num is None:
        max_epoch = 0
        for item in os.listdir(version_path):
            output = re.findall(r'\d{1,}', item)
            v_num = int(output[0])
            e_num = int(output[1])

            if e_num > max_epoch:
                max_epoch = e_num
    else:
        max_epoch = epoch_num

    model_path = os.path.join(version_path, f"v{version}_model_epoch{max_epoch}.ckpt")
    return model_path


if __name__ == '__main__':
    # Parse CLI Arguments
    parser = ArgumentParser()
    parser.add_argument('--num-samples', default=5)
    args = parser.parse_args()

    # Define Program Variables
    NUM_SAMPLES = args.num_samples

    # Declare UNETS
    model = UNet()
    train_model = UNet()

    # Get Trained Model Path
    path = get_current_model_checkpoint()
    # Load Model Checkpoint from Path
    checkpoint = torch.load(path)
    train_model.load_state_dict(checkpoint['state_dict'], strict=False)

    # Initialize Validation Dataloader
    validation_dataloader = DataLoader(ValidationData(), shuffle=True, drop_last=True, batch_size=1)

    # Define Loss Criterion
    loss_criterion = DiceBCELoss()

    for ind, batch in enumerate(validation_dataloader):
        if ind > NUM_SAMPLES:
            break

        image, mask = batch
        # Untrained Model Loss
        y_hat_untrained = model(image)
        untrained_loss = loss_criterion(y_hat_untrained, mask)

        # Trained Model
        y_hat_trained = train_model(image)
        trained_loss = loss_criterion(y_hat_trained, mask)

        img = torch.squeeze(image)
        msk_list = torch.squeeze(mask)
        ut_pred = torch.squeeze(y_hat_untrained)
        t_pred = torch.squeeze(y_hat_trained)

        plot_img_mask_pred(image, mask, y_hat_untrained, y_hat_trained, untrained_loss, trained_loss)


