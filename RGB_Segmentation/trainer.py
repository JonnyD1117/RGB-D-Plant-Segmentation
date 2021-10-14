import os
from argparse import ArgumentParser
from tqdm import tqdm

from Carvana_Dataset.CarvanaDS import CarvanaData
from Carvana_Dataset.ValDS import ValidationData
from Carvana_Dataset.Test_DS import TestData

import torch
import torchvision
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from RGB_Segmentation.off_the_shelf_unet import UNet
from RGB_Segmentation.bce_dice_loss import DiceBCELoss


if __name__ == '__main__':
    # Define Training Parameters
    EPOCHS = 10
    lr = .003
    batch_size = 4
    val_batch_size = 1
    img_height = 517
    img_width = 517

    # Initialize TB Logging
    version_num = 5
    writer = SummaryWriter(log_dir= f'C:\\Users\\Indy-Windows\\Documents\\RGB-D-Plant-Segmentation\\RGB_Segmentation\\old_school_logs\\version{version_num}')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define Model
    model = UNet().to(device)

    # Define Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=5, gamma=.1)
    # Define Loss Function
    loss_criterion = DiceBCELoss()

    # Create DataLoader
    carvana_dl = DataLoader(CarvanaData(), batch_size, shuffle=True, num_workers=4, drop_last=True)
    validation_dl = DataLoader(ValidationData(), val_batch_size, shuffle=False, num_workers=4, drop_last=True)

    global_time_step = 0

    # Loop Through All Epochs
    for epoch in range(EPOCHS):
        print()
        print(f"################################################")
        print(f"Epoch = {epoch}")
        print(f"################################################")
        # Loop Through Each Batch
        for image, mask in tqdm(carvana_dl):
            # Send Training Data to GPU
            image = image.to(device)
            mask = mask.to(device)

            # UNET Forward Pass
            pred = model(image)

            # Compute Loss
            loss = loss_criterion(pred, mask)

            # Tensorboard Logging
            writer.add_scalar('train_loss', loss, global_step=global_time_step)
            # writer.add_graph(model, image)
            # grid_img = make_grid(image)
            # grid_mask = make_grid(mask)
            # grid_pred = make_grid(pred)
            # writer.add_image('image', grid_img)
            # writer.add_image('mask', grid_mask)
            # writer.add_image('prediction', grid_pred)

            # Zero_grad/Backprop Loss/ Step Optimizer
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_time_step += 1

        scheduler.step()




        # Save Model Checkpoint
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            }

        torch.save(checkpoint, f'C:\\Users\\Indy-Windows\\Documents\\RGB-D-Plant-Segmentation\\RGB_Segmentation\\old_school_models\\model_epoch{epoch}.ckpt')

        # val_loss = []
        # # Compute Validation Loss
        # for val_img, val_mask in tqdm(validation_dl):
        #
        #     # Send Training Data to GPU
        #     val_img = val_img.to(device)
        #     val_mask = val_mask.to(device)
        #
        #     # UNET Forward Pass
        #     val_pred = model(val_img)
        #
        #     # Compute Loss
        #     val_loss.append(loss_criterion(val_pred, val_mask))
        #
        # print(f"Mean Validation Loss = {torch.mean(val_loss)} ")

