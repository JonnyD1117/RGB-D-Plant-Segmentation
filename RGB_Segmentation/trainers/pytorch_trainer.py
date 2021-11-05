import os
import re
from tqdm import tqdm
from pathlib import Path
from RGB_Segmentation.utilities.logging_utils import setup_version_logs
from RGB_Segmentation.data.Carvana_Dataset.CarvanaDS import CarvanaData
from RGB_Segmentation.data.Carvana_Dataset.ValDS import ValidationData

import torch
from torch import optim
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.functional import dice_score

# from RGB_Segmentation.models.unets.off_the_shelf_unet_w_sigmoid import UNet
from RGB_Segmentation.models.losses.bce_dice_loss import DiceBCELoss
from RGB_Segmentation.models.unets.off_the_shelf_unet import UNet
# from RGB_Segmentation.models.unets.unet import UNet


if __name__ == '__main__':
    # Define Training Parameters
    EPOCHS = 200
    lr = .00003
    batch_size = 16
    val_batch_size = 1
    img_height = 160
    img_width = 240

    # Paths
    rgb_seg_base_path = Path(__file__).parents[1]
    log_path = os.path.join(rgb_seg_base_path, f'logs\\pytorch_logs')

    # Initialize TB Logging
    version_num, version_dir, checkpoint_dir = setup_version_logs(rgb_seg_base_path, log_path)

    writer = SummaryWriter(log_dir=version_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f'Current Device = {device}')

    # Define Model
    model = UNet().to(device)

    # Define Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=10, gamma=.1)
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5,threshold=1.5)
    # Define Loss Function
    loss_criterion = DiceBCELoss().to(device)
    # loss_criterion = nn.BCELoss().to(device)

    # Create DataLoader
    carvana_dl = DataLoader(CarvanaData(test=False, image_size=(img_height, img_width), transform=False, augment=True), batch_size, shuffle=True, num_workers=4, drop_last=True)
    validation_dl = DataLoader(ValidationData(image_size=(img_height, img_width)), val_batch_size, shuffle=False, num_workers=4, drop_last=True)

    global_time_step = 0

    # Loop Through All Epochs
    for epoch in range(EPOCHS):
        print()
        print(f"################################################")
        print(f"Epoch = {epoch}")
        print(f"################################################")

        # Loop Through Each Batch
        model = model.train()
        for image, mask in tqdm(carvana_dl):
            # Send Training Data to GPU
            image = image.to(device)
            mask = mask.to(device)

            # UNET Forward Pass
            pred = model(image)

            # print(f"Img Max = {torch.max(image)}, Img Min = {torch.min(image)}")
            # print(f"Mask Max = {torch.max(mask)}, Mask Min = {torch.min(mask)}")
            # print(f"Pred Max = {torch.max(pred)}, Pred Min = {torch.min(pred)}")


            # Compute Loss
            loss = loss_criterion(torch.sigmoid(pred), torch.sigmoid(mask))
            # loss = loss_criterion(pred, torch.sigmoid(mask))
            ds = dice_score(torch.sigmoid(pred), torch.sigmoid(mask))

            # Tensorboard Logging
            writer.add_scalar('train_loss', loss, global_step=global_time_step)
            writer.add_scalar('Dice Score', ds, global_step=global_time_step)

            # Zero_grad/Backprop Loss/ Step Optimizer
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_time_step += 1

        # scheduler.step()
        if epoch % 10 == 0:
            lr *= .1

        # Save Model Checkpoint
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            }

        torch.save(checkpoint, os.path.join(checkpoint_dir, f"v{version_num}_model_epoch{epoch}.ckpt"))

        with torch.no_grad():
            mean_val_loss = 0
            val_ctr = 1.0
            # Compute Validation Loss
            for val_img, val_mask in tqdm(validation_dl):
                model = model.eval()
                # Send Training Data to GPU
                val_img = val_img.to(device)
                val_mask = val_mask.to(device)

                # UNET Forward Pass
                val_pred = model(val_img)

                # Compute Loss
                val_loss = loss_criterion(torch.sigmoid(val_pred), torch.sigmoid(val_mask))

                mean_val_loss = mean_val_loss + (val_loss - mean_val_loss)/val_ctr
                val_ctr += 1

            writer.add_scalar('validation_loss', mean_val_loss, global_step=epoch)
            writer.add_scalar('learning rate', lr, global_step=epoch)
            writer.add_scalar('epoch', epoch, global_step=epoch)


