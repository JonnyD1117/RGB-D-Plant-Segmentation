import os
import re
from tqdm import tqdm
from pathlib import Path
from RGB_Segmentation.utilities.logging_utils import setup_version_logs
from RGB_Segmentation.data.Carvana_Dataset.CarvanaDS import CarvanaData
from RGB_Segmentation.data.Carvana_Dataset.ValDS import ValidationData

import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# from RGB_Segmentation.models.unets.off_the_shelf_unet_w_sigmoid import UNet
from RGB_Segmentation.models.losses.bce_dice_loss import DiceBCELoss
from RGB_Segmentation.models.unets.off_the_shelf_unet import UNet

import matplotlib.pyplot as plt


if __name__ == '__main__':
    # Define Training Parameters
    EPOCHS = 20
    lr = .003

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define Model
    model = UNet().to(device)

    # Define DataLoader

    carvana_dl = DataLoader(CarvanaData(transform=True), batch_size=1, shuffle=False, drop_last=True)

    # Define Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Define Loss Function
    loss_criterion = DiceBCELoss().to(device)

    # Loop Through All Epochs
    for image, mask in carvana_dl:
        real_image = image[0][:][:][:]
        real_image = real_image.permute(1, 2, 0)

        plt.imshow(torch.squeeze(real_image))
        plt.show()
        break
