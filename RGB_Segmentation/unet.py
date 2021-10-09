import os
from time import sleep
import cv2
import numpy as np
from PIL import Image
import torch
import torchvision
import torch.optim as optim
from torchvision import transforms, datasets, utils
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import StepLR

class ImgSegDataSet(Dataset):
    def __init__(self, dir, img_transform=None, mask_transform=None):
        self.input_folder = dir
        self.Image_trans = img_transform
        self.Mask_trans = mask_transform

        self.image_dir = self.input_folder + "/images"
        self.mask_dir = self.input_folder + "/masks"

        self.image_list = os.listdir(self.image_dir)
        self.mask_list = os.listdir(self.mask_dir)

        self.image_path = None
        self.label_path = None

    def __len__(self):
        if len(self.image_list) == len(self.mask_list):
            return len(self.image_list)

    def __getitem__(self, item):
        # if torch.is_tensor(item):
        #     item = item.tolist()

        self.image_path = self.image_dir + "/" + self.image_list[item]
        self.label_path = self.mask_dir + "/" + self.mask_list[item]

        image = self.Image_trans(Image.open(self.image_path))
        msk = self.Mask_trans(Image.open(self.label_path))

        return image, msk

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.c1 = nn.Sequential(
             nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=(3 - 1) // 2),
             nn.ReLU(),
             nn.Dropout2d(),
             nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=(3 - 1) // 2),
             nn.ReLU(),
             nn.Dropout2d(),
         )

        self.p1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.c2 = nn.Sequential(
             nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=(3 - 1) // 2),
             nn.ReLU(),
             nn.Dropout2d(),
             nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=(3 - 1) // 2),
             nn.ReLU(),
             nn.Dropout2d(),
         )

        self.p2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.c3 = nn.Sequential(
             nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=(3 - 1) // 2),
             nn.ReLU(),
             nn.Dropout2d(),
             nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=(3 - 1) // 2),
             nn.ReLU(),
             nn.Dropout2d(),
         )

        self.p3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.c4 = nn.Sequential(
             nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=(3 - 1) // 2),
             nn.ReLU(),
             nn.Dropout2d(),
             nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=(3 - 1) // 2),
             nn.ReLU(),
             nn.Dropout2d(),
         )

        self.p4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.c4 = nn.Sequential(
             nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=(3 - 1) // 2),
             nn.ReLU(),
             nn.Dropout2d(),
             nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=(3 - 1) // 2),
             nn.ReLU(),
             nn.Dropout2d(),
         )

        self.p4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.c5 = nn.Sequential(
             nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=(3 - 1) // 2),
             nn.ReLU(),
             nn.Dropout2d(),
             nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=(3 - 1) // 2),
             nn.ReLU(),
             nn.Dropout2d(),
         )

        self.u6 = nn.ConvTranspose2d(256, 128, 2, stride=2)

        self.c6 = nn.Sequential(
             nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=(3 - 1) // 2),
             nn.ReLU(),
             nn.Dropout2d(),
             nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=(3 - 1) // 2),
             nn.ReLU(),
             nn.Dropout2d(),
         )

        self.u7 = nn.ConvTranspose2d(128, 64, 2, stride=2)

        self.c7 = nn.Sequential(
             nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=(3 - 1) // 2),
             nn.ReLU(),
             nn.Dropout2d(),
             nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=(3 - 1) // 2),
             nn.ReLU(),
             nn.Dropout2d(),
         )

        self.u8 = nn.ConvTranspose2d(64, 32, 2, stride=2)

        self.c8 = nn.Sequential(
             nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=(3 - 1) // 2),
             nn.ReLU(),
             nn.Dropout2d(),
             nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=(3 - 1) // 2),
             nn.ReLU(),
             nn.Dropout2d(),
         )

        self.u9 = nn.ConvTranspose2d(32, 16, 2, stride=2)

        self.c9 = nn.Sequential(
             nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=(3 - 1) // 2),
             nn.ReLU(),
             nn.Dropout2d(),
             nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=(3 - 1) // 2),
             nn.ReLU(),
             nn.Dropout2d(),
         )

        # self.final_out = nn.Conv2d(16, 1, 1)
        self.final_out = nn.Conv2d(in_channels=16, out_channels=2, kernel_size=3, padding=(3 - 1) // 2)


    def forward(self, x):
        #Encoder
        x1 = self.p1(self.c1(x))
        x2 = self.p2(self.c2(x1))
        x3 = self.p3(self.c3(x2))
        x4 = self.p4(self.c4(x3))
        x5 = self.c5(x4)
        #Decoder
        x6 = self.u6(x5)
        x7 = torch.cat([x6, self.c4(x3)], 1)
        x8 = self.u7(self.c6(x7))
        x9 = torch.cat([x8, self.c3(x2)], 1)
        x10 = self.u8(self.c7(x9))
        x11 = torch.cat([x10, self.c2(x1)], 1)
        x12 = self.u9(self.c8(x11))
        x13 = torch.cat([x12, self.c1(x)], 1)
        x14 = self.c9(x13)
        x_final = self.final_out(x14)

        # x_final = F.softmax(x_final, 1)
        return x_final