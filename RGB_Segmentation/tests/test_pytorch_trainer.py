#################################################
# Author: Jonathan Dorsey
# Description: Unit Testing for Pytorch Trainer
#################################################
import os
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import unittest
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as tf

from RGB_Segmentation.data.Carvana_Dataset.CarvanaDS import CarvanaData
from albumentations import (Compose, RandomCrop, Resize, HorizontalFlip, ShiftScaleRotate, RandomResizedCrop, RandomBrightnessContrast, ElasticTransform, IAAAffine, IAAPerspective, OneOf)


class TestPytorchTrainer(unittest.TestCase):

    def test_image_mask_processing(self):
        """
        This test checks if the dataloader images & masks
        are being handled correctly.
        """
        # Image & Mask Paths
        test_data_path = os.getcwd() + r"\data\test_image_mask_processing"
        img_path = os.path.join(test_data_path, r'0cdf5b5d0ce1_01.jpg')
        mask_path = os.path.join(test_data_path, r'0cdf5b5d0ce1_01_mask.jpg')

        # Image Dimensions
        image_size = [512, 512]

        # Use OpenCV to read Image/Masks from given paths
        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, 0)

        image, mask = Image.fromarray(image), Image.fromarray(mask)
        # Resize the Img & Mask
        r_img, r_msk = tf.resize(image, image_size), tf.resize(mask, image_size)

        # Convert to tensor
        image, mask = tf.to_tensor(r_img), tf.to_tensor(r_msk)


        # Instantiate Carvana DataSet & DataLoader
        carv_ds = CarvanaData()
        carvana_dl = DataLoader(carv_ds)
        car_iter = iter(carvana_dl)

        dl_img, dl_msk = next(car_iter)

        print(f"Resized Image Dims = {type(image)}")
        print(f"Resized Mask Dims = {type(mask)}")
        print(f"DL Image Dims = {type(dl_img)}")
        print(f"DL Mask Dims = {type(dl_msk)}")

        self.assertEqual(True, False)

    def test_resize_n_recrop(self):
        pass

    def test_image_normalization(self):
        pass

    def test_albumentations_transformation(self):
        pass

    def test_dice_metric(self):
        pass

    def test_loss_function(self):
        pass

    def test_dataset(self):
        pass

    def test_dataloader(self):
        pass

    def test_nn_model(self):
        pass


if __name__ == '__main__':
    unittest.main()
