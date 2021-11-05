import os
import cv2
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, datasets
import torchvision.transforms.functional as tf
from albumentations import (Compose, RandomCrop, Resize, HorizontalFlip, ShiftScaleRotate, RandomResizedCrop, RandomBrightnessContrast, ElasticTransform, IAAAffine, IAAPerspective, OneOf)
from tqdm import tqdm
import matplotlib.pyplot as plt

if __name__ == '__main__':

    Num_Epochs = 3
    base_path = Path(__file__).parents[1]
    data_path = r'data\Carvana_Dataset\data\train'

    train_path = os.path.join(base_path, data_path)
    train_img_path = os.path.join(train_path, r'images')
    train_msk_path = os.path.join(train_path, r'masks')

    aug_path = os.path.join(base_path, r'data\Carvana_Dataset\data\aug_train')
    aug_img_path = os.path.join(aug_path, r'images')
    aug_msk_path = os.path.join(aug_path, r'masks')

    # Define Default Aug Transformation
    album_transform = Compose([
        HorizontalFlip(),
        ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.2,
            rotate_limit=45,
            p=0.2),
        OneOf([
            ElasticTransform(p=.2),
            IAAPerspective(p=.35),
            RandomBrightnessContrast(p=.2)
        ], p=.35)
    ])

    counter = 0

    for epoch in range(1, (Num_Epochs + 1)):
        print(f"NUM EPOCH = {epoch}, counter = {counter}")
        img_list = os.listdir(train_img_path)
        mask_list = os.listdir(train_msk_path)

        data_list = zip(img_list, mask_list)
        for real_img_path, real_msk_path in tqdm(data_list):

            aug_msk_path_full = aug_msk_path + f'\mask_{counter}.png'
            aug_img_path_full = aug_img_path + f'\image_{counter}.png'
            counter += 1

            img_path = os.path.join(train_img_path, real_img_path)
            msk_path = os.path.join(train_msk_path, real_msk_path)

            # Use OpenCV to read Image/Masks from given paths
            img = cv2.imread(img_path)
            msk = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)

            # Apply Augmentation
            augment = album_transform(image=img, mask=msk)
            aug_img, aug_msk = augment['image'], augment['mask']

            # Save Augmented Images/Masks
            cv2.imwrite(aug_img_path_full, aug_img)
            cv2.imwrite(aug_msk_path_full, aug_msk)
