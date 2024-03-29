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


class ValidationData(Dataset):
    def __init__(self, root_dir=os.path.join(Path(__file__).parents[0], "data\\val"), transform=None, image_size=(512, 512)):

        # Initialize Directory Tree from current working directory if no directory is provided
        self.root_dir = root_dir

        # Get Image/Mask Path
        self.img_dir = os.path.join(self.root_dir, 'images')
        self.mask_dir = os.path.join(self.root_dir, 'masks')

        # Get List of Images/Masks in Carvana Directory
        self.img_list = os.listdir(self.img_dir)
        self.mask_list = os.listdir(self.mask_dir)

        # Get Number of Images/Masks
        self.num_img = len(self.img_list)
        self.num_mask = len(self.mask_list)

        # Define Transform Image Dimensions
        self.image_height = image_size[1]
        self.image_width = image_size[0]

        # Define Custom Augmentation Transform
        self.transform = transform

        # Define Default Aug Transformation
        self.album_transform = Compose([
            HorizontalFlip(),
            ShiftScaleRotate(
                shift_limit=0.0625,
                scale_limit=0.2,
                rotate_limit=45,
                p=0.2),
            OneOf([
                ElasticTransform(p=.2),
                IAAPerspective(p=.35),
            ], p=.35)
        ])

    def __len__(self):
        """
        Define the length of the dataset.
        """
        # Check if number of image/masks are equal
        if self.num_img == self.num_mask:
            return self.num_img
        else:
            raise Exception("Number of Images & GT Masks is NOT equal")

    def __getitem__(self, item):
        """
        Get the image/mask at index "item"
        :return:
        """
        # Define full image/mask path for extracting data
        img_path = os.path.join(self.img_dir, self.img_list[item])
        mask_path = os.path.join(self.mask_dir, self.mask_list[item])

        # Use OpenCV to read Image/Masks from given paths
        img = cv2.imread(img_path)
        msk = cv2.imread(mask_path, 0)

        if self.transform is not None:
            # augment = self.album_transform(image=image, mask=mask)
            augment = self.transform(image=img, mask=msk)
            img, msk = augment['image'], augment['mask']

        # Convert & Resize Image & Mask
        img, msk = Image.fromarray(img), Image.fromarray(msk)
        image_resized = tf.resize(img=img, size=[self.image_height, self.image_width])
        mask_resized = tf.resize(img=msk, size=[self.image_height, self.image_width])

        # Normalize Image but NOT mask (implicit in applied transforms)
        image_ten = tf.to_tensor(image_resized).float()
        mask_ten = tf.pil_to_tensor(mask_resized).float()
        return image_ten, mask_ten


if __name__ == '__main__':

    c_ds = ValidationData()
    dl = DataLoader(c_ds, batch_size=1, shuffle=True)

    for image, mask in dl:

        print(f"Image Shape = {image.shape}, type = {type(image)}, min = {torch.min(image)} max  = {torch.max(image)}")
        print(f"Mask Shape = {mask.shape}, type = {type(mask)}, min = {torch.min(mask)} max  = {torch.max(mask)}")
        break
