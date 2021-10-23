import os
import cv2
from pathlib import Path
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, datasets
import torchvision.transforms.functional as tf
from albumentations import (Compose, RandomCrop, Resize, HorizontalFlip, ShiftScaleRotate, RandomResizedCrop, RandomBrightnessContrast, ElasticTransform, IAAAffine, IAAPerspective, OneOf)


class CarvanaData(Dataset):
    def __init__(self, root_dir=os.path.join(Path(__file__).parents[0], "data\\train"), transform=None, image_size=(512, 512), test=False):

        # Initialize Directory Tree from current working directory if no directory is provided
        if test:
            self.root_dir = os.path.join(Path(__file__).parents[0], "data\\val")
        else:
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

    def resize_crop_transform(self, image, mask, normalize=False):
        """
        Applies Random Crop/Resize Transform to Image/Mask Tuple.
        This operation requires that both the mask and the image
        are transformed identically.
        :param normalize: (applies normalization if True)
        :param image:
        :param mask:
        :return: cropped & resized image/mask Pytorch tensors
        """
        # Define Random Seeds for Image/Mask Resize/Crop transform
        seed_top = np.random.randint(0, 568)
        seed_left = np.random.randint(0, 1408)

        # Apply random resize/crop transform to both image/mask
        image = tf.resized_crop(image, seed_top, seed_left, 512, 512, [self.image_height, self.image_width])
        mask = tf.resized_crop(mask, seed_top, seed_left, 512, 512, [self.image_height, self.image_width])

        # Convert to tensor
        image = tf.to_tensor(image)
        mask = tf.to_tensor(mask)

        # Optional Image Normalization
        if normalize:
            normalized_img = tf.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            return normalized_img, mask
        return image, mask

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

        # img = Image.open(img_path)
        # mask = Image.open(mask_path).convert('L')

        # Use OpenCV to read Image/Masks from given paths
        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, 0)

        # Apply Custom Transform
        if self.transform is not None:
            augment = self.transform(image=image, mask=mask)
            image, mask = augment['image'], augment['mask']

            image, mask = Image.fromarray(image), Image.fromarray(mask)
            image, mask = self.resize_crop_transform(image, mask)
            return image, mask
        # Apply Default Transform
        else:
            augment = self.album_transform(image=image, mask=mask)
            image, mask = augment['image'], augment['mask']
            image, mask = Image.fromarray(image), Image.fromarray(mask)
            image, mask = self.resize_crop_transform(image, mask)
            return image, mask


def show_batched_images(num_images=10):
    # from Carvana_Dataset.CarvanaDS import CarvanaData
    from torch.utils.data import DataLoader
    import torch
    import matplotlib.pyplot as plt

    batch_size = 2
    train_ds = CarvanaData()
    DL = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)

    for ind, batch in enumerate(DL):
        image, mask = batch

        real_image = image[0][:][:][:]

        real_image = real_image.permute(1, 2, 0)
        plt.imshow(real_image)
        plt.show()

        if ind >= num_images:
            break


def show_batched_masks(num_masks=10):
    # from Carvana_Dataset.CarvanaDS import CarvanaData
    from torch.utils.data import DataLoader
    import torch
    import matplotlib.pyplot as plt

    batch_size = 2
    train_ds = CarvanaData()
    DL = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)

    for ind, batch in enumerate(DL):
        image, mask = batch
        real_mask = mask[0][:][:][:]

        real_mask = real_mask.permute(1, 2, 0)
        plt.imshow(real_mask)
        plt.show()

        if ind >= num_masks:
            break


if __name__ == '__main__':

    # show_batched_images()
    show_batched_masks()



