import os
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, datasets
import torchvision.transforms.functional as tf
from albumentations import (Compose, RandomCrop, Resize, HorizontalFlip, ShiftScaleRotate, RandomResizedCrop, RandomBrightnessContrast, ElasticTransform, IAAAffine, IAAPerspective, OneOf)


class TestData(Dataset):
    def __init__(self, root_dir=r"C:\Users\Indy-Windows\Documents\RGB-D-Plant-Segmentation\Carvana_Dataset\data\test", image_size=(512, 512)):

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

    def resize_transform(self, image, mask, normalize=False):
        """
        Transform resize the image/mask and optionally normalizes the image
        """
        # Resize Image/Mask
        image = tf.resize(image, self.image_height)
        mask = tf.resize(mask, self.image_height)

        # Convert Image/Mask to Pytorch Tensors
        image = tf.to_tensor(image)
        mask = tf.to_tensor(mask)

        # Apply optional Image Normalization
        if normalize:
            normalized_img = tf.normalize(image, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            return normalized_img, mask
        return image, mask

    def __len__(self):
        """
        Given the length of the dataset
        """
        if self.num_img == self.num_mask:
            return self.num_img
        else:
            raise Exception("Number of Images & GT Masks is NOT equal")

    def __getitem__(self, item):
        """
        Return the Image/Mask for the given index "item"
        """
        # Define the full Image/Mask Paths
        img_path = os.path.join(self.img_dir, self.img_list[item])
        mask_path = os.path.join(self.mask_dir, self.mask_list[item])

        # img = Image.open(img_path)
        # mask = Image.open(mask_path).convert('L')

        # Use OpenCv to read-in image/mask from file
        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, 0)

        # Convert Image/Mask and Resize
        image, mask = Image.fromarray(image), Image.fromarray(mask)
        image, mask = self.resize_transform(image, mask)
        return image, mask

def show_batched_images(num_images=10):
    # from Carvana_Dataset.CarvanaDS import CarvanaData
    from torch.utils.data import DataLoader
    import torch
    import matplotlib.pyplot as plt

    batch_size = 2
    train_ds = TestData()
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
    train_ds = TestData()
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
