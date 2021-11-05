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

"""   
class UNetEnc(nn.Module):

    def __init__(self, in_channels, features, out_channels):
        super().__init__()

        self.up = nn.Sequential(
            nn.Conv2d(in_channels, features, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, 3),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(features, out_channels, 2, stride=2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.up(x)


class UNetDec(nn.Module):

    def __init__(self, in_channels, out_channels, dropout=False):
        super().__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers += [nn.Dropout(.5)]
        layers += [nn.MaxPool2d(2, stride=2, ceil_mode=True)]

        self.down = nn.Sequential(*layers)

    def forward(self, x):
        return self.down(x)



class UNet(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        self.dec1 = UNetDec(3, 64)
        self.dec2 = UNetDec(64, 128)
        self.dec3 = UNetDec(128, 256)
        self.dec4 = UNetDec(256, 512, dropout=True)
        self.center = nn.Sequential(
            nn.Conv2d(512, 1024, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.ConvTranspose2d(1024, 512, 2, stride=2),
            nn.ReLU(inplace=True),
        )
        self.enc4 = UNetEnc(1024, 512, 256)
        self.enc3 = UNetEnc(512, 256, 128)
        self.enc2 = UNetEnc(256, 128, 64)
        self.enc1 = nn.Sequential(
            nn.Conv2d(128, 64, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        dec1 = self.dec1(x)
        dec2 = self.dec2(dec1)
        dec3 = self.dec3(dec2)
        dec4 = self.dec4(dec3)
        center = self.center(dec4)
        enc4 = self.enc4(torch.cat([
            center, F.upsample_bilinear(dec4, center.size()[2:])], 1))
        enc3 = self.enc3(torch.cat([
            enc4, F.upsample_bilinear(dec3, enc4.size()[2:])], 1))
        enc2 = self.enc2(torch.cat([
            enc3, F.upsample_bilinear(dec2, enc3.size()[2:])], 1))
        enc1 = self.enc1(torch.cat([
            enc2, F.upsample_bilinear(dec1, enc2.size()[2:])], 1))

        return F.upsample_bilinear(self.final(enc1), x.size()[2:])

"""

if __name__ == '__main__':

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("TORCH: Running on GPU-", device)
    else:
        device = torch.device("cpu")
        print("TORCH: Running on CPU")


    img_trans = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # transforms.ColorJitter(1,1,1),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225])
    ])

    msk_trans = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()

    ])

    img_path = "C:/Users/Indy-Windows/Desktop/Data/Data/edit/with"

    out_path_1 = r"C:\Users\Indy-Windows\Desktop\val_data\image1.jpg"
    out_path_2 = r"C:\Users\Indy-Windows\Desktop\val_data\mask1.jpg"


    Segmentation_Dataset = ImgSegDataSet(img_path, img_transform=img_trans, mask_transform=msk_trans)
    dataLoader = DataLoader(Segmentation_Dataset, batch_size=20, shuffle=True, num_workers=8)

    nn_model = UNet()
    nn_model.train()

    nn_model.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(nn_model.parameters())

    EPOCH = 5

    for epoch in range(EPOCH):
        out_path_1 = r"C:\Users\Indy-Windows\Desktop\val_data\image" + str(epoch) + ".jpg"
        out_path_2 = r"C:\Users\Indy-Windows\Desktop\val_data\mask" + str(epoch) + ".jpg"

        for image, mask in tqdm(dataLoader):
            image, mask = image.to(device), mask.to(device)

            # print(f"IMAGE SHAPE = {images.shape}, IMAGE TYPE = {images.type()}")
            # print(f"MASK SHAPE = {mask.shape}, MASK TYPE = {mask.type()}")
            #
            # print("#####################################----IMAGE----##################################################")
            # print(images)
            # print("#####################################------MASK------#####################################")
            # print(mask)
            #
            # print(f"IMAGE MAX {torch.max(images)}, IMAGE MIN {torch.min(images)}, IMAGE MEAN {torch.mean(images)}")


            output = nn_model(image)
            soft_max_output = F.softmax(output, 1)
            # print(f"OUTPUT SHAPE = {output.shape}, OUTPUT TYPE = {output.type()}")
            # print(f"SM-OUTPUT SHAPE = {soft_max_output.shape}, SM-OUTPUT TYPE = {soft_max_output.type()}")

            # print("#####################################----OUTPUT----##################################################")
            # print(output)
            # print("#####################################------SOFT MAX OUTPUT------#####################################")
            # print(soft_max_output)

            # loss = loss_function(soft_max_output, mask.squeeze(dim=1).to(dtype=torch.int64))
            loss = loss_function(output, mask.squeeze(dim=1).to(dtype=torch.int64))


            n_mask = mask.squeeze(dim=1)


            # print(f"NEW MASK SHAPE = {n_mask.shape}, NEW MASK TYPE = {n_mask.type()}")



            loss.backward()
            optimizer.step()



        out_image = soft_max_output.detach().cpu()
        print(out_image.type())
        # out_image = out_image.detach()
        out_image = out_image[0,0,:,:]
        print(soft_max_output)
        # plt.imshow(out_image.numpy())
        # plt.show()

        normalizedImg = np.zeros((224, 224))
        normalizedImg = cv2.normalize(out_image.numpy(), normalizedImg, 255, 0, cv2.NORM_MINMAX)
        cv2.imwrite(out_path_2, normalizedImg)

        image = image[0,0,:,:]
        new_image = image.detach().cpu()

        norm_img = np.zeros((224, 224))
        norm_img = cv2.normalize(new_image.numpy(), normalizedImg, 255, 0, cv2.NORM_MINMAX)

        cv2.imwrite(out_path_1, norm_img)




    """   

    phase = 'train'
    with torch.set_grad_enabled(phase == 'train'):  # dynamically set gradient computation, in case of validation, this isn't needed


        model = UNet()
        model.train()

        criterion = nn.CrossEntropyLoss()

        # Only train the classifier parameters, feature parameters are frozen
        optimizer = optim.Adam(model.parameters(), lr=.001)

        scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

        model.to(device)

        total_train = []
        correct_train = []

        epochs = 2
        steps = 0
        running_loss = 0

        print(torch.cuda.get_device_name())

        for epoch in range(epochs):
            # print("EPOCH:", epoch)
            scheduler.step()
            print('Epoch:', epoch, 'LR:', scheduler.get_lr())

            for inputs, masks in tqdm(dataLoader):
                inputs, masks = inputs.to(device), masks.to(device)
                # print(f"IMAGE SHAPE: {inputs.shape} & MASKS SHAPE: {masks.shape}")

                optimizer.zero_grad()
                print("Mask Shape", masks.shape)
                masks = torch.argmax(masks, dim=1)

                print("MASK New Shape", masks.shape)
                output = model.forward(inputs.view(-1, 3, 224, 224))

                loss = criterion(output, masks)
                print("LOSS VAL:", loss.data)

                loss.backward()
                optimizer.step()
                running_loss += loss.item()

        checkPath = "C:\\Users\\Indy-Windows\\Documents\\Git Repos\\MachineLearning_Exercises\\Pytorch\\3_ImageSegmentation\\NN_Model\\Checkpoints"

        checkpoint = {
                'parameters': model.parameters,
                'state_dict': model.state_dict()
            }

        torch.save(checkpoint, checkPath + '\\ImgSeg_Cust_UNET_DOG_1_ONECHAN_04.pth')
































    # for inputs, masks in tqdm(Segmentation_Dataset):
    #     inputs, masks = inputs.to(device), masks.to(device)
    #     print("INPUT SHAPE", inputs.shape)
    #     print("Ground Truth SHAPE", masks.shape)
    #
    #     output = model.forward(inputs.view(-1, 3, 128, 128))
    #     print("OUTPUT", output.shape)
    #     break
    #
    # output = output[0, 0, : ,: ].cpu().detach().numpy()
    # print(output.shape)
    # print(type(output))
    #
    # plt.imshow(output)
    # plt.show()
    #
    #
    #
    #

"""