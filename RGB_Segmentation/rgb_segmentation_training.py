"""
Author: Jonathan Dorsey
Date: 10/9/2021
Summary:

This file contains the pytorch lightning implementation for training a UNET model for RGB carvana dataset
"""

import os
from argparse import ArgumentParser

from Carvana_Dataset.CarvanaDS import CarvanaData
from Carvana_Dataset.ValDS import ValidationData

import torch
import torchvision
from torch import optim
from torch.utils.data import DataLoader, random_split

import pytorch_lightning as pl
from pytorch_lightning.core import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.profiler import SimpleProfiler, AdvancedProfiler

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.model_checkpoint import Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from torchmetrics.functional import dice_score
from pl_bolts.losses.object_detection import iou_loss

# from RGB_Segmentation.unet import UNet
from RGB_Segmentation.off_the_shelf_unet import UNet
from RGB_Segmentation.dice_loss import DiceLoss


class CarvanaUnetModel(LightningModule):

    def __init__(self,
                 drop_prob: float = 0.35,
                 batch_size: int = 5,
                 learning_rate: float = 0.0003,
                 ):
        # init superclass
        super().__init__()
        self.model = UNet()
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.drop_prob = drop_prob
        self.ds = None
        self.vs = None
        self.ts = None

        self.loss_criterion = DiceLoss()

    def forward(self, x):
        """
        Forward method passes the input data into the NN model.
        Since Lighting is a wrapper around Pytorch this method
        will be called by Pytorch later during the training step.
        :param x: Unet Model Input Data
        :return: Unet Model Output Mask
        """
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop with the data from the training dataloader
        passed in as `batch`.
        """
        # Extract Input & Ground Truth from batched dataloader
        x, y = batch
        # Forward Pass through NN model
        y_hat = self(x)

        # Compute Loss
        loss = self.loss_criterion(y_hat, y)

        # Process Batched Images/Masks into a grid for logging
        grid_image = torchvision.utils.make_grid(x)
        grid_pred = torchvision.utils.make_grid(y_hat)
        grid_mask = torchvision.utils.make_grid(y)

        # Pass Images, Ground Truth, and Mask Prediction to Tensorboard Logger
        self.logger.experiment.add_image("Input_Image", grid_image, 0)
        self.logger.experiment.add_image("GT_Mask", grid_mask, 0)
        self.logger.experiment.add_image("Pred_Mask", grid_pred, 0)

        # self.logger.experiment.add_

        # tensorboard_logs = {'train_loss': loss, 'lr': self.learning_rate, 'train_dice': dice_s}
        #
        # output = {
        #     "loss": loss,
        #     "training_loss": loss,
        #     "progress_bar": tensorboard_logs,
        #     "log": tensorboard_logs
        # }

        self.log("loss", loss, prog_bar=True) #, "training_loss": loss, "progress_bar": tensorboard_logs, "log": tensorboard_logs)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop with the data from the validation dataloader
        passed in as `batch`.
        """
        # Extract Input & Ground Truth from batched dataloader
        x, y = batch
        # Forward Pass through NN model
        y_hat = self(x)

        # Compute Loss
        dice_s = dice_score(y_hat, y)
        val_loss = self.loss_criterion(y_hat, y)

        val_tensorboard_logs = {'val_loss': val_loss, "val_dice_loss": val_loss}

        output = {
            "val_loss": val_loss,
            "progress_bar": val_tensorboard_logs,
            "log": val_tensorboard_logs
        }

        # self.log("loss", val_loss, prog_bar=True)
        # self.log(output)
        # return output

    # def validation_epoch_end(self, outputs):
    #     """
    #     Called at the end of validation to aggregate outputs.
    #     :param outputs: list of individual outputs of each validation step.
    #     """
    #     avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
    #     avg_dice = torch.stack([x['dice_loss'] for x in outputs]).mean()
    #
    #     tensorboard_logs = {'val_loss': avg_loss, '    val_dice_score': avg_dice}
    #     print(f" \n Validation INFO: Val Loss {avg_loss.item()} Val DiceLoss {avg_dice.item()}  Val BCE {avg_loss.item()-avg_dice.item()}")
    #     return {'val_loss': avg_loss, "dice_score": avg_dice, 'log': tensorboard_logs}

    def configure_optimizers(self):
        """
        Configures the ADAM optimizer and a Learning-Rate Scheduler
        """
        # Define Optimizer & Learning Rate Scheduler
        reduce_on_plateau = False
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

        # Declare which learning rate scheduler will run
        if reduce_on_plateau:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, threshold_mode='rel', verbose=True)
            return [optimizer], [{'scheduler': scheduler, 'reduce_on_plateau': True, 'monitor': 'training_loss'}]

        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        return [optimizer], [{'scheduler': scheduler}]

    def prepare_data(self):
        """
        Setup the Training, Validation, & Testing Datasets
        """
        self.ds = CarvanaData()
        self.vs = ValidationData()
        self.ts = None

    def train_dataloader(self):
        """
        Pass training dataset to dataloader and return dataloader
        """
        # self.log.info('Training data loader called.')
        return DataLoader(self.ds, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        """
        Pass validation dataset to dataloader and return dataloader
        """
        # self.log.info('Validation data loader called.')
        return DataLoader(self.vs, batch_size=self.batch_size, shuffle=False, num_workers=4)


def main(hparams):

    tb_logger = TensorBoardLogger('logs', name='my_model')
    # cp_cb = ModelCheckpoint(filepath='models/checkpoints/{epoch}-{val_loss:.3f}', save_top_k=-1)
    model = CarvanaUnetModel()

    learner = Trainer(fast_dev_run=False, logger=tb_logger, accumulate_grad_batches=2, check_val_every_n_epoch=1,
                      min_epochs=80, max_epochs=200, gpus=1) #, checkpoint_callback=cp_cb)

    # # # Run learning rate finder
    # lr_finder = learner.lr_find(model)
    # new_lr = lr_finder.suggestion()
    # model.learning_rate = new_lr
    # print(new_lr)

    model.learning_rate = 0.0003
    learner.fit(model)


if __name__ == "__main__":
    # early_stop_callback = EarlyStopping(
    #     monitor='val_loss',
    #     min_delta=0.00,
    #     patience=50,
    #     verbose=False,
    #     mode='max'
    # )

    parser = ArgumentParser()
    args = parser.parse_args()
    main(args)
