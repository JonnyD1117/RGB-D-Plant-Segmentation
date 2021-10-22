from tqdm import tqdm

from RGB_Segmentation.data.Carvana_Dataset.CarvanaDS import CarvanaData
from RGB_Segmentation.data.Carvana_Dataset.ValDS import ValidationData

import torch
from torch import optim
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.utils.data import DataLoader

import optuna
from optuna.trial import TrialState

from RGB_Segmentation.models.unets.off_the_shelf_unet_w_sigmoid import UNet
from RGB_Segmentation.models.losses.bce_dice_loss import DiceBCELoss
from RGB_Segmentation.models.losses.dice_loss import DiceLoss


def objective(trial):
    # Define Training Parameters
    EPOCHS = trial.suggest_int('epochs', 30, 100)
    lr = trial.suggest_uniform('lr', 1e-5, 1e-1)
    batch_size = trial.suggest_int('batch_size', 5, 15)
    val_batch_size = trial.suggest_int('val_batch_size', 5, 10)

    img_height = 517
    img_width = 517

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define Model
    model = UNet().to(device)

    # Define Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    # Define Loss Function
    # loss_criterion = [DiceBCELoss().to(device), DiceLoss().to(device), nn.BCEWithLogitsLoss.to(device)][0]
    loss_criterion = DiceBCELoss().to(device)

    # Create DataLoader
    carvana_dl = DataLoader(CarvanaData(test=False), batch_size, shuffle=True, num_workers=4, drop_last=True)
    validation_dl = DataLoader(ValidationData(), val_batch_size, shuffle=False, num_workers=4, drop_last=True)

    # Loop Through All Epochs
    for epoch in range(EPOCHS):
        model = model.train()
        for image, mask in tqdm(carvana_dl):
            image = image.to(device)
            mask = mask.to(device)

            pred = model(image)
            loss = loss_criterion(pred, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            mean_val_loss = 0
            val_ctr = 1.0
            # Compute Validation Loss
            for val_img, val_mask in tqdm(validation_dl):
                model = model.eval()

                val_img = val_img.to(device)
                val_mask = val_mask.to(device)
                val_pred = model(val_img)
                val_loss = loss_criterion(val_pred, val_mask)
                mean_val_loss = mean_val_loss + (val_loss - mean_val_loss)/val_ctr
                val_ctr += 1

        scheduler.step(loss)


if __name__ == '__main__':

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=75)

    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))








