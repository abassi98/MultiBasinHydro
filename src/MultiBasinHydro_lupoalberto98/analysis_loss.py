import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
import re

# pytorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset, Subset
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

import torch.optim as optim
from pytorch_lightning.callbacks.early_stopping import EarlyStopping 
from torchvision import transforms, datasets

# user functions
from dataset import CamelDataset
from models import Hydro_LSTM_AE
from utils import Scale_Data, MetricsCallback, NSELoss



if __name__ == '__main__':

    #####################################################################
    dir = "checkpoints"
    models = ["lstm-ae-bdFalse-E27","lstm-ae-bdTrue-E4", "lstm-ae-bdTrue-E27", "lstm-bdFalse-N0", "lstm-bdFalse-N27" ]
    epochs = []
    nse = []
    fig, ax = plt.subplots(1,1,figsize=(5,5))
    ax.set_xlabel("epoch")
    ax.set_ylabel("NSE")
    for i in range(len(models)):
        name = models[i]
        path = os.path.join(dir, name, "metrics.pt")
        data = torch.load(path, map_location=torch.device('cpu'))
        epochs_mod = []
        nse_mod = []
        for key in data:
            epochs_mod.append(data[key]["epoch_num"])
            nse_mod.append(-data[key]["val_loss"])

        # reorder
        epochs_mod, nse_mod = zip(*sorted(zip(epochs_mod, nse_mod)))
        # plot
        ax.plot(epochs_mod,nse_mod, label=name)
        # find best model
        idx_ae = np.argmax(nse_mod)
        epoch_max_nse = epochs_mod[idx_ae]
        print("Best "+name+" model obtained at epoch " +str(epoch_max_nse))
    
    
    ax.legend()
    fig.savefig("hydro-lstm-ae_NSE.png")
    

    
   
   