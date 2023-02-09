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



##########################################################
# set seed
##########################################################
torch.manual_seed(42)


if __name__ == '__main__':

    ##########################################################
    # dataset 
    ##########################################################
    # Dataset
    """
    dates = ["1989/10/01", "2009/09/30"]
    camel_dataset = CamelDataset(dates)
    #dataset.adjust_dates() # adjust dates if necessary
    camel_dataset.load_data() # load data
    num_basins = camel_dataset.__len__()
    seq_len = camel_dataset.seq_len
    print("Number of basins: %d" %num_basins)
    print("Number of points: %d" %seq_len)
    """
    data_ae = glob.glob("checkpoints/lstm-ae/*.ckpt")
    data_lstm = glob.glob("checkpoints/lstm/*.ckpt")
    ae_nse = []
    lstm_nse = []
    epochs_ae = []
    epochs_lstm = []
    for file in data_ae:
        epoch = re.findall(r'\b\d+\b', file)
        epochs_ae.append(int(epoch[0]))
        checkpoint_ae = torch.load(file, map_location=lambda storage, loc: storage)
        ae_nse.append(-checkpoint_ae["callbacks"]["ModelCheckpoint{'monitor': 'val_loss', 'mode': 'min', 'every_n_train_steps': 0, 'every_n_epochs': 1, 'train_time_interval': None}"]["current_score"].item())

    for file in data_lstm:
        epoch = re.findall(r'\b\d+\b', file)
        epochs_lstm.append(int(epoch[0]))
        checkpoint_ae = torch.load(file, map_location=lambda storage, loc: storage)
        lstm_nse.append(-checkpoint_ae["callbacks"]["ModelCheckpoint{'monitor': 'val_loss', 'mode': 'min', 'every_n_train_steps': 0, 'every_n_epochs': 1, 'train_time_interval': None}"]["current_score"].item())
        
    fig2, ax2 = plt.subplots(1,2,figsize=(10,10), sharey=True)
    epochs_ae, ae_nse = zip(*sorted(zip(epochs_ae, ae_nse)))
    epochs_lstm, lstm_nse = zip(*sorted(zip(epochs_lstm, lstm_nse)))
    ax2[0].plot(epochs_ae,ae_nse, label="LSTM-AE")
    ax2[1].plot(epochs_lstm,lstm_nse, label="LSTM")
    ax2[0].set_xlabel("epoch")
    ax2[0].set_ylabel("NSE")
    ax2[1].set_xlabel("epoch")
    ax2[1].set_ylabel("NSE")
    fig2.savefig("hydro-lstm-ae_NSE.png")
    
    # # find maximum
    # index = np.argmax(val_loss)
    # print(index)
    # max_epoch = epochs[index]
    # print("Best model attained at epoch: %d" %max_epoch)
    # print("Best model NSE: %d"%-val_loss[index])
 
   