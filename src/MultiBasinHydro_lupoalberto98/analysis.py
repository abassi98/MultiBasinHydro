import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob

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
    data_ae = glob.glob("chcekpoints/lstm-ae/*.ckpt)")
    data_lstm = glob.glob("chcekpoints/lstm/*.ckpt)")
    dir_ae="checkpoints/lstm-ae/"
    dir_lstm="checkpoints/lstm/"
    ae_nse = []
    lstm_nse = []
    epochs = []
    for file in data_ae:
        # epoch = str(i*10 +9).rjust(2,"0") 
        # file_ae="hydro-lstm-ae-epoch="+epoch+".ckpt"
        # path_ae = os.path.join(dir_ae, file_ae)
        # file_lstm="hydro-lstm-epoch="+epoch+".ckpt"
        # path_lstm = os.path.join(dir_lstm, file_lstm)
        
        # retrieve validation loss for plotting
        checkpoint_ae = torch.load(file, map_location=lambda storage, loc: storage)
        ae_nse.append(-checkpoint_ae["callbacks"]["ModelCheckpoint{'monitor': 'val_loss', 'mode': 'min', 'every_n_train_steps': 0, 'every_n_epochs': 1, 'train_time_interval': None}"]["current_score"].item())
        # checkpoint_lstm = torch.load(path_lstm, map_location=lambda storage, loc: storage)
        # lstm_nse.append(-checkpoint_lstm["callbacks"]["ModelCheckpoint{'monitor': 'val_loss', 'mode': 'min', 'every_n_train_steps': 0, 'every_n_epochs': 1, 'train_time_interval': None}"]["current_score"].item())
        # epochs.append(float(epoch))

        # # plot weights of linear feature space
        # model = Hydro_LSTM_AE.load_from_checkpoint(path)
        # enc_liner_2 = model.encoder.encoder_lin[4].weight # select last linear encoder layer
        # fig, axs = plt.subplots(9,3, figsize=(18,6), sharex=True)
        # for j in range(3):
        #     for i in range(9):
        #         ax = axs[i,j]
        #         feature = j*9 + i 
        #         ax.set_xlabel(str(feature))
        #         ax.hist(enc_liner_2.detach().numpy()[feature,:], density=True)
        # path_hist = os.path.join("hist/","hist-epoch="+epoch+".png")
        # fig.savefig(path_hist)
        
    for file in data_lstm:
        checkpoint_ae = torch.load(file, map_location=lambda storage, loc: storage)
        ae_nse.append(-checkpoint_ae["callbacks"]["ModelCheckpoint{'monitor': 'val_loss', 'mode': 'min', 'every_n_train_steps': 0, 'every_n_epochs': 1, 'train_time_interval': None}"]["current_score"].item())
        
    fig2, ax2 = plt.subplots(1,1,figsize=(10,10))
    ax2.plot(epochs,ae_nse, label="LSTM-AE")
    ax2.plot(epochs,lstm_nse, label="LSTM")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("NSE")
    fig2.savefig("hydro-lstm-ae_NSE.png")
    
    # # find maximum
    # index = np.argmax(val_loss)
    # print(index)
    # max_epoch = epochs[index]
    # print("Best model attained at epoch: %d" %max_epoch)
    # print("Best model NSE: %d"%-val_loss[index])
 
   