import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

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
    dirpath="checkpoints/lstm-ae/"
    val_loss = []
    for i in range(14):
        epoch = str(i*10 +9).rjust(2,"0") 
        filename="hydro-lstm-ae-epoch="+epoch+".ckpt"
        path = os.path.join(dirpath, filename)
        
        # retrieve validation loss for plotting
        checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
        val_loss.append(-checkpoint["callbacks"]["ModelCheckpoint{'monitor': 'val_loss', 'mode': 'min', 'every_n_train_steps': 0, 'every_n_epochs': 1, 'train_time_interval': None}"]["current_score"].item())

        # plot weights of linear feature space
        model = Hydro_LSTM_AE.load_from_checkpoint(path)
        enc_liner_2 = model.encoder.encoder_lin[4].weight # select last linear encoder layer
        fig, axs = plt.subplots(9,3, figsize=(18,6), sharex=True)
        for j in range(3):
            for i in range(9):
                ax = axs[i,j]
                feature = j*9 + i 
                ax.set_xlabel(str(feature))
                ax.hist(enc_liner_2.detach().numpy()[feature,:], density=True)
        path_hist = os.path.join("hist/","hist-epoch="+epoch+".png")
        fig.savefig(path_hist)
        
    fig2, ax2 = plt.subplots(1,1,figsize=(10,10))
    ax2.plot(val_loss)
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("NSE")
    fig2.savefig("hydro-lstm-ae_NSE.png")

 
 
   