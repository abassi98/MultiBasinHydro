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

    # data files
    data_ae = glob.glob("checkpoints/lstm-ae/*.ckpt")
    data_lstm = glob.glob("checkpoints/lstm/*.ckpt")
    data_lstm_noise = glob.glob("checkpoints/lstm-noise-dim27/*.ckpt")
    data_bidir = glob.glob("checkpoints/lstm-ae-bidirectional/*.ckpt")
    data_bdir_nf5 = glob.glob("checkpoints/lstm-ae-bdTrue-E5/*.ckpt")
    # nse containers
    ae_nse = []
    lstm_nse = []
    lstm_noise_nse = []
    bidir_nse = []
    bdir_nf5_nse = []
    # epochs count container
    epochs_ae = []
    epochs_lstm = []
    epochs_lstm_noise = []
    epochs_bidir = []
    epochs_bdir_nf5 = []

    dict_ae = {}

    for file in data_ae:
        epoch = int(re.findall(r'\b\d+\b', file)[0])
        epochs_ae.append(epoch)
        checkpoint_ae = torch.load(file, map_location=lambda storage, loc: storage)
        #ae_nse.append(-checkpoint_ae["callbacks"]["ModelCheckpoint{'monitor': 'val_loss', 'mode': 'min', 'every_n_train_steps': 0, 'every_n_epochs': 1, 'train_time_interval': None}"]["current_score"].item())
        val_loss = checkpoint_ae["callbacks"]["ModelCheckpoint{'monitor': 'val_loss', 'mode': 'min', 'every_n_train_steps': 0, 'every_n_epochs': 1, 'train_time_interval': None}"]["current_score"].item()
        #epoch = checkpoint_ae["callbacks"]["ModelCheckpoint{'monitor': 'val_loss', 'mode': 'min', 'every_n_train_steps': 0, 'every_n_epochs': 1, 'train_time_interval': None}"]["epoch"].item()
        dict_ae["Epoch: "+str(epoch)] = {"val_loss" : val_loss}

    torch.save(dict_ae, "checkpoints/lstm-ae/metrics.pt")
    
    # for file in data_lstm:
    #     epoch = re.findall(r'\b\d+\b', file)
    #     epochs_lstm.append(int(epoch[0]))
    #     checkpoint_ae = torch.load(file, map_location=lambda storage, loc: storage)
    #     lstm_nse.append(-checkpoint_ae["callbacks"]["ModelCheckpoint{'monitor': 'val_loss', 'mode': 'min', 'every_n_train_steps': 0, 'every_n_epochs': 1, 'train_time_interval': None}"]["current_score"].item())
        
    # for file in data_lstm_noise:
    #     epoch = re.findall(r'\b\d+\b', file)
    #     epochs_lstm_noise.append(int(epoch[0]))
    #     checkpoint = torch.load(file, map_location=lambda storage, loc: storage)
    #     lstm_noise_nse.append(-checkpoint["callbacks"]["ModelCheckpoint{'monitor': 'val_loss', 'mode': 'min', 'every_n_train_steps': 0, 'every_n_epochs': 1, 'train_time_interval': None}"]["current_score"].item())
        
    # for file in data_bidir:
    #     epoch = re.findall(r'\b\d+\b', file)
    #     epochs_bidir.append(int(epoch[0]))
    #     checkpoint = torch.load(file, map_location=lambda storage, loc: storage)
    #     bidir_nse.append(-checkpoint["callbacks"]["ModelCheckpoint{'monitor': 'val_loss', 'mode': 'min', 'every_n_train_steps': 0, 'every_n_epochs': 1, 'train_time_interval': None}"]["current_score"].item())
        
    # for file in data_bdir_nf5:
    #     epoch = re.findall(r'\b\d+\b', file)
    #     epochs_bdir_nf5.append(int(epoch[0]))
    #     checkpoint = torch.load(file, map_location=lambda storage, loc: storage)
    #     bdir_nf5_nse.append(-checkpoint["callbacks"]["ModelCheckpoint{'monitor': 'val_loss', 'mode': 'min', 'every_n_train_steps': 0, 'every_n_epochs': 1, 'train_time_interval': None}"]["current_score"].item())
        


    # fig2, ax2 = plt.subplots(1,1,figsize=(10,10))
    # epochs_ae, ae_nse = zip(*sorted(zip(epochs_ae, ae_nse)))
    # epochs_lstm, lstm_nse = zip(*sorted(zip(epochs_lstm, lstm_nse)))
    # epochs_lstm_noise, lstm_noise_nse = zip(*sorted(zip(epochs_lstm_noise, lstm_noise_nse)))
    # epochs_bidir, bidir_nse = zip(*sorted(zip(epochs_bidir, bidir_nse)))
    # epochs_bdir_nf5, bdir_nf5_nse = zip(*sorted(zip(epochs_bdir_nf5, bdir_nf5_nse)))

    # ax2.plot(epochs_ae,ae_nse, label="LSTM-AE 27 Features")
    # ax2.plot(epochs_lstm,lstm_nse, label="LSTM")
    # ax2.plot(epochs_lstm_noise,lstm_noise_nse, label="LSTM + 27 Noise")
    # ax2.plot(epochs_bidir,bidir_nse, label="LSTM Bidirectional AE (27 Features)")
    # ax2.plot(epochs_bdir_nf5,bdir_nf5_nse, label="LSTM Bidirectional AE (5 Features)")

    # ax2.set_xlabel("epoch")
    # ax2.set_ylabel("NSE")
    # ax2.legend()
    # fig2.savefig("hydro-lstm-ae_NSE.png")
    
    # ### find the best models
    # idx_ae = np.argmax(ae_nse)
    # epoch_max_nse = epochs_ae[idx_ae]
    # print("Best LSTM-AE (27 features) model obtained at epoch %d"%epoch_max_nse)
    # idx_lstm = np.argmax(lstm_nse)
    # epoch_max_nse = epochs_lstm[idx_lstm]
    # print("Best LSTM model obtained at epoch %d"%epoch_max_nse)
    # idx_noise = np.argmax(lstm_noise_nse)
    # epoch_max_nse = epochs_lstm_noise[idx_noise]
    # print("Best LSTM + Noise (27 features) model obtained at epoch %d"%epoch_max_nse)
    # idx_noise = np.argmax(bidir_nse)
    # epoch_max_nse = epochs_bidir[idx_noise]
    # print("Best LSTM-AE-BIDIR (27 features) model obtained at epoch %d"%epoch_max_nse)
    # idx_noise = np.argmax(bdir_nf5_nse)
    # epoch_max_nse = epochs_bdir_nf5[idx_noise]
    # print("Best LSTM-AE-BIDIR (5 features) model obtained at epoch %d"%epoch_max_nse)