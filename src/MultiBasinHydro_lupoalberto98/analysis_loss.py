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
    data = glob.glob("checkpoints/lstm/*.ckpt")
    data.remove("checkpoints/lstm/last.ckpt")
    epochs = []
    nse = []
    for file in data:
        epoch = re.findall(r'\b\d+\b', file)
        epoch = int(epoch[0])
        epochs.append(epoch)
        checkpoint = torch.load(file, map_location=torch.device('cpu'))
        val_loss = checkpoint["callbacks"]["ModelCheckpoint{'monitor': 'val_loss', 'mode': 'min', 'every_n_train_steps': 0, 'every_n_epochs': 1, 'train_time_interval': None}"]["current_score"].item()
        nse.append(-val_loss)
        

    best_model_idx = np.argpartition(nse, -10)[-10:]
    best_model_idx = np.sort(best_model_idx)
    best_data = [data[i] for i in best_model_idx]
    data_to_delete = list(set(data) - set(best_data))
    print(data_to_delete)
    for file in data_to_delete:
        os.remove(file)


    # #####################################################################
    # # data = glob.glob("checkpoints/lstm-ae/*.ckpt")
    # data = torch.load("checkpoints/lstm-ae/metrics.pt")
    # dict = {}
    # #for file in data:
    # #epoch = re.findall(r'\b\d+\b', file)
    # #epoch = int(epoch[0])
    # epochs_ae = []
    # ae_nse = []
    # for key in data:
    #     epochs_ae.append(data[key]["epoch_num"])
    #     ae_nse.append(-data[key]["val_loss"])
    
        
    # #val_loss = checkpoint["callbacks"]["ModelCheckpoint{'monitor': 'val_loss', 'mode': 'min', 'every_n_train_steps': 0, 'every_n_epochs': 1, 'train_time_interval': None}"]["current_score"].item()
    # #epoch = checkpoint_ae["callbacks"]["ModelCheckpoint{'monitor': 'val_loss', 'mode': 'min', 'every_n_train_steps': 0, 'every_n_epochs': 1, 'train_time_interval': None}"]["epoch"].item()
    # #dict["Epoch: "+str(epoch)] = {"val_loss" : val_loss, "epoch_num" : epoch}


    # #####################################################################
    # data = torch.load("checkpoints/lstm/metrics.pt")
    # epochs_lstm = []
    # lstm_nse = []
    # for key in data:
    #     epochs_lstm.append(data[key]["epoch_num"])
    #     lstm_nse.append(-data[key]["val_loss"])


    # #####################################################################
    # data = torch.load("checkpoints/lstm-noise-dim27/metrics.pt")
    # epochs_lstm_noise = []
    # lstm_noise_nse = []
    # for key in data:
    #     epochs_lstm_noise.append(data[key]["epoch_num"])
    #     lstm_noise_nse.append(-data[key]["val_loss"])
        
    # fig2, ax2 = plt.subplots(1,1,figsize=(10,10))
    # epochs_ae, ae_nse = zip(*sorted(zip(epochs_ae, ae_nse)))
    # epochs_lstm, lstm_nse = zip(*sorted(zip(epochs_lstm, lstm_nse)))
    # epochs_lstm_noise, lstm_noise_nse = zip(*sorted(zip(epochs_lstm_noise, lstm_noise_nse)))
    

    # ax2.plot(epochs_ae,ae_nse, label="LSTM-AE 27 Features")
    # ax2.plot(epochs_lstm,lstm_nse, label="LSTM")
    # ax2.plot(epochs_lstm_noise,lstm_noise_nse, label="LSTM + 27 Noise")
 
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
   
   