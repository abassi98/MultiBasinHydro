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

    dirpath="checkpoints"
    filename="hydro-lstm-epoch=199.ckpt"
    path = os.path.join(dirpath, filename)

    loss_fn = NSELoss()
    model = Hydro_LSTM_AE.load_from_checkpoint(path)
    