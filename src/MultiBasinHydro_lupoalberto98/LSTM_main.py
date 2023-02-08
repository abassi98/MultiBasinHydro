import numpy as np
import pandas as pd
import os
import multiprocessing

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
from models import Hydro_LSTM
from utils import Scale_Data, MetricsCallback, NSELoss



##########################################################
# set seed
##########################################################
torch.manual_seed(42)

if __name__ == '__main__':
    
    ##########################################################
    # dataset and dataloaders
    ##########################################################
    # Dataset
    #dates = ["1989/10/01", "2009/09/30"] 
    dates = ["1980/10/01", "2010/09/30"] # interval dates to pick
    force_attributes = ["prcp(mm/day)", "srad(W/m2)", "tmin(C)", "tmax(C)", "vp(Pa)"] # force attributes to use
    camel_dataset = CamelDataset(dates, force_attributes)
    #dataset.adjust_dates() # adjust dates if necessary
    camel_dataset.load_data() # load data
    num_basins = camel_dataset.__len__()
    seq_len = camel_dataset.seq_len
    print("Number of basins: %d" %num_basins)
    print("Sequence length: %d" %seq_len)

    ### Set proper device and train
    # check cpus and gpus available
    num_cpus = multiprocessing.cpu_count()
    print("Num of cpus: %d"%num_cpus)
    num_gpus = torch.cuda.device_count()
    print("Num of gpus: %d"%num_gpus)
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Training device: {device}")

    ### Dataloader
    batch_size = 32
    # split 80/10/10
    num_workers = 0
    print("Number of workers: %d"%num_workers)
    

    num_train_data = int(num_basins * 0.8) 
    num_val_data = num_basins - num_train_data
    #num_test_data = num_basins - num_train_data - num_val_data
    print("Train basins: %d" %num_train_data)
    print("Validation basins: %d" %num_val_data)
    #print("Test basins: %d" %num_test_data)
    train_dataset, val_dataset = random_split(camel_dataset, (num_train_data, num_val_data))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True,  drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=num_val_data, num_workers=num_workers, shuffle=False)
    #test_dataloader = DataLoader(val_dataset, batch_size=num_test_data, num_workers=8, shuffle=False)

    ##########################################################
    # initialize the Hydro LSTM Auto Encoder
    ##########################################################
    # define the model
    loss_fn = NSELoss()
    # possibly adjust kernel sizes according to seq_len
    model = Hydro_LSTM(lstm_hidden_units = 256, 
                 bidirectional = False,
                 layers_num = 2,
                 act = nn.LeakyReLU, 
                 loss_fn = loss_fn,
                 drop_p = 0.5, 
                 seq_len = seq_len,
                 lr = 1e-4,
                 weight_decay = 0.0,
                 num_force_attributes = len(force_attributes),
                )

    ##########################################################
    # training 
    ##########################################################

    ### Set proper device and train
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Training device: {device}")

    # define callbacks
    metrics_callback = MetricsCallback()
    early_stopping = EarlyStopping(monitor="val_loss", patience = 10, mode="min")
    checkpoint_callback = ModelCheckpoint(
        save_top_k=100,
        monitor="val_loss",
        mode="min",
        dirpath="checkpoints/lstm/",
        filename="hydro-lstm-{epoch:02d}",
    )

    
    # define trainer 
    trainer = pl.Trainer(max_epochs=3000, callbacks=[checkpoint_callback], accelerator=str(device), check_val_every_n_epoch=10, logger=False)
    
    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders = val_dataloader)
    