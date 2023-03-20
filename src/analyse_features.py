import numpy as np
import pandas as pd

# pytorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset, Subset
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

from pytorch_lightning.callbacks.early_stopping import EarlyStopping 

# user functions
from dataset import FeatureDataset
from models import Hydro_FFNet




if __name__ == '__main__':
    ##########################################################
    # set seed
    ##########################################################
    torch.manual_seed(35)
    

    ##########################################################
    # dataset and dataloaders
    ##########################################################
    # Dataset
    #dates = ["1989/10/01", "2009/09/30"] 
    filename = "encoded_features_lstm_ae.txt"
    dataset = FeatureDataset(filename)
    num_basins = dataset.__len__()
    print("Number of basins: %d" %num_basins)

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
    num_workers = 4 # 4 times the number of gpus
    print("Number of workers: %d"%num_workers)

    num_train_data = int(num_basins * 0.7) 
    num_val_data = int(num_basins * 0.15) 
    num_test_data = num_basins - num_train_data - num_val_data
    print("Train basins: %d" %num_train_data)
    print("Validation basins: %d" %num_val_data)
    print("Test basins: %d" %num_test_data)
    train_dataset, val_dataset, test_dataset = random_split(dataset, (num_train_data, num_val_data, num_test_data))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True,  drop_last=False)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    loss_fn = nn.MSELoss()

    model = Hydro_FFNet(n_inputs = 27, 
                    n_outputs = 27, 
                    hidden_layers = [64,64,64,64,64,64], 
                    drop_p = 0.5, 
                    lr = 1e-4, 
                    activation = nn.LeakyReLU,
                    weight_decay = 0.0 )
        
    
    