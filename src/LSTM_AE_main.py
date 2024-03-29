import numpy as np
import pandas as pd
import sys
import argparse 

# pytorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping 
import multiprocessing


# user functions
from dataset import CamelDataset
from models import Hydro_LSTM_AE
from utils import MetricsCallback, NSELoss



def parse_args():
    parser=argparse.ArgumentParser(description="Arguments for Convolutional LSTM autoencoder")
    parser.add_argument('--num_features', type=int, default=27, help="Number of features in the encoded space")
    parser.add_argument('--bidirectional', type=int, default=1, help="Bidirectionality of LSTM decoder. 0 False, else True")
    parser.add_argument('--debug', type=int, default=0, help="If debug mode is on load only 15 basins. 0 False, else True")
    args=parser.parse_args()
    return args

if __name__ == '__main__':
    ##########################################################
    # set seed
    ##########################################################
    torch.manual_seed(42)
    np.random.seed(42)
    args = parse_args()


    ##########################################################
    # dataset and dataloaders
    ##########################################################
    # Dataset
    #dates = ["1989/10/01", "2009/09/30"] 
    dates = ["1980/10/01", "2010/09/30"] # interval dates to pick
    force_attributes = ["PRCP(mm/day)", "SRAD(W/m2)", "Tmin(C)", "Tmax(C)", "Vp(Pa)"] # force attributes to use
    camel_dataset = CamelDataset(dates, force_attributes)
    print("Bidirectional LSTM: ", bool(args.bidirectional))

    #dataset.adjust_dates() # adjust dates if necessary
    camel_dataset.load_data() # load data
    camel_dataset.load_hydro()
    camel_dataset.load_statics()

    num_basins = camel_dataset.__len__()
    seq_len = camel_dataset.seq_len
    print("Number of basins: %d" %num_basins)
    print("Total number of days in Camels data: %d" %seq_len)

    

    ### Set proper device and train
    # check cpus and gpus available
    num_cpus = multiprocessing.cpu_count()
    print("Num of cpus: %d"%num_cpus)
    num_gpus = torch.cuda.device_count()
    print("Num of gpus: %d"%num_gpus)
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Training device: {device}")


    ### split data
    num_train_data = int(num_basins * 0.7) 
    num_val_data = int(num_basins * 0.15) 
    num_test_data = num_basins - num_train_data - num_val_data
    train_dataset, val_dataset, test_dataset = random_split(camel_dataset, (num_train_data, num_val_data, num_test_data)) 
    ### Dataloader
    batch_size = 32
    # split 80/10/10
    num_workers = 0 # 4 times the number of gpus
    print("Number of workers: %d"%num_workers)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True,  drop_last=False)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
  
    ##########################################################
    # initialize the Hydro LSTM Auto Encoder
    ##########################################################
    # define the model
    loss_fn = NSELoss()
    # possibly adjust kernel sizes according to seq_len
    model = Hydro_LSTM_AE(in_channels=(1,8,16), 
                    out_channels=(8,16,32), 
                    kernel_sizes=(6,7,4), 
                    encoded_space_dim=args.num_features,
                    drop_p=0.5,
                    seq_len=seq_len,
                    lr = 1e-5,
                    act=nn.LeakyReLU,
                    loss_fn=loss_fn,
                    lstm_hidden_units=256,
                    layers_num=2,
                    bidirectional = bool(args.bidirectional),
                    linear=512,
                    num_force_attributes = len(force_attributes),
                    warmup = 45)
    
    print("Training and Validation lengths (days): %d"%model.seq_len)
    print("Warmup days: %d"%model.warmup)
    ##########################################################
    # training 
    ##########################################################

    
    # define callbacks
    early_stopping = EarlyStopping(monitor="val_loss", patience = 10, mode="min")
    max_epochs = 20000
    check_val_every_n_epoch = 10
    save_top_k = int(max_epochs/check_val_every_n_epoch)

    dirpath = "checkpoints/lstm-ae-bd"+str(bool(args.bidirectional))+"-E"+str(args.num_features)+"/"
    
    metrics_callback = MetricsCallback(
        dirpath=dirpath,
        filename="metrics.pt",
    )

    checkpoint_model = ModelCheckpoint(
            save_top_k=10,
            save_last=True,
            monitor="val_loss",
            mode="min",
            dirpath=dirpath,
            filename="model-{epoch:02d}",
        )
    

 
    # define trainer 
    trainer = pl.Trainer(max_epochs=max_epochs, callbacks=[checkpoint_model,metrics_callback], accelerator=str(device),devices=1, check_val_every_n_epoch=check_val_every_n_epoch, logger=False, gradient_clip_val=1.0, gradient_clip_algorithm="value")
    
    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders = val_dataloader)
   