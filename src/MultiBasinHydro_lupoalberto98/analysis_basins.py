import numpy as np
import pandas as pd
import os
import argparse
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import datetime

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
import multiprocessing


# user functions
from dataset import CamelDataset
from models import Hydro_LSTM_AE, Hydro_LSTM
from utils import Scale_Data, MetricsCallback, NSELoss

def parse_args():
    parser=argparse.ArgumentParser(description="Take model id and best model epoch to analysis on test dataset")
    parser.add_argument('--model_id', type=str, required=True, help="Identity of the model to analyize")
    parser.add_argument('--best_epoch', type=int, required=True, help="Epoch where best model (on validation dataset) is obtained")
    args=parser.parse_args()
    return args


if __name__ == '__main__':
    ##########################################################
    # set seed
    ##########################################################
    torch.manual_seed(42)
    
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

    num_train_data = int(num_basins * 0.7) 
    num_val_data = int(num_basins * 0.15) 
    num_test_data = num_basins - num_train_data - num_val_data
    print("Train basins: %d" %num_train_data)
    print("Validation basins: %d" %num_val_data)
    print("Test basins: %d" %num_test_data)
    train_dataset, val_dataset, test_dataset = random_split(camel_dataset, (num_train_data, num_val_data, num_test_data))
    #train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True,  drop_last=False)
    #val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    # entire test dataset as one batch
    test_dataloader = DataLoader(test_dataset, batch_size=num_test_data, num_workers=num_workers, shuffle=False)
    split_indices = test_dataset.indices
    basin_names = [camel_dataset.trimmed_basin_names[idx] for idx in split_indices]
    print("Split indices for test dataset: ", split_indices)

    # load best model
    args = parse_args()
    start_date = datetime.datetime.strptime(dates[0], '%Y/%m/%d').date()
    path = os.path.join("checkpoints", args.model_id,"hydro-"+args.model_id+"-epoch="+str(args.best_epoch)+".ckpt")
    if args.model_id =="lstm-ae":
        model = Hydro_LSTM_AE.load_from_checkpoint(path)
        model.eval()
        # compute squeezed encoded representation and reconstruction
        x, y = next(iter(test_dataloader))
        enc, rec = model(x,y)
        enc = enc.squeeze().detach().numpy() # tensor of size (num_test_data, encoded_space_dim)
        rec = rec.squeeze().detach().numpy() # tensor of size (num_test_data, seq_len)
        x = x.squeeze().detach().numpy()
        # perform tsne over encoded space
        enc_embedded = TSNE(n_components=2, perplexity=1.0).fit_transform(enc)

        fig1, ax1 = plt.subplots(1,1,figsize=(5,5))
        ax1.scatter(enc_embedded[:,0], enc_embedded[:,1])
        fig1.savefig("encoded_space-"+args.model_id+"-epoch="+str(args.best_epoch)+".png")

        # plot some sequences
        length_to_plot = 730 # 2 years
        basins_n = 6
        fig, axs = plt.subplots(basins_n,basins_n, figsize=(30,30), sharey=True)
        for i in range(basins_n):
            for j in range(basins_n):
                ax = axs[i,j]
                basin_idx = np.random.randint(0,num_test_data)
                basin_name = basin_names[basin_idx]
                start_seq = np.random.randint(0, seq_len-length_to_plot)
                date = start_date + datetime.timedelta(days=start_seq)
                time = date.strftime("%Y/%m/%d")
                ax.plot(x[basin_idx, start_seq:start_seq+length_to_plot], label="True")
                ax.plot(rec[basin_idx, start_seq:start_seq+length_to_plot], label="Reconstructed")
                ax.set_title("Start date: "+time, style='italic')
                ax.text(-0.1, 0.8, basin_name, style='italic')

        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center')
        fig.text(0.5, 0.04, 'Time (days)', ha='center')
        fig.text(0.04, 0.5, 'Runoff', va='center', rotation='vertical')
        fig.tight_layout
        fig.savefig("reconstructed-"+args.model_id+"-epoch="+str(args.best_epoch)+".png")

    elif args.model_id =="lstm":
        model = Hydro_LSTM.load_from_checkpoint(path)

    

    

    
    
    