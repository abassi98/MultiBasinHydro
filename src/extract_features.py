import numpy as np
import pandas as pd

# pytorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import multiprocessing


# user functions
from dataset import CamelDataset
from models import Hydro_LSTM_AE
from utils import Globally_Scale_Data, find_best_epoch


if __name__ == '__main__':
    ##########################################################
    # set seed
    ##########################################################
    torch.manual_seed(42)
    np.random.seed(42)
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
    camel_dataset.load_statics() # load statics
    camel_dataset.save_statics("statics.txt") #save statics attributes
    camel_dataset.load_hydro()
    camel_dataset.save_hydro("hydro.txt")

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
    

    # define dataloader for all dataset
    num_workers = 0
    print("Number of workers: %d"%num_workers)
    dataloader = DataLoader(camel_dataset, batch_size=num_basins, num_workers=num_workers, shuffle=False)

    # extract forcing and streamflow
    x, y, statics = next(iter(dataloader))
    print(x.shape)
    print(y.shape)

    # retrieve best epoch
    model_id = "lstm-ae-bdTrue-E4"
    best_epoch = find_best_epoch(model_id)
    ckpt_path = "checkpoints/"+model_id+"/model-epoch="+str(best_epoch)+".ckpt"

    model = Hydro_LSTM_AE.load_from_checkpoint(ckpt_path)
    model.eval()
    enc = torch.zeros((x.shape[0],model.encoded_space_dim))
    with torch.no_grad():
        for i in range(x.shape[0]):
            print(i)
            enc_temp, _ = model(x[i].unsqueeze(0),y[i].unsqueeze(0))
            enc[i] = enc_temp.squeeze()

    # pass thorugh sigmoid
    enc = nn.Sigmoid()(enc)
    
    # save encoded features
    enc = enc.detach().squeeze().numpy() # size (562,27)
    print(enc.shape)
    filename = "encoded_features_"+model_id+".txt"
    df = pd.DataFrame()
    df["basin_id"] = camel_dataset.loaded_basin_ids

    for i in range(enc.shape[1]):
        df["E"+str(i)] = enc[:,i]

    df.to_csv(filename, sep=" ")