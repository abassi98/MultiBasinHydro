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
    camel_dataset = CamelDataset(dates, force_attributes, debug=True)
    #dataset.adjust_dates() # adjust dates if necessary
    camel_dataset.load_data() # load data
    camel_dataset.load_statics() # load statics
    transform_statics = Globally_Scale_Data(camel_dataset.min_statics, camel_dataset.max_statics)
    camel_dataset.statics_data = transform_statics(camel_dataset.statics_data)
    camel_dataset.save_statics("statics.txt") #save statics attributes

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
    with torch.no_grad():
        enc, rec = model(x,y)

    # pass thorugh sigmoid
    enc = nn.Sigmoid()(enc)
    
    # save encoded features
    enc = enc.detach().squeeze().numpy() # size (562,27)
    print(enc.shape)
    filename = "encoded_features_"+model_id+".txt"
    df = pd.DataFrame()
    df["basin__id"] = camel_dataset.loaded_basin_ids

    for i in range(enc.shape[1]):
        df["E"+str(i)] = enc[:,i]

    df.to_csv(filename, sep=" ")