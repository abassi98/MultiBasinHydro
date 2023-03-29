import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

# pytorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import multiprocessing


# user functions
from dataset import CamelDataset
from models import Hydro_LSTM_AE
from utils import find_best_epoch, NSELoss


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
    
    # Load latitude and longitude
    file_topo = "basin_dataset_public_v1p2/camels_topo.txt"
    df_topo = pd.read_csv(file_topo, sep=";")
    topo_basin_ids = df_topo.iloc[:,0]
   
    lat_topo = df_topo["gauge_lat"]
    lon_topo = df_topo["gauge_lon"]

    lat = []
    lon = []
 
    for i in range(len(camel_dataset.loaded_basin_ids)):
        for j in range(len(topo_basin_ids)):
            if topo_basin_ids[j] == int(camel_dataset.loaded_basin_ids[i]):
                lat.append(lat_topo[j])
                lon.append(lon_topo[j])

    # define dataloader for all dataset
    num_workers = 0
    print("Number of workers: %d"%num_workers)
    dataloader = DataLoader(camel_dataset, batch_size=num_basins, num_workers=num_workers, shuffle=False) # all dataset

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
    nse = []
    loss_fn = NSELoss()
    with torch.no_grad():
        for i in range(x.shape[0]):
            print(i)
            enc_temp, rec = model(x[i].unsqueeze(0),y[i].unsqueeze(0))
            enc[i] = enc_temp.squeeze()
            nse.append(-loss_fn(rec.squeeze().unsqueeze(0), x[i].squeeze().unsqueeze(0)).item())

    
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

    ### plot nse over us map
    # initialize an axis
    fig, ax= plt.subplots(1,1, figsize=(8,8), sharex=True, sharey=True)

    # plot map on axis
    countries = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))

    ax.set_xlim(-128, -65)
    ax.set_ylim(24, 50)
    ax.set_title("NSE", fontsize=20)
    ax.set_axis_off()
    countries[countries["name"] == "United States of America"].plot(color="lightgrey", ax=ax)
    im = ax.scatter(x=lon, y=lat,c=nse, cmap="YlOrRd", s=1)
    fig.colorbar(im, ax=ax, fraction=0.028, pad=0.02, location="bottom")
    
    save_file = "plot_NSEmap_"+model_id+".png"
    fig.savefig(save_file)