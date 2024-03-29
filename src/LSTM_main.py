import multiprocessing
import argparse
import numpy as np

# pytorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from pytorch_lightning.callbacks.early_stopping import EarlyStopping 


# user functions
from dataset import CamelDataset, YearlyCamelsDataset
from models import Hydro_LSTM
from utils import MetricsCallback, NSELoss



def parse_args():
    parser=argparse.ArgumentParser(description="If add to LSTM some noise features")
    parser.add_argument('--noise_dim', type=int, default=0, help="How many random noise components")
    parser.add_argument('--statics', type=int, default=0, help="Include 27 Camels statics features")
    parser.add_argument('--hydro', type=int, default=0, help="Include Camels Hydrological signatures")
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
    force_attributes =  ["PRCP(mm/day)", "SRAD(W/m2)", "Tmin(C)", "Tmax(C)", "Vp(Pa)"] # force attributes to use
    camel_dataset = CamelDataset(dates, force_attributes, debug=bool(args.debug))
    print("Debug mode: ", bool(camel_dataset.debug))
    print("Bidirectional LSTM: ", bool(args.bidirectional))
    print("Use static features: ", bool(args.statics))
    print("Use hydro signatures: ", bool(args.hydro))
    
    

    #dataset.adjust_dates() # adjust dates if necessary
    camel_dataset.load_data() # load data
    loaded_basin_ids = camel_dataset.loaded_basin_ids
    camel_dataset.load_statics() # load statics attributes
    camel_dataset.load_hydro() # load hydrological signatures
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

    ### split data
    num_train_data = int(num_basins * 0.7) 
    num_val_data = int(num_basins * 0.15) 
    num_test_data = num_basins - num_train_data - num_val_data

    index_basins = np.arange(num_basins)
    np.random.shuffle(index_basins)
    shuffled_basin_ids = [loaded_basin_ids[i] for i in index_basins]

    train_basins = shuffled_basin_ids[:num_train_data]
    train_indeces = index_basins[:num_train_data]
    train_bool = [False] * num_basins
    for idx in train_indeces:
        train_bool[idx] = True

    val_basins = shuffled_basin_ids[num_train_data:num_train_data+num_val_data]
    val_indeces = index_basins[num_train_data:num_train_data+num_val_data]
    val_bool = [False] * num_basins
    for idx in val_indeces:
        val_bool[idx] = True
    test_basins = shuffled_basin_ids[num_train_data+num_val_data:]
    test_indeces = index_basins[num_train_data+num_val_data:]
    test_bool = [False] * num_basins
    for idx in test_indeces:
        test_bool[idx] = True

    
    print("Train basins: %d" %num_train_data)
    print(train_basins)
    print("Validation basins: %d" %num_val_data)
    print(val_basins)
    print("Test basins: %d" %num_test_data)
    print(test_basins)
    
    # select datasets (train, val, test are separated in space and time)
    train_start = "1980/10/01"
    train_end = "1995/09/27"
    train_dataset = YearlyCamelsDataset(train_indeces, train_start, train_end, camel_dataset)
    val_start = "1995/10/01"
    val_end = "2010/09/26"
    val_dataset = YearlyCamelsDataset(val_indeces, val_start, val_end, camel_dataset)
    test_dataset = YearlyCamelsDataset(test_indeces, val_start, val_end, camel_dataset)
    
    print(train_dataset.__len__())
    ### Dataloader
    batch_size = 1024
    # split 80/10/10
    num_workers = 0 # 4 times the number of gpus
    print("Number of workers: %d"%num_workers)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True,  drop_last=False)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    # x,y, statics, hydro, ids = next(iter(train_dataloader))

    # print(torch.amin(x))
    # print(torch.amax(x))
    # print(torch.amin(y))
    # print(torch.amax(y))

    ##########################################################
    # initialize the Hydro LSTM Auto Encoder
    ##########################################################
    # define the model
    loss_fn = NSELoss()
    assert args.noise_dim >= 0
    # possibly adjust kernel sizes according to seq_len
    model = Hydro_LSTM(lstm_hidden_units = 256, 
                 bidirectional = bool(args.bidirectional),
                 layers_num = 2,
                 act = nn.LeakyReLU, 
                 loss_fn = loss_fn,
                 drop_p = 0.5, 
                 seq_len = 365,
                 lr = 1e-5,
                 weight_decay = 0.0,
                 num_force_attributes = len(force_attributes),
                 noise_dim = args.noise_dim,
                 statics = bool(args.statics),
                 hydro =  bool(args.hydro),
                 warmup = 45)
                

    ##########################################################
    # training 
    ##########################################################

    ### Set proper device and train
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Training device: {device}")

    # define callbacks
    early_stopping = EarlyStopping(monitor="val_loss", patience = 10, mode="min")
    max_epochs = 20000
    check_val_every_n_epoch = 10
    save_top_k = int(max_epochs/check_val_every_n_epoch)

    # select dirpath according to noise features added
    dirpath="checkpoints/lstm-bd"+str(bool(args.bidirectional))+"-N"+str(args.noise_dim)+"-S"+str(bool(args.statics))+"-H"+str(bool(args.hydro))+"/"
        
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
    

    # # retrieve checkpoints and continue training
    # ckpt_path = "checkpoints/lstm-bdTrue-N0/last.ckpt"

    # define trainer 
    # , gradient_clip_val=1.0, gradient_clip_algorithm="value"
    trainer = pl.Trainer(max_epochs=max_epochs, callbacks=[checkpoint_model,metrics_callback], accelerator=str(device), devices=1, check_val_every_n_epoch=check_val_every_n_epoch, logger=False, gradient_clip_val=1.0, gradient_clip_algorithm="value")
    
    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders = val_dataloader)
    