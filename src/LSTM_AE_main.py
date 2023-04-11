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
from dataset import CamelDataset,YearlyCamelsDataset
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
    camel_dataset = CamelDataset(dates, force_attributes, debug=bool(args.debug))
    print("Debug mode: ", bool(camel_dataset.debug))
    print("Bidirectional LSTM: ", bool(args.bidirectional))

    #dataset.adjust_dates() # adjust dates if necessary
    camel_dataset.load_data() # load data
    loaded_basin_ids = camel_dataset.loaded_basin_ids

    num_basins = camel_dataset.__len__()
    seq_len = camel_dataset.seq_len
    print("Number of basins: %d" %num_basins)
    print("Total number of days in Camels data: %d" %seq_len)

    camel_dataset.save_dataset()

    # ### Set proper device and train
    # # check cpus and gpus available
    # num_cpus = multiprocessing.cpu_count()
    # print("Num of cpus: %d"%num_cpus)
    # num_gpus = torch.cuda.device_count()
    # print("Num of gpus: %d"%num_gpus)
    
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # print(f"Training device: {device}")


    # ### split data
    # num_train_data = int(num_basins * 0.7) 
    # num_val_data = int(num_basins * 0.15) 
    # num_test_data = num_basins - num_train_data - num_val_data

    # index_basins = np.arange(num_basins)
 
   
    # np.random.shuffle(index_basins)
    # shuffled_basin_ids = [loaded_basin_ids[i] for i in index_basins]

    # train_basins = shuffled_basin_ids[:num_train_data]
    # train_indeces = index_basins[:num_train_data]
    # train_bool = [False] * num_basins
    # for idx in train_indeces:
    #     train_bool[idx] = True

    # val_basins = shuffled_basin_ids[num_train_data:num_train_data+num_val_data]
    # val_indeces = index_basins[num_train_data:num_train_data+num_val_data]
    # val_bool = [False] * num_basins
    # for idx in val_indeces:
    #     val_bool[idx] = True
    # test_basins = shuffled_basin_ids[num_train_data+num_val_data:]
    # test_indeces = index_basins[num_train_data+num_val_data:]
    # test_bool = [False] * num_basins
    # for idx in test_indeces:
    #     test_bool[idx] = True

    
    # print("Train basins: %d" %num_train_data)
    # print(train_basins)
    # print("Validation basins: %d" %num_val_data)
    # print(val_basins)
    # print("Test basins: %d" %num_test_data)
    # print(test_basins)
    
    # # select datasets (train, val, test are separated in space and time)
    # train_start = "1980/10/01"
    # train_end = "1995/09/27"
    # train_dataset = YearlyCamelsDataset(train_indeces, train_start, train_end, camel_dataset)
    # val_start = "1995/10/01"
    # val_end = "2010/09/26"
    # val_dataset = YearlyCamelsDataset(val_indeces, val_start, val_end, camel_dataset)
    # test_dataset = YearlyCamelsDataset(test_indeces, val_start, val_end, camel_dataset)
    
  
    # print(train_dataset.__len__())
    # ### Dataloader
    # batch_size = 1024
    # # split 80/10/10
    # num_workers = 0 # 4 times the number of gpus
    # print("Number of workers: %d"%num_workers)

    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True,  drop_last=False)
    # val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    # test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    # # tar,_,_,_,_ = next(iter(train_dataloader)) 
    # # tar = tar.squeeze()
    # # print(tar.shape)
    # # delta =  torch.sum((torch.abs( tar - torch.mean(tar, dim=-1, keepdims=True)))**2, dim=-1)
    # # print(delta.shape)
    # # condition = torch.logical_or(delta == 0.0, delta == 0)
    # # condition_np = np.array(condition)
    # # indeces = condition.nonzero().squeeze()
    # # indeces = [indeces[i].item() for i in range(len(indeces))]
    # # print(np.array(train_dataset.loaded_basin_ids)[condition_np])
    # # print(indeces)

    
    # # x,y, _, _, ids = next(iter(train_dataloader))
    # # print(torch.amax(x), torch.amin(x))
   
    # # print(sys.getsizeof(camel_dataset.input_data))
    # # print(sys.getsizeof(camel_dataset.output_data))
    
    # # torch.cuda.empty_cache()

    # ##########################################################
    # # initialize the Hydro LSTM Auto Encoder
    # ##########################################################
    # # define the model
    # loss_fn = NSELoss()
    # # possibly adjust kernel sizes according to seq_len
    # model = Hydro_LSTM_AE(in_channels=(1,8,16), 
    #                 out_channels=(8,16,32), 
    #                 kernel_sizes=(8,4,3), 
    #                 encoded_space_dim=args.num_features,
    #                 drop_p=0.5,
    #                 seq_len=365,
    #                 lr = 1e-5,
    #                 act=nn.LeakyReLU,
    #                 loss_fn=loss_fn,
    #                 lstm_hidden_units=256,
    #                 layers_num=2,
    #                 bidirectional = bool(args.bidirectional),
    #                 linear=512,
    #                 num_force_attributes = len(force_attributes),
    #                 warmup = 45)
    
    # print("Training and Validation lengths (days): %d"%model.seq_len)
    # print("Warmup days: %d"%model.warmup)
    # ##########################################################
    # # training 
    # ##########################################################

    
    # # define callbacks
    # early_stopping = EarlyStopping(monitor="val_loss", patience = 10, mode="min")
    # max_epochs = 20000
    # check_val_every_n_epoch = 10
    # save_top_k = int(max_epochs/check_val_every_n_epoch)

    # dirpath = "checkpoints/lstm-ae-bd"+str(bool(args.bidirectional))+"-E"+str(args.num_features)+"/"
    
    # metrics_callback = MetricsCallback(
    #     dirpath=dirpath,
    #     filename="metrics.pt",
    # )

    # checkpoint_model = ModelCheckpoint(
    #         save_top_k=10,
    #         save_last=True,
    #         monitor="val_loss",
    #         mode="min",
    #         dirpath=dirpath,
    #         filename="model-{epoch:02d}",
    #     )
    

    # # retrieve checkpoints and continue training
    # #ckpt_path = "checkpoints/lstm-ae-bdTrue-E3/last.ckpt"

    # # define trainer 
    # trainer = pl.Trainer(max_epochs=max_epochs, callbacks=[checkpoint_model,metrics_callback], accelerator=str(device),devices=1, check_val_every_n_epoch=check_val_every_n_epoch, logger=False, gradient_clip_val=1.0, gradient_clip_algorithm="value")
    
    # trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders = val_dataloader)
   