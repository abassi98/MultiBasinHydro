import multiprocessing
import argparse

# pytorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from pytorch_lightning.callbacks.early_stopping import EarlyStopping 


# user functions
from dataset import CamelDataset
from models import Hydro_LSTM
from utils import MetricsCallback, NSELoss



def parse_args():
    parser=argparse.ArgumentParser(description="If add to LSTM some noise features")
    parser.add_argument('--noise_dim', type=int, required=True, help="How many random noise components")
    parser.add_argument('--statics', type=int, required=True, help="How many random noise components")
    parser.add_argument('--bidirectional', type=int, default=0, help="Bidirectionality of LSTM decoder. 0 False, else True")
    parser.add_argument('--debug', type=int, default=0, help="If debug mode is on load only 15 basins. 0 False, else True")
    args=parser.parse_args()
    return args



if __name__ == '__main__':
    ##########################################################
    # set seed
    ##########################################################
    torch.manual_seed(42)
    args = parse_args()
    ##########################################################
    # dataset and dataloaders
    ##########################################################
    # Dataset
    #dates = ["1989/10/01", "2009/09/30"] 
    dates = ["1980/10/01", "2010/09/30"] # interval dates to pick
    force_attributes = ["prcp(mm/day)", "srad(W/m2)", "tmin(C)", "tmax(C)", "vp(Pa)"] # force attributes to use
    camel_dataset = CamelDataset(dates, force_attributes, debug=bool(args.debug))
    print("Debug mode: ", bool(camel_dataset.debug))
    print("Bidirectional LSTM: ", bool(args.bidirectional))
    print("Use static features: ", bool(args.statics))
    

    #dataset.adjust_dates() # adjust dates if necessary
    camel_dataset.load_data() # load data
    camel_dataset.load_statics() # load statics attributes
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
    # split 70/15/15
    num_workers = 0
    print("Number of workers: %d"%num_workers)
    

    num_train_data = int(num_basins * 0.7) 
    num_val_data = int(num_basins * 0.15) 
    num_test_data = num_basins - num_train_data - num_val_data
    print("Train basins: %d" %num_train_data)
    print("Validation basins: %d" %num_val_data)
    print("Test basins: %d" %num_test_data)
    train_dataset, val_dataset, test_dataset = random_split(camel_dataset, (num_train_data, num_val_data, num_test_data))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True,  drop_last=False)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

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
                 seq_len = int(seq_len/2),
                 lr = 1e-6,
                 weight_decay = 0.01,
                 num_force_attributes = len(force_attributes),
                 noise_dim = args.noise_dim,
                 statics = bool(args.statics),
                 warmup = 730)
                

    ##########################################################
    # training 
    ##########################################################

    ### Set proper device and train
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Training device: {device}")

    # define callbacks
    early_stopping = EarlyStopping(monitor="val_loss", patience = 10, mode="min")
    max_epochs = 10000
    check_val_every_n_epoch = 10
    save_top_k = int(max_epochs/check_val_every_n_epoch)

    # select dirpath according to noise features added
    dirpath="checkpoints/lstm-bd"+str(bool(args.bidirectional))+"-N"+str(args.noise_dim)+"-S"+str(bool(args.statics))+"/"
        
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
    trainer = pl.Trainer(max_epochs=max_epochs, callbacks=[checkpoint_model,metrics_callback], accelerator=str(device), devices=1, check_val_every_n_epoch=check_val_every_n_epoch, logger=False)
    
    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders = val_dataloader)
    