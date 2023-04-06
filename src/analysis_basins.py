import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import datetime
import torch
from torch.utils.data import DataLoader, random_split
import multiprocessing


# user functions
from dataset import CamelDataset
from models import Hydro_LSTM_AE, Hydro_LSTM
from utils import NSELoss, PFAB, Globally_Scale_Data, find_best_epoch

# def parse_args():
#     parser=argparse.ArgumentParser(description="Take model id and best model epoch to analysis on test dataset")
#     parser.add_argument('--model_ids', type=list, required=True, help="Identity of the model to analyize")
#     parser.add_argument('--best_epochs', type=list, required=True, help="Epoch where best model (on validation dataset) is obtained")
#     args=parser.parse_args()
#     return args


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
    camel_dataset = CamelDataset(dates, force_attributes, debug=False)
    #dataset.adjust_dates() # adjust dates if necessary
    camel_dataset.load_data() # load data
    camel_dataset.load_statics()
    camel_dataset.load_hydro()
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

    ### define reverse transformation
    transform_input = Globally_Scale_Data(camel_dataset.min_flow, camel_dataset.max_flow)
    transform_output = Globally_Scale_Data(camel_dataset.min_force, camel_dataset.max_force)

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
    basin_names = [camel_dataset.loaded_basin_names[idx] for idx in split_indices]
    print("Indices for training dataset: ", train_dataset.indices)
    print("Indices for validation dataset: ", val_dataset.indices)
    print("Indices for test dataset: ", split_indices)

    # load best model
    model_ids =["lstm-ae-bdTrue-E27","lstm-ae-bdTrue-E4", "lstm-ae-bdTrue-E3", "lstm-bdTrue-N0-STrue", "lstm-bdTrue-N0"]
    num_models = len(model_ids)
   
    start_date = datetime.datetime.strptime(dates[0], '%Y/%m/%d').date()
    # get data 
    x, y, statics, hydro = next(iter(test_dataloader))
   
    x_unnorm = transform_input.reverse_transform(x.detach()).squeeze().numpy()
    # build figure
    length_to_plot = 730 # 2 years
    basins_n = 6
    fig, axs = plt.subplots(basins_n,basins_n, figsize=(30,30), sharey=True, sharex=True)
    fig1, axs1 = plt.subplots(basins_n,basins_n, figsize=(30,30), sharey=True, sharex=True)
    
    # basin idxs and start sequences
    start_sequences_list = np.random.randint(0, seq_len-length_to_plot, size=basins_n**2)

    # define figure
    fig_stat, axs_stat = plt.subplots(num_models,3, figsize=(20,10))
   
    # define metric functions
    loss_NSE = NSELoss(reduction=None)
    loss_mNSE = NSELoss(alpha=1, reduction=None)
    loss_PFAB = PFAB(ex_prob=0.01, reduction=None)
    nse_df = pd.DataFrame()
    mnse_df = pd.DataFrame()
    pfab_df = pd.DataFrame()

    ###################################################################################
    # PLOT
    ###################################################################################
    # plot true one
    # plot some sequences
    # for i in range(basins_n):
    #     for j in range(basins_n):
    #         ax = axs[i,j]
    #         ax1 = axs1[i,j]
    #         val = i*basins_n + j
    #         basin_name = basin_names[val]
    #         start_seq = start_sequences_list[val]
    #         date = start_date + datetime.timedelta(days=int(start_seq))
    #         time = date.strftime("%Y/%m/%d")
    #         ax.plot(x_unnorm[val, start_seq:start_seq+length_to_plot], label="Camel")
    #         ax.set_title("Start date: "+time, style='italic')
    #         at = AnchoredText(basin_name,loc='upper left', prop=dict(size=8), frameon=True)
    #         ax.add_artist(at)
    #         ax1.set_title("Start date: "+time, style='italic')
    #         at1 = AnchoredText(basin_name,loc='upper left', prop=dict(size=8), frameon=True)
    #         ax1.add_artist(at1)

    for count in range(num_models):
        model_id = model_ids[count]
        best_epoch = find_best_epoch(model_id)
        # open best model
        dirpath = os.path.join("checkpoints", model_id)
        filename = "model-epoch="+str(best_epoch)+".ckpt"
        path_best  = os.path.join(dirpath, filename)
        
        if model_id.find("lstm-ae") != -1:
            model = Hydro_LSTM_AE.load_from_checkpoint(path_best)
            model.eval()
            # compute squeezed encoded representation and reconstruction
            with torch.no_grad():
                enc, rec = model(x,y)

        else:
            model = Hydro_LSTM.load_from_checkpoint(path_best)
            print(model.statics, model.hydro)
            with torch.no_grad():
                rec = model(y, statics, hydro)

        # compute NSE, mNSE and save in dataframe
        nse_df[model_id] = - loss_NSE(x.squeeze(), rec.squeeze()).detach().numpy() # array of size (num_test_data)
        mnse_df[model_id] = - loss_mNSE(x.squeeze(), rec.squeeze()).detach().numpy() # array of size (num_test_data)
        pfab_df[model_id] = loss_PFAB(x.squeeze(), rec.squeeze()).detach().numpy() # array of size (num_test_data)
        
        # unnormalize input and output
        rec = transform_input.reverse_transform(rec.detach()).squeeze().numpy()
        # # perform tsne over encoded space
        # enc_embedded = TSNE(n_components=2, perplexity=1.0).fit_transform(enc)

        # fig1, ax1 = plt.subplots(1,1,figsize=(5,5))
        # ax1.scatter(enc_embedded[:,0], enc_embedded[:,1])
        # fig1.savefig("encoded_space-"+args.model_id+"-epoch="+str(args.best_epoch)+".png")

        # # plot some sequences
        # for i in range(basins_n):
        #     for j in range(basins_n):
        #         ax = axs[i,j]
        #         ax1 = axs1[i,j]
        #         val = i*basins_n + j
        #         start_seq = start_sequences_list[val]
        #         ax.plot(rec[val, start_seq:start_seq+length_to_plot], label=model_id)
        #         ax1.semilogy(np.absolute(rec[val, start_seq:start_seq+length_to_plot]-x_unnorm[val, start_seq:start_seq+length_to_plot]), label=model_id)

    # compute and plot statistics
    # NSE 
    stat_NSE = nse_df.describe()
    print("NSE statistics")
    print(stat_NSE)
    # mNSE 
    stat_mNSE = mnse_df.describe()
    print("mNSE statistics")
    print(stat_mNSE)
    # PFAB
    stat_pfab = pfab_df.describe()
    print("PFAB statistics")
    print(stat_pfab)
    axs_stat[0,0].set_title("NSE", fontsize=30)
    axs_stat[0,1].set_title("mNSE", fontsize=30)
    axs_stat[0,2].set_title("PFAB", fontsize=30)

    # plot distributions
    for count in range(num_models):
        model_id = model_ids[count]
        #NSE
        nse_df[model_id].plot(kind="kde", ax=axs_stat[count,0])
        axs_stat[count,0].grid()
        axs_stat[count,0].axvline(stat_NSE.loc["25%",model_id], ls="--", lw=1, c="black")
        axs_stat[count,0].axvline(stat_NSE.loc["50%", model_id], ls="-", lw=2, c="black")
        axs_stat[count,0].axvline(stat_NSE.loc["75%", model_id], ls="--", lw=1, c="black")
        axs_stat[count,0].set_ylabel(model_id, fontsize=15)
        axs_stat[count,0].set_xlim(-0.5,1.0)
        
        # mNSE
        mnse_df[model_id].plot(kind="kde", ax=axs_stat[count,1])
        axs_stat[count,1].grid()
        axs_stat[count,1].axvline(stat_mNSE.loc["25%",model_id], ls="--", lw=1, c="black")
        axs_stat[count,1].axvline(stat_mNSE.loc["50%", model_id], ls="-", lw=2, c="black")
        axs_stat[count,1].axvline(stat_mNSE.loc["75%", model_id], ls="--", lw=1, c="black")
        axs_stat[count,1].set_xlim(-0.5,1.0)

        # PFAB
        pfab_df[model_id].plot(kind="kde", ax=axs_stat[count,2])
        axs_stat[count,2].grid()
        axs_stat[count,2].axvline(stat_pfab.loc["25%",model_id], ls="--", lw=1, c="black")
        axs_stat[count,2].axvline(stat_pfab.loc["50%", model_id], ls="-", lw=2, c="black")
        axs_stat[count,2].axvline(stat_pfab.loc["75%", model_id], ls="--", lw=1, c="black")
        axs_stat[count,2].set_xlim(0.0,100.0)

        
   
    file = "plot/stat_distr.png"
    fig_stat.savefig(file)

    # # return and save the figure of runoff
    # handles, labels = ax.get_legend_handles_labels()
    # fig.legend(handles, labels, loc='upper left', fontsize=50)
    # fig.text(0.5, 0.04, 'Time (days)', ha='center', fontsize=50)
    # fig.text(0.04, 0.5, 'Streamflow (mm/day)', va='center', rotation='vertical', fontsize=50)
    # fig.suptitle('Streamflow of best models compared to Camel data', fontsize=100)
    # fig.tight_layout
    # fig.savefig("plot/reconstructed-best-epochs.png")

    # # return and save the figure of runoff
    # handles, labels = ax1.get_legend_handles_labels()
    # fig1.legend(handles, labels, loc='upper left', fontsize=50)
    # fig1.text(0.5, 0.04, 'Time (days)', ha='center', fontsize=50)
    # fig1.text(0.04, 0.5, 'Delta Streamflow (mm/day)', va='center', rotation='vertical', fontsize=50)
    # fig1.suptitle('Absolute Streamflow difference betwen best models and Camel data', fontsize=100)
    # fig1.tight_layout
    # fig1.savefig("plot/abs-diff-best-epochs.png")


   
    

    

    