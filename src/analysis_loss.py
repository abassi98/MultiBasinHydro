import numpy as np
import matplotlib.pyplot as plt
import os
import torch


if __name__ == '__main__':

    #####################################################################
    dir = "checkpoints"
    models = ["lstm-ae-bdTrue-E27","lstm-ae-bdTrue-E4", "lstm-ae-bdTrue-E3", "lstm-bdTrue-N0-STrue", "lstm-bdTrue-N0"]
    epochs = []
    nse = []
    fig, ax = plt.subplots(1,1,figsize=(5,5))
    ax.set_xlabel("epoch")
    ax.set_ylabel("NSE")
    for i in range(len(models)):
        name = models[i]
        path = os.path.join(dir, name, "metrics.pt")
        data = torch.load(path, map_location=torch.device('cpu'))
        epochs_mod = []
        nse_mod = []
        for key in data:
            epoch_num = data[key]["epoch_num"]
            nse = -data[key]["val_loss"]
            if isinstance(epoch_num, int):
                epochs_mod.append(epoch_num)
                nse_mod.append(nse)
            else:
                epochs_mod.append(int(epoch_num.item()))
                nse_mod.append(nse.item())

        # reorder
        epochs_mod, nse_mod = zip(*sorted(zip(epochs_mod, nse_mod)))
        # plot
        ax.plot(epochs_mod,nse_mod, label=name)
        # find best model
        idx_ae = np.argmax(nse_mod)
        epoch_max_nse = epochs_mod[idx_ae]
        print("Best "+name+" model obtained at epoch " +str(epoch_max_nse))
    
    
    ax.legend(fontsize=10)
    fig.suptitle("Validation NSE (alpha=2)")
    fig.savefig("hydro-lstm-ae_NSE.png")
    

    
   
   