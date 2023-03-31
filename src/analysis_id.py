import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dadapy import data

S = np.loadtxt("statics.txt")
E = pd.read_csv("encoded_features_lstm_ae.txt", sep=" ", skiprows=1).iloc[:,4::]
E = np.array(E)

def compute_ids_scaling(X, range_max = 2048, N_min = 20):
    "instantiate data class"
    _data = data.Data(coordinates = X,
                     maxk = 100)
    "compute ids scaling gride"
    ids_gride, ids_err_gride, rs_gride = _data.return_id_scaling_gride(range_max=range_max)
    "compute ids with twoNN + decimation"
    ids_twoNN, ids_err_twoNN, rs_twoNN = _data.return_id_scaling_2NN(N_min = N_min)
    return ids_gride, ids_twoNN, rs_gride, rs_twoNN

#*************************************************************

ids_S_gride, ids_S_twoNN, rs_S_gride, rs_S_twoNN = compute_ids_scaling(S,range_max = S.shape[0]-1,N_min = 1)
ids_E_gride, ids_E_twoNN, rs_E_gride, rs_E_twoNN = compute_ids_scaling(E,range_max = S.shape[0]-1,N_min = 1)

fig, ax = plt.subplots(1,2,figsize = (16, 8), sharey=True)


xrange = min(len(ids_S_gride), len(ids_S_twoNN))
sns.lineplot(ax = ax[0], x=rs_S_gride, y = ids_S_gride, label = 'Gride', marker = 'o')
sns.lineplot(ax = ax[0], x=rs_S_twoNN, y = ids_S_twoNN, label = 'twoNN', marker = 'o')
ax[0].set_xscale('log')
ax[0].set_title('Statics attributes', fontsize = 20)
ax[0].set_ylabel('ID', fontsize = 20)
#ax.axvline(noise_plane, color = 'black', alpha = 1, label = 'noise scale', linewidth = 0.8, linestyle= 'dotted')
ax[0].legend(fontsize = 20)


xrange = min(len(ids_E_gride), len(ids_E_twoNN))
sns.lineplot(ax = ax[1], x=rs_E_gride, y = ids_E_gride, label = 'Gride', marker = 'o')
sns.lineplot(ax = ax[1], x=rs_E_twoNN, y = ids_E_twoNN, label = 'twoNN', marker = 'o')
ax[1].set_xscale('log')
ax[1].set_title('Encoded attributes', fontsize = 20)
ax[1].set_ylabel('ID', fontsize = 20)
#ax.axvline(noise_plane, color = 'black', alpha = 1, label = 'noise scale', linewidth = 0.8, linestyle= 'dotted')
ax[1].legend(fontsize = 20)

fig.text(0.5, 0.05, 'distance range', ha='center', fontsize=20)

#fig.tight_layout()
fig.savefig("id_hydro.png")