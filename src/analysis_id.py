import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dadapy import data

A = pd.read_csv("statics.txt", sep=" ", skiprows=1).iloc[:,2::]
A = np.array(A)
H = pd.read_csv("hydro.txt", sep=" ", skiprows=1).iloc[:,2::]
H = np.array(H)

S = np.concatenate((A,H), axis=1)
print(S.shape)
E = pd.read_csv("encoded_features_lstm-ae-bdTrue-E27.txt", sep=" ", skiprows=1).iloc[:,2::]
E4 = pd.read_csv("encoded_features_lstm-ae-bdTrue-E4.txt", sep=" ", skiprows=1).iloc[:,2::]
E3 = pd.read_csv("encoded_features_lstm-ae-bdTrue-E3.txt", sep=" ", skiprows=1).iloc[:,2::]

E = np.array(E)
E4 = np.array(E4)
E3 = np.array(E3)


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

ids_S_gride, ids_S_twoNN, rs_S_gride, rs_S_twoNN = compute_ids_scaling(S,range_max = S.shape[0]-1,N_min = 10)
ids_E_gride, ids_E_twoNN, rs_E_gride, rs_E_twoNN = compute_ids_scaling(E,range_max = S.shape[0]-1,N_min = 10)
ids_E4_gride, ids_E4_twoNN, rs_E4_gride, rs_E4_twoNN = compute_ids_scaling(E4,range_max = S.shape[0]-1,N_min = 10)
ids_E3_gride, ids_E3_twoNN, rs_E3_gride, rs_E3_twoNN = compute_ids_scaling(E3,range_max = S.shape[0]-1,N_min = 10)

fig, ax = plt.subplots(1,4,figsize = (24, 8), sharey=True)


xrange = min(len(ids_S_gride), len(ids_S_twoNN))
sns.lineplot(ax = ax[0], x=rs_S_gride, y = ids_S_gride, label = 'Gride', marker = 'o')
sns.lineplot(ax = ax[0], x=rs_S_twoNN, y = ids_S_twoNN, label = 'twoNN', marker = 'o')
ax[0].set_xscale('log')
ax[0].set_title('Statics + Hydro attributes', fontsize = 20)
ax[0].set_ylabel('ID', fontsize = 20)
#ax.axvline(noise_plane, color = 'black', alpha = 1, label = 'noise scale', linewidth = 0.8, linestyle= 'dotted')
ax[0].legend(fontsize = 20)


xrange = min(len(ids_E_gride), len(ids_E_twoNN))
sns.lineplot(ax = ax[1], x=rs_E_gride, y = ids_E_gride, label = 'Gride', marker = 'o')
sns.lineplot(ax = ax[1], x=rs_E_twoNN, y = ids_E_twoNN, label = 'twoNN', marker = 'o')
ax[1].set_xscale('log')
ax[1].set_title('Encoded attributes (27)', fontsize = 20)
ax[1].set_ylabel('ID', fontsize = 20)
#ax.axvline(noise_plane, color = 'black', alpha = 1, label = 'noise scale', linewidth = 0.8, linestyle= 'dotted')
ax[1].legend(fontsize = 20)

xrange = min(len(ids_E4_gride), len(ids_E4_twoNN))
sns.lineplot(ax = ax[2], x=rs_E4_gride, y = ids_E4_gride, label = 'Gride', marker = 'o')
sns.lineplot(ax = ax[2], x=rs_E4_twoNN, y = ids_E4_twoNN, label = 'twoNN', marker = 'o')
ax[2].set_xscale('log')
ax[2].set_title('Encoded attributes (4)', fontsize = 20)
ax[2].set_ylabel('ID', fontsize = 20)
#ax.axvline(noise_plane, color = 'black', alpha = 1, label = 'noise scale', linewidth = 0.8, linestyle= 'dotted')
ax[2].legend(fontsize = 20)

xrange = min(len(ids_E3_gride), len(ids_E3_twoNN))
sns.lineplot(ax = ax[3], x=rs_E3_gride, y = ids_E3_gride, label = 'Gride', marker = 'o')
sns.lineplot(ax = ax[3], x=rs_E3_twoNN, y = ids_E3_twoNN, label = 'twoNN', marker = 'o')
ax[3].set_xscale('log')
ax[3].set_title('Encoded attributes (3)', fontsize = 20)
ax[3].set_ylabel('ID', fontsize = 20)
#ax.axvline(noise_plane, color = 'black', alpha = 1, label = 'noise scale', linewidth = 0.8, linestyle= 'dotted')
ax[3].legend(fontsize = 20)

fig.text(0.5, 0.05, 'distance range', ha='center', fontsize=20)

#fig.tight_layout()
fig.savefig("plot/id_hydro.png")