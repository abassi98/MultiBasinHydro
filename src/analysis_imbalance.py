import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dadapy.plot import plot_inf_imb_plane
from dadapy.metric_comparisons import MetricComparisons
import seaborn as sns
# retrieve data
df_S = pd.read_csv("statics.txt", sep=" ").iloc[:,2::]

model_id = "lstm-ae-bdTrue-E4"
filename = "encoded_features_"+model_id+".txt"
df_E = pd.read_csv(filename, sep=" ").iloc[:,1::]
noise = pd.DataFrame(np.random.normal(size=(562,5)),columns= df_E.columns)
df_noise = pd.concat([noise, df_S], axis=1)
df = pd.concat([df_E, df_S], axis=1)

columns = df.iloc[:,1::].columns
corr = df.iloc[:,1::].corr()
corr_noise = df_noise.iloc[:,1::].corr()
fig, axs = plt.subplots(1,1,figsize=(10,10))
g = sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, cmap="bwr", ax=axs)
g.set_xticklabels(g.get_xticklabels(), rotation = 70, fontsize = 8)
g.set_yticklabels(g.get_yticklabels(), rotation = 30, fontsize = 8)
#sns.heatmap(corr_noise, xticklabels=corr.columns, yticklabels=corr.columns, cmap="bwr", ax=axs[1])
file_corr = "plot_corrmat_"+model_id+".png"
fig.savefig(file_corr)


X = np.array(df.iloc[:,1::])
print(X.shape) # first 4 are encoded features, last 27 columns ar static attributes
# define an instance of the MetricComparisons class
d = MetricComparisons(X)

coor_E = np.arange(4)
coor_S = np.arange(4, 31)
coor_tot = np.arange(31)
print(coor_S)

imb_E = d.return_inf_imb_two_selected_coords(coords1= coor_tot, coords2= coor_E)
imb_S = d.return_inf_imb_two_selected_coords(coords1= coor_tot, coords2= coor_S)
imb_ES = d.return_inf_imb_two_selected_coords(coords1= coor_E, coords2= coor_S)

fig, ax = plt.subplots(1,1,figsize=(10,10))


for i in range(27):
    imb_singles =  d.return_inf_imb_two_selected_coords(coords1= coor_E, coords2= [coor_S[i]])
    ax.scatter(imb_singles[0], imb_singles[1], label = "Im E/"+columns[i+4])



ax.scatter(imb_E[0], imb_E[1], label = 'Im tot/E', s=200, c="red")
ax.scatter(imb_S[0], imb_S[1], label = 'Im tot/S', s=200, c="blue")
ax.scatter(imb_ES[0], imb_ES[1], label = 'Im E/S', s=200, c="black")
ax.plot([0, 1], [0, 1], 'k--')
ax.plot([0, 1], [0, 1], 'k--')
ax.set_xlabel(r'$\Delta(x_1 \rightarrow x_2) $')
ax.set_ylabel(r'$\Delta(x_2 \rightarrow x_1) $')

fig.legend(bbox_to_anchor=(0.5, 0.2, 0.5, 0.5))
file_corr = "plot_imbalance_"+model_id+".png"
fig.savefig(file_corr)