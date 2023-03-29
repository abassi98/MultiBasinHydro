import matplotlib as mpl
from matplotlib import cycler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dadapy.plot import plot_inf_imb_plane
from dadapy.metric_comparisons import MetricComparisons
import seaborn as sns
# retrieve statics
df_S = pd.read_csv("statics.txt", sep=" ").iloc[:,1::]
S_ids = df_S.iloc[:,0]
df_S = df_S.iloc[:,1::]
df_S = (df_S- df_S.min())/(df_S.max()-df_S.min())

# retrieve encoded features
model_id = "lstm-ae-bdTrue-E4"
filename = "encoded_features_"+model_id+".txt"
df_E = pd.read_csv(filename, sep=" ").iloc[:,1::]
E_ids = df_E.iloc[:,0]
df_E = df_E.iloc[:,1::]
df_E = (df_E -df_E .min())/(df_E .max()- df_E .min())

# retrieve hydro features
df_H = pd.read_csv("hydro.txt", sep=" ").iloc[:,1::]
H_ids = df_H.iloc[:,0]
df_H = df_H.iloc[:,1::]
df_H = (df_H- df_H.min())/(df_H.max()-df_H.min())

# assert
assert((S_ids==E_ids).all())
assert((S_ids==H_ids).all())
assert((E_ids==H_ids).all())

# concat ES
df_ES = pd.concat([df_E, df_S], axis=1)
df_ES.insert(0, "basin_id", S_ids)

# concat EH
df_EH = pd.concat([df_E, df_H], axis=1)
df_EH.insert(0, "basin_id", H_ids)

# concat SH
df_SH = pd.concat([df_S, df_H], axis=1)
df_SH.insert(0, "basin_id", H_ids)


# plot correaltion matrix ES
columns = df_ES.iloc[:,1::].columns
corr = np.array(np.abs(df_ES.iloc[:,1::].corr()))[:4,4:]

fig, axs = plt.subplots(1,1,figsize=(10,10))
g = sns.heatmap(corr, xticklabels=columns[4:], yticklabels=columns[:4], cmap="YlOrRd", ax=axs, vmin=0, vmax=1)
g.set_xticklabels(g.get_xticklabels(), rotation = 70, fontsize = 10)
g.set_yticklabels(g.get_yticklabels(), rotation = 30, fontsize = 10)
#sns.heatmap(corr_noise, xticklabels=corr.columns, yticklabels=corr.columns, cmap="bwr", ax=axs[1])
file_corr = "plot_corrES_"+model_id+".png"
axs.set_title("Correlation matrix Encoded Features/Statics attributes(Kratzert)", fontsize=15)
fig.savefig(file_corr)

# plot correaltion matrix EH
columns = df_EH.iloc[:,1::].columns
corr = np.array(np.abs(df_EH.iloc[:,1::].corr()))[:4,4:]

fig, axs = plt.subplots(1,1,figsize=(10,10))
g = sns.heatmap(corr, xticklabels=columns[4:], yticklabels=columns[:4], cmap="YlOrRd", ax=axs, vmin=0, vmax=1)
g.set_xticklabels(g.get_xticklabels(), rotation = 70, fontsize = 10)
g.set_yticklabels(g.get_yticklabels(), rotation = 30, fontsize = 10)
#sns.heatmap(corr_noise, xticklabels=corr.columns, yticklabels=corr.columns, cmap="bwr", ax=axs[1])
file_corr = "plot_corrEH_"+model_id+".png"
axs.set_title("Correlation matrix Encoded Features/Hydrological Signatures", fontsize=15)
fig.savefig(file_corr)

# plot correaltion matrix SH
columns = df_SH.iloc[:,1::].columns
corr = np.array(np.abs(df_SH.iloc[:,1::].corr()))[:13,13:]
fig, axs = plt.subplots(1,1,figsize=(10,10))
g = sns.heatmap(corr, xticklabels=columns[13:], yticklabels=columns[:13], cmap="YlOrRd", ax=axs, vmin=0, vmax=1)
g.set_xticklabels(g.get_xticklabels(), rotation = 70, fontsize = 10)
g.set_yticklabels(g.get_yticklabels(), rotation = 30, fontsize = 10)
#sns.heatmap(corr_noise, xticklabels=corr.columns, yticklabels=corr.columns, cmap="bwr", ax=axs[1])
file_corr = "plot_corrSH_"+model_id+".png"
axs.set_title("Correlation matrix Static attributes(Kratzert)/Hydrological Signatures", fontsize=15)
fig.savefig(file_corr)


# information imbalance between ES
X = np.array(df_ES.iloc[:,1::])

# define an instance of the MetricComparisons class
d = MetricComparisons(X)

coor_E = np.arange(4)
coor_S = np.arange(4, 31)
coor_tot = np.arange(31)
print(coor_S)

imb_ES = d.return_inf_imb_two_selected_coords(coords1= coor_E, coords2= coor_S)

fig, ax = plt.subplots(1,1,figsize=(10,10))

imb_singles = []
trimmed_indexes = []
for i in range(27):
    imb_singles.append(d.return_inf_imb_two_selected_coords(coords1= coor_E, coords2= [coor_S[i]]))
    ax.scatter(imb_singles[i][0], imb_singles[i][1], label = "E ->"+columns[coor_S[i]])
    if imb_singles[i][0] > 0.8 and imb_singles[i][1] > 0.8:
        print(columns[coor_S[i]])
        trimmed_indexes.append(i)

mask = np.ones(len(coor_S), dtype=bool)
print(trimmed_indexes)
mask[trimmed_indexes] = False
trimmed_coor_S = coor_S[mask]
print(trimmed_coor_S)
imb_trimmed = d.return_inf_imb_two_selected_coords(coords1= coor_E, coords2= trimmed_coor_S)


ax.scatter(imb_ES[0], imb_ES[1], label = 'E -> S', s=200, c="black")
ax.scatter(imb_trimmed[0], imb_trimmed[1], label = 'E -> S trimmed', s=200, c="blue")

ax.plot([0, 1], [0, 1], 'k--')
ax.plot([0, 1], [0, 1], 'k--')
ax.set_xlabel(r'$\Delta(x_1 \rightarrow x_2) $')
ax.set_ylabel(r'$\Delta(x_2 \rightarrow x_1) $')
cmap = plt.cm.coolwarm
mpl.rcParams['axes.prop_cycle'] = cycler(color=cmap(np.linspace(0, 1, 27)))

ax.legend(bbox_to_anchor=(0.5, 0.2, 0.5, 0.5))
file_corr = "plot_imbalanceES_"+model_id+".png"
fig.savefig(file_corr)