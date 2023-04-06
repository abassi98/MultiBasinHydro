import matplotlib as mpl
from matplotlib import cycler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dadapy.plot import plot_inf_imb_plane
from dadapy.metric_comparisons import MetricComparisons
import seaborn as sns
import copy

# retrieve statics
df_S = pd.read_csv("statics.txt", sep=" ").iloc[:,1::]
S_ids = df_S.iloc[:,0]
df_S = df_S.iloc[:,1::]
df_S = (df_S- df_S.min())/(df_S.max()-df_S.min())
print(df_S)
# retrieve encoded features
E_dim = 27
model_id = "lstm-ae-bdTrue-E"+str(E_dim)

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


# # plot correaltion matrix ES
# columns = df_ES.iloc[:,1::].columns
# corr = np.array(np.abs(df_ES.iloc[:,1::].corr()))[:E_dim ,E_dim :]

# fig, axs = plt.subplots(1,1,figsize=(10,10))
# g = sns.heatmap(corr, xticklabels=columns[E_dim :], yticklabels=columns[:E_dim ], cmap="YlOrRd", ax=axs, vmin=0, vmax=1)
# g.set_xticklabels(g.get_xticklabels(), rotation = 70, fontsize = 10)
# g.set_yticklabels(g.get_yticklabels(), rotation = 30, fontsize = 10)
# #sns.heatmap(corr_noise, xticklabels=corr.columns, yticklabels=corr.columns, cmap="bwr", ax=axs[1])
# file_corr = "plot/plot_corrES_"+model_id+".png"
# axs.set_title("Correlation matrix Encoded Features/Statics attributes(Kratzert)", fontsize=15)
# fig.savefig(file_corr)

# # plot correaltion matrix EH
# columns = df_EH.iloc[:,1::].columns
# corr = np.array(np.abs(df_EH.iloc[:,1::].corr()))[:E_dim ,E_dim :]

# fig, axs = plt.subplots(1,1,figsize=(10,10))
# g = sns.heatmap(corr, xticklabels=columns[E_dim :], yticklabels=columns[:E_dim ], cmap="YlOrRd", ax=axs, vmin=0, vmax=1)
# g.set_xticklabels(g.get_xticklabels(), rotation = 70, fontsize = 10)
# g.set_yticklabels(g.get_yticklabels(), rotation = 30, fontsize = 10)
# #sns.heatmap(corr_noise, xticklabels=corr.columns, yticklabels=corr.columns, cmap="bwr", ax=axs[1])
# file_corr = "plot/plot_corrEH_"+model_id+".png"
# axs.set_title("Correlation matrix Encoded Features/Hydrological Signatures", fontsize=15)
# fig.savefig(file_corr)

# # plot correaltion matrix SH
# columns = df_SH.iloc[:,1::].columns
# corr = np.array(np.abs(df_SH.iloc[:,1::].corr()))[:27,27:]
# fig, axs = plt.subplots(1,1,figsize=(10,10))
# g = sns.heatmap(corr, xticklabels=columns[27:], yticklabels=columns[:27], cmap="YlOrRd", ax=axs, vmin=0, vmax=1)
# g.set_xticklabels(g.get_xticklabels(), rotation = 70, fontsize = 10)
# g.set_yticklabels(g.get_yticklabels(), rotation = 30, fontsize = 10)
# #sns.heatmap(corr_noise, xticklabels=corr.columns, yticklabels=corr.columns, cmap="bwr", ax=axs[1])
# file_corr = "plot/plot_corrSH_"+model_id+".png"
# axs.set_title("Correlation matrix Static attributes(Kratzert)/Hydrological Signatures", fontsize=15)
# fig.savefig(file_corr)

# cmap = plt.cm.get_cmap("Oranges")
# mpl.rcParams['axes.prop_cycle'] = cycler(color=cmap(np.linspace(0, 1, 27)))

# # information imbalance between ES
# X = np.array(df_ES.iloc[:,1::])

# # define an instance of the MetricComparisons class
# d = MetricComparisons(X)

coor_E = np.arange(E_dim)
coor_S = np.arange(E_dim, E_dim+27)
coor_tot = np.arange(E_dim+27)


# imb_ES = d.return_inf_imb_two_selected_coords(coords1= coor_E, coords2= coor_S)

# fig, ax = plt.subplots(1,1,figsize=(10,10))

# imb_singles = []
# trimmed_indexes = []
# for i in range(27):
#     imb_singles.append(d.return_inf_imb_two_selected_coords(coords1= coor_E, coords2= [coor_S[i]]))
#     ax.scatter(imb_singles[i][0], imb_singles[i][1], label = "E ->"+df_ES.iloc[:,1::].columns[coor_S[i]])
#     # if imb_singles[i][0] > 0.8 and imb_singles[i][1] > 0.8:
#     #     print(columns[coor_S[i]])
#     #     trimmed_indexes.append(i)

# mask = np.ones(len(coor_S), dtype=bool)
# print(trimmed_indexes)
# mask[trimmed_indexes] = False
# trimmed_coor_S = coor_S[mask]
# print(trimmed_coor_S)
# imb_trimmed = d.return_inf_imb_two_selected_coords(coords1= coor_E, coords2= trimmed_coor_S)


# ax.scatter(imb_ES[0], imb_ES[1], label = 'E -> S', s=200, c="black")
# ax.scatter(imb_trimmed[0], imb_trimmed[1], label = 'E -> S trimmed', s=200, c="blue")

# ax.plot([0, 1], [0, 1], 'k--')
# ax.plot([0, 1], [0, 1], 'k--')
# ax.set_xlabel(r'$\Delta(x_1 \rightarrow x_2) $')
# ax.set_ylabel(r'$\Delta(x_2 \rightarrow x_1) $')

# ax.legend(bbox_to_anchor=(0.5, 0.2, 0.5, 0.5))
# file_corr = "plot/plot_imbalanceES_"+model_id+".png"
# fig.savefig(file_corr)

# ###### information imbalance between EH
# X = np.array(df_EH.iloc[:,1::])

# # define an instance of the MetricComparisons class
# mpl.rcParams['axes.prop_cycle'] = cycler(color=cmap(np.linspace(0, 1, 13)))

# d = MetricComparisons(X)

# coor_E = np.arange(3)
# coor_H = np.arange(3, 16)
# coor_tot = np.arange(16)
# print(coor_H)

# imb_EH = d.return_inf_imb_two_selected_coords(coords1= coor_E, coords2= coor_H)

# fig, ax = plt.subplots(1,1,figsize=(10,10))

# imb_singles = []
# trimmed_indexes = []
# for i in range(13):
#     imb_singles.append(d.return_inf_imb_two_selected_coords(coords1= coor_E, coords2= [coor_H[i]]))
#     ax.scatter(imb_singles[i][0], imb_singles[i][1], label = "E ->"+df_EH.iloc[:,1::].columns[coor_H[i]])
#     # if imb_singles[i][0] > 0.8 and imb_singles[i][1] > 0.8:
#     #     print(columns[coor_H[i]])
#     #     trimmed_indexes.append(i)

# mask = np.ones(len(coor_H), dtype=bool)
# print(trimmed_indexes)
# mask[trimmed_indexes] = False
# trimmed_coor_H = coor_H[mask]
# print(trimmed_coor_H)
# imb_trimmed = d.return_inf_imb_two_selected_coords(coords1= coor_E, coords2= trimmed_coor_H)


# ax.scatter(imb_EH[0], imb_EH[1], label = 'E -> H', s=200, c="black")
# ax.scatter(imb_trimmed[0], imb_trimmed[1], label = 'E -> H trimmed', s=200, c="blue")

# ax.plot([0, 1], [0, 1], 'k--')
# ax.plot([0, 1], [0, 1], 'k--')
# ax.set_xlabel(r'$\Delta(x_1 \rightarrow x_2) $')
# ax.set_ylabel(r'$\Delta(x_2 \rightarrow x_1) $')

# ax.legend(bbox_to_anchor=(0.5, 0.2, 0.5, 0.5))
# file_corr = "plot/plot_imbalanceEH_"+model_id+".png"
# fig.savefig(file_corr)


X = np.array(df_S)
# define an instance of the MetricComparisons class
d = MetricComparisons(X)
d.compute_distances(X.shape[0]-1)
fig, ax = plt.subplots(1,2,figsize=(10,10))

# ### assess which subset of static features is more contained in learned AE featues
# all_min_x = []
# all_min_y = []
# all_min_subsests_x = []
# x_axis = range(1,6)
# coor_S = np.arange(27)
# l = []

    
# for i in range(27):
#     subsets = coor_S - l
#     k_imbalance_x = []
#     k_imbalance_y = []
#     new_l = l
#     for new in subsets:
#         new_l.append(new)
#         imb = d.return_inf_imb_two_selected_coords(coords1= coor_S, coords2= new_l)
#         k_imbalance_x.append(imb[0])
#         k_imbalance_y.append(imb[1])

#     min_x = min(k_imbalance_x)
#     index_min_x = k_imbalance_x.index(min_x)
#     min_subset_x = subsets[index_min_x]

#     all_min_x.append(min_x)
#     all_min_subsests_x.append(min_subset_x)
#     all_min_y.append(k_imbalance_y[index_min_x])


best_sets, best_imbs, all_imbs = d.greedy_feature_selection_target(target_ranks=d.dist_indices, n_coords=27, n_best=1, k=1)
print(df_S.columns)
for set in best_sets:
    print(df_S.columns[set])

x_axis = np.arange(27)
ax[0].scatter(x_axis, np.log(best_imbs[:,0]))
ax[0].set_xlabel("k Features out of S")
ax[0].set_ylabel(r'$\Delta(E \rightarrow S_k) $')
ax[0].set_xticks(x_axis)

for i in range(best_imbs.shape[0]):
    ax[1].scatter(best_imbs[i,0], best_imbs[i,1], label="k = "+str(x_axis[i]))

ax[1].legend()
#ax[0].set_ylim(0,1)
ax[1].set_xlim(0,1)
ax[1].set_ylim(0,1)
ax[1].plot([0, 1], [0, 1], 'k--')
ax[1].set_xlabel(r'$\Delta(E \rightarrow S_k) $')
ax[1].set_ylabel(r'$\Delta(S_k \rightarrow E) $')


file_save = "plot/plot_iterative_imbalance_fullS.png"
fig.savefig(file_save)