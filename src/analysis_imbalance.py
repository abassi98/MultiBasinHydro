import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dadapy.plot import plot_inf_imb_plane
from dadapy.metric_comparisons import MetricComparisons

# retrieve data
S = np.loadtxt("statics.txt")
E = pd.read_csv("encoded_features_lstm_ae.txt", sep=" ", skiprows=1).iloc[:,4::]
E = np.array(E)

# define an instance of the MetricComparisons class
d = MetricComparisons(S)

# list of the coordinate names
labels = ['x', 'y', 'z']

# list of the the subsets of coordinates for which the imbalance should be computed
coord_list = [[0,], [1,], [2,], [0,1], [0,2], [1, 2]]