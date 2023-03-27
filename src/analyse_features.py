import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt




if __name__ == '__main__':
    ##########################################################
    # Load encoded features of chosen LSTM-AE model
    ##########################################################
    # Load encoded features
    model_id = "lstm-ae-bdTrue-E4"
    filename = "encoded_features_"+model_id+".txt"
    df = pd.read_csv(filename, sep=" ")
    features = df.iloc[:,2::]
    features_basin_ids = df.iloc[:,1]
    # Load latitude and longitude
    file_topo = "basin_dataset_public_v1p2/camels_topo.txt"
    df_topo = pd.read_csv(file_topo, sep=";")
    topo_basin_ids = df_topo.iloc[:,0]
   
    lat_topo = df_topo["gauge_lat"]
    lon_topo = df_topo["gauge_lon"]

    lat = []
    lon = []
    for i in range(len(features_basin_ids)):
        for j in range(len(topo_basin_ids)):
            if topo_basin_ids[j] == features_basin_ids[i]:
                lat.append(lat_topo[j])
                lon.append(lon_topo[j])

    df["lat"] = lat
    df["lon"] = lon
   
    # initialize an axis
    fig = plt.figure(1, figsize = (6,6))
    plt.subplots_adjust(wspace=0.5, right=0.8, top=0.9, bottom=0.1)

    # plot map on axis
    countries = gpd.read_file(  
        gpd.datasets.get_path("naturalearth_lowres"))
    
    num_features = 4
    for i in range(num_features):
        plt.subplot(2,2,i+1)
        ax = plt.gca()
        ax.set_xlim(-128, -65)
        ax.set_ylim(24, 50)
        countries[countries["name"] == "United States of America"].plot(color="lightgrey", ax=ax)
        df.plot.scatter(x="lon", y="lat",c="E"+str(i), colormap="YlOrRd", title="E"+str(i), ax=ax, s=1, colorbar=False)
    
  
    fig.tight_layout()
    save_file = "plot_encoded_"+model_id+".png"
    fig.savefig(save_file)