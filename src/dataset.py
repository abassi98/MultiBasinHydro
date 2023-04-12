"""
@author : Alberto Bassi
"""

#!/usr/bin/env python3
import numpy as np
import pandas as pd
import os
import datetime

import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class CamelDataset(Dataset):
    def __init__(self, dates: list, force_attributes: list,  data_path: str = "basin_dataset", source_data_set: str = "nldas_extended", debug=False) -> None:
        super().__init__()
     
        self.data_path = data_path
        self.source_data_set = source_data_set
        self.basin_list = np.loadtxt(data_path+"/basin_list.txt", dtype=str)
        self.basin_list = [str(x).rjust(8, "0") for x in self.basin_list] # convert to string and pad
        self.len_dataset = len(self.basin_list)
        self.debug = debug # debug mode default off
      
        # static attributes
        clim_attr = ["p_mean", "pet_mean", "p_seasonality", "frac_snow", "aridity", "high_prec_freq", "high_prec_dur","low_prec_freq", "low_prec_dur"] # 9 features
        df_clim = pd.read_csv(data_path+"/camels_attributes_v2.0/camels_clim.txt", sep=";")[clim_attr]
        geol_attr = ["carbonate_rocks_frac", "geol_permeability"] # 2 attributes
        df_geol = pd.read_csv(data_path+"/camels_attributes_v2.0/camels_geol.txt", sep=";")[geol_attr]
        topo_attr = ["elev_mean","slope_mean","area_gages2"] # 3 attributes
        df_topo = pd.read_csv(data_path+"/camels_attributes_v2.0/camels_topo.txt", sep=";")[topo_attr] 
        vege_attr = ["frac_forest","lai_max","lai_diff","gvf_max","gvf_diff"] # 5 attributes
        df_vege = pd.read_csv(data_path+"/camels_attributes_v2.0/camels_vege.txt", sep=";")[vege_attr]
        soil_attr = ["soil_depth_pelletier","soil_depth_statsgo","soil_porosity","soil_conductivity","max_water_content","sand_frac","silt_frac","clay_frac"] # 8 features
        df_soil = pd.read_csv(data_path+"/camels_attributes_v2.0/camels_soil.txt", sep=";")[soil_attr] 

        # hydrological signaures(normalized)
        self.df_hydro = pd.read_csv(data_path+"/camels_attributes_v2.0/camels_hydro.txt", sep=";").iloc[:,1:]
        self.hydro_attributes = self.df_hydro.shape[1] # as many as Kratzert
        self.hydro_ids = np.array(pd.read_csv(data_path+"/camels_attributes_v2.0/camels_hydro.txt", sep=";")["gauge_id"]).astype(int)

        # statics attributes(normalized)
        self.df_statics = pd.concat([df_clim, df_geol, df_topo, df_vege, df_soil], axis=1)
        self.static_attributes = self.df_statics.shape[1] # as many as Kratzert
        self.statics_ids = np.array(pd.read_csv(data_path+"/camels_attributes_v2.0/camels_clim.txt", sep=";")["gauge_id"]).astype(int)
     
        # convert string dates to datetime format
        self.start_date = datetime.datetime.strptime(dates[0], '%Y/%m/%d').date()
        self.end_date = datetime.datetime.strptime(dates[1], '%Y/%m/%d').date()

        # initialize dates and sequence length
        self.dates = [self.start_date +datetime.timedelta(days=x) for x in range((self.end_date-self.start_date).days+1)]
        self.seq_len = len(self.dates)
        self.force_attributes = force_attributes
        self.num_force_attributes = len(self.force_attributes) 
    
        self.input_data = torch.zeros(self.len_dataset, 1, self.seq_len, 1)
        self.output_data = torch.zeros(self.len_dataset, 1, self.seq_len, self.num_force_attributes)
        self.statics_data = torch.zeros(self.len_dataset,1, 1, self.static_attributes)
        self.hydro_data = torch.zeros(self.len_dataset,1, 1, self.hydro_attributes)

        
    def load_data(self, ):
        # run over trimmed basins
        print("Loading Camels ...")
        # len(self.trimmed_basin_ids)
        count = 0
        
        for i in tqdm(range(self.len_dataset)):
            # retrieve data
            basin_id = self.basin_list[i]
            path_forcing_data = os.path.join(self.data_path, self.source_data_set, basin_id + "_nldas.txt")
            path_flow_data = os.path.join(self.data_path, "streamflow", basin_id + "_streamflow.txt")
            
            # read data
            df_streamflow = pd.read_csv(path_flow_data, sep=" ")
            flow_data = df_streamflow.iloc[:,4].to_numpy()
            df_forcing = pd.read_csv(path_forcing_data,sep=" ")
            force_data = torch.tensor(df_forcing.iloc[:,4:].to_numpy(), dtype=torch.float32).unsqueeze(0) # shape (1, seq_len, feature_dim=4)
            flow_data = torch.tensor(flow_data, dtype=torch.float32).unsqueeze(1).unsqueeze(0) # shape (1, seq_len, feature_dim=1)
        
            # append
            self.input_data[i] = flow_data
            self.output_data[i] = force_data

       
        # normalize
        self.min_flow = torch.amin(self.input_data, dim=(0,2), keepdim=True).squeeze()
        self.max_flow = torch.amax(self.input_data, dim=(0,2), keepdim=True).squeeze()
        delta_input = torch.amax(self.input_data, dim=(0,2), keepdim=True)-torch.amin(self.input_data, dim=(0,2), keepdim=True)
        self.input_data = (self.input_data - torch.amin(self.input_data, dim=(0,2), keepdim=True))/delta_input
        self.min_force = torch.amin(self.output_data, dim=(0,2), keepdim=True).squeeze()
        self.max_force = torch.amax(self.output_data, dim=(0,2), keepdim=True).squeeze()
        delta_output = torch.amax(self.output_data, dim=(0,2), keepdim=True)-torch.amin(self.output_data, dim=(0,2), keepdim=True)
        self.output_data = (self.output_data - torch.amin(self.output_data, dim=(0,2), keepdim=True))/delta_output

        print("... done.")

    def save_dataset(self,):
        np.savetxt("basin_list.txt", np.array(self.loaded_basin_ids, dtype=str),  fmt='%s')
        dir_force = "basin_dataset/nldas_extended"
        dir_flow = "basin_dataset/streamflow"
        for i in range(len(self.loaded_basin_ids)):
            file_force = os.path.join(dir_force, self.loaded_basin_ids[i]+"_nldas.txt")
            df_force = pd.DataFrame()
            df_force["Year"] = [self.dates[i].year for i in range(len(self.dates))]
            df_force["Month"] = [self.dates[i].month for i in range(len(self.dates))]
            df_force["Day"] = [self.dates[i].day for i in range(len(self.dates))]
            df_force["PRCP(mm/day)"] = np.array(self.output_data[i].squeeze())[:,0]
            df_force["SRAD(W/m2)"] = np.array(self.output_data[i].squeeze())[:,1]
            df_force["Tmin(C)"] = np.array(self.output_data[i].squeeze())[:,2]
            df_force["Tmax(C)"] = np.array(self.output_data[i].squeeze())[:,3]
            df_force["Vp(Pa)"] = np.array(self.output_data[i].squeeze())[:,4]
            df_force.to_csv(file_force, sep=" ")

            file_flow = os.path.join(dir_flow, self.loaded_basin_ids[i]+"_streamflow.txt")
            df_flow = pd.DataFrame()
            df_flow["Year"] = [self.dates[i].year for i in range(len(self.dates))]
            df_flow["Month"] = [self.dates[i].month for i in range(len(self.dates))]
            df_flow["Day"] = [self.dates[i].day for i in range(len(self.dates))]
            df_flow["Streamflow(mm/day)"] = np.array(self.input_data[i].squeeze())
            df_flow.to_csv(file_flow, sep=" ")

    def load_statics(self):
        """
        Load static catchment features
        """
        print("Loading statics attributes...")
        for i in tqdm(range(len(self.basin_list))):
            for j in range(len(self.statics_ids)):
                if  int(self.basin_list[i]) == self.statics_ids[j]:
                    statics_data = torch.tensor(self.df_statics.iloc[j,:], dtype=torch.float32).unsqueeze(0).unsqueeze(0) # shape (1, seq_len, feature_dim=1)
                    self.statics_data[i] = statics_data
                  
        # renormalize
        delta = torch.amax(self.statics_data, dim=0, keepdim=True)-torch.amin(self.statics_data, dim=0, keepdim=True)
        delta[delta<10e-8] = 10e-8 # stabilize numerically
        self.statics_data = (self.statics_data- torch.amin(self.statics_data, dim=0, keepdim=True))/delta
        print("...done.")


    def load_hydro(self):
        """
        Load hydrological fingerprints features
        """
        print("Loading hydrological signatures...")
        for i in tqdm(range(len(self.basin_list))):
            for j in range(len(self.hydro_ids)):
                if  int(self.basin_list[i]) == self.hydro_ids[j]:
                    hydro_data = torch.tensor(self.df_hydro.iloc[j,:], dtype=torch.float32).unsqueeze(0).unsqueeze(0) # shape (1, seq_len, feature_dim=1)
                    self.hydro_data[i] = hydro_data
        # renormalize
        delta = torch.amax(self.hydro_data, dim=0, keepdim=True)-torch.amin(self.hydro_data, dim=0, keepdim=True)
        delta[delta<10e-8] = 10e-8 # stabilize numerically
        self.hydro_data = (self.hydro_data- torch.amin(self.hydro_data, dim=0, keepdim=True))/delta
        print("...done.")
                  
    def save_statics(self, filename):
        np_data =  self.statics_data.squeeze().cpu().numpy()
        df = pd.DataFrame(np_data, columns=self.df_statics.columns)
        df.insert(0, "basin_id", self.loaded_basin_ids)
        df.to_csv(filename, sep=" ")


    def save_hydro(self, filename):
        np_data =  self.hydro_data.squeeze().cpu().numpy()
        df = pd.DataFrame(np_data, columns=self.df_hydro.columns)
        df.insert(0, "basin_id", self.loaded_basin_ids)
        df.to_csv(filename, sep=" ")
    
    
    def __len__(self):
        assert len(self.input_data)==len(self.output_data)
        return len(self.input_data)

    def __getitem__(self, idx):
        x_data = self.input_data[idx]
        y_data = self.output_data[idx]
        statics = self.statics_data[idx]
        hydro = self.hydro_data[idx]
        
        return x_data, y_data, statics, hydro



