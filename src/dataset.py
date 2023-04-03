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
from utils import Globally_Scale_Data



class CamelDataset(Dataset):
    def __init__(self, dates: list, force_attributes: list,  data_path: str = "basin_dataset_public_v1p2", source_data_set: str = "daymet", debug=False, transform_input=None, transform_output=None) -> None:
        super().__init__()
     
        self.data_path = data_path
        self.source_data_set = source_data_set
        self.basins_with_missing_data = [1208990, 1613050, 2051000, 2235200, 2310947,
        2408540,2464146,3066000,3159540,3161000,3187500,3281100,3300400,3450000,
        5062500,5087500,6037500,6043500,6188000,6441500,7290650,7295000,7373000,
        7376000,7377000,8025500,8155200,9423350,9484000,9497800,9505200,10173450,
        10258000,12025000,12043000,12095000,12141300,12374250,13310700]
        self.basins_with_missing_data = [str(x).rjust(8, "0") for x in self.basins_with_missing_data] # convert to string and pad
        self.debug = debug # debug mode default off
        self.transform_input = transform_input
        self.transform_output = transform_output
       

        # static attributes
        clim_attr = ["p_mean", "pet_mean", "p_seasonality", "frac_snow", "aridity", "high_prec_freq", "high_prec_dur","low_prec_freq", "low_prec_dur"] # 9 features
        df_clim = pd.read_csv(data_path+"/camels_clim.txt", sep=";")[clim_attr]
        geol_attr = ["carbonate_rocks_frac", "geol_permeability"] # 2 attributes
        df_geol = pd.read_csv(data_path+"/camels_geol.txt", sep=";")[geol_attr]
        topo_attr = ["elev_mean","slope_mean","area_gages2"] # 3 attributes
        df_topo = pd.read_csv(data_path+"/camels_topo.txt", sep=";")[topo_attr] 
        vege_attr = ["frac_forest","lai_max","lai_diff","gvf_max","gvf_diff"] # 5 attributes
        df_vege = pd.read_csv(data_path+"/camels_vege.txt", sep=";")[vege_attr]
        soil_attr = ["soil_depth_pelletier","soil_depth_statsgo","soil_porosity","soil_conductivity","max_water_content","sand_frac","silt_frac","clay_frac"] # 8 features
        df_soil = pd.read_csv(data_path+"/camels_soil.txt", sep=";")[soil_attr] 

        # hydrological signaures(normalized)
        self.df_hydro = pd.read_csv(data_path+"/camels_hydro.txt", sep=";").iloc[:,1:]
        self.hydro_attributes = self.df_hydro.shape[1] # as many as Kratzert
        self.hydro_ids = np.array(pd.read_csv(data_path+"/camels_hydro.txt", sep=";")["gauge_id"]).astype(int)

        # statics attributes(normalized)
        self.df_statics = pd.concat([df_clim, df_geol, df_topo, df_vege, df_soil], axis=1)
        self.static_attributes = self.df_statics.shape[1] # as many as Kratzert
        self.statics_ids = np.array(pd.read_csv(data_path+"/camels_clim.txt", sep=";")["gauge_id"]).astype(int)
     
        # convert string dates to datetime format
        self.start_date = datetime.datetime.strptime(dates[0], '%Y/%m/%d').date()
        self.end_date = datetime.datetime.strptime(dates[1], '%Y/%m/%d').date()
        

        # initialize dates and sequence length
        self.dates = [self.start_date +datetime.timedelta(days=x) for x in range((self.end_date-self.start_date).days+1)]
        self.seq_len = len(self.dates)
        self.force_attributes = force_attributes
        self.num_force_attributes = len(self.force_attributes)
        
        # define max/min forcing and flow tensors
        self.min_flow = 1000*torch.ones((1,), dtype=torch.float32)
        self.max_flow = -1000*torch.ones((1,), dtype=torch.float32)
        self.min_force = 1000*torch.ones((self.num_force_attributes,), dtype=torch.float32)
        self.max_force = -1000*torch.ones((self.num_force_attributes,), dtype=torch.float32)
        
        
        
        # ==========================================================================
        # gauge_information has to be read first to obtain correct HUC (hydrologic unit code)
        path_gauge_meta = os.path.join(self.data_path, "basin_metadata","gauge_information.txt")
        gauge_meta = pd.read_csv(path_gauge_meta, sep="\t", header=6, names=["HUC_02","GAGE_ID","GAGE_NAME","LAT","LONG","DRAINAGE AREA (KM^2)"])

        # retrieve basin_ids and basin_hucs and convert to string, possibly padded
        all_basin_ids = [str(gauge_meta.loc[i,"GAGE_ID"]).rjust(8,"0") for i in range(gauge_meta.shape[0])]
        all_basin_hucs = [str(gauge_meta.loc[i,"HUC_02"]).rjust(2,"0") for i in range(gauge_meta.shape[0])] 
        all_basin_names = [str(gauge_meta.loc[i,"GAGE_NAME"]) for i in range(gauge_meta.shape[0])] 

        # get rid of basin with missing data
        missing_data_indexes = []
        for i in range(len(all_basin_ids)):
            missing_data = False
            for j in range(len(self.basins_with_missing_data)):
                if all_basin_ids[i] == self.basins_with_missing_data[j]:
                    missing_data = True
            if missing_data==False:
                missing_data_indexes.append(i)

        self.trimmed_basin_ids = [all_basin_ids[i] for i in missing_data_indexes]
        self.trimmed_basin_hucs = [all_basin_hucs[i] for i in missing_data_indexes]
        self.trimmed_basin_names = [all_basin_names[i] for i in missing_data_indexes]

        # set container for hucs, ids and names effectively loaded
        self.len_dataset = 562 #len(self.trimmed_basin_ids)
        if self.debug:
            self.len_dataset = 15
      
        self.loaded_basin_hucs = []
        self.loaded_basin_ids = []
        self.loaded_basin_names = []

        # containers for data
        self.input_data = torch.zeros(self.len_dataset, 1, self.seq_len, 1)
        self.output_data = torch.zeros(self.len_dataset, 1, self.seq_len, self.num_force_attributes)
        self.statics_data = torch.zeros(self.len_dataset,1, 1, self.static_attributes)
        self.hydro_data = torch.zeros(self.len_dataset,1, 1, self.hydro_attributes)
        
    def adjust_dates(self, ):
        """
        Redefine the start and end date as the the maximum of 
        the start dates and the minimum of the end dates respectively.
        Then trim according to the missing values. Each catchment
        must have the same sequence length to be fed into CNN.
        To run the first time to find the correct dates
        """
        
        # run over trimmed basins
        print("Adjusting start and end dates ...")
        
        for i in range(len(self.trimmed_basin_ids)):
            # retrieve dates
            basin_id = self.trimmed_basin_ids[i]
            basin_huc = self.trimmed_basin_hucs[i]
            path_forcing_data = os.path.join(self.data_path, "basin_mean_forcing", self.source_data_set, basin_huc, basin_id + "_lump_cida_forcing_leap.txt")
            path_flow_data = os.path.join(self.data_path, "usgs_streamflow", basin_huc, basin_id + "_streamflow_qc.txt")
            
            # read flow and forcing dates
            df_streamflow = pd.read_csv(path_flow_data,delim_whitespace=True, header=None)
            flow_data = df_streamflow.iloc[:,4].to_numpy()
            flow_dates = np.array([datetime.date(df_streamflow.iloc[i,1], df_streamflow.iloc[i,2], df_streamflow.iloc[i,3]) for i in range(len(df_streamflow))])
            df_forcing = pd.read_csv(path_forcing_data,delim_whitespace=True, skiprows=3)
            force_dates = np.array([datetime.date(df_forcing.iloc[i,0], df_forcing.iloc[i,1], df_forcing.iloc[i,2]) for i in range(len(df_forcing))])
            
            bool_false_values = flow_data!= -999.0
            if all(bool_false_values): # take only catchments with no missing values
                # adjust dates
                if self.start_date < flow_dates[0]:
                    self.start_date = flow_dates[0]
                if self.start_date < force_dates[0]:
                    self.start_date = force_dates[0]

                if flow_dates[-1] < self.end_date:
                    self.end_date = flow_dates[-1]
                if force_dates[-1] < self.end_date:
                    self.end_date = force_dates[-1]
                
        # redefine dates and sequence length
        self.dates = [self.start_date +datetime.timedelta(days=x) for x in range((self.end_date-self.start_date).days+1)]
        self.seq_len = len(self.dates)
        print("...done.")
        
        print("New start date: " + self.start_date.strftime("%Y/%m/%d"))
        print("New end date: " + self.end_date.strftime("%Y/%m/%d"))
   
        
    def load_data(self, ):
        # run over trimmed basins
        print("Loading Camel ...")
        # len(self.trimmed_basin_ids)
        count = 0
        self.num_iter = len(self.trimmed_basin_ids)
        if self.debug:
            self.num_iter = 15
    
        for i in range(self.num_iter):
            # retrieve data
            basin_id = self.trimmed_basin_ids[i]
            basin_huc = self.trimmed_basin_hucs[i]
            path_forcing_data = os.path.join(self.data_path, "basin_mean_forcing", self.source_data_set, basin_huc, basin_id + "_lump_cida_forcing_leap.txt")
            path_flow_data = os.path.join(self.data_path, "usgs_streamflow", basin_huc, basin_id + "_streamflow_qc.txt")
            
            # read cathcment area
            with open(path_forcing_data) as myfile:
                first_force_lines = [next(myfile) for y in range(3)]
        
            area = float(first_force_lines[-1]) # area in meter squared

            # read flow data
            df_streamflow = pd.read_csv(path_flow_data,delim_whitespace=True, header=None)
            flow_data = df_streamflow.iloc[:,4].to_numpy()
            flow_dates = np.array([datetime.date(df_streamflow.iloc[i,1], df_streamflow.iloc[i,2], df_streamflow.iloc[i,3]) for i in range(len(flow_data))])
            
            # read forcing data 
            df_forcing = pd.read_csv(path_forcing_data,delim_whitespace=True, skiprows=3)
            force_dates = np.array([datetime.date(df_forcing.iloc[i,0], df_forcing.iloc[i,1], df_forcing.iloc[i,2]) for i in range(len(df_forcing))])
            
            # control length and assert they are equal
            #assert len(flow_data) == len(df_forcing)
            
            # get rid of cathcments with false values and whose dates range is not the
            interval_force_dates_bool = force_dates[0] <= self.start_date and force_dates[-1] >= self.end_date
            interval_flow_dates_bool = flow_dates[0] <= self.start_date and flow_dates[-1] >= self.end_date

            if interval_force_dates_bool and interval_flow_dates_bool:
                # adjust dates
                bool_flow_dates = np.logical_and(self.start_date <= flow_dates, flow_dates <= self.end_date)
                flow_data = flow_data[bool_flow_dates]
                flow_dates  = flow_dates[bool_flow_dates]
                bool_force_dates = np.logical_and(self.start_date <= force_dates, force_dates <= self.end_date)
                df_forcing = df_forcing[bool_force_dates]
                force_dates = force_dates[bool_force_dates]

                # control that dates are the same
                #print("Basin %d: " %count, basin_huc, basin_id)
                assert len(flow_data) == len(df_forcing)
                assert all(force_dates == flow_dates)

                # check that basins have with no missing data in the interval chosen
                bool_false_values = flow_data!= -999.0
                assert all(bool_false_values) == True

                # transfer from cubic feet (304.8 mm) per second to mm/day (normalized by basin area)
                flow_data = flow_data * (304.8**3)/(area * 10**6) * 86400

                # add tmean(C)
                # df_forcing["tmean(C)"] = (df_forcing["tmin(C)"] + df_forcing["tmax(C)"])/2.0
                # rescale day to hours
                df_forcing["dayl(s)"] = df_forcing["dayl(s)"]/3600.0
                df_forcing.rename(columns = {'dayl(s)':'dayl(h)'}, inplace = True)

                # take data
                force_data = torch.tensor(df_forcing.loc[:,self.force_attributes].to_numpy(), dtype=torch.float32).unsqueeze(0) # shape (1, seq_len, feature_dim=4)
                flow_data = torch.tensor(flow_data, dtype=torch.float32).unsqueeze(1).unsqueeze(0) # shape (1, seq_len, feature_dim=1)

                # take current max/min, tensor of size (feature_dim)
                current_min_flow = torch.amin(flow_data.squeeze(dim=0), dim=0)
                current_max_flow = torch.amax(flow_data.squeeze(dim=0), dim=0)
                current_min_force = torch.amin(force_data.squeeze(dim=0), dim=0)
                current_max_force = torch.amax(force_data.squeeze(dim=0), dim=0)
                
                # reset max/min force/flow
                self.min_flow = torch.minimum(self.min_flow, current_min_flow)
                self.max_flow = torch.maximum(self.max_flow, current_max_flow)
                self.min_force = torch.minimum(self.min_force, current_min_force)
                self.max_force = torch.maximum(self.max_force, current_max_force)
                
                # append
                self.input_data[count] = flow_data
                self.output_data[count] = force_data
                self.loaded_basin_hucs.append(self.trimmed_basin_hucs[i])
                self.loaded_basin_ids.append(self.trimmed_basin_ids[i])
                self.loaded_basin_names.append(self.trimmed_basin_names[i])
                count += 1
            
        # redefine transformations
        #self.transform_input = Globally_Scale_Data(self.min_flow, self.max_flow)
        #self.transform_output = Globally_Scale_Data(self.min_force, self.max_force)

        # normalize
        self.input_data = (self.input_data - torch.amin(self.input_data, dim=(0,2), keepdim=True))/(torch.amax(self.input_data, dim=(0,2), keepdim=True)-torch.amin(self.input_data, dim=(0,2), keepdim=True))
        self.output_data = (self.output_data - torch.amin(self.output_data, dim=(0,2), keepdim=True))/(torch.amax(self.output_data, dim=(0,2), keepdim=True)-torch.amin(self.output_data, dim=(0,2), keepdim=True))
        print("... done.")


    def load_statics(self):
        """
        Load static catchment features
        """
        print("Loading statics attributes...")
        for i in range(len(self.loaded_basin_ids)):
            for j in range(len(self.statics_ids)):
                if  int(self.loaded_basin_ids[i]) == self.statics_ids[j]:
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
        print("Loading statics attributes...")
        for i in range(len(self.loaded_basin_ids)):
            for j in range(len(self.hydro_ids)):
                if  int(self.loaded_basin_ids[i]) == self.hydro_ids[j]:
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

    


class YearlyCamelsDataset(Dataset):
    """
    Take a full range Camels Dataset (between 1980 and 2010) and split in sequences of 1 year
    """
    def __init__(self, indeces, start_date, end_date, dataset, num_years=15):
        super().__init__()
        
        # retrieve data
        self.start_date = datetime.datetime.strptime(start_date, '%Y/%m/%d').date()
        self.end_date = datetime.datetime.strptime(end_date, '%Y/%m/%d').date()
        self.num_years = num_years
        indeces.sort()
        self.basin_ids = [dataset.loaded_basin_ids[i] for i in indeces]

        # check if dates contains start and end date
        
        assert(dataset.dates.count(self.start_date)>0)
        assert(dataset.dates.count(self.end_date)>0)
        self.dates = np.array(dataset.dates)
        # trim dates
        bool_dates = np.logical_and(self.start_date <= self.dates, self.dates <= self.end_date)
        self.dates = self.dates[bool_dates]

        # select elements only on those dates for meterological data and flows
        flow_data = torch.index_select(dataset.input_data[:,:,bool_dates,:], 0, torch.tensor(indeces))
        force_data = torch.index_select(dataset.output_data[:,:,bool_dates,:], 0, torch.tensor(indeces))

        # take sequence length
        self.raw_seq_len = flow_data.shape[2]

        # load static data
        statics_features = torch.index_select(dataset.statics_data, 0, torch.tensor(indeces))
        hydro_features = torch.index_select(dataset.hydro_data, 0, torch.tensor(indeces))

        # take number of basins
        assert(flow_data.shape[0] == len(self.basin_ids))
        assert(flow_data.shape[0] == force_data.shape[0])
        assert(flow_data.shape[0] == statics_features.shape[0])
        assert(flow_data.shape[0] == hydro_features.shape[0])
        self.num_basins = len(self.basin_ids)
       

        # divide dataset
        self.input_data = []
        self.output_data = []
        self.hydro_data = []
        self.statics_data = []
        self.loaded_basin_ids = []

        for i in range(self.num_basins):
            for y in range(self.num_years): # number years of data for basins
                self.input_data.append(flow_data[i,:,y*365:(y+1)*365,:])
                self.output_data.append(force_data[i,:,y*365:(y+1)*365,:])
                self.hydro_data.append(hydro_features[i,:,:,:])
                self.statics_data.append(statics_features[i,:,:,:])
                self.loaded_basin_ids.append(self.basin_ids[i])


    def __len__(self):
        assert len(self.input_data)==len(self.output_data)
        return len(self.input_data)

    def __getitem__(self, idx):
        x_data = self.input_data[idx]
        y_data = self.output_data[idx]
        statics = self.statics_data[idx]
        hydro = self.hydro_data[idx]
        basin_id = self.loaded_basin_ids[idx]

        return x_data, y_data, statics, hydro, basin_id


    