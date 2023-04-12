import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from pytorch_lightning import Callback
import os
import copy
from typing import Tuple

# NLDAS mean/std calculated over all basins in period 01.10.1999 until 30.09.2008
SCALER = {
    'input_means': np.array([3.015, 357.68, 10.864, 10.864, 1055.533]),
    'input_stds': np.array([7.573, 129.878, 10.932, 10.932, 705.998]),
    'output_mean': np.array([1.49996196]),
    'output_std': np.array([3.62443672])
}

class Scale_Data:
    def __init__(self, min : Tensor, max : Tensor) -> None:
        """
        min and max are tensor of shape (feature_dim,)
        """
        self.min = min.unsqueeze(0).unsqueeze(0)
        self.max = max.unsqueeze(0).unsqueeze(0)
        
    def __call__(self, x : torch.Tensor):
        """
        Scale x tensor in the range (min, max)
        Arguments
        ---------
            x : tensor of shape (1, seq_len, feature_dim)
        """
        # check dimensions
        # min/max lengths should match last dimension of x
        assert self.min.shape[-1] == x.shape[-1]
        assert self.max.shape[-1] == x.shape[-1]
        
        max_x = torch.amax(x, dim=1, keepdim=True) # shape (1, 1, feature_dim)
        min_x = torch.amin(x, dim=1, keepdim=True) # shape (1, 1, feature_dim)
        
        x = (x - min_x)/(max_x - min_x)
        x = x * (self.max - self.min) + self.min
        return x
    
class Globally_Scale_Data:
    def __init__(self, min : Tensor, max : Tensor) -> None:
        """
        min and max are tensor of shape (feature_dim,)
        """
        self.min = min.unsqueeze(0).unsqueeze(0)
        self.max = max.unsqueeze(0).unsqueeze(0)
        
    def __call__(self, x : torch.Tensor):
        """
        Scale x tensor in the range (min, max) globally
        Arguments
        ---------
            x : tensor of shape (1, seq_len, feature_dim)
        """
        # check dimensions
        # min/max lengths should match last dimension of x
        assert self.min.shape[-1] == x.shape[-1]
        assert self.max.shape[-1] == x.shape[-1]
        
        x = (x - self.min) / (self.max - self.min) 
        return x
    def reverse_transform(self, x : torch.Tensor):
        """
        Inverse of __call__ transformation
        Arguments
        ---------
            x : tensor of shape (1, seq_len, feature_dim)
        """
        # check dimensions
        # min/max lengths should match last dimension of x
        assert self.min.shape[-1] == x.shape[-1]
        assert self.max.shape[-1] == x.shape[-1]
        x = x * (self.max - self.min) + self.min
        return x

### callbacks
class MetricsCallback(Callback):
    """
    PyTorch Lightning metric callback.
    Save logged metrics
    """

    def __init__(self, dirpath, filename):
        super().__init__()
        self.dirpath = dirpath
        self.filename = filename
        self.path = os.path.join(dirpath, filename)
        exists = os.path.exists(self.path)
        # if already exists a saving, load it and update
        if exists:
            self.dict_metrics = torch.load(self.path, map_location=torch.device('cpu'))
        else:
            os.makedirs(self.dirpath, exist_ok = True) 
            self.dict_metrics = {}
            
        
    def on_validation_epoch_end(self,trainer, pl_module):
        epoch_num = int(trainer.logged_metrics["epoch_num"].cpu().item())
        self.dict_metrics["Epoch: "+str(epoch_num)] = copy.deepcopy(trainer.logged_metrics)
        torch.save(self.dict_metrics, self.path)


class NSELoss(nn.Module):
    def __init__(self, alpha = 2, reduction = "mean") -> None:
        super().__init__()
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, tar : Tensor, obs : Tensor) -> Tensor:
        """
        Arguments
        _________
            tar : target values, tensor of size (batch_size, seq_len) "true values"
            obs : observed values, tensor of size (batch_size, seq_len) "simulated values"
        Returns
        _______
            - NSE : minus the Nash-Sutcliffe efficiency, tensor of size ()
        """
        # compute NSE as batch
        assert(tar.shape==obs.shape)
        NSE_num = torch.sum((torch.abs(tar - obs))**self.alpha, dim=-1) # tensor of size (batch_size,)
        NSE_den = torch.sum((torch.abs(tar - torch.mean(obs, dim=-1, keepdims=True)))**self.alpha, dim=-1) # tensor of size (batch_size,)
        NSE_tensor = 1.0 - NSE_num / NSE_den
        if self.reduction == "mean":
            NSE = torch.mean(NSE_tensor)
        elif self.reduction == "sum":
            NSE = torch.sum(NSE_tensor)
        elif self.reduction == None:
            NSE = NSE_tensor
        else:
            raise Exception("Invalid reduction provided. Allowed 'mean', 'sum', None")
            
        return - NSE
 
 

 

class PFAB():
    """
    Compute absolute FHV, percent bias in flow duration curve high-segment volume
    """
    def __init__(self, ex_prob=0.01, reduction=None):
        self.ex_prob = ex_prob # exceedance probability
        self.reduction = reduction # reduction over batch dimension

    def __call__(self, tar : Tensor, obs : Tensor) -> Tensor:
        assert(tar.shape==obs.shape)
        seq_len = tar.shape[-1] # take sequence lenght
        
        high_peak_length = int(seq_len * self.ex_prob)
        
        sorted_tar, _ = torch.sort(tar, dim=-1, descending=True)
        sorted_tar = sorted_tar[:,:high_peak_length]
        sorted_obs, _= torch.sort(obs, dim=-1, descending=True)
        sorted_obs = sorted_obs[:,:high_peak_length]
        num = torch.sum(sorted_obs - sorted_tar, dim=-1, keepdim=False)
        den = torch.sum(sorted_obs, dim=-1, keepdim=False)
        out = num/den
        if self.reduction == "mean":
            out = torch.mean(out)
        elif self.reduction == "sum":
            NSE = torch.sum(out)
        elif self.reduction==None:
            pass
        else:
            raise Exception("Invalid reduction provided. Allowed 'mean', 'sum', None")
        
        return torch.abs(out)*100
            



def find_best_epoch(model_id):
        """
        Find the epoch at which the validation error is minimized, or quivalently
        when thevalidation NSE is maximized
        Returns
        -------
            best_epoch : (int)
        """
        dirpath = os.path.join("checkpoints", model_id)
        path_metrics = os.path.join(dirpath, "metrics.pt")
        data = torch.load(path_metrics, map_location=torch.device('cpu'))
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
        idx_ae = np.argmax(nse_mod)
        return int(epochs_mod[idx_ae])


def reshape_data(x: np.ndarray, y: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """Reshape data into LSTM many-to-one input samples
    Parameters
    ----------
    x : np.ndarray
        Input features of shape [num_samples, num_features]
    y : np.ndarray
        Output feature of shape [num_samples, 1]
    seq_length : int
        Length of the requested input sequences.
    Returns
    -------
    x_new: np.ndarray
        Reshaped input features of shape [num_samples*, seq_length, num_features], where 
        num_samples* is equal to num_samples - seq_length + 1, due to the need of a warm start at
        the beginning
    y_new: np.ndarray
        The target value for each sample in x_new
    """
    num_samples, num_features = x.shape

    x_new = np.zeros((num_samples - seq_length + 1, seq_length, num_features))
    y_new = np.zeros((num_samples - seq_length + 1, 1))

    for i in range(0, x_new.shape[0]):
        x_new[i, :, :num_features] = x[i:i + seq_length, :]
        y_new[i, :] = y[i + seq_length - 1, 0]

    return x_new, y_new


def normalize_features(feature: np.ndarray, variable: str) -> np.ndarray:
    """Normalize features using global pre-computed statistics.
    Parameters
    ----------
    feature : np.ndarray
        Data to normalize
    variable : str
        One of ['inputs', 'output'], where `inputs` mean, that the `feature` input are the model
        inputs (meteorological forcing data) and `output` that the `feature` input are discharge
        values.
    Returns
    -------
    np.ndarray
        Normalized features
    Raises
    ------
    RuntimeError
        If `variable` is neither 'inputs' nor 'output'
    """

    if variable == 'inputs':
        feature = (feature - SCALER["input_means"]) / SCALER["input_stds"]
    elif variable == 'output':
        feature = (feature - SCALER["output_mean"]) / SCALER["output_std"]
    else:
        raise RuntimeError(f"Unknown variable type {variable}")

    return feature


def rescale_features(feature: np.ndarray, variable: str) -> np.ndarray:
    """Rescale features using global pre-computed statistics.
    Parameters
    ----------
    feature : np.ndarray
        Data to rescale
    variable : str
        One of ['inputs', 'output'], where `inputs` mean, that the `feature` input are the model
        inputs (meteorological forcing data) and `output` that the `feature` input are discharge
        values.
    Returns
    -------
    np.ndarray
        Rescaled features
    Raises
    ------
    RuntimeError
        If `variable` is neither 'inputs' nor 'output'
    """
    if variable == 'inputs':
        feature = feature * SCALER["input_stds"] + SCALER["input_means"]
    elif variable == 'output':
        feature = feature * SCALER["output_std"] + SCALER["output_mean"]
    else:
        raise RuntimeError(f"Unknown variable type {variable}")

    return feature