import torch
import torch.nn as nn
from torch import Tensor
from pytorch_lightning import Callback
import os
import copy

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
            self.dict_metrics = torch.load(self.path)
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
            tar : target values, tensor of size (batch_size, seq_len)
            obs : observed values, tensor of size (batch_size, seq_len)
        Returns
        _______
            - NSE : minus the Nash-Sutcliffe efficiency, tensor of size ()
        """
        # compute NSE as batch
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
