import torch
import torch.nn as nn
from torch import Tensor
from pytorch_lightning import Callback

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
    """PyTorch Lightning metric callback."""

    def __init__(self):
        super().__init__()
        self.train_loss_log = []
        self.val_loss_log = []        
        
    def on_train_epoch_end(self, trainer, pl_module):
        train_loss = trainer.logged_metrics["train_loss"].cpu()
        self.train_loss_log.append(train_loss)
    
        
    def on_validation_epoch_end(self,trainer, pl_module):
        val_loss = trainer.logged_metrics["val_loss"].cpu()            
        self.val_loss_log.append(val_loss)


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


    