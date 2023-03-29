import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim
from utils import NSELoss
from torch.optim.lr_scheduler import MultiStepLR

### Convolutional LSTM Autoencoder 
class ConvEncoder(nn.Module):
    
    def __init__(self,
                 in_channels, 
                 out_channels,
                 kernel_sizes,
                 padding = (0,0,0),
                 encoded_space_dim = 2, 
                 drop_p = 0.5,
                 act = nn.LeakyReLU,
                 seq_len = 100,
                 linear = 256,
                ):
        """
        Convolutional Network with three convolutional and two dense layers
        Args:
            in_channels : inputs channels
            out_channels : output channels
            kernel_sizes : kernel sizes
            padding : padding added to edges
            encoded_space_dim : dimension of encoded space
            drop_p : dropout probability
            act : activation function
            seq_len : length of input sequences 
            weight_decay : l2 regularization constant
            linea : linear layer units
        """
        super().__init__()
    
        # Retrieve parameters
        self.in_channels = in_channels #tuple of int, input channels for convolutional layers
        self.out_channels = out_channels #tuple of int, of output channels 
        self.kernel_sizes = kernel_sizes #tuple of tuples of int kernel size, single integer or tuple itself
        self.padding = padding
        self.encoded_space_dim = encoded_space_dim
        self.drop_p = drop_p
        self.act = act
        self.seq_len = seq_len
        self.linear = linear 
      
        ### Network architecture
        # First convolutional layer (2d convolutional layer
        self.first_conv = nn.Sequential(
            nn.Conv1d(self.in_channels[0], self.out_channels[0], self.kernel_sizes[0], padding=self.padding[0]), 
            nn.BatchNorm1d(self.out_channels[0]),
            self.act(inplace = True),
            nn.Dropout(self.drop_p, inplace = False),
            nn.AvgPool1d(4)
        )
        
        # Second convolution layer
        self.second_conv = nn.Sequential(
            nn.Conv1d(self.in_channels[1], self.out_channels[1], self.kernel_sizes[1], padding=self.padding[1]), 
            nn.BatchNorm1d(self.out_channels[1]),
            self.act(inplace = True),
            nn.Dropout(self.drop_p, inplace = False),
            nn.AvgPool1d(4)
        )
        
        # Third convolutional layer
        self.third_conv = nn.Sequential(
            nn.Conv1d(self.in_channels[2], self.out_channels[2], self.kernel_sizes[2], padding=self.padding[2]), 
            nn.BatchNorm1d(self.out_channels[2]),
            self.act(inplace = True),
            nn.Dropout(self.drop_p, inplace = False),
            nn.AvgPool1d(4)
        )


        # Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        
        # Liner dimension after 2 convolutional layers
        self.lin_dim = int((((self.seq_len-self.kernel_sizes[0]+1)/4.+1-self.kernel_sizes[1])/4.+1-self.kernel_sizes[2])/4.)
        
        # Linear encoder
        self.encoder_lin = nn.Sequential(
            # First linear layer
            nn.Linear(self.out_channels[2]*self.lin_dim, self.linear),
            nn.BatchNorm1d(self.linear),
            self.act(inplace = True),
            nn.Dropout(self.drop_p, inplace = False),
            # Second linear layer
            nn.Linear(self.linear, self.encoded_space_dim)
        )

       
    def forward(self, x):
        # Apply first convolutional layer
        x = self.first_conv(x)
        # Apply second convolutional layer
        x = self.second_conv(x)
        # Apply third conv layer
        x = self.third_conv(x)
        # Flatten 
        x = self.flatten(x)
        # Apply linear encoder layer
        x = self.encoder_lin(x)

        return x

class Hydro_LSTM_AE(pl.LightningModule):
    """
    Autoencoder with a convolutional encoder and a LSTM decoder
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_sizes,
                 padding = (0,0,0),
                 encoded_space_dim = 2,
                 lstm_hidden_units = 100, 
                 bidirectional = False,
                 layers_num = 2,
                 act = nn.LeakyReLU, 
                 loss_fn = nn.MSELoss(),
                 drop_p = 0.5, 
                 seq_len = 100,
                 lr = 1e-4,
                 linear = 512,
                 weight_decay = 0.0,
                 num_force_attributes = 5,
                 warmup = 730, # 2 years
                ):
        
        """
        Convolutional Symmetric Autoencoder
        Args:
            in_channels : inputs channels
            out_channels : output channels
            kernel_sizes : kernel sizes
            padding : padding added to edges
            encoded_space_dim : dimension of encoded space
            lstm_hidden_units : hidden units of LSTM, 
            bidirectional : if LSTMs are bidirectional or not,
            layers_num : number of LSTM layers,
            drop_p : dropout probability
            act : activation function
            seq_len : length of input sequences 
            lr : learning rate
        """
        
        super().__init__()
        self.save_hyperparameters(ignore=['loss_fn']) # save hyperparameters for chekpoints
        
        # Parameters
        self.seq_len = seq_len
        self.lr = lr
        self.encoded_space_dim = encoded_space_dim
        self.weight_decay = weight_decay
        self.sigmoid = nn.Sigmoid()
        self.loss_fn = loss_fn
        self.num_force_attributes = num_force_attributes 
        self.warmup = warmup # warmup days

        # Encoder
        self.encoder = ConvEncoder(in_channels, out_channels, kernel_sizes,padding, encoded_space_dim, 
                 drop_p=drop_p, act=act, seq_len=seq_len, linear=linear)
     
                    
        ### LSTM decoder
        self.lstm = nn.LSTM(input_size=encoded_space_dim+num_force_attributes, 
                           hidden_size=lstm_hidden_units,
                           num_layers=layers_num,
                           dropout=drop_p,
                           batch_first=True,
                          bidirectional=bidirectional)
        
        self.dropout = nn.Dropout(drop_p, inplace = False)
        if bidirectional:
            D = 2
        else:
            D = 1
            
        self.out = nn.Linear(D * lstm_hidden_units, 1)

        print("Convolutional LSTM Autoencoder initialized")

        
    def forward(self, x, y):
        # Encode data and keep track of indexes
        enc = self.encoder(x.squeeze(dim=-1))
        enc_expanded = self.sigmoid(enc.unsqueeze(1).expand(-1, self.seq_len, -1))
        # concat data
        input_lstm = torch.cat((enc_expanded.squeeze(), y.squeeze()),dim=-1)
        # Decode data
        hidd_rec, _ = self.lstm(input_lstm)
        #hidd_rec = self.dropout(hidd_rec)
        # Fully connected output layer, forced in [0,1]
        rec = self.out(hidd_rec)
        rec = self.sigmoid(rec)
        # Reinsert channel dimension
        rec = rec.unsqueeze(1)
        return enc, rec
        
    def training_step(self, batch, batch_idx):        
        ### Unpack batch
        x, y = batch
        # select past period
        x_past = x[:,:,:self.seq_len,:]
        y_past = y[:,:,:self.seq_len,:]
        # select future (validation) period
        x_fut = x[:,:,1+self.seq_len:,:]
        y_fut = y[:,:,1+self.seq_len:,:]
        # forward pass
        _, rec_past = self.forward(x_past,y_past)
        # Logging to TensorBoard by default
        train_loss = self.loss_fn(x_past.squeeze()[:,self.warmup:], rec_past.squeeze()[:,self.warmup:])
        self.log("train_loss", train_loss, prog_bar=True)
        # compute future (training) loss and log
        with torch.no_grad():
            _, rec_fut = self.forward(x_fut,y_fut)
        train_fut_loss = self.loss_fn(x_fut.squeeze()[:,self.warmup:], rec_fut.squeeze()[:,self.warmup:])
        self.log("train_fut_loss", train_fut_loss)

        #print(self.lr_scheduler.get_last_lr())
        return train_loss
    
    def validation_step(self, batch, batch_idx):
        ### Unpack batch
        x, y = batch
        # select past period
        x_past = x[:,:,:self.seq_len,:]
        y_past = y[:,:,:self.seq_len,:]
        # select future (validation) period
        x_fut = x[:,:,1+self.seq_len:,:]
        y_fut = y[:,:,1+self.seq_len:,:]
        
 
        # forward pass
        _, rec_fut = self.forward(x_fut,y_fut)
        # Logging to TensorBoard by default
        val_loss = self.loss_fn(x_fut.squeeze()[:,self.warmup:], rec_fut.squeeze()[:,self.warmup:])
        # Logging to TensorBoard by default
        self.log("val_loss", val_loss, prog_bar=True)
        self.log("epoch_num", float(self.current_epoch),prog_bar=True)
        
        # compute past (validation) loss
        with torch.no_grad():
            _, rec_past = self.forward(x_past,y_past)
        val_past_loss = self.loss_fn(x_past.squeeze()[:,self.warmup:], rec_past.squeeze()[:,self.warmup:])
        self.log("val_past_loss", val_past_loss)
        
        return val_loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr = self.lr, weight_decay = self.weight_decay)
        lr_scheduler = MultiStepLR(optimizer, milestones=[300,], gamma=0.1)
        return {"optimizer":optimizer, "lr_scheduler":lr_scheduler}



class Hydro_LSTM(pl.LightningModule):
    """
    LSTM decoder
    """
    def __init__(self,
                 lstm_hidden_units = 100, 
                 bidirectional = False,
                 layers_num = 2,
                 act = nn.LeakyReLU, 
                 loss_fn = nn.MSELoss(),
                 drop_p = 0.5, 
                 seq_len = 100,
                 lr = 1e-4,
                 weight_decay = 0.0,
                 num_force_attributes = 5,
                 noise_dim = 0,
                 statics = False,
                ):
        
        """
        Args:
            lstm_hidden_units : hidden units of LSTM, 
            bidirectional : if LSTMs are bidirectional or not,
            layers_num : number of LSTM layers,
            drop_p : dropout probability
            act : activation function
            seq_len : length of input sequences 
            lr : learning rate
        """
        
        super().__init__()
        self.save_hyperparameters(ignore=['loss_fn']) # save hyperparameters for chekpoints
        
        # Parameters
        self.seq_len = seq_len
        self.lr = lr
        self.weight_decay = weight_decay
        self.sigmoid = nn.Sigmoid()
        self.loss_fn = loss_fn
        self.num_force_attributes = num_force_attributes 
        self.noise_dim = noise_dim
        self.statics = statics

        ### LSTM decoder
        if self.statics:
            input_size = num_force_attributes + 27
        else:
            input_size = num_force_attributes + noise_dim
            
        self.lstm = nn.LSTM(input_size=input_size, 
                           hidden_size=lstm_hidden_units,
                           num_layers=layers_num,
                           dropout=drop_p,
                           batch_first=True,
                          bidirectional=bidirectional)
        
        self.dropout = nn.Dropout(drop_p, inplace = False)
        if bidirectional:
            D = 2
        else:
            D = 1
            
        self.out = nn.Linear(D * lstm_hidden_units, 1)

        print("LSTM initialized")

        
    def forward(self, y, statics): 
        # Decode data
        if self.noise_dim == 0:
            if self.statics:
                input_lstm = torch.cat((statics.squeeze(1).repeat(1,self.seq_len,1), y.squeeze()),dim=-1)
                hidd_rec, _ = self.lstm(input_lstm)
            else:
                hidd_rec, _ = self.lstm(y.squeeze(1))
        else:
            y_shape = y.squeeze().shape # size (batch_size, seq_len, force_attributes)
            noise = self.sigmoid(torch.randn(size=(y_shape[0], self.seq_len, self.noise_dim), device=self.device))
            # concat data
            input_lstm = torch.cat((noise, y.squeeze()),dim=-1)
            hidd_rec, _ = self.lstm(input_lstm)

        #hidd_rec = self.dropout(hidd_rec)
        # Fully connected output layer, forced in [0,1]
        rec = self.out(hidd_rec)
        rec = self.sigmoid(rec)
        # Reinsert channel dimension
        rec = rec.unsqueeze(1)
        return rec
        
    def training_step(self, batch, batch_idx):        
        ### Unpack batch
        x, y, statics = batch
        # forward pass
        rec = self.forward(y, statics)
        # Logging to TensorBoard by default
        train_loss = self.loss_fn(x.squeeze(), rec.squeeze())
        self.log("train_loss", train_loss, prog_bar=True)
        return train_loss
    
    def validation_step(self, batch, batch_idx):
        ### Unpack batch
        x, y, statics = batch
            
        # forward pass
        rec = self.forward(y, statics)
        # Logging to TensorBoard by default
        val_loss = self.loss_fn(x.squeeze(), rec.squeeze())
        # Logging to TensorBoard by default
        self.log("val_loss", val_loss, prog_bar=True)
        self.log("epoch_num", float(self.current_epoch),prog_bar=True)
        
        return val_loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr = self.lr, weight_decay = self.weight_decay)
        #lr_scheduler = MultiStepLR(optimizer, milestones=[300,], gamma=0.1)
        return optimizer #{"optimizer":optimizer, "lr_scheduler":lr_scheduler}


class Hydro_FFNet(pl.LightningModule):

    def __init__(self, 
                 n_inputs, 
                 n_outputs, 
                 hidden_layers, 
                 drop_p = 0.5, 
                 lr = 1e-4, 
                 activation = nn.LeakyReLU,
                 weight_decay = 0,
                ):
        """
        Initialize a typical feedforward network with different hidden layers
        The input is typically a mnist image, given as a torch tensor of size = (1,784),
        or a sequence, torch.tensor of size (1, seq_length, 3)
        Args:
            n_inputs : input features
            n_outputs : output features
            hidden_layers : list of sizes of the hidden layers
            drop_p : dropout probability
            lr : learning rate
            activation : activation function
            weight_decay : l2 regularization constant
        """
        super().__init__()
        # Parameters
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.hidden_layers = hidden_layers
        self.num_hidden_layers = len(self.hidden_layers)
        self.drop_p = drop_p
        self.lr = lr
        self.activation = activation
        self.weight_decay = weight_decay
      
        ### Network architecture
        layers = []
        
        # input layer
        layers.append(nn.Linear(self.n_inputs, self.hidden_layers[0]))
        layers.append(nn.BatchNorm1d(self.hidden_layers[0]))
        layers.append(nn.Dropout(self.drop_p, inplace = False))
        layers.append(self.activation(inplace=True))
        
        # hidden layers
        for l in range(self.num_hidden_layers-1):
            layers.append(nn.Linear(self.hidden_layers[l], self.hidden_layers[l+1]))
            layers.append(nn.BatchNorm1d(self.hidden_layers[l+1]))
            layers.append(nn.Dropout(self.drop_p, inplace = False))
            layers.append(self.activation(inplace=True))
        
        # output layer
        layers.append(nn.Linear(self.hidden_layers[-1], self.n_outputs))
        
        self.layers = nn.ModuleList(layers)
                          
        print("Feedforward Network initialized")
                  

    def forward(self, x):
        """
        Input tensor of size (batch_size, features)
        """
        for l in range(len(self.layers)):
            x = self.layers[l](x)

        return x
    

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr = self.lr, weight_decay = self.weight_decay)
        #lr_scheduler = MultiStepLR(optimizer, milestones=[300,], gamma=0.1)
        return optimizer #{"optimizer":optimizer, "lr_scheduler":lr_scheduler}