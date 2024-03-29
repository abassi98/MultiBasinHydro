import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim
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
        self.pool_division = 4
      
        ### Network architecture
        # First convolutional layer (2d convolutional layer
        self.first_conv = nn.Sequential(
            nn.Conv1d(self.in_channels[0], self.out_channels[0], self.kernel_sizes[0], padding=self.padding[0]), 
            nn.BatchNorm1d(self.out_channels[0]),
            self.act(inplace = True),
            nn.Dropout(self.drop_p, inplace = False),
            nn.AvgPool1d(self.pool_division)
        )
        
        # Second convolution layer
        self.second_conv = nn.Sequential(
            nn.Conv1d(self.in_channels[1], self.out_channels[1], self.kernel_sizes[1], padding=self.padding[1]), 
            nn.BatchNorm1d(self.out_channels[1]),
            self.act(inplace = True),
            nn.Dropout(self.drop_p, inplace = False),
            nn.AvgPool1d(self.pool_division)
        )
        
        # Third convolutional layer
        self.third_conv = nn.Sequential(
            nn.Conv1d(self.in_channels[2], self.out_channels[2], self.kernel_sizes[2], padding=self.padding[2]), 
            nn.BatchNorm1d(self.out_channels[2]),
            self.act(inplace = True),
            nn.Dropout(self.drop_p, inplace = False),
            nn.AvgPool1d(self.pool_division)
        )


        # Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        
        # Liner dimension after 2 convolutional layers
        self.lin_dim = int((((self.seq_len-self.kernel_sizes[0]+1)/self.pool_division+1-self.kernel_sizes[1])/self.pool_division+1-self.kernel_sizes[2])/self.pool_division)
        
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
                 warmup = 45, # 2 years
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
        input_lstm = torch.cat((enc_expanded, y.squeeze(1)),dim=-1) # squeeze channel dimension for input to lstm
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
        x, y, _, _ = batch
        # # select past period
        # x_past = x[:,:,:self.seq_len,:]
        # y_past = y[:,:,:self.seq_len,:]
        # # select future (validation) period
        # x_fut = x[:,:,1+self.seq_len:,:]
        # y_fut = y[:,:,1+self.seq_len:,:]
        # forward pass
        _, rec = self.forward(x,y)
        # Logging to TensorBoard by default
        train_loss = self.loss_fn(x.squeeze()[:,self.warmup:], rec.squeeze()[:,self.warmup:])
        self.log("train_loss", train_loss, prog_bar=True)
        # # compute future (training) loss and log
        # with torch.no_grad():
        #     _, rec_fut = self.forward(x_fut,y_fut)
        # train_fut_loss = self.loss_fn(x_fut.squeeze()[:,self.warmup:], rec_fut.squeeze()[:,self.warmup:])
        # self.log("train_fut_loss", train_fut_loss)

        #print(self.lr_scheduler.get_last_lr())
        return train_loss
    
    def validation_step(self, batch, batch_idx):
        ### Unpack batch
        x, y, _, _ = batch
        # # select past period
        # x_past = x[:,:,:self.seq_len,:]
        # y_past = y[:,:,:self.seq_len,:]
        # # select future (validation) period
        # x_fut = x[:,:,1+self.seq_len:,:]
        # y_fut = y[:,:,1+self.seq_len:,:]
        
 
        # forward pass
        _, rec = self.forward(x,y)
        # Logging to TensorBoard by default
        val_loss = self.loss_fn(x.squeeze()[:,self.warmup:], rec.squeeze()[:,self.warmup:])
        # Logging to TensorBoard by default
        self.log("val_loss", val_loss, prog_bar=True)
        self.log("epoch_num", float(self.current_epoch),prog_bar=True)
        
        # # compute past (validation) loss
        # with torch.no_grad():
        #     _, rec_past = self.forward(x_past,y_past)
        # val_past_loss = self.loss_fn(x_past.squeeze()[:,self.warmup:], rec_past.squeeze()[:,self.warmup:])
        # self.log("val_past_loss", val_past_loss)
        
        return val_loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr = self.lr, weight_decay = self.weight_decay)
        #lr_scheduler = MultiStepLR(optimizer, milestones=[300,], gamma=0.1)
        return optimizer #{"optimizer":optimizer, "lr_scheduler":lr_scheduler}



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
                 hydro = False,
                 warmup = 45,
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
        self.hydro = hydro
        self.warmup = warmup

        ### LSTM decoder
        input_size = num_force_attributes + noise_dim
        if self.statics:
            input_size += 27
        if self.hydro:
            input_size += 13
            
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

        
    def forward(self, y, statics, hydro): 
        input_lstm = y.squeeze()
    
        # Append statics/hydro
        if self.statics:
            input_lstm = torch.cat((input_lstm, statics.squeeze(1).repeat(1,self.seq_len,1)),dim=-1)
        if self.hydro:
            input_lstm = torch.cat((input_lstm, hydro.squeeze(1).repeat(1,self.seq_len,1)),dim=-1)
        
        # append noise
        batch_size = input_lstm.shape[0] # size (batch_size, seq_len, force_attributes)
        noise = self.sigmoid(torch.randn(size=(batch_size, self.seq_len, self.noise_dim), device=self.device))
        input_lstm = torch.cat((input_lstm, noise),dim=-1)
       
        #print("input_lstm shape: ", input_lstm.shape)
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
        x, y, statics, hydro = batch
        # # select past period
        # x_past = x[:,:,:self.seq_len,:]
        # y_past = y[:,:,:self.seq_len,:]
      
        # # select future (validation) period
        # x_fut = x[:,:,1+self.seq_len:,:]
        # y_fut = y[:,:,1+self.seq_len:,:]
    
        # forward pass
        rec = self.forward(y, statics, hydro)
        # Logging to TensorBoard by default
        train_loss = self.loss_fn(x.squeeze()[:,self.warmup:], rec.squeeze()[:,self.warmup:])
        self.log("train_loss", train_loss, prog_bar=True)

        # # compute future (training) loss and log
        # with torch.no_grad():
        #     rec_fut = self.forward(y_fut, statics)
        # train_fut_loss = self.loss_fn(x_fut.squeeze()[:,self.warmup:], rec_fut.squeeze()[:,self.warmup:])
        # self.log("train_fut_loss", train_fut_loss)

        return train_loss
    
    def validation_step(self, batch, batch_idx):
        ### Unpack batch
        x, y, statics, hydro = batch
        # # select past period
        # x_past = x[:,:,:self.seq_len,:]
        # y_past = y[:,:,:self.seq_len,:]
        
        # # select future (validation) period
        # x_fut = x[:,:,1+self.seq_len:,:]
        # y_fut = y[:,:,1+self.seq_len:,:]
        
        # forward pass
        rec = self.forward(y, statics, hydro)
        # Logging to TensorBoard by default
        val_loss = self.loss_fn(x.squeeze()[:,self.warmup:], rec.squeeze()[:,self.warmup:])
        # Logging to TensorBoard by default
        self.log("val_loss", val_loss, prog_bar=True)
        self.log("epoch_num", float(self.current_epoch),prog_bar=True)
        
        #  # compute past (validation) loss
        # with torch.no_grad():
        #     rec_past = self.forward(y_past,statics)
        # val_past_loss = self.loss_fn(x_past.squeeze()[:,self.warmup:], rec_past.squeeze()[:,self.warmup:])
        # self.log("val_past_loss", val_past_loss)
        
        return val_loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr = self.lr, weight_decay = self.weight_decay)
        #lr_scheduler = MultiStepLR(optimizer, milestones=[300,], gamma=0.1)
        return optimizer #{"optimizer":optimizer, "lr_scheduler":lr_scheduler}


