import torch
import torch.nn as nn
import torch_geometric
import numpy as np

from models.models import BaseFloodModel, activation_functions

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, bias=False, batch_norm=True, activation='relu'):
        super().__init__()
        
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias)]
        if batch_norm:
            layers.append(nn.BatchNorm2d(num_features=out_channels))
        layers.append(activation_functions(activation))
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias))
                
        self.cnnblock = nn.Sequential(*layers)

    def forward(self, x):
        return self.cnnblock(x)

class Encoder(nn.Module):
    def __init__(self, channels=[32, 64, 128], kernel_size=3, padding=1, bias=False, batch_norm=True, activation='relu'):
        super().__init__()

        self.enc_blocks = nn.ModuleList([
            CNNBlock(channels[block], channels[block+1], kernel_size, padding, bias, 
                     batch_norm=batch_norm, activation=activation) 
            for block in range(len(channels)-1)]
            )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        outs = []
        for block in self.enc_blocks:
            x = block(x)
            outs.append(x)
            x = self.pool(x)
        return outs

class Decoder(nn.Module):
    def __init__(self, channels=[128, 64, 32], kernel_size=3, padding=1, bias=False, batch_norm=True, activation='relu'):
        super().__init__()
        self.channels = channels
        self.upconvs = nn.ModuleList([
            nn.ConvTranspose2d(channels[block], channels[block+1], kernel_size=2, padding=0, stride=2) 
            for block in range(len(channels)-1)]
            )
        self.dec_blocks = nn.ModuleList([
            CNNBlock(channels[block], channels[block+1], kernel_size, padding, bias, 
                     batch_norm=batch_norm, activation=activation)
             for block in range(len(channels)-1)]
             )
        
    def forward(self, x, x_skips):
        for i in range(len(x_skips)):
            x = self.upconvs[i](x)
            x = torch.cat((x, x_skips[-(1+i)]), dim=1)
            x = self.dec_blocks[i](x)

        x = self.dec_blocks[-1](x)
        return x

class UNet(nn.Module):
    def __init__(self, input_channels, output_channels, hidden_channels=[32,64,128], kernel_size=3, 
                 padding=1, bias=False, batch_norm=True, activation='relu'):
        super().__init__()
        encoder_channels = [input_channels]+hidden_channels
        decoder_channels = list(reversed(hidden_channels))+[output_channels]
        self.encoder = Encoder(encoder_channels, kernel_size=kernel_size, padding=padding, 
                               bias=bias, batch_norm=batch_norm, activation=activation)
        self.decoder = Decoder(decoder_channels, kernel_size=kernel_size, padding=padding, 
                               bias=bias, batch_norm=batch_norm, activation=activation)
                               
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x[-1], x[:-1])
        return x


class CNN(BaseFloodModel):
    def __init__(self, node_features, n_downsamples=3, initial_hid_dim=64, batch_norm=True, 
                 bias=True, activation='prelu', **base_model_kwargs):
        super(CNN, self).__init__(**base_model_kwargs)
        self.type_model = 'CNN'
        self.kernel_size = 3
        self.node_features = node_features
        self.DEM_index = node_features - self.previous_t*self.out_dim
        hidden_channels = [initial_hid_dim*2**i for i in range(n_downsamples)]
                
        self.main = UNet(node_features, self.out_dim, hidden_channels=hidden_channels, 
                         kernel_size=self.kernel_size, padding=1, bias=bias, batch_norm=batch_norm, 
                         activation=activation)
        
    def forward(self, data):

        if isinstance(data, torch_geometric.data.batch.Batch):
            number_grids = int(np.sqrt(data.num_nodes/data.num_graphs))
        else:
            number_grids = int(np.sqrt(data.num_nodes))
        
        x0 = data.x.clone()
        
        if self.with_WL:
            # Add water level
            WL = x0[:,self.DEM_index] + x0[:,-self.out_dim]
            x0 = torch.cat((x0, WL.unsqueeze(-1)), 1)

        x = get_x_for_CNN(data, number_grids=number_grids, node_features=self.node_features)
               
        x = self.main(x)
        
        x = resize_for_GNN(x, node_features=self.out_dim)
        
        # Add residual connections (THE CNN WORKS BEST WITHOUT)
        # x = x + x0[:,-self.out_dim:]

        # Mask very small water depth
        x = self._mask_small_WD(x, epsilon=0.001)
                
        return x


def get_x_for_CNN(graph, number_grids, node_features):
    '''
    Transforms rectangular mesh 'x' with shape [batch * number of nodes, node features]
    into shape [batch, channels, height, width] required by the CNN
    ------
    x: torch.Tensor
        input tensor to be reshaped
    number_grids: int
        number of grids per side of the squared domain
    node_features: int
        number of input node features
    '''
    if isinstance(graph, torch_geometric.data.batch.Batch):
        x1 = graph.x.reshape(graph.num_graphs,number_grids,number_grids,node_features)
    else:
        x1 = graph.x.reshape(1,number_grids,number_grids,node_features)

    x2 = torch.permute(x1, (0, 3, 1, 2))

    return x2


def resize_for_GNN(x, node_features):
    '''
    Transforms rectangular mesh 'x' with shape [batch, channels, height, width]
    into shape [batch * number of nodes, node features] required by the GNN plots
    ------
    x: torch.Tensor
        input tensor to be reshaped
    number_grids: int
        number of grids per side of the squared domain
    node_features: int
        number of input node features
    '''
    x1 = torch.permute(x, (0, 2, 3, 1)).reshape(-1,node_features)

    return x1