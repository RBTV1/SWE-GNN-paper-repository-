# Libraries
import torch
import torch.nn as nn
from torch.nn import Sequential as Seq, Linear as Lin
from torch.nn import ReLU, PReLU, ELU, GELU, Sigmoid, Dropout, Tanh, LeakyReLU

class BaseFloodModel(nn.Module):
    '''Base class for modelling flood inundation
    ------
    previous_t: int
        dataset-specific parameter that indicates the number of previous times steps given as input
    seed: int
        seed used for replicability
    '''
    def __init__(self, previous_t=1, seed=42, with_WL=True, device='cpu'):
        super().__init__()
        torch.manual_seed(seed)
        self.previous_t = previous_t
        self.with_WL = with_WL
        self.device = device
        self.out_dim = 2
            
    def _mask_small_WD(self, x, epsilon=0.001):        
        x[:,0][x[:,0].abs() < epsilon] = 0

        # Mask velocities where there is no water
        x[:,1:][x[:,0] == 0] = 0

        return x

def add_norm_dropout_activation(hidden_size, layer_norm=False, dropout=0, activation='relu', 
                                device='cpu'):
    '''Add LayerNorm, Dropout, and activation function'''
    layers = []
    if layer_norm:
        layers.append(nn.LayerNorm(hidden_size, eps=1e-5, device=device))
    if dropout:
        layers.append(Dropout(dropout))
    if activation is not None:
        layers.append(activation_functions(activation, device=device))
    return layers


def init_weights(layer):
    if isinstance(layer, Lin):
        torch.nn.init.xavier_normal_(layer.weight)
        if layer.bias is not None:
            torch.nn.init.normal_(layer.bias)

def make_mlp(input_size, output_size, hidden_size=32, n_layers=2, bias=False, 
             activation='relu', dropout=0, layer_norm=False, device='cpu'):
    """Builds an MLP"""
    layers = []
    if n_layers==1:
        layers.append(Lin(input_size, output_size, bias=bias, device=device))
        layers = layers + add_norm_dropout_activation(
            output_size, layer_norm=layer_norm, dropout=dropout, activation=activation)
    else:
        layers.append(Lin(input_size, hidden_size, bias=bias, device=device))
        layers = layers + add_norm_dropout_activation(
            hidden_size, layer_norm=layer_norm, dropout=dropout, activation=activation)
            
        for layer in range(n_layers-2):
            layers.append(Lin(hidden_size, hidden_size, bias=bias, device=device))
            layers = layers + add_norm_dropout_activation(
                hidden_size, layer_norm=layer_norm, dropout=dropout, activation=activation)

        layers.append(Lin(hidden_size, output_size, bias=bias, device=device))
        layers = layers + add_norm_dropout_activation(
            output_size, layer_norm=layer_norm, dropout=dropout, activation=activation)

    mlp = Seq(*layers)
    # mlp.apply(init_weights)

    return mlp


def activation_functions(activation_name, device='cpu'):
    '''Returns an activation function given its name'''
    if activation_name == 'relu':
        return ReLU()
    elif activation_name == 'prelu':
        return PReLU(device=device)
    elif activation_name == 'leakyrelu':
        return LeakyReLU(0.1)
    elif activation_name == 'elu':
        return ELU()
    elif activation_name == 'gelu':
        return GELU()
    elif activation_name == 'sigmoid':
        return Sigmoid()
    elif activation_name == 'tanh':
        return Tanh()
    elif activation_name is None:
        return None
    else:
        raise AttributeError('Please choose one of the following options:\n'\
            '"relu", "prelu", "leakyrelu", "elu", "gelu", "sigmoid", "tanh"')