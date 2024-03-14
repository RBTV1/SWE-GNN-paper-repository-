# Libraries
import torch
import torch.nn as nn
from torch.nn import Sequential as Seq
from models.models import BaseFloodModel, make_mlp, activation_functions
from torch_geometric.nn import ChebConv, TAGConv, GATConv
from torch import Tensor
from torch_scatter import scatter
from torch.linalg import vector_norm

class GNN(BaseFloodModel):
    '''
    GNN encoder-processor-decoder
    ------
    node_features: int
        number of features per node
    edge_features: int
        number of features per edge
    hid_features: int (default=32)
        number of features per node (and edge) in the GNN layers
    K: int (default=2)
        K-hop neighbourhood
    n_GNN_layers: int (default=2)
        number of GNN layers
    dropout: float (default=0)
        add dropout layer in decoder
    type_GNN: str (default='SWEGNN')
        specifies the type of GNN model
        options: 
            "GNN_A" : Adjacency as graph shift operator 
            "GNN_L" : Laplacian as graph shift operator
            "GAT"   : Graph Attention, i.e., learned shift operator
            "SWEGNN": learned graph shift operator
    edge_mlp: bool (default=True)
        adds MLP as edge encoder (valid only for 'SWEGNN')
    '''
    def __init__(self, node_features, edge_features, hid_features=32, K=2, n_GNN_layers=2, type_GNN="SWEGNN", 
                 mlp_layers=1, mlp_activation='prelu', gnn_activation='prelu', dropout=0, 
                 with_WL=False, normalize=True, with_filter_matrix=False, edge_mlp=True,
                 with_gradient=False, **base_model_kwargs):
        super(GNN, self).__init__(**base_model_kwargs)
        self.type_model = "GNN"
        self.hid_features = hid_features
        self.node_features = node_features
        self.edge_features = edge_features
        self.type_GNN = type_GNN
        self.edge_mlp = edge_mlp
        self.with_WL = with_WL
        self.gnn_activation = gnn_activation
        self.dynamic_node_features = self.previous_t*self.out_dim
        self.static_node_features = node_features - self.dynamic_node_features + self.with_WL
        
        # Edge encoder
        if type_GNN == "SWEGNN" and edge_mlp:
            self.edge_features = hid_features
            self.edge_encoder = make_mlp(edge_features, hid_features, hid_features, n_layers=2, bias=True,
                                         activation=mlp_activation, device=self.device)
        
        # Node encoder
        if type_GNN == "SWEGNN":
            self.dynamic_node_encoder = make_mlp(self.dynamic_node_features, hid_features, hid_features, n_layers=2,
                                        activation=mlp_activation, device=self.device)
    
            self.static_node_encoder = make_mlp(
                self.static_node_features, hid_features, hid_features, n_layers=2, bias=True,
                activation=mlp_activation, device=self.device)
        else:
            self.node_encoder = make_mlp(node_features + self.with_WL, hid_features, hid_features, n_layers=2, bias=True,
                                        activation=mlp_activation, device=self.device)
        
        # GNN
        self.gnn_processor = self._make_gnn(hid_features, K_hops=K, n_GNN_layers=n_GNN_layers, n_layers=mlp_layers, 
                                            activation=mlp_activation, bias=True, type_GNN=type_GNN, 
                                            normalize=normalize, with_filter_matrix=with_filter_matrix,
                                            with_gradient=with_gradient)

        self.gnn_activations = Seq(*[activation_functions(gnn_activation, 
                                device=self.device)]*n_GNN_layers)
        
        # Decoder
        self.node_decoder = make_mlp(hid_features, self.out_dim, hid_features, n_layers=2, dropout=dropout,
                                     activation=mlp_activation, device=self.device)

    def _make_gnn(self, hidden_size, K_hops=1, n_GNN_layers=1, type_GNN='SWEGNN', **swegnn_kwargs):
        """Builds GNN module"""
        convs = nn.ModuleList()
        for l in range(n_GNN_layers):
            if type_GNN == "GNN_L":
                convs.append(ChebConv(hidden_size, hidden_size, K=K_hops))
            elif type_GNN == "GNN_A":
                convs.append(TAGConv(hidden_size, hidden_size, K=K_hops))
            elif type_GNN == "GAT":
                convs.append(GATConv(hidden_size, hidden_size, heads=1))
            elif type_GNN == "SWEGNN":
                convs.append(SWEGNN(hidden_size, hidden_size, self.edge_features, K=K_hops, 
                            device=self.device, **swegnn_kwargs))
            else:
                raise("Only 'GNN_A', 'GNN_L', 'GAT', and 'SWEGNN' are valid for now")
        return convs
    
    def _forward_block(self, x, edge_index, edge_attr):
        """Build encoder-decoder block"""
        # 1. Node and edge encoder
        if self.type_GNN == "SWEGNN" and self.edge_mlp:
            edge_attr = self.edge_encoder(edge_attr)
        
        x0 = x
        x_s = x[:,:self.static_node_features-self.with_WL]
        x_t = x[:,self.static_node_features-self.with_WL:]

        if self.with_WL:
            # Add water level as static input
            WL = x_s[:,-1] + x_t[:,-self.out_dim]
            x_s = torch.cat((x_s, WL.unsqueeze(-1)), 1)
        
        if self.type_GNN == "SWEGNN":
            x_s = self.static_node_encoder(x_s)
            x = x_t = self.dynamic_node_encoder(x_t)
        else:
            x = self.node_encoder(torch.cat((x_s, x_t), 1))

        # 2. Processor 
        for i, conv in enumerate(self.gnn_processor):
            if self.type_GNN == "SWEGNN":
                x = conv(x_s, x_t, edge_index, edge_attr)
            else:
                x = conv(x=x, edge_index=edge_index)

            # Add non-linearity
            if self.gnn_activation is not None:
                x = self.gnn_activations[i](x)

            x_t = x

        # 3. Decoder
        x = self.node_decoder(x)
                    
        # Add residual connections
        x = x + x0[:,-self.out_dim:]
        
        # Mask very small water depth
        x = self._mask_small_WD(x, epsilon=0.001)

        return x

    def forward(self, graph):
    
        x = graph.x.clone()
        edge_index = graph.edge_index.clone()
        edge_attr = graph.edge_attr.clone()
        
        x = self._forward_block(x, edge_index, edge_attr)
        
        return x

class SWEGNN(nn.Module):
    r"""Shallow Water Equations inspired Graph Neural Network

    .. math::
        \mathbf{x}^{\prime}_ti = \mathbf{x}_ti + \sum_{j \in \mathcal{N}(i)} 
        \mathbf{w}_{ij} \cdot (\mathbf{x}_tj - \mathbf{x}_ti)

        \mathbf{w}_{ij} = MLP \left(\mathbf{x}_si, \mathbf{x}_sj,
        \mathbf{x}_ti, \mathbf{x}_tj,
        \mathbf{e}_{ij}\right)
    """
    def __init__(self, static_node_features: int, dynamic_node_features: int, edge_features: int, 
                 K: int = 2, normalize=True, with_filter_matrix=True, with_gradient=True, device='cpu', **mlp_kwargs):
        super().__init__()
        self.edge_features = edge_features
        self.edge_input_size = edge_features + static_node_features*2 + dynamic_node_features*2
        self.edge_output_size = dynamic_node_features
        hidden_size = self.edge_output_size*2
        self.normalize = normalize
        self.K = K
        self.with_filter_matrix = with_filter_matrix
        self.device = device
        self.with_gradient = with_gradient
        
        self.edge_mlp = make_mlp(self.edge_input_size, self.edge_output_size,
                                hidden_size=hidden_size, device=device, **mlp_kwargs)

        if with_filter_matrix:
            self.filter_matrix = torch.nn.ModuleList([
                nn.Linear(dynamic_node_features, dynamic_node_features, bias=False) for _ in range(K+1)
            ])


    def forward(self, x_s: Tensor, x_t: Tensor, edge_index: Tensor, 
                edge_attr: Tensor) -> Tensor:
        '''x_s: static node features,
        x_t: dynamic node features'''
        row = edge_index[0]
        col = edge_index[1]
        num_nodes = x_t.size(0)
        if self.with_filter_matrix:
            out = self.filter_matrix[0].forward(x_t.clone())
        else:
            out = x_t.clone()
        
        for k in range(self.K):
            # Filter out zero values
            mask = out.sum(1) != 0
            mask_row = mask[row]
            mask_col = mask[col]
            edge_index_mask = mask_row + mask_col

            # Edge update
            e_ij = torch.cat([x_s[row][edge_index_mask], 
                            x_s[col][edge_index_mask], 
                            out[row][edge_index_mask], 
                            out[col][edge_index_mask], 
                            edge_attr[edge_index_mask]], 1)
            w_ij = self.edge_mlp(e_ij)
            
            if self.normalize:
                w_ij = w_ij/vector_norm(w_ij, dim=1, keepdim=True)
                w_ij.masked_fill_(torch.isnan(w_ij), 0)

            # Node update
            if self.with_gradient:
                shift_sum = (out[col][edge_index_mask]-out[row][edge_index_mask])*w_ij
            else:
                shift_sum = w_ij

            scattered = scatter(shift_sum, col[edge_index_mask], reduce='sum', 
                          dim=0, dim_size=num_nodes)

            if self.with_filter_matrix:
                out = out + self.filter_matrix[k+1].forward(scattered)
            else:
                out = out + scattered
        
        return out

    def __repr__(self):
        return '{}(node_features={}, edge_features={}, K={}, with_filter_matrix={},\
                with_gradient={})'.format(
            self.__class__.__name__, self.edge_output_size, 
            self.edge_features, self.K, self.with_filter_matrix,
            self.with_gradient)