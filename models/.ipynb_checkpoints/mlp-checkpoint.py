# Libraries
import torch_geometric
from models.models import BaseFloodModel, make_mlp
import torch

class MLP(BaseFloodModel):
    def __init__(self, num_nodes, node_features, hid_features, n_layers=2, dropout=0, layer_norm=True, 
                 activation='relu', **base_model_kwargs):
        super(MLP, self).__init__(**base_model_kwargs)
        self.type_model = 'MLP'
        self.hid_features = hid_features
        self.dropout = dropout
        self.num_nodes = num_nodes
        self.input_size = num_nodes*(node_features + self.with_WL)
        self.output_size = num_nodes*self.out_dim
        self.DEM_index = node_features - self.previous_t*self.out_dim

        self.main = make_mlp(
            self.input_size, self.output_size, hidden_size=hid_features, n_layers=n_layers, 
            dropout=dropout, layer_norm=layer_norm, activation=activation, device=self.device)
        
    def forward(self, data):

        x0 = data.x.clone()

        if self.with_WL:
            # Add water level
            WL = x0[:,self.DEM_index] + x0[:,-self.out_dim]
            x0 = torch.cat((x0, WL.unsqueeze(-1)), 1)
        
        if isinstance(data, torch_geometric.data.batch.Batch):
            x = x0.reshape(data.num_graphs,-1)
        else:
            x = x0.reshape(-1)
        x = self.main(x)
        x = x.reshape(-1, self.out_dim)
        
        # Add residual connections
        x = x + x0[:,-self.out_dim:]
        
        # Mask very small water depth
        x = self._mask_small_WD(x, epsilon=0.001)
                
        return x