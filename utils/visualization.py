## Libraries
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import torch
from torch_geometric.utils import to_networkx
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
import sys

from utils.miscellaneous import add_null_time_start, get_rollouts, plot_line_with_deviation, get_model_name
from utils.miscellaneous import get_time_vector, get_rollout_loss, get_CSI, get_F1
from utils.dataset import get_input_water, get_breach_coordinates
from training.loss import get_mean_error
from utils.scaling import get_none_scalers

WD_color = LinearSegmentedColormap.from_list('', ['white', 'MediumBlue'])
V_color = LinearSegmentedColormap.from_list('', ['white', 'darkviolet'])
diff_color = LinearSegmentedColormap.from_list('', ['red', 'white', 'green'])
diff_color_positive = LinearSegmentedColormap.from_list('', ['white', 'green'])
diff_color_negative = LinearSegmentedColormap.from_list('', ['red', 'white'])
FAT_color = LinearSegmentedColormap.from_list('', ['MediumBlue', 'white'])

def get_coords(pos):
    '''
    Returns array of dimensions (n_nodes, 2) containing x and y coordinates of each node
    ------
    pos: dict
        keys: (x,y) index of every node
        values: spatial x and y positions of each node
    '''
    return np.array([xy for xy in pos.values()])

def get_info_from_pyg(data):
    ''' 
    Returns pos, graph, and mesh given a Data object
    ------
    data: torch_geometric.data.data.Data
        Data object as defined in create_dataset.py
    '''
	
    pos = {i:(x,y) for i, (x,y) in enumerate(data.pos.numpy())}
    mesh = None #this version doesn't work with irregular meshes
    graph = to_networkx(data, to_undirected=True)
    return pos, graph, mesh

def get_corners(pos):
    '''
    Returns the coordinates of the corners of a grid
    ------
    pos: dict
        keys: (x,y) index of every node
        values: spatial x and y positions of each node
    '''    
    BL = min(pos.values()) #bottom-left
    TR = max(pos.values()) #top-right
    BR = (BL[0], TR[1]) #bottom-right
    TL = (TR[0], BL[1]) #top-left
    
    return BL, TR, BR, TL

def plot_loss(train_losses, val_losses=None, scale='log'):
    '''
    Plot losses after training
    ------
    *_losses: list
        training (and validation) losses during training
    name: str
        give a name to save the plot as and image
    scale: str
        options: "linear", "log", "symlog", "logit", ...
    '''    
    plt.plot(train_losses, 'b-')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.yscale(scale)
    
    if val_losses is not None:
        plt.plot(val_losses, 'r-')
        plt.legend(['Training', 'Validation'], loc='upper right')
    plt.title('Loss vs. No. of epochs')
        
    return None

class BasePlotMap(object):
    '''
    Base class for plotting a map defined by either a graph (graph) or a triangular mesh (mesh)

    ------
    map_: np.array or torch.tensor (shape [N], [N, 1] or [N_x, N_y])
        represents a single feature for each point in the domain
    pos: dict
        keys: (x,y) index of every node
        values: spatial x and y positions of each node
    graph: networkx.classes.graph.Graph
        networkx graph with nodes and edges
    mesh: scipy.spatial.qhull.Delaunay
        triangular mesh
    scaler: sklearn.preprocessing._data scaler
        scaler object used for normalizing the data
    colorbar: bool (default=True)
        if True, shows colorbar in plot
    '''
    def __init__(self, map, pos, graph=None, mesh=None, scaler=None, edge_index=None, 
                 difference_plot=False, **kwargs):
        self.map = map
        self.scaler = scaler
        self.pos = pos
        self.graph = graph
        self.mesh = mesh
        self.edge_index = edge_index
        self.kwargs = {**kwargs}
        self.difference_plot = difference_plot
    
        self.map = self._check_device(self.map)
        self._check_map_type()

    def _check_map_dimension(self, map):
        '''map must be of dimension [N] when plotting'''
        if len(map.shape)>1:
            map = map.reshape(-1)
        return map

    def _scale_map(self, map):
        '''Scales back map, given scaler'''
        if self.scaler is not None:
            if len(map.shape)==1:
                map = map.reshape(-1, 1)
            map = self.scaler.inverse_transform(map)
        map = self._check_map_dimension(map)
        return map

    def _check_device(self, map):
        '''Convert map to cpu'''
        if isinstance(map, torch.Tensor):
            if map.device.type != 'cpu':
                map = map.to('cpu')
            map = map.numpy()
        return map

    def _check_map_type(self):
        if self.graph is None and self.mesh is None:
            raise AttributeError("BasePlotMap must receive either a graph 'graph' or a triangular mesh 'mesh'")

    def _get_vmin(self, map):
        if 'vmin' not in self.kwargs:
            self.kwargs['vmin'] = map.min()
            
    def _get_vmax(self, map):
        if 'vmax' not in self.kwargs:
            self.kwargs['vmax'] = map.max()

    def _create_axes(self, ax=None):
        if ax is None:
            ax = plt.gca()
        return ax

    def _get_cmap(self):
        if self.difference_plot:
            if self.kwargs['vmin'] >= 0:
                self.kwargs['vmin'] = 0
                self.kwargs['cmap'] = diff_color_positive
            elif self.kwargs['vmax'] <= 0:
                self.kwargs['vmax'] = 0
                self.kwargs['cmap'] = diff_color_negative
            else:
                self.kwargs['cmap'] = diff_color
        elif 'cmap' not in self.kwargs:
            self.kwargs['cmap'] = plt.cm.plasma

    def _add_colorbar(self, ax=None, colorbar=True):
        if colorbar:
            self.kwargs['vmax'] = self._check_device(self.kwargs['vmax'])
            self.kwargs['vmin'] = self._check_device(self.kwargs['vmin'])
            if self.difference_plot:
                if self.kwargs['vmin'] >= 0:
                    ticks_interval = np.linspace(0, self.kwargs['vmax'], 6, endpoint=True)
                    plt.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(vmin = 0, vmax=self.kwargs['vmax']), 
                                cmap=self.kwargs['cmap']), ticks=np.sign(ticks_interval)*np.floor(np.abs(ticks_interval)*10)/10,
                                fraction=0.05, shrink=0.9, ax=ax)
                elif self.kwargs['vmax'] <= 0:
                    ticks_interval = np.linspace(self.kwargs['vmin'], 0, 6, endpoint=True)
                    plt.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(vmin=self.kwargs['vmin'], vmax=0), 
                                cmap=self.kwargs['cmap']), ticks=np.sign(ticks_interval)*np.floor(np.abs(ticks_interval)*10)/10,
                                fraction=0.05, shrink=0.9, ax=ax)
                else:
                    ticks_interval = np.linspace(self.kwargs['vmin'], self.kwargs['vmax'], 7, endpoint=True)
                    plt.colorbar(plt.cm.ScalarMappable(norm=TwoSlopeNorm(
                                vmin=self.kwargs['vmin'], vcenter=0, vmax=self.kwargs['vmax']), 
                                cmap=self.kwargs['cmap']), ticks=np.sign(ticks_interval)*np.floor(np.abs(ticks_interval)*10)/10,
                                fraction=0.05, shrink=0.9, ax=ax)
            else:
                ticks_interval = np.linspace(self.kwargs['vmin'], self.kwargs['vmax'], 6, endpoint=True)
                plt.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(vmin = self.kwargs['vmin'], vmax=self.kwargs['vmax']), 
                                cmap=self.kwargs['cmap']), ticks=np.sign(ticks_interval)*np.floor(np.abs(ticks_interval)*10)/10,
                                fraction=0.05, shrink=0.9, ax=ax)
    
    def plot_map(self, ax=None, colorbar=True):
        self.map = self._scale_map(self.map)
        self._get_vmin(self.map)
        self._get_vmax(self.map)
        ax = self._create_axes(ax=ax)
        self._get_cmap()
        self._add_colorbar(ax=ax, colorbar=colorbar)
        
        if self.graph is not None:
            '''Plot map as graph'''
            nx.draw_networkx_nodes(self.graph, pos=self.pos, node_color=self.map, node_shape='s', node_size=20, 
                                   ax=ax, **self.kwargs)
        elif self.mesh is not None:
            '''Plot map as mesh'''
            coordinates = get_coords(self.pos)
            X = coordinates[:,0]
            Y = coordinates[:,1]
            ax.tricontourf(X, Y, self.mesh.simplices, self.map, levels=48,**self.kwargs)
            ax.triplot(X, Y, self.mesh.simplices, linewidth=0.1, c='black')
            
        return ax

    def plot_edge_map(self, ax=None, colorbar=True):
        self.map = self._scale_map(self.map)
        self._get_vmin(self.map)
        self._get_vmax(self.map)
        ax = self._create_axes(ax=ax)
        self._get_cmap()
        self._add_colorbar(ax=ax, colorbar=colorbar)
        self.kwargs['edge_vmin'] = self.kwargs.pop('vmin')
        self.kwargs['edge_vmax'] = self.kwargs.pop('vmax')
        self.kwargs['edge_cmap'] = self.kwargs.pop('cmap')
        edge_list = self.edge_index.T.numpy()
        
        if self.graph is None:
            raise NotImplementedError("This function only works with graphs as input")
        else:
            '''Plot edges of a graph'''
            nx.draw_networkx_edges(self.graph, pos=self.pos, edgelist=edge_list, 
                edge_color=self.map, ax=ax, **self.kwargs)
            
        return ax

class TemporalPlotMap(BasePlotMap):
    '''Plot class for maps with temporal attributes

    ------
    map: np.array-like (shape [N, T] or [N, 1])
        temporal matrix of the map to be plotted
    time_step: int
        time step at which to plot the map
    temporal_res: int, [minutes]
        temporal resolution of the temporal dataset
    '''
    def __init__(self, map, temporal_res, time_start=0, **map_kwargs):
        super().__init__(map, **map_kwargs)
        self.temporal_res = temporal_res
        self.time_start = time_start
        self.total_time = self.map.shape[1]

    def _get_map_at_time_step(self, map):
        if self.total_time > 1:
            map = map[:, self.time_step]
        return map

    def _get_current_time_step(self):
        self.time_in_minutes = (self.time_start + 1 + self.time_step%self.total_time)*self.temporal_res
        self.time_in_hours = self.time_in_minutes/60
    
    def plot_map(self, time_step, ax=None, colorbar=True):
        self.time_step = time_step
        self._get_current_time_step()
        map = self._get_map_at_time_step(self.map)
        
        map = self._scale_map(map)
        self._get_vmin(map)
        self._get_vmax(map)
        ax = self._create_axes(ax=ax)
        self._get_cmap()
        self._add_colorbar(ax=ax, colorbar=colorbar)
        
        if self.graph is not None:
            '''Plot map as graph'''
            nx.draw_networkx_nodes(self.graph, pos=self.pos, node_color=map, node_shape='s', node_size=20, 
                                   ax=ax, **self.kwargs)
        elif self.mesh is not None:
            '''Plot map as mesh'''
            coordinates = get_coords(self.pos)
            X = coordinates[:,0]
            Y = coordinates[:,1]
            ax.tricontourf(X, Y, self.mesh.simplices, map, levels=48,**self.kwargs)
            ax.triplot(X, Y, self.mesh.simplices, linewidth=0.1, c='black')

        return ax

class DEMPlotMap(BasePlotMap):
    '''Plot digital elevation model(DEM)'''
    def __init__(self, map, **map_kwargs):
        super().__init__(map, **map_kwargs)
        self.kwargs['cmap'] = 'terrain'
        
    def _add_axes_info(self, ax):
        ax.set_title('DEM (m)')
        ax.set_xlabel('x distance [km]')
        ax.set_ylabel('y distance [km]')
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

        if self.map.shape[0] > 4096:
            ax.set_xticks([0,40,80,120]) 
            ax.set_xticklabels([0,4,8,12])
            ax.set_yticks([0,40,80,120]) 
            ax.set_yticklabels([0,4,8,12])
        else:
            ax.set_xticks([0,20,40,60]) 
            ax.set_xticklabels([0,2,4,6])
            ax.set_yticks([0,20,40,60]) 
            ax.set_yticklabels([0,2,4,6])

    def _add_breach_location(self, ax, breach_coordinates:list=[[0.5,0.5]]):
        for breach in breach_coordinates:
            ax.scatter(breach[0], breach[1], s=200, c='r', marker='x', zorder=3, linewidths=5)

def plot_rollout_diff_in_time_all(diff_rollout, temporal_res, type_loss='RMSE', 
                              time_start=0, ax=None):
    '''Plot average node error distribution across time for a given simulation'''
    ax = ax or plt.gca()

    # WD plot
    lns = plot_rollout_diff_in_time_var(
        diff_rollout, temporal_res, type_loss, dim=0, 
        time_start=time_start, ax=ax, label='h', c='royalblue')

    ax.set_ylabel(f'h {type_loss} [m]')
    ax.set_xlabel('Time [h]')
    ax.set_xlim(0)

    ax2 = ax.twinx()
    V_symbol = "|q|" 
    V_unit = "$m^2$/s"
    
    lin_V = plot_rollout_diff_in_time_var(
        diff_rollout, temporal_res, type_loss, dim=1, 
        time_start=time_start, ax=ax2, label=V_symbol, c='purple')
    lns = lns + lin_V
    ax2.set_ylabel(f'{V_symbol} {type_loss} [{V_unit}]')

    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs)
    
    return ax, ax2

def plot_rollout_diff_in_time_var(diff_rollout, temporal_res, type_loss='RMSE', dim=0, 
                                  time_start=0, ax=None, **plot_kwargs):
    '''Plot average node error distribution for a variable across time for a given simulation
    Variable axis is identified by dim: 0 = WD, 1 = Q
    '''
    diff_rollout = diff_rollout[:,dim,:].to('cpu')

    ax = ax or plt.gca()
    
    time_stop = diff_rollout.shape[-1]
    time_vector = np.linspace(0, (time_start+time_stop)*temporal_res/60, time_stop+time_start+1)
        
    average_diff_t = get_mean_error(diff_rollout, type_loss).numpy()

    average_diff_t = add_null_time_start(time_start, average_diff_t)
    
    return ax.plot(time_vector, average_diff_t, marker='.', **plot_kwargs)

def plot_breach_distribution(dataset, ax=None, with_label=True):
    assert isinstance(dataset, list), "This function works for a list of simulations"

    ax = ax or plt.gca()

    pos, _, _ = get_info_from_pyg(dataset[0])
    corners = get_corners(pos)
    [ax.scatter(x,y,c='w') for (x,y) in corners]

    ax.set_xlabel('x distance [km]')
    ax.set_ylabel('y distance [km]')
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

    if dataset[0].DEM.shape[0] > 4096:
        ax.set_xticks([0,40,80,120]) 
        ax.set_xticklabels([0,4,8,12])
        ax.set_yticks([0,40,80,120]) 
        ax.set_yticklabels([0,4,8,12])
    else:
        ax.set_xticks([0,20,40,60]) 
        ax.set_xticklabels([0,2,4,6])
        ax.set_yticks([0,20,40,60]) 
        ax.set_yticklabels([0,2,4,6])
    
    for i, data in enumerate(dataset):
        breach_coordinates = get_breach_coordinates(data.WD, pos)
        for breach in breach_coordinates:
            ax.scatter(breach[0], breach[1], s=50, c='r', marker='x', zorder=3)
            if with_label:
                ax.annotate(i, (breach[0], breach[1]))
    
    return ax

class PlotRollout():
    '''Explore predictions vs real simulations'''
    def __init__(self, model, dataset, scalers=None, type_loss='RMSE', **temporal_test_dataset_parameters):
        super().__init__()
        self.time_start = temporal_test_dataset_parameters['time_start']
        self.time_stop = temporal_test_dataset_parameters['time_stop']
        self.temporal_res = dataset.temporal_res
        self.V_unit = "$m^2$/s"
        self.V_label = "Discharge"
        self.V_symbol = "q"
        self.model = model
        self.dataset = dataset
        self.type_loss = type_loss
        
        if scalers is None:
            scalers = get_none_scalers()
        self.scalers = scalers

        pos, graph, mesh = get_info_from_pyg(dataset)
        self.default_plot_kwargs = {'pos':pos, 'graph':graph, 'mesh':mesh}
        self.default_temporal_plot_kwargs = self.default_plot_kwargs|\
            {'time_start': self.time_start, 'temporal_res':self.temporal_res}
        
        self.predicted_rollout, self.real_rollout, self.prediction_time = get_rollouts(
            model, dataset, **temporal_test_dataset_parameters)
        self.diff_rollout = self.predicted_rollout - self.real_rollout
        self.input_water = get_input_water(dataset, self.time_start)

        self.breach_coordinates = get_breach_coordinates(self.dataset.WD, pos)

        self._get_maxs(self.real_rollout, self.predicted_rollout, self.diff_rollout)
        
        self.real_WD, self.predicted_WD, self.difference_WD = self._get_WDPlots(
            self.real_rollout, self.predicted_rollout, self.diff_rollout)

        self.real_V, self.predicted_V, self.difference_V = self._get_VPlots(
            self.real_rollout, self.predicted_rollout, self.diff_rollout)

    def _get_maxs(self, real_rollout, predicted_rollout, diff_rollout):
        self.WD_max = max(real_rollout[:,0,:].max(), predicted_rollout[:,0,:].max())

        self.max_diff_WD = diff_rollout[:,0,:].abs().max()
        self.last_max_diff_WD = diff_rollout[:,0,-1].abs().max()
        
        self.V_max = max(abs(predicted_rollout[:,1:,:]).max(), abs(real_rollout[:,1:,:]).max())
        self.last_V_max = max(abs(predicted_rollout[:,1:,-1]).max(), abs(real_rollout[:,1:,-1]).max())

        self.max_diff_V = diff_rollout[:,1:,:].abs().max()
        self.last_max_diff_V = diff_rollout[:,1:,-1].abs().max()
        
    def _plot_temporal_errors(self, diff_rollout, ax):
        axs = plot_rollout_diff_in_time_all(diff_rollout, ax=ax, 
            type_loss=self.type_loss, temporal_res=self.temporal_res, 
            time_start=self.time_start)
        return axs

    def _plot_DEM(self, ax):
        self.DEM = self.dataset.DEM
        DEMPlot = DEMPlotMap(self.DEM, **self.default_plot_kwargs)
        DEMPlot.plot_map(ax=ax)
        DEMPlot._add_axes_info(ax=ax)
        DEMPlot._add_breach_location(ax=ax, breach_coordinates=self.breach_coordinates)

    def _get_WDPlots(self, real_rollout, predicted_rollout, diff_rollout):
        real_WD = TemporalPlotMap(real_rollout[:,0,:], 
            **self.default_temporal_plot_kwargs, scaler=self.scalers['WD_scaler'], 
            cmap=WD_color, vmax=self.WD_max)

        predicted_WD = TemporalPlotMap(predicted_rollout[:,0,:], 
            **self.default_temporal_plot_kwargs, scaler=self.scalers['WD_scaler'], 
            cmap=WD_color, vmin=0, vmax=self.WD_max)

        difference_WD = TemporalPlotMap(diff_rollout[:,0,:],  
            **self.default_temporal_plot_kwargs, scaler=self.scalers['WD_scaler'],
            difference_plot=True, vmin=-self.max_diff_WD, vmax=self.max_diff_WD)
            
        self.init_WD = TemporalPlotMap(self.input_water[:,0,:],
            **self.default_plot_kwargs|{'time_start': -1, 'temporal_res':self.temporal_res}, 
            cmap=WD_color, vmax=self.WD_max)

        return real_WD, predicted_WD, difference_WD
            
    def _get_VPlots(self, real_rollout, predicted_rollout, diff_rollout):
        real_V = TemporalPlotMap(real_rollout[:,1,:], 
            **self.default_temporal_plot_kwargs, scaler=self.scalers['V_scaler'], 
            cmap=V_color, vmax=self.V_max)

        predicted_V = TemporalPlotMap(predicted_rollout[:,1,:], 
            **self.default_temporal_plot_kwargs, scaler=self.scalers['V_scaler'], 
            cmap=V_color, vmin=0, vmax=self.V_max)

        difference_V = TemporalPlotMap(diff_rollout[:,1,:],  
            **self.default_temporal_plot_kwargs, scaler=self.scalers['V_scaler'],
            difference_plot=True, vmin=-self.max_diff_V, vmax=self.max_diff_V)
            
        self.init_V = TemporalPlotMap(self.input_water[:,1,:], 
            **self.default_plot_kwargs|{'time_start': -1, 'temporal_res':self.temporal_res}, 
            cmap=V_color, vmax=self.V_max)

        return real_V, predicted_V, difference_V
            
    def explore_rollout(self):
        fig, axs = plt.subplots(2, 4, figsize=(6*4, 11), facecolor='white', 
                            gridspec_kw={'width_ratios': [1, 1, 1, 1]},
                            constrained_layout = True)

        self._plot_DEM(ax=axs[0,0])

        # water depth
        self.real_WD.plot_map(time_step=-1, ax=axs[0,1], colorbar=False)
        self.predicted_WD.plot_map(time_step=-1, ax=axs[0,2])
        self.difference_WD.plot_map(time_step=-1, ax=axs[0,3])

        axs[0,1].set_ylabel('Water depth [m]')
        axs[0,1].set_title('Ground-truth')
        axs[0,2].set_title('Predicted')
        axs[0,3].set_title('Difference')

        self._plot_temporal_errors(self.diff_rollout, ax=axs[1,0])

        # velocities
        axs[1,1].set_ylabel(f'{self.V_label} [{self.V_unit}]')

        self.real_V.plot_map(time_step=-1, ax=axs[1,1], colorbar=False)
        self.predicted_V.plot_map(time_step=-1, ax=axs[1,2])
        self.difference_V.plot_map(time_step=-1, ax=axs[1,3])
                            
        return fig
    
    def compare_h_rollout(self, plot_times=[1,6,24,40]):
        self.plot_times = plot_times + [-1] #add final time step
        
        n_plots = len(self.plot_times)+1
        fig, axs = plt.subplots(3, n_plots, figsize=(n_plots*4, 12), facecolor='white')

        colorbar = False
        self.init_WD.plot_map(time_step=-1, ax=axs[0,0], colorbar=colorbar)
        for axs_index, time_step in enumerate(self.plot_times, start=1):
            if time_step == -1:
                colorbar = True
            self.real_WD.plot_map(time_step=time_step, ax=axs[0,axs_index], colorbar=colorbar)
            self.predicted_WD.plot_map(time_step=time_step, ax=axs[1,axs_index], colorbar=colorbar)
            self.difference_WD.plot_map(time_step=time_step, ax=axs[2,axs_index], colorbar=colorbar)
            axs[0,axs_index].set_title(f'time: {self.real_WD.time_in_hours:.1f} h')
        
        for ax in axs[1:,0]:
            ax.axis('off')

        axs[0,0].set_title(f'time: {self.init_WD.time_in_hours:.1f} h')
        axs[0,0].set_ylabel(f'Ground-truth [m]')
        axs[1,1].set_ylabel(f'Predictions [m]')
        axs[2,1].set_ylabel(f'Difference [m]')    

        fig.subplots_adjust(wspace=0, hspace=0)
            
        return None
    
    def compare_v_rollout(self, plot_times=[1,6,24,40]):
        self.plot_times = plot_times + [-1] #add final time step

        n_plots = len(self.plot_times)+1
        fig, axs = plt.subplots(3, n_plots, figsize=(n_plots*4, 12), facecolor='white')

        colorbar = False
        self.init_V.plot_map(time_step=-1, ax=axs[0,0], colorbar=colorbar)
        for axs_index, time_step in enumerate(self.plot_times, start=1):
            if time_step == -1:
                colorbar = True
            self.real_V.plot_map(time_step=time_step, ax=axs[0,axs_index], colorbar=colorbar)
            self.predicted_V.plot_map(time_step=time_step, ax=axs[1,axs_index], colorbar=colorbar)
            self.difference_V.plot_map(time_step=time_step, ax=axs[2,axs_index], colorbar=colorbar)
            axs[0,axs_index].set_title(f'time: {self.real_V.time_in_hours:.1f} h')
        
        for ax in axs[1:,0]:
            ax.axis('off')
            
        axs[0,0].set_title(f'time: {self.init_V.time_in_hours:.1f} h')
        axs[0,0].set_ylabel(f'Ground-truth [{self.V_unit}]')
        axs[1,1].set_ylabel(f'Predictions [{self.V_unit}]')
        axs[2,1].set_ylabel(f'Difference [{self.V_unit}]')

        fig.subplots_adjust(wspace=0, hspace=0)
            
        return None

    def create_video(self, interval=200, blit=False, **anim_kwargs):
        '''
        This function seems to work only on Jupyter webpages (not on Visual Studio Code)
        For more information on how to roll please refer to http://news.rr.nihalnavath.com/posts/rollout-d628137f
        ------
        save: str
            name to give to save the simulation
        '''
        from IPython.display import clear_output
        from matplotlib.animation import FuncAnimation

        fig, axs = plt.subplots(2, 4, figsize=(6*4, 11), facecolor='white', 
                            gridspec_kw={'width_ratios': [1, 1, 1, 1]},
                            constrained_layout = True)

        self._plot_DEM(ax=axs[0,0])

        axs[1,0].set_ylabel('MAE')
        axs[1,0].set_xlabel('Time [h]')
        average_diff_t = get_mean_error(self.diff_rollout, self.type_loss).numpy()
        max_avg_WD = average_diff_t[0].max()
        max_avg_V = average_diff_t[1].max()

        self.add_initial_colorbars(axs)

        def animate(time_step): 
            '''
            Function used to create video of the simulation
            '''
            for axx in axs:
                for ax in axx[1:]:
                    ax.cla()

            axs[1,0].cla()

            ax, axv = self._plot_temporal_errors(self.diff_rollout[:,:,:time_step], ax=axs[1,0])

            ax.set_xlim(0, (self.real_WD.total_time+self.time_start)*self.temporal_res/60)
            ax.set_ylim(0, max_avg_WD*1.1)
            axv.set_ylim(0, max_avg_V*1.1)
            axv.ticklabel_format(style='sci', scilimits=(-1,3), useMathText=True)
            ax.ticklabel_format(style='sci', scilimits=(-1,3), useMathText=True)

            # water depth
            self.real_WD.plot_map(time_step=time_step, ax=axs[0,1], colorbar=False)
            self.predicted_WD.plot_map(time_step=time_step, ax=axs[0,2], colorbar=False)
            self.difference_WD.plot_map(time_step=time_step, ax=axs[0,3], colorbar=False)

            current_time = self.real_WD.time_in_hours
            axs[0,1].set_title(f'Ground-truth h [m]\ntime {current_time} h')
            axs[0,2].set_title(f'Predicted h [m]\ntime {current_time} h')
            axs[0,3].set_title(f'Difference h [m]\ntime {current_time} h')

            # velocities
            self.real_V.plot_map(time_step=time_step, ax=axs[1,1], colorbar=False)
            self.predicted_V.plot_map(time_step=time_step, ax=axs[1,2], colorbar=False)
            self.difference_V.plot_map(time_step=time_step, ax=axs[1,3], colorbar=False)
            axs[1,1].set_title(f'Ground-truth |{self.V_symbol}| [{self.V_unit}]\ntime {current_time} h')
            axs[1,2].set_title(f'Predicted |{self.V_symbol}| [{self.V_unit}]\ntime {current_time} h')
            axs[1,3].set_title(f'Difference |{self.V_symbol}| [{self.V_unit}]\ntime {current_time} h')
                    
            fig.subplots_adjust(wspace=0.4, hspace=0.3)

            clear_output(wait=True)
            print ('It: %i'%time_step)
            sys.stdout.flush()
            return (fig)
        
        frames = self.real_WD.total_time
        self.anim = FuncAnimation(fig, animate, frames=frames, interval=interval, blit=blit, **anim_kwargs)
        plt.close()

    def add_initial_colorbars(self, axs):
        self.predicted_WD.kwargs['vmin'] = 0
        self.predicted_WD.kwargs['vmax'] = self.WD_max
        self.predicted_WD._get_cmap()
        self.predicted_WD._add_colorbar(ax=axs[0,2], colorbar=True)
        
        self.difference_WD._get_cmap()
        self.difference_WD._add_colorbar(ax=axs[0,3], colorbar=True)

        self.predicted_V.kwargs['vmin'] = 0
        self.predicted_V.kwargs['vmax'] = self.V_max
        self.predicted_V._get_cmap()
        self.predicted_V._add_colorbar(ax=axs[1,2], colorbar=True)
        
        self.difference_V._get_cmap()
        self.difference_V._add_colorbar(ax=axs[1,3], colorbar=True)

    def save_video(self, path, fps=5, dpi=250, **save_kwargs):
        self.anim.save(f'{path}.mp4', writer='ffmpeg', fps=fps, dpi=dpi, 
        metadata={'title':'test_dataset', 'artist':'Roberto Bentivoglio'}, **save_kwargs)

    def HTML_plot(self):
        from IPython.display import HTML
        HTML(self.anim.to_html5_video())

    def _get_CSI(self, water_threshold=0):
        return get_CSI(self.predicted_rollout, self.real_rollout, water_threshold=water_threshold)
        
    def _get_F1(self, water_threshold=0):
        return get_F1(self.predicted_rollout, self.real_rollout, water_threshold=water_threshold)

    def _plot_metric(self, metric_name='CSI', water_thresholds=[0.05, 0.3], ax=None):
        '''Plots metric in time for different water_thresholds
        -------
        metric_function: 
            options: CSI, F1
        '''
        metrics_dict = {'CSI': self._get_CSI,
                        'F1': self._get_F1}
        metric_function = metrics_dict[metric_name]

        ax = ax or plt.gca()
        time_vector = get_time_vector(self.dataset, time_stop=self.time_stop)

        for wt in water_thresholds:
            metric = metric_function(water_threshold=wt).to('cpu').numpy()
            metric = add_null_time_start(self.time_start, metric)
            plot_line_with_deviation(time_vector, metric, label=f'{metric_name}_{wt}')
            
        ax.set_xlabel('Time [h]')
        ax.set_ylabel(f'{metric_name} score')
        ax.set_ylim(0,1)
        ax.grid()
        ax.legend(loc=4)
        
        return ax

    def _get_rollout_loss(self, type_loss='RMSE', only_where_water=False):
        return get_rollout_loss(self.predicted_rollout, self.real_rollout, type_loss=type_loss, 
                                only_where_water=only_where_water)

class DoublePlotRollout(PlotRollout):
    '''Use this function to compare 2 models side by side'''
    
    def __init__(self, model1, model2, dataset, **temporal_test_dataset_parameters):
        super().__init__(model=model1, dataset=dataset, scalers=None, 
                         type_loss='RMSE', **temporal_test_dataset_parameters)
        self.model1 = model1
        self.model2 = model2

        self.model1_name = get_model_name(model1)
        self.model2_name = get_model_name(model2)
        
        self.predicted_rollout1, self.real_rollout, self.prediction_time1 = get_rollouts(
            model1, dataset, **temporal_test_dataset_parameters)
        self.diff_rollout1 = self.predicted_rollout1 - self.real_rollout
        
        self.predicted_rollout2, _, self.prediction_time2 = get_rollouts(
            model2, dataset, **temporal_test_dataset_parameters)
        self.diff_rollout2 = self.predicted_rollout2 - self.real_rollout

        self._get_maxs_12(self.real_rollout, self.predicted_rollout1, self.diff_rollout1,
                       self.predicted_rollout2, self.diff_rollout2)
        
        self.real_WD, self.predicted_WD1, self.difference_WD1 = self._get_WDPlots(
            self.real_rollout, self.predicted_rollout1, self.diff_rollout1)
        _, self.predicted_WD2, self.difference_WD2 = self._get_WDPlots(
            self.real_rollout, self.predicted_rollout2, self.diff_rollout2)

        self.real_V, self.predicted_V1, self.difference_V1 = self._get_VPlots(
            self.real_rollout, self.predicted_rollout1, self.diff_rollout1)
        _, self.predicted_V2, self.difference_V2 = self._get_VPlots(
            self.real_rollout, self.predicted_rollout2, self.diff_rollout2)

    def _get_maxs_12(self, real_rollout, predicted_rollout1, diff_rollout1,
                        predicted_rollout2, diff_rollout2):
        self.WD_max = max(real_rollout[:,0,:].max(), predicted_rollout1[:,0,:].max(), predicted_rollout2[:,0,:].max())

        self.max_diff_WD = max(abs(diff_rollout1[:,0,:]).max(), abs(diff_rollout2[:,0,:]).max())
        self.last_max_diff_WD = max(abs(diff_rollout1[:,0,-1]).max(), abs(diff_rollout2[:,0,-1]).max())
                
        self.V_max = max(abs(predicted_rollout1[:,1:,:]).max(), abs(predicted_rollout2[:,1:,:]).max(), abs(real_rollout[:,1:,:]).max())
        self.last_V_max = max(abs(predicted_rollout1[:,1:,-1]).max(), abs(predicted_rollout2[:,1:,-1]).max(), abs(real_rollout[:,1:,-1]).max())

        self.max_diff_V = max(abs(diff_rollout1[:,1:,:]).max(), abs(diff_rollout2[:,1:,:]).max())
        self.last_max_diff_V = max(abs(diff_rollout1[:,1:,-1]).max(), abs(diff_rollout2[:,1:,-1]).max())

    def explore_rollout(self):
        fig, axs = plt.subplots(2, 6, figsize=(6*6, 12), facecolor='white', 
                            gridspec_kw={'width_ratios': [1, 1, 1, 1, 1, 1]},
                            constrained_layout = True)

        self._plot_DEM(ax=axs[0,0])

        # water depth
        self.real_WD.plot_map(time_step=-1, ax=axs[0,1], colorbar=False)
        self.predicted_WD1.plot_map(time_step=-1, ax=axs[0,2], colorbar=False)
        self.predicted_WD2.plot_map(time_step=-1, ax=axs[0,3])
        self.difference_WD1.plot_map(time_step=-1, ax=axs[0,4], colorbar=False)
        self.difference_WD2.plot_map(time_step=-1, ax=axs[0,5])

        axs[0,1].set_ylabel('Water depth [m]')
        axs[0,1].set_title('Ground-truth')
        axs[0,2].set_title(f'{self.model1_name}\nPrediction')
        axs[0,3].set_title(f'{self.model2_name}\nPrediction')
        axs[0,4].set_title(f'{self.model1_name}\nDifference')
        axs[0,5].set_title(f'{self.model2_name}\nDifference')

        # velocities
        axs[1,1].set_ylabel(f'{self.V_label} [{self.V_unit}]')

        self.real_V.plot_map(time_step=-1, ax=axs[1,1], colorbar=False)
        self.predicted_V1.plot_map(time_step=-1, ax=axs[1,2], colorbar=False)
        self.predicted_V2.plot_map(time_step=-1, ax=axs[1,3])
        self.difference_V1.plot_map(time_step=-1, ax=axs[1,4], colorbar=False)
        self.difference_V2.plot_map(time_step=-1, ax=axs[1,5])

        axs[1,0].axis('off')

    
    def compare_h_rollout(self, plot_times=[1,6,24,40]):
        self.plot_times = plot_times + [-1] #add final time step
        
        n_plots = len(self.plot_times)+1
        fig, axs = plt.subplots(5, n_plots, figsize=(n_plots*4, 20), facecolor='white')

        colorbar = False
        self.init_WD.plot_map(time_step=-1, ax=axs[0,0], colorbar=colorbar)
        for axs_index, time_step in enumerate(self.plot_times, start=1):
            if time_step == -1:
                colorbar = True
            self.real_WD.plot_map(time_step=time_step, ax=axs[0,axs_index], colorbar=colorbar)
            self.predicted_WD1.plot_map(time_step=time_step, ax=axs[1,axs_index], colorbar=colorbar)
            self.difference_WD1.plot_map(time_step=time_step, ax=axs[2,axs_index], colorbar=colorbar)
            self.predicted_WD2.plot_map(time_step=time_step, ax=axs[3,axs_index], colorbar=colorbar)
            self.difference_WD2.plot_map(time_step=time_step, ax=axs[4,axs_index], colorbar=colorbar)
            axs[0,axs_index].set_title(f'time: {self.real_WD.time_in_hours:.1f} h')
        
        for ax in axs[1:,0]:
            ax.axis('off')

        axs[0,0].set_title(f'time: {self.init_WD.time_in_hours:.1f} h')
        axs[0,0].set_ylabel(f'Ground-truth [m]')
        axs[1,1].set_ylabel(f'{self.model1_name}\nPredictions [m]')
        axs[2,1].set_ylabel(f'{self.model1_name}\nDifference [m]') 
        axs[3,1].set_ylabel(f'{self.model2_name}\nPredictions [m]')
        axs[4,1].set_ylabel(f'{self.model2_name}\nDifference [m]')

        fig.subplots_adjust(wspace=0, hspace=0)
            
        return None
    
    def compare_v_rollout(self, plot_times=[1,6,24,40]):
        self.plot_times = plot_times + [-1] #add final time step

        n_plots = len(self.plot_times)+1
        fig, axs = plt.subplots(5, n_plots, figsize=(n_plots*4, 20), facecolor='white')

        colorbar = False
        self.init_V.plot_map(time_step=-1, ax=axs[0,0], colorbar=colorbar)

        for axs_index, time_step in enumerate(self.plot_times, start=1):
            if time_step == -1:
                colorbar = True
            self.real_V.plot_map(time_step=time_step, ax=axs[0,axs_index], colorbar=colorbar)
            self.predicted_V1.plot_map(time_step=time_step, ax=axs[1,axs_index], colorbar=colorbar)
            self.difference_V1.plot_map(time_step=time_step, ax=axs[2,axs_index], colorbar=colorbar)
            self.predicted_V2.plot_map(time_step=time_step, ax=axs[3,axs_index], colorbar=colorbar)
            self.difference_V2.plot_map(time_step=time_step, ax=axs[4,axs_index], colorbar=colorbar)
            axs[0,axs_index].set_title(f'time: {self.real_V.time_in_hours:.1f} h')
        
        for ax in axs[1:,0]:
            ax.axis('off')
            
        axs[0,0].set_title(f'time: {self.init_V.time_in_hours:.1f} h')
        axs[0,0].set_ylabel(f'Ground-truth [{self.V_unit}]')
        axs[1,1].set_ylabel(f'{self.model1_name}\nPredictions [{self.V_unit}]')
        axs[2,1].set_ylabel(f'{self.model1_name}\nDifference [{self.V_unit}]')
        axs[3,1].set_ylabel(f'{self.model2_name}\nPredictions [{self.V_unit}]')
        axs[4,1].set_ylabel(f'{self.model2_name}\nDifference [{self.V_unit}]')

        fig.subplots_adjust(wspace=0, hspace=0)
            
        return None