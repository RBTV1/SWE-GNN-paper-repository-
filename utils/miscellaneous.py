import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

from training.test import get_rollouts
from training.loss import mask_on_water, get_mean_error
from models.gnn import GNN
from models.mlp import MLP
from models.cnn import CNN

def get_model(model_name):
    models_dict = {'GNN': GNN,
                   'CNN': CNN,
                   'MLP': MLP}
    return models_dict[model_name]

def get_time_vector(datasets, time_stop):
    '''Returns array with correct temporal stamps'''
    if isinstance(datasets, list):
        temporal_res = datasets[0].temporal_res
        if time_stop == -1:
            time_stop = datasets[0].WD.shape[-1]
    else:
        temporal_res = datasets.temporal_res
        if time_stop == -1:
            time_stop = datasets.WD.shape[-1]
    time_vector = np.linspace(0, (time_stop)*temporal_res/60, time_stop)
    return time_vector

def add_null_time_start(time_start, temporal_array):
    if temporal_array.ndim == 1: # [T]
        new_temporal_array = np.concatenate((np.nan*np.empty(time_start+1), temporal_array))
    elif temporal_array.ndim == 2: # [N_datasets, T]
        new_temporal_array = np.concatenate((np.nan*np.empty((
            temporal_array.shape[0], time_start+1)), temporal_array), axis=1)
    else:
        raise ValueError("Wrong temporal array dimensions")

    return new_temporal_array

def get_model_name(model):
    if model.type_model == 'GNN':
        return model.type_GNN
    else:
        return model.type_model

def get_numerical_times(test_size, temporal_res, maximum_time,
                        overview_file='database/raw_datasets/overview.csv',
                        **temporal_test_dataset_parameters):

    time_start = temporal_test_dataset_parameters['time_start']
    time_stop = temporal_test_dataset_parameters['time_stop']

    final_time = time_stop%maximum_time + (time_stop==-1)
    
    assert final_time != -1, "I'm not sure how to interpret final_time value of -1"

    numerical_simulation_overview = pd.read_csv(overview_file, sep=',')

    small_test_id = numerical_simulation_overview['seed'].isin(np.arange(500,520))
    random_breach_test_id = numerical_simulation_overview['seed'].isin(np.arange(10001,10021))
    big_random_breach_test_id = numerical_simulation_overview['seed'].isin(np.arange(15001,15011))

    if test_size == 'random_breach':
        ids = random_breach_test_id
        test_size = 20
    elif test_size == 'big_random_breach':
        ids = big_random_breach_test_id
        test_size = 10
    else:
        ids = small_test_id

    computation_time = numerical_simulation_overview.loc[ids]['computation_time[s]']

    simulated_times = numerical_simulation_overview.loc[ids]['simulation_time[h]']
    model_simulated_times = (final_time-time_start)*temporal_res/60

    time_ratio = model_simulated_times/simulated_times

    return (computation_time*time_ratio).iloc[:test_size]

def calculate_speed_ups(numerical_times, model_times):
    '''Calculate speed up as ratio between simulation time and DL model time'''
    speed_up = numerical_times/model_times

    return speed_up.mean(), speed_up.std()

def get_binary_rollouts(predicted_rollout, real_rollout, water_threshold=0):
    '''Converts flood simulation into a binary map (1=flood, 0=no flood) for classification purposes
    ------
    water_threshold: float
        Threshold for the binary map creation, i.e., 'flood' if WD>threshold
    '''
    if predicted_rollout.dim() == 4:
        predicted_rollout_flood = predicted_rollout[:,:,0,:]>water_threshold
        real_roll_flood = real_rollout[:,:,0,:]>water_threshold
    elif predicted_rollout.dim() == 3:
        predicted_rollout_flood = predicted_rollout[:,0,:]>water_threshold
        real_roll_flood = real_rollout[:,0,:]>water_threshold

    return predicted_rollout_flood, real_roll_flood

def get_rollout_confusion_matrix(predicted_rollout, real_rollout, water_threshold=0):
    predicted_rollout_flood, real_roll_flood = get_binary_rollouts(predicted_rollout, real_rollout, water_threshold=water_threshold)

    if predicted_rollout.dim() == 4:
        nodes_dim = 1
    elif predicted_rollout.dim() == 3:
        nodes_dim = 0

    TP = (predicted_rollout_flood & real_roll_flood).sum(nodes_dim) #true positive
    TN = (~predicted_rollout_flood & ~real_roll_flood).sum(nodes_dim) #true negative
    FP = (predicted_rollout_flood & ~real_roll_flood).sum(nodes_dim) #false positive
    FN = (~predicted_rollout_flood & real_roll_flood).sum(nodes_dim) #false negative

    return TP, TN, FP, FN

def get_CSI(predicted_rollout, real_rollout, water_threshold=0):
    '''Returns the Critical Success Index (CSI) in time for a given water_threshold'''
    TP, TN, FP, FN = get_rollout_confusion_matrix(predicted_rollout, real_rollout, water_threshold=water_threshold)

    CSI = TP / (TP + FN + FP)

    return CSI

def get_F1(predicted_rollout, real_rollout, water_threshold=0):
    '''Returns the Critical Success Index (CSI) in time for a given water_threshold'''
    TP, TN, FP, FN = get_rollout_confusion_matrix(predicted_rollout, real_rollout, water_threshold=water_threshold)

    F1 = TP / (TP + 0.5*(FN + FP))

    return F1

def get_masked_diff(diff_roll, where_water):
    masked_diff = torch.stack([diff_roll[:,water_variable,:][where_water] 
                                    for water_variable in range(diff_roll.shape[1])])
                                
    return masked_diff

def get_rollout_loss(predicted_rollout, real_rollout, type_loss='RMSE', only_where_water=False):
    diff_roll = predicted_rollout - real_rollout

    if diff_roll.dim() == 4: #multiple simulations
        nodes_dim = 1
        water_axis = 2
    elif diff_roll.dim() == 3: #single simulation
        nodes_dim = 0
        water_axis = 1

    if only_where_water:
        where_water = mask_on_water(predicted_rollout, real_rollout, water_axis=water_axis)
        
        if diff_roll.dim() == 4:
            roll_loss = torch.stack([get_mean_error(
                get_masked_diff(diff_roll[id_dataset], where_water[id_dataset]), 
                type_loss, nodes_dim=-1) for id_dataset in range(diff_roll.shape[0])])
        elif diff_roll.dim() == 3:
            roll_loss = get_mean_error(get_masked_diff(diff_roll, where_water), type_loss, nodes_dim=-1)
    else:
        roll_loss = get_mean_error(diff_roll, type_loss, nodes_dim=nodes_dim).mean(-1)
    
    return roll_loss

def plot_line_with_deviation(time_vector, variable, ax=None, **plt_kwargs):
    ax = ax or plt.gca()

    df = pd.DataFrame(np.vstack((time_vector, variable))).T
    df = df.rename(columns={0: "time"})
    df = df.set_index('time')

    mean = df.mean(1)
    std = df.std(1)
    under_line = (mean - std)
    over_line = (mean + std)

    p = ax.plot(mean, linewidth=2, marker='o', **plt_kwargs)
    color = p[0].get_color()
    ax.fill_between(std.index, under_line, over_line, color=color, alpha=.3)
    return p

def get_pareto_front(df, objective_function1, objective_function2, ascending=False):
    sorted_df = df.sort_values(by=[objective_function1, objective_function2], ascending=ascending)[[objective_function1, objective_function2]]

    pareto_front = sorted_df.values[0].reshape(1,-1)
    for var1, var2 in sorted_df.values[1:]:
        if var2 >= pareto_front[-1,1]:
            pareto_front = np.concatenate((pareto_front, np.array([[var1, var2]])), axis=0)
    
    return pareto_front

class SpatialAnalysis():
    def __init__(self, model, dataset, **temporal_test_dataset_parameters):
        '''This class saves model predictions and real values and provides tools to analyse them'''
        self.dataset = dataset
        self.time_start = temporal_test_dataset_parameters['time_start']
        self.time_stop = temporal_test_dataset_parameters['time_stop']
        self.time_vector = get_time_vector(dataset, self.time_stop)
        self.DEMs = self._get_DEMS(dataset)
        self.predicted_rollout, self.real_rollout, self.prediction_times = self._get_rollouts(
            model, dataset, **temporal_test_dataset_parameters)
        self.diff_rollout = self.predicted_rollout - self.real_rollout

    def _get_rollouts(self, model, dataset, **temporal_test_dataset_parameters):
        if isinstance(dataset, list):
            all_rollouts = [get_rollouts(model, data, **temporal_test_dataset_parameters) for data in dataset]
            predicted_rollout = torch.stack([roll[0] for roll in all_rollouts])
            real_rollout = torch.stack([roll[1] for roll in all_rollouts])
            prediction_times = np.array([roll[2] for roll in all_rollouts])
        else:
            predicted_rollout, real_rollout, prediction_times = get_rollouts(model, dataset, **temporal_test_dataset_parameters)
            predicted_rollout = predicted_rollout.unsqueeze(0)
            real_rollout = real_rollout.unsqueeze(0)
            prediction_times = prediction_times

        return predicted_rollout, real_rollout, prediction_times

    def _get_DEMS(self, dataset):
        if isinstance(dataset, list):
            DEMs = torch.stack([data.DEM for data in dataset])
        else:
            DEMs = dataset.DEM
        return DEMs

    def _plot_metric_rollouts(self, metric_name, metric_function, water_thresholds=[0.05, 0.3]):
        '''Plots metric in time for different water_thresholds
        -------
        metric_function: 
            options: get_CSI, get_F1
        '''
        fig, ax = plt.subplots(figsize=(7,5))

        all_metric = []
        for wt in water_thresholds:
            metric = metric_function(self.predicted_rollout, self.real_rollout, water_threshold=wt).to('cpu').numpy()
            all_metric.append(metric)
            metric = add_null_time_start(self.time_start, metric)
            plot_line_with_deviation(self.time_vector, metric, ax=ax, label=f'{metric_name}_{wt}')
            # plt.legend()
            
        ax.set_xlabel('Time [h]')
        ax.set_ylabel(f'{metric_name} score')
        ax.set_ylim(0,1)
        ax.grid()
        ax.legend(loc=4)
        
        return fig, np.array(all_metric)
    
    def _plot_rollouts(self, type_loss):
        '''Plots loss in time for the different water variables'''
        fig, ax = plt.subplots(figsize=(7,5))

        water_labels = ['h [m]', '|q| [m^2/s]']
        var_colors = ['royalblue', 'purple']
        lines = []

        ax2 = ax.twinx()
        axx = ax

        for var in range(self.diff_rollout.shape[2]):
            average_diff_t = get_mean_error(self.diff_rollout, type_loss, nodes_dim=1)[:,var,:].to('cpu').numpy()
            average_diff_t = add_null_time_start(self.time_start, average_diff_t)
            lines.append(plot_line_with_deviation(self.time_vector, average_diff_t, ax=axx,
                                        label=water_labels[var], c=var_colors[var])[0])
            axx = ax2
        
        axx = ax
        ax.set_xlabel('Time [h]')
        ax.set_title(type_loss)

        labs = [l.get_label() for l in lines]
        ax.legend(lines, labs, loc=1)
        
        return fig

    def _get_CSI(self, water_threshold=0):
        return get_CSI(self.predicted_rollout, self.real_rollout, water_threshold=water_threshold)
        
    def _get_F1(self, water_threshold=0):
        return get_F1(self.predicted_rollout, self.real_rollout, water_threshold=water_threshold)

    def plot_CSI_rollouts(self, water_thresholds=[0.05, 0.3]):
        return self._plot_metric_rollouts('CSI', get_CSI, water_thresholds=water_thresholds)

    def plot_F1_rollouts(self, water_thresholds=[0.05, 0.3]):
        return self._plot_metric_rollouts('F1', get_F1, water_thresholds=water_thresholds)

    def _get_rollout_loss(self, type_loss='RMSE', only_where_water=False):
        return get_rollout_loss(self.predicted_rollout, self.real_rollout, type_loss=type_loss, 
                                only_where_water=only_where_water)

    def plot_loss_per_simulation(self, type_loss='RMSE', water_thresholds=[0.05, 0.3], 
                                 ranking='loss', only_where_water=False, figsize=(20,12)):
        '''Plot sorted loss for each simulation in a dataset
        ranking: criterion to sort simulations
            options: 'loss', 'CSI'
        '''
        rollout_loss = self._get_rollout_loss(type_loss=type_loss, only_where_water=only_where_water)
        CSIs = torch.stack([self._get_CSI(wt) for wt in water_thresholds], 1).mean(2).to('cpu')

        assert rollout_loss.dim() == 2, "rollout_loss should have dimension [S, O]"\
            "where S is the number of simulations and O is the output dimension"
        if rollout_loss.shape[0] == 1:
            raise ValueError("This plot works only for multiple simulations")
        if isinstance(rollout_loss, torch.Tensor):
            rollout_loss = rollout_loss.to('cpu').numpy()

        fig, axs = plt.subplots(3, 1, figsize=figsize, sharex='col')

        if ranking == 'loss':
            sorted_ids = rollout_loss.mean(1).argsort()
        elif ranking == 'CSI':
            sorted_ids = CSIs.mean(1).argsort().flip(-1).numpy()
        else:
            raise ValueError("ranking can only be either 'loss' or 'CSI'")

        positions=range(len(sorted_ids))
        
        water_labels = ['h [m]', '|q| [m^2/s]']
        var_colors = ['royalblue', 'purple']
            
        axs[0].set_title(f'{ranking} ranking for test simulations')
        n_x_ticks = range(len(sorted_ids))
        axs[0].boxplot(self.DEMs[sorted_ids], positions=positions)
        axs[0].set_ylabel(r'DEM [m]')

        for i, (color, label) in enumerate(zip(var_colors, water_labels)):
            axs[1].plot(rollout_loss[sorted_ids, i], 'o--', label=label, c=color)
        axs[1].set_ylabel(type_loss)
        axs[1].set_yscale('log')
        axs[1].legend()
        
        axs[2].set_xticks(n_x_ticks)
        axs[2].set_xticklabels(sorted_ids)
        axs[2].plot(CSIs[sorted_ids], 'o--', label=[f'CSI_{str(wt)}' for wt in water_thresholds])
        axs[2].set_ylim(0,1)
        axs[2].set_xlabel('Simulation id')
        axs[2].set_ylabel('CSI')
        axs[2].legend()

        fig.subplots_adjust(wspace=0, hspace=0.05)

        return sorted_ids

    def plot_summary(self, numerical_times, type_loss='RMSE', water_thresholds=[0.05, 0.3], 
                     only_where_water=False, figsize=(10,5)):

        fig, axs = plt.subplots(1, 3, figsize=figsize)

        RMSE = self._get_rollout_loss(type_loss=type_loss, only_where_water=only_where_water).to('cpu')
        CSIs = [self._get_CSI(wt).mean(1).to('cpu') for wt in water_thresholds]

        axs[0].boxplot(CSIs)
        axs[0].set_ylim(0,1)
        axs[0].set_xticklabels([r'$\tau$'f'={wt}m' for wt in water_thresholds])
        axs[0].set_title(r'CSI_$\tau$ [-]')

        axs[1].boxplot((RMSE[:,0], RMSE[:,1:].mean(1)))
        axs[1].set_xticklabels((r'$h [m]$', r'$|q| [m^2/s]$'))
        axs[1].set_title(f'{type_loss}')
        axs[1].set_yscale('log')

        axs[2].boxplot((self.prediction_times, numerical_times))
        axs[2].set_title('Execution times [sec]')
        axs[2].set_xticklabels(('DL', 'Numerical'))
        axs[2].set_ylim(0)

        plt.tight_layout()

        return fig