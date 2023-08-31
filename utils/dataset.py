# Libraries
import torch
from torch_geometric.data import Data

from utils.load import load_dataset
from utils.scaling import get_scalers
from sklearn.model_selection import train_test_split

def process_attr(attribute, scaler=None, to_min=False, device='cpu'):
    '''
    processes an attribute for dataset creation (shift to min, shape, scale)
    ------
    scaler: 
        if present, used to scale the attribute
    to_min: bool
        if True, shifts minimum to zero before scaling
    '''
    assert isinstance(attribute, torch.Tensor), "Input attribute is not a tensor"
    
    if attribute.dim() == 1:
        attribute = attribute.reshape(-1,1)

    attr = attribute.clone()

    if to_min:
        attr = attr - attr.min()

    if scaler is not None:
        attr = torch.cat([torch.FloatTensor(scaler.transform(attr[:,col:col+1])) for col in range(attr.shape[1])], dim=1)

    assert attribute.shape == attr.shape, "Shape has changed during processing: \n"\
        f"Before it was {attribute.shape}, now it is {attr.shape}"
        
    return attr.to(device)

def slopes_from_DEM(DEM):
    '''
    Calculate slope in x and y directions, given a DEM
    '''
    slope_x, slope_y = torch.gradient(DEM)
    return slope_x.reshape(-1), slope_y.reshape(-1)

def get_temporal_res(matrix, temporal_res=30):
    '''
    extracts a sub-matrix with time_step [min] from a temporal matrix [N, T]
    ------
    matrix: torch.tensor
        input temporal matrix with temporal resolution in the second column
    time_step: int
        selects the desired time step for the temporal resolution
    '''
    selected_times = torch.arange(0, matrix.shape[-1], temporal_res/30, dtype=int)
        
    return matrix[:, selected_times]

def get_node_features(data=None, scalers=None, slope_x=True, slope_y=True, 
                      area=True, DEM=True, device='cpu'):
    '''Return the static node features

    data: torch_geometric.data.data.Data
        dataset sample, containing numerical simulation
    scalers: None, dict
        define how to scale DEM and slopes
        (default=None)
    device: str
        device used to store dataset (default='cpu')

    selected_node_features: dict (of bool)
        slopex, slopey: topographic slopes in x and y directions (default=True)
        area: mesh area (default=True)
        DEM: digital elevation model of the topography (default=False)
    '''
    node_features = {}
    
    if scalers is None:
        scalers = {
            'DEM_scaler' : None, 
            'slope_scaler' : None
        }

    if slope_x or slope_y:
        number_grids = int(data.DEM.shape[0]**0.5)
        slopex, slopey = slopes_from_DEM(data.DEM.reshape(number_grids, number_grids))

        if slope_x:
            node_features['slope_x'] = process_attr(slopex, scaler=scalers['slope_scaler'], device=device)
        
        if slope_y:
            node_features['slope_y'] = process_attr(slopey, scaler=scalers['slope_scaler'], device=device)
        
        slopex, slopey = 0, 0

    if DEM:
        node_features['DEM'] = process_attr(data.DEM, scaler=scalers['DEM_scaler'], to_min=True, device=device)
    
    if area:
        try:
            node_features['area'] = 1/process_attr(data.area, device=device)
        except:
            node_features['area'] = torch.ones(data.num_nodes, 1).to(device)
    
    selected_node_features = locals()

    selected_nodes = [node_features[key] for key, value in selected_node_features.items() if value==True]
    
    if len(selected_nodes) == 0:
        node_features = torch.ones(data.num_nodes, 1).to(device)
    else:
        node_features = torch.cat(selected_nodes, 1).to(device)
    
    return node_features

def get_edge_features(data=None, scalers=None, cell_length=True, normal_x=True, normal_y=True, 
                      device='cpu'):
    '''Return the static edge features

    data: torch_geometric.data.data.Data
        dataset sample, containing numerical simulation
    device: str
        device used to store dataset (default='cpu')

    selected_edge_features: dict (of bool)
        cell_length: length of the cell side $l_{ij}$ (default=True)
        normal_x, normal_y: x and y components of the cell outward unit normal vector $n_{ij}$ (default=True)
        delta_DEM: difference in DEM between neighbouring nodes (default=False)
    '''
    edge_features = {
        'cell_length' : data.edge_distance, # valid only for grids but for now it's fine because it's the same thing
        'normal_x' : data.edge_relative_distance[:,0]/data.edge_distance,
        'normal_y' : data.edge_relative_distance[:,1]/data.edge_distance,
    }
    
    selected_edge_features = locals()

    selected_edges = [edge_features[key].reshape(-1,1).to(device) for key, value in selected_edge_features.items() if value==True]
    
    if len(selected_edges) == 0:
        edge_features = torch.ones(data.num_edges, 1).to(device)
    else:
        edge_features = torch.cat(selected_edges, 1).float().to(device)
    
    return edge_features

def get_output_features(data=None, temporal_res=60, scalers=None, device='cpu'):
    '''Return the static edge features

    data: torch_geometric.data.data.Data
        dataset sample, containing numerical simulation
    temporal_res: int [min]
        temporal resolution for the dataset (default=60)
    scalers: None, dict
        define how to scale water depth and velocities
        (default=None)
    device: str
        device used to store dataset (default='cpu')

    output_features has shape [num_nodes, num_output_variables, time_steps]
    '''
    if scalers is None:
        scalers = {
            'WD_scaler' : None, 
            'V_scaler' : None
        }

    WD = process_attr(data.WD, scaler=scalers['WD_scaler'], device=device)
    temporal_WD = get_temporal_res(WD, temporal_res=temporal_res)

    VX = process_attr(data.VX, scaler=scalers['V_scaler'], device=device)*WD
    VY = process_attr(data.VY, scaler=scalers['V_scaler'], device=device)*WD

    V = torch.sqrt(VX**2 + VY**2)
    temporal_V = get_temporal_res(V, temporal_res=temporal_res)
    output_features = torch.stack([temporal_WD, temporal_V], 1)
    
    return output_features

def create_data_attr(datasets, scalers=None, temporal_res=60, device='cpu', **selected_features):
    '''
    Creates x, y, and edge_attr from Data object
    ------
    datasets : list
        each element in the list is a torch_geometric.data.data.Data object
    scalers: dict
        sklearn.preprocessing._data scaler used for normalizing the data
    temporal_res: int [min]
        selects the desired time step for the temporal resolution
    selected_features:
        selected_node_features: dict (of bool)
            dictionary that specifies node features
        selected_edge_features: dict (of bool)
            dictionary that specifies edge features
    '''
    new_dataset = []
    selected_node_features = {
        key:selected_features[key] for key in 
        ['slope_x', 'slope_y', 'area', 'DEM']}

    selected_edge_features = {
        key:selected_features[key] for key in 
        ['cell_length', 'normal_x', 'normal_y']}

    for data in datasets:
        temp = Data()

        temp.edge_index = data.edge_index.to(device)
        temp.edge_attr = get_edge_features(data, scalers=scalers, **selected_edge_features, device=device)
        temp.x = get_node_features(data, **selected_node_features, scalers=scalers, device=device)
        temp.y = get_output_features(data, temporal_res=temporal_res, scalers=scalers, device=device)
        
        temp.DEM = data.DEM
        temp.WD = get_temporal_res(data.WD, temporal_res=temporal_res)
        VX = get_temporal_res(data.VX, temporal_res=temporal_res)*temp.WD
        VY = get_temporal_res(data.VY, temporal_res=temporal_res)*temp.WD
        temp.V = torch.sqrt(VX**2 + VY**2)
        temp.pos = data.pos
        temp.temporal_res = temporal_res
        try:
            temp.area = 1/process_attr(data.area, device=device)
        except:
            temp.area = torch.ones(data.num_nodes).to(device)*100*100

        new_dataset.append(temp)
    
    return new_dataset


def create_model_dataset(dataset_name, train_size=100, val_prcnt=0.3, test_size=50, 
                         dataset_folder='database\\datasets',
                         scalers=None, seed=42, device='cpu', **dataset_parameters):
    '''
    Create dataset with scaled node and edge attributes
    Return training, validation, and testing datasets
    ------
    dataset_name: str
        name of the dataset to be loaded
        options: 'grid' or 'mesh', or ['grid', 'mesh'] for both at the same time
    train_size: int
        number of samples to be considered for training (default=100)
    test_size: int
        number of samples to be considered for testing (default=50)
    val_prcnt: float
        percentage of the training dataset used for validation (default=0.3)
    seed: int
        fixed randomness for replicability in dataset splits and shuffling
    dataset_parameters:
        selected_node_features: dict (of bool)
            dictionary that specifies node features
        selected_edge_features: dict (of bool)
            dictionary that specifies edge features
        temporal_res: int [min]
            selects the desired time step for the temporal resolution
    '''
    # Load datasets
    train_dataset = load_dataset(dataset_name, train_size, seed, dataset_folder+'/train')
    # create validation dataset from training
    train_dataset, val_dataset = train_test_split(train_dataset, test_size=val_prcnt, random_state=seed)
    if test_size == 'big' or test_size == 'big_random_breach':
        test_dataset = load_dataset('big_random_breach_grid', 10, seed=0, dataset_folder=dataset_folder+'/test')
    elif test_size == 'random_breach':
        test_dataset = load_dataset('random_breach_grid', 20, seed=0, dataset_folder=dataset_folder+'/test')
    else:
        test_dataset = load_dataset(dataset_name, test_size, seed=0, dataset_folder=dataset_folder+'/test')
    
    # Normalization
    scalers = get_scalers(train_dataset, scalers)
    
    # Create x, edge_attr, y
    train_dataset = create_data_attr(train_dataset, scalers=scalers, device=device, **dataset_parameters)
    val_dataset = create_data_attr(val_dataset, scalers=scalers, device=device, **dataset_parameters)
    test_dataset = create_data_attr(test_dataset, scalers=scalers, device=device, **dataset_parameters)
    
    return train_dataset, val_dataset, test_dataset, scalers

def aggregate_WD_V(WD, V, init_time):
    '''Create a tensor that concatenates water depth and velocity (module)
    
    WD and V are taken for a interval [init_time:init_time+1]
    
    Output shape: [num_nodes, 2]
    '''
    return torch.cat((WD[:,init_time:init_time+1], 
                      V[:,init_time:init_time+1]), 1)

def get_previous_steps(aggregate_function, init_time, previous_t, *water_variables_args):
    '''Return tensor with next rollout time steps'''
    prev_steps = torch.cat([aggregate_function(*water_variables_args, step) for step in range(init_time,init_time+previous_t)], -1)
    return prev_steps

def get_next_steps(aggregate_function, init_time, rollout_steps, *water_variables_args):
    '''Return tensor with next rollout time steps'''
    next_steps = torch.stack([aggregate_function(*water_variables_args, step) for step in range(init_time,init_time+rollout_steps)], -1)
    assert next_steps.shape[2] == rollout_steps, f"The output dimension is wrong: {next_steps.shape}"
    return next_steps

def add_dry_bed_condition(variable, previous_t):
    '''Add blank previous time steps (dry bed conditions)'''
    num_nodes = variable.shape[0]
    return torch.cat((torch.zeros(num_nodes, previous_t-1), variable), 1)

def get_node_feature_matrix(x, prev_steps, with_x=True):
    '''Return tensor used as node feature matrix X

    x: static node features (e.g., slopes) 
        dim [num_nodes, num_static_features]
    prev_steps: dynamic node features (input previous time steps) 
        dim [num_nodes, num_previous_steps*output_dimension]
    with_x: bool, include static node features
    '''
    num_nodes = x.shape[0]
    if with_x:
        node_feature_matrix = torch.cat((x, prev_steps), 1)
    else:
        node_feature_matrix = prev_steps

    assert node_feature_matrix.shape[0] == num_nodes, "number of nodes changed"
        
    return node_feature_matrix

def get_temporal_samples_size(maximum_time, time_start=0, time_stop=-1, rollout_steps=1):
    '''Returns the number of samples generated when creating the temporal dataset

    maximum_time: int
        temporal size of the given dataset (e.g., 48*1h)
    time_start: int (default=0)
        initial time step given as input
    time_stop: int (default=-1)
        final time step of the simulation
        if -1, takes all the simulation
    rollout_steps: int (default=1)
        number of times the output is predicted
    '''
    if maximum_time <= 0:
        raise ValueError('The temporal size of the dataset is zero')

    if time_stop == maximum_time:
        time_stop = maximum_time
    elif time_stop > maximum_time:
        raise ValueError('time_stop cannot be higher than the temporal size of the dataset')
    else:
        time_stop = time_stop%maximum_time-time_start+1 #add 1 because rollout_steps starts from 1

    if time_start > time_stop:
        raise ValueError('time_start cannot be higher than the last selected time')

    if rollout_steps > time_stop:
        raise ValueError('Number of rollout_steps is too high')
        
    temporal_sample_size = time_stop-rollout_steps%time_stop

    if temporal_sample_size < 0:
        raise ValueError('Something went wrong here')

    return temporal_sample_size

def to_temporal(data, previous_t=2, time_start=0, time_stop=-1, 
                rollout_steps=1, with_x=True):
    '''Converts Data object with temporal signal on graph into multiple graphs
    
    previous_t: int (default=2)
        number of previous time steps given as input
    time_start: int (default=0)
        initial time step given as input
    time_stop: int (default=-1)
        final time step of the simulation
        if -1, takes all the simulation
    rollout_steps: int (default=1)
        number of times the output is predicted
    with_x: bool (default=False)
        if True, add static node attributes obtained in create_model_dataset
    '''
    temporal_data = []
    device = data.x.device
    maximum_time = data.WD.shape[1]
    temporal_samples_size = get_temporal_samples_size(maximum_time, 
        time_start=time_start, time_stop=time_stop, rollout_steps=rollout_steps)
    rollout_steps = rollout_steps%(time_stop%maximum_time-time_start+1)

    WD = add_dry_bed_condition(data.WD, previous_t)
    V = add_dry_bed_condition(data.V, previous_t)

    for init_time in range(time_start, time_start+temporal_samples_size):
        temp = Data()
        
        temp.edge_index = data.edge_index
        temp.edge_attr = data.edge_attr
        temp.pos = data.pos
        temp.area = data.area
        temp.temporal_res = data.temporal_res
        
        prev_steps = get_previous_steps(aggregate_WD_V, init_time, previous_t, WD, V)
        next_steps = get_next_steps(aggregate_WD_V, init_time+previous_t, rollout_steps, WD, V)

        output_dim = 2
        assert prev_steps.shape[1] == output_dim*previous_t, f"The output dimension is wrong: {prev_steps.shape}"
        assert next_steps.shape[1] == output_dim, f"The output dimension is wrong: {next_steps.shape}"
        if (prev_steps[:,-output_dim:] != 0).all(): # Except when everything is zero, then no problem
            assert ~torch.isclose(prev_steps[:,-output_dim:], next_steps[:,:,0]).all(), "You're copying last time step and output"
        
        temp.x = get_node_feature_matrix(data.x, prev_steps.to(device), with_x=with_x)
        temp.y = next_steps.to(device)
        
        temp.time = init_time
        temp.previous_t = previous_t
        temp.with_x = with_x

        temporal_data.append(temp)
        
    return temporal_data

def to_temporal_dataset(datasets, **temporal_dataset_parameters):
    '''
    Converts dataset into a list of temporal Data objects
    '''
    new_dataset = []
    for data in datasets:
        new_dataset += to_temporal(data, **temporal_dataset_parameters)

    return new_dataset
    
def use_prediction(x, pred, previous_t):
    '''
    Creates a new input given the model's prediction
    '''
    out_dim = 2
    dynaminc_vars = previous_t*out_dim
    input_size = x.shape[1]
    static_vars = input_size-dynaminc_vars

    if static_vars != 0:    # equivalent to with_x == True
        if previous_t == 1:
            temp = torch.cat((x[:,:static_vars], 
                              pred), 1)
        else:
            temp = torch.cat((x[:,:static_vars], 
                              x[:,-dynaminc_vars+out_dim:], 
                              pred), 1)
        assert temp.shape[1] == input_size, 'Something went wrong with predictions'
    else:
        if previous_t == 1:
            temp = pred
        else:
            temp = torch.cat((x[:,out_dim:], 
                              pred), 1)
        assert temp.shape[1] == dynaminc_vars, 'Something went wrong with predictions'

    assert temp.shape == x.shape, 'Something went wrong with predictions'

    return temp

def get_real_rollout(dataset, time_start, time_stop):
    '''Return real rollout for the selected time interval
    '''
    if time_stop == -1:
        real_rollout = dataset.y[:,:,time_start+1:].clone()
    else:
        real_rollout = dataset.y[:,:,time_start+1:time_stop].clone()
    
    return real_rollout

def get_input_water(dataset, time_start):
    '''Return real rollout for the selected time interval
    '''
    input_water = dataset.y[:,:,:time_start+1].clone()

    return input_water

def get_temporal_test_dataset_parameters(config, temporal_dataset_parameters):
    try:
        temporal_test_dataset_parameters = config['temporal_test_dataset_parameters']
        temporal_test_dataset_parameters['with_x'] = temporal_dataset_parameters['with_x']
    except:
        temporal_test_dataset_parameters = temporal_dataset_parameters.copy()
        temporal_test_dataset_parameters.pop('rollout_steps')
        temporal_test_dataset_parameters.pop('previous_t')

    return temporal_test_dataset_parameters

def get_breach_coordinates(WD, pos):
    breach_locations = [loc.item() for loc in torch.where(WD[:,0] != 0)]

    breach_coordinates = [pos[loc] for loc in breach_locations]

    return breach_coordinates

def load_val_test_dataset(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset_parameters = config['dataset_parameters']
    scalers = config['scalers']
    selected_node_features = config['selected_node_features']
    selected_edge_features = config['selected_edge_features']

    _, val_dataset, test_dataset, scalers = create_model_dataset(
        'grid', scalers=scalers, device=device, **dataset_parameters,
        **selected_node_features, **selected_edge_features
    )

    temporal_dataset_parameters = config['temporal_dataset_parameters']
    temporal_test_dataset_parameters = get_temporal_test_dataset_parameters(
        config, temporal_dataset_parameters)

    return val_dataset, test_dataset, temporal_test_dataset_parameters