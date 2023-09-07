# Libraries
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def get_none_scalers():
    none_scalers = {'DEM_scaler' : None, 
                    # 'slope_scaler' : None, 
                    'WD_scaler' : None, 
                    'edge_slope_scaler' : None, 
                    'V_scaler' : None}
    return none_scalers

def stack_attrs(dataset, attr, to_min=False):
    '''
    Returns a vector containing all features 'attr' contained in dataset
    '''
    stacked_map = torch.stack([data[attr] - data[attr].min()*to_min for data in dataset])
    
    return stacked_map.reshape(-1,1)

def scaler(train_database, attr: str or list or tuple, type_scaler='minmax', to_min=False):
    '''
    Returns Scaler for a 2D map
    This function should consider only the training dataset
    -------
    train_database : list
        each element in the list is a torch_geometric.data.data.Data object
    attr: str or list
        name of the feature to be scaled
        if list, scaling applies to more features
        if tuple, elements are used for vector norm
    type_scaler: str
        name of the scaler
        options: 'minmax', 'minmax_neg', or 'standard'
    to_min: bool
        subtract the minimum value before scaling (then the minimum is always 0)
    '''
    if type_scaler == 'minmax':
        scaler = MinMaxScaler(feature_range=(0,1))
    elif type_scaler == 'minmax_neg':
        scaler = MinMaxScaler(feature_range=(-1,1))
    elif type_scaler == 'standard':
        scaler = StandardScaler()
    elif type_scaler is None:
        return None
    else:
        raise 'type_scaler can be only "minmax", "minmax_neg", or "standard"'
    
    if isinstance(attr, list):
        all_attrs = torch.cat([stack_attrs(train_database, att, to_min=to_min) for att in attr])
    elif isinstance(attr, tuple):
        all_attrs = torch.cat([stack_attrs(train_database, att, to_min=to_min)**2 for att in attr], 1)
        all_attrs = all_attrs.sum(1).sqrt().reshape(-1,1)
    else:
        all_attrs = stack_attrs(train_database, attr, to_min=to_min)

    scaler.fit(all_attrs)
        
    return scaler

def get_scalers(dataset, scalers: dict):
    '''
    Returns scaler dictionary with scaler objects as values
    ------
    dataset: list
        training dataset used to obtain the scalers
    scalers: dict
        dict with the type of scaling used for every variable
    '''
    if scalers is None:
        scalers = get_none_scalers()

    scalers['DEM_scaler'] = scaler(dataset, 'DEM', type_scaler=scalers['DEM_scaler'], to_min=True)
    # scalers['slope_scaler'] = scaler(dataset, ['slope_x', 'slope_y'], type_scaler=scalers['slope_scaler'])
    scalers['WD_scaler'] = scaler(dataset, 'WD', type_scaler=scalers['WD_scaler'])
    scalers['V_scaler'] = scaler(dataset, ('VX', 'VY'), type_scaler=scalers['V_scaler'])

    return scalers