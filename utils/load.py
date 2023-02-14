import pickle
import yaml
import random

def read_config(config_file):
    '''
    Creates configuration variables from file
    ------
    config_file: .yaml file
        file containing dictionary with dataset creation information
    '''   
    
    with open(config_file) as f:
        cfg = yaml.safe_load(f)
        
    return cfg

    
def load_dataset(dataset_name, size, seed=42, dataset_folder='database/datasets'):
    '''
    Loads dataset, composed by a list of pytorch geometric data objects
    ------
    dataset_name: str
        name of the dataset to be loaded
        options: 'grid' or 'mesh'
    size: int
        number of simulations selected
    '''
    if size > 80:
        raise FileNotFoundError('maximum training dataset size is 80')
        
    path = f"{dataset_folder}/{dataset_name}.pkl"
    
    with open(path, 'rb') as file:
        dataset = pickle.load(file)

    if seed!=0:
        random.seed(seed)
        random.shuffle(dataset)
    
    return dataset[:size]