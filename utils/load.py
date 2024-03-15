import pickle
import yaml
import random
import os

def read_config(config_file):
    '''
    Creates configuration variables from file
    ------
    config_file: .yaml file
        file containing dictionary with dataset creation information
    '''   
    
    try:
        with open(config_file, 'r') as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"The configuration file was not found: {config_file}")
    except yaml.YAMLError as exc:
        raise Exception(f"An error occurred during YAML parsing: {exc}")
    return cfg

def load_dataset(dataset_name, size, seed=42, dataset_folder='SWE-GNN-paper-repository-/database/datasets'):
    '''
    Loads a dataset, composed of a list of PyTorch Geometric data objects.
    ------
    dataset_name: str
        Name of the dataset to be loaded.
        Options: 'grid' or 'mesh'.
    size: int
        Number of simulations selected.
    seed: int, optional
        Seed for shuffling the dataset. Default is 42.
    dataset_folder: str, optional
        Base folder where datasets are stored. Default is 'database/datasets'.
    '''
    if size > 80:
        raise FileNotFoundError('Maximum training dataset size is 80.')

    # Correctly constructing the path using os.path.join
    path = os.path.join(dataset_folder, f"{dataset_name}.pkl")
    
    try:
        with open(path, 'rb') as file:
            dataset = pickle.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"No such file or directory: '{path}'")

    if seed != 0:
        random.seed(seed)
        random.shuffle(dataset)
    
    return dataset[:size]

