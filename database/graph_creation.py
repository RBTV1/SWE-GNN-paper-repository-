import numpy as np
import networkx as nx
import os
import pickle
from tqdm import tqdm
import torch
from torch_geometric.data import Data

def center_grid_graph(dim1, dim2):
    '''
    Create graph from a rectangular grid of dimensions dim1 x dim2
    Returns networkx graph connecting the grid centers and corresponding 
    node positions
    ------
    dim1: int
        number of grids in the x direction
    dim2: int
        number of grids in the y direction
    '''
    G = nx.grid_2d_graph(dim1, dim2, create_using=nx.DiGraph)
    # for the position, it is assumed that they are located in the centre of each grid
    pos = {i:(x+0.5,y+0.5) for i, (x,y) in enumerate(G.nodes())}
    
    #change keys from (x,y) format to i format
    mapping = dict(zip(G, range(0, G.number_of_nodes())))
    G = nx.relabel_nodes(G, mapping)

    return G, pos

def get_coords(pos):
    '''
    Returns array of dimensions (n_nodes, 2) containing x and y coordinates of each node
    ------
    pos: dict
        keys: (x,y) index of every node
        values: spatial x and y positions of each node
    '''
    return np.array([xy for xy in pos.values()])
	

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

def get_contour(pos):
    '''
    Returns a dictionary with the contours of a grid
    ------
    pos: dict
        keys: (x,y) index of every node
        values: spatial x and y positions of each node
    '''
    BL, TR, BR, TL = get_corners(pos)
    
    x_pos = np.arange(BL[0], TR[0]+1)
    y_pos = np.arange(BL[1], TR[1]+1)
    
    bottom = [(x, BL[1]) for x in x_pos]
    left = [(BL[0], y) for y in y_pos]
    right = [(TR[0], y) for y in y_pos]
    top = [(x, TR[1]) for x in x_pos]
    
    contour = {}

    for point in (bottom + left + right + top):
        key = list(pos.keys())[list(pos.values()).index(point)]
        contour[point] = pos[key]
    
    return contour

def reorder_dict(dictt):
    '''
    Change the key of a dictionary and sorts it by values order
    '''
    new_dict = {}
    
    #sort to exclude double values and order it
    dictt = dict(sorted(dictt.items()))

    #change keys from (x,y) format to i format
    for i, key in enumerate(dictt.keys()):
        new_dict[i] = dictt[key]
        
    return new_dict

def convert_to_pyg(graph, pos, DEM, WD, VX, VY):
    '''Converts a graph or mesh into a PyTorch Geometric Data type 
    Then, add position, DEM, and water variables to data object'''
    DEM = DEM.reshape(-1)

    edge_index = torch.LongTensor(list(graph.edges)).t().contiguous()
    row, col = edge_index

    data = Data()

    delta_DEM = torch.FloatTensor(DEM[col]-DEM[row])
    coords = torch.FloatTensor(get_coords(pos))
    edge_relative_distance = coords[col] - coords[row]
    edge_distance = torch.norm(edge_relative_distance, dim=1)
    edge_slope = delta_DEM/edge_distance

    data.edge_index = edge_index
    data.edge_distance = edge_distance
    data.edge_slope = edge_slope
    data.edge_relative_distance = edge_relative_distance

    data.num_nodes = graph.number_of_nodes()
    data.pos = torch.tensor(list(pos.values()))
    data.DEM = torch.FloatTensor(DEM)
    data.WD = torch.FloatTensor(WD.T)
    data.VX = torch.FloatTensor(VX.T)
    data.VY = torch.FloatTensor(VY.T)
        
    return data

def create_grid_dataset(dataset_folder, n_sim, start_sim=1, number_grids=64):
    '''
    Creates a pytorch geometric dataset with n_sim simulations
    returns a regular grid graph dataset
    ------
    dataset_folder: str, path-like
        path to raw dataset location
    n_sim: int
        number of simulations used in the dataset creation
    '''
    assert os.path.exists(dataset_folder), "There is no raw dataset folder"
    grid_dataset = []

    graph, pos = center_grid_graph(number_grids,number_grids)
    
    for i in tqdm(range(start_sim,start_sim+n_sim)):

        DEM = np.loadtxt(f"{dataset_folder}\\DEM\\DEM_{i}.txt")[:,2]
        WD = np.loadtxt(f"{dataset_folder}\\WD\\WD_{i}.txt")
        VX = np.loadtxt(f"{dataset_folder}\\VX\\VX_{i}.txt")
        VY = np.loadtxt(f"{dataset_folder}\\VY\\VY_{i}.txt")
        
        grid_i = convert_to_pyg(graph, pos, DEM, WD, VX, VY)
        grid_dataset.append(grid_i)
    
    return grid_dataset

def save_database(dataset, name, out_path='datasets'):
    '''
    This function saves the geometric database into a pickle file
    The name of the file is given by the type of graph and number of simulations
    ------
    dataset: list
        list of geometric datasets for grid and mesh
    names: str
        name of saved dataset
    out_path: str, path-like
        output file location
    '''
    n_sim = len(dataset)
    path = f"{out_path}/{name}.pkl"
    
    if os.path.exists(path):
        os.remove(path)
    elif not os.path.exists(out_path):
        os.mkdir(out_path)
    
    pickle.dump(dataset, open(path, "wb" ))
        
    return None