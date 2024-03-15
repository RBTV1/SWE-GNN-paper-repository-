
import numpy as np 
from perlin_noise import PerlinNoise
import random
from dfm_tools.get_nc import get_ncmodeldata
from netCDF4 import Dataset
import subprocess
import time
import os
import pandas as pd
import xarray as xr
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

def grid_DEM_generator(seed, number_grids=64, noise_octave=10, initial_octave=4, DEM_multiplier=5, initial_DEM=1):
    '''
    Generates a random digital elevation model (DEM) based on Perlin noise
    ------
    seed: int, replicable randomness for Perlin noise and magnitude randomizer
    number_grids: int, number of grid cells in x and y directions
    noise_octave: float, Perlin noise octave magnitude
    DEM_multiplier: float, multiplier for max and min DEM values
    '''
    random.seed(seed)

    octaves = noise_octave*random.random()+initial_octave
    noise = PerlinNoise(octaves=octaves, seed=seed)
    DEM_multiplier = DEM_multiplier*random.random()+initial_DEM

    xpix, ypix = number_grids, number_grids
    DEM = np.array([[noise([i/xpix, j/ypix])*DEM_multiplier for j in range(xpix)] for i in range(ypix)])

    return DEM

def change_nc_file(grid_file, varname, new_value):
    '''
    Change NETCDF variable 'varname' with 'new_value'
    '''
    ncfile = Dataset(grid_file,mode='r+') 

    ncfile.variables[varname][:] = new_value

    ncfile.close()
    return None

def save_DEM(DEM, grid_size, dst_file, pos=None):
    '''
    Saves DEM file as .xyz or .txt file for numerical simulations
    ------
    DEM: np.array, elevetion map in x and y directions
    grid_size: float, length of each DEM cell [m]
    dst_file: str (path-like), destination file for saving DEM
    pos: dict, node positions (x, y)
    '''
    if pos is None:
        number_grids = DEM.shape[0] #int(len(pos)**0.5)
        y_grid = np.array([[(i+0.5)*grid_size for j in range(number_grids)] for i in range(number_grids)])
        x_grid = y_grid.T
        xyz = np.array([[x, y, z] for x, y, z in zip(x_grid.reshape(-1,1), y_grid.reshape(-1,1), DEM.reshape(-1,1))]).squeeze()
    else:
        xyz = np.array([[x*grid_size, y*grid_size, z] for (x, y), z in zip(list(pos.values()), DEM)])
    
    #creating xyz file for DEM with proper number of decimals
    np.savetxt(dst_file, xyz, fmt = ('%1.1f', '%1.1f', '%1.5f'))

    return None

def change_breach_location(breach_polygon_file, point1=[0, 0.5], point2=[0.5, 0], grid_size=100):
    '''Creates a 2-point segment that represents a dike breach'''
    y1, x1 = np.array(point1)*grid_size
    y2, x2 = np.array(point2)*grid_size

    assert point1 != point2, "The two points have the same location"

    replacement = 'ConstantQ\n'\
              '    2    2\n'\
              f'{x1}    {y1}     ConstantQ_0001\n'\
              f'{x2}    {y2}     ConstantQ_0002'

    # print(replacement)
    
    with open(breach_polygon_file, "w") as f:
        f.write(replacement)
    
    return None

def get_random_breach_location(breach_polygon_file, number_grids, grid_size):
    x = int(random.random()*number_grids)
    y = int(random.random()*number_grids)
    border = random.randint(0,1)*number_grids
    
    if random.random()>0.5:
        point1=[x, border]
        point2=[x+1, border]
    else:
        point1=[border, y]
        point2=[border, y+1]

    change_breach_location(breach_polygon_file, point1, point2, grid_size)

    return None

def run_simulation(model_folder):
    '''
    Run D-Hydro simulation, give model folder location
    Returns computational time
    '''
    #paths
    input_folder = f'{model_folder}\\input'
    execution_file = f'{input_folder}\\run.bat'

    start_time = time.time()

    #Run D-Hydro, let Python wait till D-Hydro is done
    command = subprocess.Popen(execution_file, cwd = input_folder)
    command.wait()

    computation_time = round(time.time() - start_time, 4)

    return computation_time


def from_output_nc_to_txt(output_map, save_folder, seed):
    '''
    Converts numerical simulation in .nc file (output_map) to water depth and velocities .txt files
    ------
    output_map: str (path-like), netcdf output file location
    save_folder: str (path-like), folder location for saving results
    seed: int, simulation number
    '''
    #retrieve map data
    Mesh2d_face_x = get_ncmodeldata(file_nc=output_map, varname='Mesh2d_face_x', silent=True)
    Mesh2d_face_y = get_ncmodeldata(file_nc=output_map, varname='Mesh2d_face_y', silent=True)
    df = pd.DataFrame({'xloc': Mesh2d_face_x, 'yloc': Mesh2d_face_y})
    order = df.sort_values(['xloc', 'yloc']).index

    #extract water depth and velocities
    waterdepth = get_ncmodeldata(file_nc=output_map, varname='Mesh2d_waterdepth', timestep='all', silent=True)[:,order]
    velocity_x = get_ncmodeldata(file_nc=output_map, varname='Mesh2d_ucx', timestep='all', silent=True)[:,order]
    velocity_y = get_ncmodeldata(file_nc=output_map, varname='Mesh2d_ucy', timestep='all', silent=True)[:,order]
    
    #saving water depth and velocities
    np.savetxt(f'{save_folder}/WD/WD_{seed}.txt', waterdepth, fmt='%1.4f')
    np.savetxt(f'{save_folder}/VX/VX_{seed}.txt', velocity_x, fmt='%1.4f')
    np.savetxt(f'{save_folder}/VY/VY_{seed}.txt', velocity_y, fmt='%1.4f')

    return None

def create_dataset_folders(dataset_folder='datasets'):
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)

    train_folder = os.path.join(dataset_folder, 'train')
    test_folder = os.path.join(dataset_folder, 'test')

    if not os.path.exists(train_folder):
        os.makedirs(train_folder)

    if not os.path.exists(test_folder):
        os.makedirs(test_folder)    

def run_simulations(n_sim, model_folder, save_folder, start_sim=1, number_grids:int=64, 
                    noise_octave=8, initial_octave=4, DEM_multiplier=5, initial_DEM=1, 
                    grid_size=100, random_breach=True):
    '''
    Run multiple hydraulic simulations
    ------
    DEM: np.array, elevetion map in x and y directions
    grid_size: float, length of each DEM cell [m]
    s_time: float, simulation time [h]
    '''
    output_map = f'{model_folder}\\input\\dflowfm\\output\\SWE_GNN_dataset_map.nc'
    grid_file = f'{model_folder}\\input\\dflowfm\\SWE_GNN_dataset_net.nc'
    breach_polygon_file = f'{model_folder}\\input\\dflowfm\\Boundary01.pli'

    time_steps = xr.open_dataset(output_map).indexes['time']
    simulated_time = time_steps[-1] - time_steps[0]
    simulated_time_hours = simulated_time.days * 24 + simulated_time.seconds // 3600
    temporal_resolution = time_steps[1].minute

    simulation_stats = []

    for sim in tqdm(range(start_sim, start_sim+n_sim)):
        if random_breach:
            get_random_breach_location(breach_polygon_file, number_grids, grid_size)
        else:
            change_breach_location(breach_polygon_file, grid_size=grid_size)
        
        # generate and save random DEM
        DEM = grid_DEM_generator(sim, number_grids, noise_octave, initial_octave, DEM_multiplier, initial_DEM)
        save_DEM(DEM, grid_size, f'{model_folder}\\input\\dflowfm\\DEM.xyz')
        change_nc_file(grid_file, 'Mesh2d_face_z', DEM.reshape(-1))
        save_DEM(DEM.T, grid_size, f'{save_folder}\\DEM\\DEM_{sim}.txt')
        
        # run simulation
        computation_time = run_simulation(model_folder)

        # save results in files and overview folder
        from_output_nc_to_txt(output_map, save_folder, sim)
        simulation_stats.append([sim, grid_size, number_grids, simulated_time_hours, computation_time])

    df = pd.DataFrame(simulation_stats, columns=['seed', 'grid_size[m]', 'number_grids', 'simulation_time[h]', 'computation_time[s]'])
    df.to_csv(f'{save_folder}\\overview.csv', mode='a', sep = ',', index = False)

    return df