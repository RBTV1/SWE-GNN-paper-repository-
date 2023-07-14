# SWE-GNN (paper repository)
Code repository for paper "Rapid Spatio-Temporal Flood Modelling via Hydraulics-Based Graph Neural Networks"

(Version 1.1 (post-review) - June. 21st, 2023)

![summary_figure](summary_figure.png)

## Overview

All test video simulations can be found at <https://dx.doi.org/10.5281/zenodo.7652663>.

To begin with running the experiments, you must first download the dataset from <https://dx.doi.org/10.5281/zenodo.7764418>.
After downloading it, run the **create_dataset.ipynb** notebook inside the **database** folder and you will be good to go!

For reproducing the paper's results, you can run **plot_results.ipynb**

For training the model run **main.py** (which uses **config.yaml** as configuration file with all model's specifications)

For exploring the trained models, run **try_model.ipynb**

### Environment setup

The required libraries are in requirements.txt. For installing PyTorch-Geometric libraries, follow the steps on <https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html>. Consider the compatibility with your version of PyTorch.

### Repository 

The repository is divided in the following folders:

* **database:** Creation of hydrodynamic simulations (**D-Hydro simulations.ipynb**) [requires the license and installation of "D-HYDRO Suite 2023.01 1D2D"] and conversion of the NETCDF output files into PyTorch Geometric-friendly data (**create_dataset.ipynb**).
Also contains the output of the hydrodynamic simulations (**raw_datasets**: for downloading this dataset go to <https://dx.doi.org/10.5281/zenodo.7764418>). This is converted into Pickle files that are then stored and separated into training and testing datasets in **datasets**.

* **models:**  Deep learning models developed for surrogating the hydraulic one: contains MLP, CNN, and GNNs, as well as a base class with common inputs and functions.

* **results:** Contains trained models and respective configuration files, used for the paper's results.

* **training:** Contains loss functions, Trainer object, and testing functions.

* **utils:** Contains Python functions for loading, creating and scaling the dataset. There are also other miscellaneous functions and visualization functions.
