# SWE-GNN (paper repository)
Code repository for paper "Generalized Spatio-Temporal Rapid Flood Modelling via Hydraulics-Inspired Graph Neural Networks"

(Version 1.0 (review) - Feb. 15th, 2023)

![Summary figure](Summary figure.png)

## Overview

For the viewing all simulations with the best model, check <https://drive.google.com/drive/folders/1OsXgFvAUa2ylwFm-qQMmjWqScxR0xJGK?usp=sharing>.

For reproducing the paper's results, explore **plot_results.ipynb**

For training the model run **main.py** (uses **config.yaml** as reference configuration file)
For exploring the trained models, run **try_model.ipynb**

The repository is divided in the following folders:

* **database:** Creation of hydrodynamic simulations (**D-Hydro simulations.ipynb**) [requires the license and installation of "D-HYDRO Suite 2023.01 1D2D"] and conversion of the NETCDF output files into PyTorch Geometric-friendly data (**create_dataset.ipynb**).
Also contains the output of the hydrodynamic simulations (**raw_datasets**: for downloading this dataset go to <https://zenodo.placeholder>). This is converted into Pickle files that are then stored and separated into training and testing datasets in **datasets**.

* **models:**  Deep learning models developed for surrogating the hydraulic one: contains MLP, CNN, and GNNs, as well as a base class with common inputs and functions.

* **results:** Contains trained models and respective configuration files, used for the paper's results.

* **training:** Contains loss functions, Trainer object, and testing functions.

* **utils:** Contains Python functions for loading, creating and scaling the dataset. There are also other miscellaneous functions and visualization functions.

## Environment setup

The required libraries are in requirements.txt. For installing PyTorch-Geometric libraries, follow the steps on <https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html>. Consider the compatibility with your version of PyTorch.
