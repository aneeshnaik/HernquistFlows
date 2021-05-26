# HernquistFlows


## Summary

This code was used to generate the results in Section 5 ('Implementation') of An, Naik, Evans, and Burrage (2021). That paper describes a new method for calculating gravitational accelerations from a known stellar distribution function. In Section 5, we provide a demonstration of this technique, applying it to mock datasets generated from isotropic and anisotropic distribution functions corresponding to the Hernquist model.

In particular, we demonstrate a two-stage procedure:
1. We use normalising flows to 'learn' the distribution function of a 6D mock dataset.
2. We then convert these learned DFs to accelerations using our new technique.

Please see our paper for more details about the technique.


## Prerequisites

*COMING SOON*


## Structure

This code is structured as follows:
- `/data` contains the mock datasets, as well as the code used to generate them.
- `/figures` contains the plotting scripts used for the paper.
- `/nflow_models` contains the trained normalising flow models, as well as the scripts used to generate them.
- `/src` contains all of the 'source code', including code underlying the normalising flows, the Hernquist DFs, the conversion to accelerations, and various other utility functions etc. 

The various subsections below describe these components in further detail.

### `/data`

This directory contains the mock datasets and the scripts used to generate the mock datasets.

The noiseless isotropic and anisotropic datasets are respectively `hq_iso_orig.npz` and `hq_aniso_orig.npz`. These are `numpy` archive files, and can be opened with a simple `np.load` command. They each comprise just two arrays: `pos` and `vel`, containing positions and velocities (in SI units) of the stars.

The two datasets were generated by running the scripts `sample_hq_iso.py` and `sample_hq_aniso.py`. These work by using an MCMC sampler to generate samples from the exact distribution functions as defined in `src/hernquist`.

There are two more datasets: `hq_iso_1pc_noise.npz` and `hq_iso_10pc_noise.npz`. These are the same as `hq_iso_orig.npz`, but with added random Gaussian noise at the 1% and 10% levels respectively. They were created by running `add_noise.py` (after having already run `sample_hq_iso.py` to create the noiseless dataset).

### `/figures`

This directory contains the four figures from our paper (in .pdf format), along with the scripts used to generate them. The general file pattern is:
- `fign_x.py`
- `fign_x.pdf`
- `fign_data.npz`

Here, `x` is some brief descriptor describing the figure. The python script generates both the `.npz` file containing the figure data and the figure itself. On subsequent runs, the script will find the `.npz` file it has previously created and generate the figure directly from the saved figure data, saving the trouble of generating the data anew.

These scripts give different example use cases for how to read and analyse the normalising flow models and how to convert their learned DFs into accelerations.

### `/nflow_models`

There are four subdirectories in here, one for each dataset in `/data`. Each subdirectory contains:
- An ensemble of trained flow models, `x_best.pth`, with `x` being some integer. 
- For each flow model, a saved numpy array containing training losses, `x_losses.npy`
- One script (total, not one per `.pth` model), `train_nflow.py`: running this script kicks off the training procedure which generates the other files.

The runscript `train_nflow.py` works by loading the appropriate dataset, rescaling it, then feeding it to a randomly initialised normalising flow which then trains from it. It does this with one main underlying function call to `train_flow`, defined in `src/ml.py`.

The only difference between the different members of the flow ensemble (i.e. the different `.pth` files in a given subdirectory) is the random initialisation of the flow. This is set by a random seed, which is taken as a mandatory argument by `train_flow.py`. For example, one could run the command:

    python train_nflow.py 243
 
This would then take 243 as a random seed to initialise the normalising flow, then train the flow and (eventually) save the best fit model as `243_best.pth`, and the losses as `243_losses.npy`.

### `/src`

There are 5 python files in this subdirectory, and these form the main workhorse of the codebase. A brief summary of the contents of each file is given here, but all of the objects contained within these files are themselves well-documented, so further details can be found there.

- `cbe.py`: this deals with calculating accelerations from gradients of a DF (via an inversion of the CBE). It contains only one function: `calc_accel_CBE`.
- `constants.py`: this contains various physical constants and unit definitions
- `hernquist.py`: this contains exact definitions for the Hernquist isotropic and anisotropic DFs, via the functions `calc_DF_iso` and `calc_DF_aniso` respectively.
- `ml.py`: this contains a number of functions relating to the training and subsequent analysis of normalising flows.
    1. `setup_MAF`: Initialises a masked autoregressive flow.
    2. `load_flow`: Loads a saved flow model.
    3. `calc_total_loss`: Compute total loss of model on data.
    4. `train_epoch`: Train flow model for one epoch.
    5. `calc_DF_model`: Given a flow model, evaluate DF at desired phase points.
    6. `calc_DF_ensemble`: Given an ensemble of flow models, evaluate DF at desired phase points, averaging across ensemble.
    7. `train_flow`: Train a normalising flow on data. This is the main function called by the runscript `train_nflow.py` in `/nflow_models`.
- `utils.py`: this is essentially an oddbin containing utility functions that don't belong anywhere else. There are three of these:
    1. `get_rescaled_tensor`: given a dataset as in `/data`, this function rescales, shuffles, and stacks the data, then returns it as a torch tensor ready to be trained on. 
    2. `diff_DF`: given a DF, this function calculates its spatial and velocity derivatives. 
    3. `sample_velocities`: this function samples a number of velocities at a given position (for use in calculating accelerations).


