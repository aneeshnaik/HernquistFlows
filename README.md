# HernquistFlows

## Summary

This code was used to generate the results in Section 5 ('Implementation') of An, Naik, Evans, and Burrage (2021). That paper describes a new method for calculating gravitational accelerations from a known stellar distribution function. In Section 5, we provide a demonstration of this technique, applying it to mock datasets generated from isotropic and anisotropic distribution functions corresponding to the Hernquist model.

In particular, we demonstrate a two-stage procedure:
1. We use normalising flows to 'learn' the distribution function of a 6D mock dataset.
2. We then convert these learned DFs to accelerations using our new technique.

Please see our paper for more details about the technique.

This code is structured as follows:
- `/data` contains the mock datasets, as well as the code used to generate them.
- `/figures` contains the plotting scripts used for the paper.
- `/nflow_models` contains the trained normalising flow models, as well as the scripts used to generate them.
- `/src` contains all of the 'source code', including code underlying the normalising flows, the Hernquist DFs, the conversion to accelerations, and various other utility functions etc. 

The various sections below describe some of these components in further detail.

## Prerequisites

*COMING SOON*

## Usage

### `/data`

*COMING SOON*

### `/nflow_models`

*COMING SOON*
