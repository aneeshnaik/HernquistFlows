#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Add Gaussian 'errors' to the 6D Hernquist samples.

Created: May 2021
Author: A. P. Naik
"""
import numpy as np


if __name__ == '__main__':

    # set up RNG
    rng = np.random.default_rng(42)

    # loop over isotropic and anisotropic datasets
    for df in ['hq_iso', 'hq_aniso']:

        # load original dataset
        data = np.load(df + "_orig.npz")
        pos = data['pos']
        vel = data['vel']

        # loop over 10% and 1% noise
        for i in range(2):

            # generate noise
            scale = [0.1, 0.01][i]
            pos_new = pos * rng.normal(loc=1.0, scale=scale, size=pos.shape)
            vel_new = vel * rng.normal(loc=1.0, scale=scale, size=vel.shape)

            # save
            fname_suffix = ['_10pc_noise', '_1pc_noise'][i]
            np.savez(df + fname_suffix, pos=pos_new, vel=vel_new)
