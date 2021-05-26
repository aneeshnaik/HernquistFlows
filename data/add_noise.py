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

    # load original dataset
    data = np.load("hq_iso_orig.npz")
    pos = data['pos']
    vel = data['vel']

    # loop over 10% and 1% noise
    for i in range(2):

        # generate noise
        scale = [0.1, 0.01][i]
        pos_new = pos * rng.normal(loc=1.0, scale=scale, size=pos.shape)
        vel_new = vel * rng.normal(loc=1.0, scale=scale, size=vel.shape)

        # save
        fname = ['hq_iso_10pc_noise', 'hq_iso_1pc_noise'][i]
        np.savez(fname, pos=pos_new, vel=vel_new)
