#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train normalising flow.

Created: May 2021
Author: A. P. Naik
"""
import numpy as np
import sys

sys.path.append("../../src")
from constants import kpc, M_sun, G
from ml import train_flow
from utils import get_rescaled_tensor


if __name__ == '__main__':

    # load data
    M = 1e+10 * M_sun
    a = 5 * kpc
    u_pos = 10 * a
    u_vel = np.sqrt(2 * G * M / a)
    data = get_rescaled_tensor("../../data/hq_iso_10pc_noise.npz", u_pos, u_vel)

    # parse arguments
    assert len(sys.argv) == 2
    seed = int(sys.argv[1])

    # train flow
    train_flow(data, seed, n_layers=8, n_hidden=64)
