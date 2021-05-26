#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Various utility functions.

Created: May 2021
Author: A. P. Naik
"""
import numpy as np
import torch
from torch.autograd import grad
from constants import pi


def get_rescaled_tensor(dfile, u_pos, u_vel):
    """Load data, rescale, shuffle, and make into torch tensor."""
    # load data
    print("Loading data...")
    data = np.load(dfile)
    pos = data['pos']
    vel = data['vel']

    # rescale data
    pos = pos / u_pos
    vel = vel / u_vel

    # stack and shuffle data
    data = np.hstack((pos, vel))
    rng = np.random.default_rng(42)
    rng.shuffle(data)

    # make torch tensor
    data_tensor = torch.from_numpy(data.astype(np.float32))
    return data_tensor


def diff_DF(q, p, df_func, df_args):
    """Get q and p derivs of DF."""

    # check if 1D
    oneD = False
    if q.ndim == 1:
        oneD = True
        q = q[None]
        p = p[None]

    # convert to torch tensors
    q = torch.tensor(q, requires_grad=True)
    p = torch.tensor(p, requires_grad=True)

    # evaluate DF
    f = df_func(q, p, **df_args)

    # calculate f gradients; dfdq and dfdp both have shape (Nv, 3)
    dfdq = grad(f, q, torch.ones_like(f), create_graph=True)[0]
    dfdp = grad(f, p, torch.ones_like(f), create_graph=True)[0]
    if oneD:
        dfdq = dfdq[0]
        dfdp = dfdp[0]

    return dfdq.detach().numpy(), dfdp.detach().numpy()


# =============================================================================
# def sample_velocities(Nv, v_max, v_min=0, v_mean=np.zeros(3)):
#     """
#     Uniformly sample Nv velocities in ball of radius v_max centred on
#     v_mean.
#     """
#     # magnitude
#     v_mag = v_min + (v_max - v_min) * np.random.rand(Nv)
# 
#     # orientation
#     phi = 2 * pi * np.random.rand(Nv)
#     theta = np.arccos(2 * np.random.rand(Nv) - 1)
# 
#     # convert to Cartesian
#     vx = v_mag * np.sin(theta) * np.cos(phi)
#     vy = v_mag * np.sin(theta) * np.sin(phi)
#     vz = v_mag * np.cos(theta)
# 
#     # stack
#     vel = v_mean + np.stack((vx, vy, vz), axis=-1)
#     return vel
# =============================================================================


def sample_velocities(Nv, v_max, pos, vt_min=0, vx_min=0, v_mean=np.zeros(3)):
    """
    Uniformly sample Nv velocities in ball of radius v_max centred on
    v_mean.
    """
    # magnitude
    v_mag = v_max * np.random.rand(10 * Nv)

    # orientation
    phi = 2 * pi * np.random.rand(10 * Nv)
    theta = np.arccos(2 * np.random.rand(10 * Nv) - 1)

    # convert to Cartesian
    vx = v_mag * np.sin(theta) * np.cos(phi)
    vy = v_mag * np.sin(theta) * np.sin(phi)
    vz = v_mag * np.cos(theta)

    # stack
    vel = v_mean + np.stack((vx, vy, vz), axis=-1)

    # downsample to exclude vels in disallowed regions
    r = np.linalg.norm(pos, axis=-1)
    rhat = pos / r
    vr = np.sum(vel * rhat, axis=-1)
    vt = np.linalg.norm(vel - vr[:, None] * rhat, axis=-1)
    vx2 = vel[:, 0]**2
    vy2 = vel[:, 1]**2
    vz2 = vel[:, 2]**2
    c = vx_min**2
    mask = (vt > vt_min) & (vx2 > c) & (vy2 > c) & (vz2 > c)
    if mask.sum() < Nv:
        assert False
    inds = np.random.choice(np.arange(10 * Nv)[mask], Nv, replace=False)
    vel = vel[inds]

    return vel
