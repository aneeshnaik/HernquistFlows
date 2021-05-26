#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Various functions relating to the Hernquist model.

Created: May 2021
Author: A. P. Naik
"""
import numpy as np
from constants import pi, G
import torch


def calc_DF_iso(q, p, M, a):
    """
    Evaluate isotropic DF corresponding to Hernquist model.

    Parameters
    ----------
    q : np.array or torch.tensor, shape (N, 3) or (3)
        Positions at which to evaluate DF. Either an array shaped (N, 3) for N
        different phase points, or shape (3) for single phase point.
        UNITS: metres.
    p : np.array or torch.tensor, shape (N, 3) or (3)
        Velocities at which to evaluate DF. UNITS: m/s.
    M : float
        Mass of Hernquist profile. UNITS: kg.
    a : float
        Scale radius of Hernquist profile. UNITS: m.

    Returns
    -------
    f : np.array or torch.tensor (shape (N)) or float
        DF evaluated at given phase points. If inputs are 1D, f is float. If
        inputs are 2D, f is either np.array or torch.tensor, matching type of
        input. Gradient information is propagated if torch.tensor.
        UNITS: m^-6 s^3.
    """
    # check input shapes match
    assert q.shape == p.shape

    # check if inputs are 1D or 2D and whether np or torch
    oneD = False
    np_array = False
    if q.ndim == 1:
        oneD = True
    if type(q) == np.ndarray:
        np_array = True

    # convert inputs to torch tensors if nec.
    if np_array:
        q = torch.tensor(q)
        p = torch.tensor(p)

    # get normed r, v
    r = q.norm(dim=-1)
    v = p.norm(dim=-1)

    # get energy (KE + PE)
    E = 0.5 * v**2 - G * M / (r + a)
    x = E / (G * M / a)
    B = torch.abs(x)

    # calculate DF
    const = 1 / (np.sqrt(2) * (2 * pi)**3 * (G * M * a)**(3 / 2))
    prefac = torch.sqrt(B) / (1 - B)**2 * const
    term1 = (1 - 2 * B) * (8 * B**2 - 8 * B - 3)
    term2 = 3 * torch.arcsin(torch.sqrt(B)) / torch.sqrt(B * (1 - B))
    f = prefac * (term1 + term2)

    # zero out unbound and unphysical
    if oneD:
        if (x.item() > 0) or (x.item() < -1):
            return 0.
    else:
        f[(x > 0) | (x < -1)] = 0

    # sort out format of output
    if oneD:
        f = f.item()
    elif np_array:
        f = f.detach().numpy()
    return f


def calc_DF_aniso(q, p, M, a):
    """
    Evaluate anisotropic Hernquist distribution function.

    Parameters
    ----------
    q : np.array or torch.tensor, shape (N, 3) or (3)
        Positions at which to evaluate DF. Either an array shaped (N, 3) for N
        different phase points, or shape (3) for single phase point.
        UNITS: metres.
    p : np.array or torch.tensor, shape (N, 3) or (3)
        Velocities at which to evaluate DF. UNITS: m/s.
    M : float
        Mass of Hernquist profile. UNITS: kg.
    a : float
        Scale radius of Hernquist profile. UNITS: m.

    Returns
    -------
    f : np.array or torch.tensor (shape (N)) or float
        DF evaluated at given phase points. If inputs are 1D, f is float. If
        inputs are 2D, f is either np.array or torch.tensor, matching type of
        input. Gradient information is propagated if torch.tensor.
        UNITS: m^-6 s^3.
    """
    # check input shapes match
    assert q.shape == p.shape

    # check if inputs are 1D or 2D and whether np or torch
    oneD = False
    np_array = False
    if q.ndim == 1:
        oneD = True
    if type(q) == np.ndarray:
        np_array = True

    # convert inputs to torch tensors if nec.
    if np_array:
        q = torch.tensor(q)
        p = torch.tensor(p)

    # get normed r, v
    r = q.norm(dim=-1)
    v = p.norm(dim=-1)

    # get energy and AM
    E = 0.5 * v**2 - G * M / (r + a)
    x = E / (G * M / a)
    L = torch.cross(q, p).norm(dim=-1)

    # calculate DF
    prefac = (3 * a) / (4 * pi**3)
    f = prefac * E**2 / (G**3 * M**3 * L)

    # zero out unbound and unphysical
    if oneD:
        if (x.item() > 0) or (x.item() < -1):
            return 0.
    else:
        f[(x > 0) | (x < -1)] = 0

    # sort out format of output
    if oneD:
        f = f.item()
    elif np_array:
        f = f.detach().numpy()
    return f
