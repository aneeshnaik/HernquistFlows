#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Various physical functions.

Created: May 2021
Author: A. P. Naik
"""
import numpy as np


def calc_accel_CBE(pos, vel, gradxf, gradvf, coords='cartesian'):
    """
    Convert DF gradients to acceleration via CBE inversion

    Parameters
    ----------
    pos : TYPE
        DESCRIPTION.
    vel : TYPE
        DESCRIPTION.
    gradxf : TYPE
        DESCRIPTION.
    gradvf : TYPE
        DESCRIPTION.
    coords : TYPE, optional
        DESCRIPTION. The default is 'cartesian'.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    """. vel, gradxf and gradvf all np arrays shape (N, 3)."""
    assert coords in ['cartesian', 'cylindrical']
    S = np.sum(vel * gradxf, axis=-1)
    if coords == 'cylindrical':
        r = pos[:, 0]
        t1 = (vel[:, 1] * vel[:, 1] / r) * gradvf[:, 0]
        t2 = (vel[:, 0] * vel[:, 1] / r) * gradvf[:, 1]
        S += t1 - t2
    R = np.sum(S[:, None] * gradvf, axis=0)
    A = gradvf[:, None] * gradvf[..., None]
    A = np.sum(A, axis=0)
    Ainv = np.linalg.inv(A)
    gphi = np.matmul(Ainv, R)
    return -gphi
