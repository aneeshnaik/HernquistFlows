#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sample 10^6 particles from isotropic Hernquist DF.

Created: February 2021
Author: A. P. Naik
"""
import sys
from emcee import EnsembleSampler as Sampler
import numpy as np

sys.path.append("../src")
from constants import G, M_sun, kpc
from hernquist import calc_DF_iso


def hernquist_df_iso(theta, M, a):
    """
    Evaluate isotropic Hernquist distribution function.

    Calculates log-probability of given phase space position theta. Functional
    form is Eq. (43) in Naik et al., (2020).

    Parameters
    ----------
    theta: array-like, shape (6,)
        Array containing phase space position (x, y, z, vx, vy, vz). UNITS:
        metres and metres/second for positions/velocities respectively.
    M: float
        Total mass of Hernquist blob. UNITS: kilograms.
    a: float
        Scale radius of Hernquist blob. UNITS: metres.

    Returns
    -------
    lnf: float
        Unnormalised ln-probability associated with phase space position.
    """
    q = theta[:3]
    p = theta[3:]

    f = calc_DF_iso(q, p, M, a)
    if f == 0:
        return -1e+20
    else:
        lnf = np.log(f)
        return lnf


def sample(N, M, a):
    """
    Sample N particles from isotropic Hernquist distribution function.

    Takes either isotropic or anisotropic DF, parametrised by mass M and scale
    radius a.

    Sampler uses 50 MCMC walkers, each taking N iterations (after burn-in).
    These samples are then thinned by an interval of 50, giving N
    quasi-independent samples.

    Parameters
    ----------
    N: int
        Number of particles to sample. Note: this needs to be a multiple of 50.
    M: float
        Total mass of Hernquist blob. UNITS: kilograms.
    a: float
        Scale radius of Hernquist blob. UNITS: metres.

    Returns
    -------
    pos: (N, 3) array
        Positions of sampled particles, in Cartesian coordinates. UNITS:
        metres.
    vel: (N, 3) array
        Velocities of sampled particles, in Cartesian coordinates. UNITS:
        metres/second.
    """

    # set up sampler
    df_function = hernquist_df_iso
    nwalkers, ndim = 100, 6
    n_burnin = 1000
    assert N % nwalkers == 0
    n_iter = N
    s = Sampler(nwalkers, ndim, df_function, args=[M, a])

    # set up initial walker positions
    v_sig = 0.5 * np.sqrt(G * M / a) / np.sqrt(3)
    sig = np.array([0.3 * a, 0.3 * a, 0.3 * a, v_sig, v_sig, v_sig])
    p0 = -sig + 2 * sig * np.random.rand(nwalkers, ndim)

    # burn in
    print("\nBurning in...", flush=True)
    s.run_mcmc(p0, n_burnin, progress=True)

    # take final sample
    p0 = s.chain[:, -1, :]
    s.reset()
    print("\n\nTaking final sample...", flush=True)
    s.run_mcmc(p0, n_iter, progress=True, thin=100)
    pos = s.flatchain[:, :3]
    vel = s.flatchain[:, 3:]

    return pos, vel


def downsample(pos, vel, a, x_truncation):
    """Downsample from truncated Hernquist."""
    r = np.linalg.norm(pos, axis=-1)
    allowed = np.where(r < x_truncation * a)[0]
    inds = np.random.choice(allowed, size=N)

    pos = pos[inds]
    vel = vel[inds]
    return pos, vel


if __name__ == '__main__':

    M = 1e+10 * M_sun
    a = 5 * kpc
    N = 1000000

    pos, vel = sample(2 * N, M, a)
    pos, vel = downsample(pos, vel, a, x_truncation=200)

    np.savez("hq_iso_orig", pos=pos, vel=vel)
