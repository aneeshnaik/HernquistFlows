#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 1: Isotropic Hernquist DF.

Created: May 2021
Author: A. P. Naik
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import copy
from os.path import exists

sys.path.append("../src")
from constants import M_sun, kpc, G, pi
from hernquist import calc_DF_iso
from ml import load_flow, calc_DF_ensemble


def get_f_exact(rgrid, vgrid, M, a):
    """Get exact Hernquist DF evaluated on r/v grids."""
    # deproject grids into 6D
    N_bins = rgrid.shape[0]
    dr = np.diff(rgrid[0, :], axis=0)[0]
    dv = np.diff(vgrid[:, 0], axis=0)[0]
    x = rgrid.reshape(N_bins**2)
    vx = vgrid.reshape(N_bins**2)
    zeroes = np.zeros_like(x)
    q = np.stack((x, zeroes, zeroes), axis=-1)
    p = np.stack((vx, zeroes, zeroes), axis=-1)

    # evaluate DF
    f_exact = calc_DF_iso(q, p, M, a).reshape(N_bins, N_bins)

    # renormalise; expected fraction of particles in each bin
    f_exact = 16 * pi**2 * f_exact * rgrid**2 * vgrid**2 * dr * dv
    return f_exact


def get_f_data(r_bin_edges, v_bin_edges):
    """Get mock Hernquist sample histogrammed into r v grids."""
    # load data
    data = np.load('../data/hq_iso_orig.npz')
    pos = data['pos']
    vel = data['vel']

    # get histogram
    r = np.linalg.norm(pos, axis=-1)
    v = np.linalg.norm(vel, axis=-1)
    bins = [r_bin_edges / kpc, v_bin_edges / 1000]
    H, _, _ = np.histogram2d(r / kpc, v / 1000, bins=bins)
    f_data = H.T / 1e+6
    return f_data


def get_f_model(rgrid, vgrid, M, a):
    """Get reconstructed Hernquist DF evaluated on r/v grids."""
    # deproject grids into 6D
    N_bins = rgrid.shape[0]
    dr = np.diff(rgrid[0, :], axis=0)[0]
    dv = np.diff(vgrid[:, 0], axis=0)[0]
    x = rgrid.reshape(N_bins**2)
    vx = vgrid.reshape(N_bins**2)
    zeroes = np.zeros_like(x)
    q = np.stack((x, zeroes, zeroes), axis=-1)
    p = np.stack((vx, zeroes, zeroes), axis=-1)

    # units
    u_q = 10 * a
    u_p = np.sqrt(2 * G * M / a)
    u_f = u_q**3 * u_p**3

    # load flows
    n_flows = 10
    flows = []
    for i in range(n_flows):
        fname = f"../nflow_models/hq_iso_orig/{i}_best.pth"
        flows.append(load_flow(fname, 6, 8, 64))

    # evaluate DF
    f_model = calc_DF_ensemble(q, p, u_q, u_p, flows).reshape(N_bins, N_bins)

    # renormalise; expected fraction of particles in each bin
    f_model = 16 * pi**2 * f_model * rgrid**2 * vgrid**2 * dr * dv / u_f
    return f_model


if __name__ == '__main__':

    # Hernquist params and scaling units
    M = 1e+10 * M_sun
    a = 5 * kpc
    u_pos = 10 * a
    u_vel = np.sqrt(2 * G * M / a)

    # grid dims
    r_max = 5.5 * a
    v_max = np.sqrt(2 * G * M / a)
    N_bins = 128

    # check if plot data exists, otherwise generate
    dfile = "fig1_data.npz"
    if not exists(dfile):

        # define r/v bins in which to evaluate DF
        r_bin_edges = np.linspace(0, r_max, N_bins + 1)
        v_bin_edges = np.linspace(0, v_max, N_bins + 1)
        r_cen = 0.5 * (r_bin_edges[1:] + r_bin_edges[:-1])
        v_cen = 0.5 * (v_bin_edges[1:] + v_bin_edges[:-1])
        rgrid, vgrid = np.meshgrid(r_cen, v_cen)
        dr = r_max / N_bins
        dv = v_max / N_bins

        # f_ref
        x0 = np.array([a, 0, 0])
        v0 = np.array([v_max / 4, 0, 0])
        f_ref = calc_DF_iso(x0, v0, M, a)
        f_ref = 16 * pi**2 * f_ref * a**2 * (v_max / 4)**2 * dr * dv

        # get various DFs
        f_exact = get_f_exact(rgrid, vgrid, M, a) / f_ref
        f_data = get_f_data(r_bin_edges, v_bin_edges) / f_ref
        f_model = get_f_model(rgrid, vgrid, M, a) / f_ref

        # calculate residuals
        with np.errstate(divide='ignore', invalid='ignore'):
            res = np.divide((f_model - f_exact), f_exact)

        # save data file
        np.savez(
            dfile, f_exact=f_exact, f_data=f_data, f_model=f_model, res=res
        )
    else:
        # load data file
        data = np.load(dfile)
        f_exact = data['f_exact']
        f_model = data['f_model']
        f_data = data['f_data']
        res = data['res']

    # set up figure
    fig = plt.figure(figsize=(6.9, 3), dpi=150)
    left = 0.065
    right = 0.98
    bottom = 0.125
    top = 0.83
    dX = (right - left) / 4
    dY = (top - bottom)
    CdY = 0.05

    # plot settings
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 9
    plt.rcParams['ytick.labelsize'] = 8
    plt.rcParams['xtick.labelsize'] = 8
    labels = ['Exact', 'Data', 'Model', 'Residuals']
    cmap = copy.copy(plt.cm.bone)
    cmap.set_under('white')
    vmin = 0.00001
    vmax = 1.3
    extent = [0, r_max / a, 0, 1]
    iargs1 = {'origin': 'lower', 'cmap': cmap, 'vmin': vmin, 'vmax': vmax,
              'extent': extent, 'aspect': 'auto'}
    iargs2 = {'origin': 'lower', 'extent': extent, 'vmin': -0.75, 'vmax': 0.75,
              'cmap': 'Spectral_r', 'aspect': 'auto'}

    # loop over panels
    for i in range(4):

        # set up axes
        ax = fig.add_axes([left + i * dX, top - dY, dX, dY])

        # get relevant DF
        if i == 0:
            f = np.copy(f_exact)
        elif i == 1:
            f = np.copy(f_data)
        elif i == 2:
            f = np.copy(f_model)
        else:
            f = np.copy(res)

        # plot DF
        if i == 3:
            im1 = ax.imshow(res, **iargs2)
        else:
            im0 = ax.imshow(f, **iargs1)

        # text
        ax.text(0.97, 0.96, labels[i], ha='right', va='top',
                transform=ax.transAxes)

        # ticks, axis labels etc.
        ax.tick_params(top=True, right=True, direction='inout')
        if i == 0:
            ax.set_ylabel(r"$v\ /\ v_\mathrm{esc}(r=0)$")
        else:
            ax.tick_params(labelleft=False)
        if i == 2:
            ax.set_xlabel(r"$r\ /\ a$")
            ax.xaxis.set_label_coords(0, -0.1)

    # colourbars
    cax0 = fig.add_axes([left, top, 3 * dX, CdY])
    cax1 = fig.add_axes([left + 3 * dX, top, dX, CdY])
    plt.colorbar(im0, cax=cax0, orientation='horizontal')
    plt.colorbar(im1, cax=cax1, orientation='horizontal')
    cax0.set_xlabel(r"$F / F_\mathrm{ref}$")
    cax1.set_xlabel(r"Model / Exact - 1")
    for cax in [cax0, cax1]:
        cax.xaxis.set_ticks_position('top')
        cax.xaxis.set_label_position('top')

    # save
    fig.savefig("fig1_iso.pdf")
