#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 4: Anisotropic Hernquist DF

Created: May 2021
Author: A. P. Naik
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import copy
from os.path import exists

sys.path.append('../src')
from hernquist import calc_DF_aniso
from constants import M_sun, kpc, G
from ml import load_flow, calc_DF_ensemble
from scipy.integrate import trapezoid as trapz


def normalise_DF(f, vR, vT):
    """Normalise 2D PDF in vR-vT space, defined by 1D arrays vR vT."""
    N = np.size(vR)
    norm = trapz(np.array([trapz(f[:, i], vR) for i in range(N)]), vT)
    f = f / norm
    return f


if __name__ == "__main__":

    # Hernquist params and scalings
    M = 1e+10 * M_sun
    a = 5 * kpc
    u_q = 10 * a
    u_p = np.sqrt(2 * G * M / a)
    v_esc = np.sqrt(G * M / a)

    # grid size
    N_bins = 64
    v_lim = 1.1 * v_esc

    # check if plot data exists, otherwise generate
    dfile = "fig4_data.npz"
    if not exists(dfile):

        # fixed position for velocity grids
        r = a

        # set up velocity grid
        vR_edges = np.linspace(-v_lim, v_lim, N_bins + 1)
        vT_edges = np.linspace(0, v_lim, N_bins + 1)
        vR_cen = 0.5 * (vR_edges[1:] + vR_edges[:-1])
        vT_cen = 0.5 * (vT_edges[1:] + vT_edges[:-1])
        dvR = np.diff(vR_cen)[0]
        dvT = np.diff(vT_cen)[0]

        # position and velocity arrays to feed to DF functions
        vx = vR_cen
        vz = vT_cen
        vx_grid, vz_grid = np.meshgrid(vx, vz, indexing='ij')
        vy_grid = np.zeros_like(vx_grid)
        vel = np.stack((vx_grid, vy_grid, vz_grid), axis=-1)
        vel = vel.reshape(N_bins**2, 3)
        pos = np.array([r, 0, 0])
        pos = np.tile(pos[None], reps=[N_bins**2, 1])

        # exact DF
        f_exact = calc_DF_aniso(pos, vel, M, a)
        f_exact = f_exact.reshape(N_bins, N_bins) * vT_cen
        f_exact = normalise_DF(f_exact, vR_cen, vT_cen)

        # load flows
        n_flows = 30
        flows = []
        for j in range(n_flows):
            fname = f"../nflow_models/hq_aniso_orig/{j}_best.pth"
            flows.append(load_flow(fname, 6, 8, 64))

        # get f_model
        df_args = {'u_q': u_q, 'u_p': u_p, 'flows': flows}
        f_model = calc_DF_ensemble(pos, vel, u_q=u_q, u_p=u_p, flows=flows)
        f_model = f_model.reshape(N_bins, N_bins) * vT_cen
        f_model = normalise_DF(f_model, vR_cen, vT_cen)

        # load data
        data = np.load("../data/hq_aniso_orig.npz")
        pos = data['pos']
        vel = data['vel']

        # derive vr and vt from data
        r = np.linalg.norm(pos, axis=-1)
        rhat = pos / r[:, None]
        vR = np.sum(vel * rhat, axis=-1)
        vT = np.linalg.norm(vel - vR[:, None] * rhat, axis=-1)

        # only keep data within small radial slice
        inds = np.abs(r - a) < 0.5 * kpc
        vR = vR[inds]
        vT = vT[inds]

        # get f_data
        bins = [vR_edges, vT_edges]
        f_data = np.histogram2d(vR, vT, bins=bins, density=True)[0]

        # reference value
        f_ref = f_exact[N_bins // 2, N_bins // 2]
        f_exact /= f_ref
        f_model /= f_ref
        f_data /= f_ref

        # calculate residuals
        with np.errstate(divide='ignore', invalid='ignore'):
            res = np.divide((f_model - f_exact), f_exact)

        # save data file
        np.savez(
            dfile, f_exact=f_exact, f_model=f_model, f_data=f_data, res=res
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
    right = 0.985
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
    vmin = 0.001
    vmax = 2.2
    extent = [-v_lim / v_esc, v_lim / v_esc, 0, v_lim / v_esc]
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
            im1 = ax.imshow(res.T, **iargs2)
        else:
            im0 = ax.imshow(f.T, **iargs1)

        # text
        ax.text(0.97, 0.96, labels[i], ha='right', va='top',
                transform=ax.transAxes)

        # ticks, axis labels etc.
        ax.tick_params(top=True, right=True, direction='inout')
        if i == 0:
            ax.set_ylabel(r"$v_t\ /\ v_\mathrm{esc}(r=a)$")
        else:
            ax.tick_params(labelleft=False)
        if i == 2:
            ax.set_xlabel(r"$v_r\ /\ v_\mathrm{esc}(r=a)$")
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

    # save figure
    fig.savefig("fig4_aniso.pdf")
