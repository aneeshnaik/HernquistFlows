#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 2: Gradients of isotropic Hernquist DF and reconstructions.

Created: May 2021
Author: A. P. Naik
"""
import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from os.path import exists


sys.path.append("../src")
from constants import M_sun, kpc, G
from hernquist import calc_DF_iso
from utils import diff_DF
from ml import load_flow, calc_DF_ensemble


# hernquist params and scalings
M = 1e+10 * M_sun
a = 5 * kpc
u_q = 10 * a
u_p = np.sqrt(2 * G * M / a)

# coord grid extents
x_lim = 3 * a
vx_lim = 1.1 * np.sqrt(2 * G * M / a)

# check if plot data exists, otherwise generate
dfile = "fig2_data.npz"
if not exists(dfile):

    # load flows
    n_flows = 20
    flows = []
    for i in range(n_flows):
        fname = f"../nflow_models/hq_iso_orig/{i}_best.pth"
        flows.append(load_flow(fname, 6, 8, 64))

    # coordinate grid for gradient panels
    N_grid = 128
    x_arr = np.linspace(-x_lim, x_lim, N_grid)
    vx_arr = np.linspace(-vx_lim, vx_lim, N_grid)
    x_grid, vx_grid = np.meshgrid(x_arr, vx_arr, indexing='ij')
    x_grid = x_grid.reshape(N_grid**2)
    vx_grid = vx_grid.reshape(N_grid**2)
    zeros = np.zeros_like(x_grid)
    pos = np.stack((x_grid, zeros, zeros), axis=-1)
    vel = np.stack((vx_grid, zeros, zeros), axis=-1)

    # reference point
    x0 = np.array([a, 0, 0])
    v0 = np.array([0.5 * np.sqrt(G * M / (2 * a)), 0, 0])

    # evaluate exact DF gradients on grid
    df_args = {'M': M, 'a': a}
    dx_ref, dvx_ref = diff_DF(x0, v0, df_func=calc_DF_iso, df_args=df_args)
    gradxf, gradvf = diff_DF(pos, vel, calc_DF_iso, df_args)
    gradxf /= dx_ref[0]
    gradvf /= dvx_ref[0]
    dfdx0 = gradxf[:, 0].reshape((N_grid, N_grid))
    dfdz0 = gradxf[:, 2].reshape((N_grid, N_grid))
    dfdvx0 = gradvf[:, 0].reshape((N_grid, N_grid))
    dfdvz0 = gradvf[:, 2].reshape((N_grid, N_grid))

    # ditto model gradients
    df_args = {'u_q': u_q, 'u_p': u_p, 'flows': flows}
    dx_ref, dvx_ref = diff_DF(x0, v0, df_func=calc_DF_ensemble, df_args=df_args)
    gradxf, gradvf = diff_DF(pos, vel, df_func=calc_DF_ensemble, df_args=df_args)
    gradxf /= dx_ref[0]
    gradvf /= dvx_ref[0]
    dfdx1 = gradxf[:, 0].reshape((N_grid, N_grid))
    dfdz1 = gradxf[:, 2].reshape((N_grid, N_grid))
    dfdvx1 = gradvf[:, 0].reshape((N_grid, N_grid))
    dfdvz1 = gradvf[:, 2].reshape((N_grid, N_grid))

    # residuals
    with np.errstate(divide='ignore', invalid='ignore'):
        resdx = np.divide((dfdx1 - dfdx0), dfdx0)
        resdvx = np.divide((dfdvx1 - dfdvx0), dfdvx0)

    np.savez(dfile, dfdx0=dfdx0, dfdvx0=dfdvx0, dfdx1=dfdx1, dfdvx1=dfdvx1,
             resdx=resdx, resdvx=resdvx)

else:

    data = np.load(dfile)
    dfdx0 = data['dfdx0']
    dfdvx0 = data['dfdvx0']
    dfdx1 = data['dfdx1']
    dfdvx1 = data['dfdvx1']
    resdx = data['resdx']
    resdvx = data['resdvx']


# set up figure
fig = plt.figure(figsize=(3.3, 4.5), dpi=150)
bottom = 0.07
top = 0.94
left = 0.135
right = 0.775
CdX = 0.035
dX = (right - left) / 2
dY = (top - bottom) / 3

# plot settings
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 9
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['xtick.labelsize'] = 8
extent = [-x_lim / a, x_lim / a, -vx_lim / u_p, vx_lim / u_p]
labels = ['Exact', 'Model', 'Residuals']
norm = SymLogNorm(1e-6, vmax=2e+6, vmin=-2e+6, base=10)
imargs1 = {
    'extent': extent, 'origin': 'lower', 'aspect': 'auto',
    'cmap': 'BrBG', 'norm': norm
}
imargs2 = {
    'extent': extent, 'origin': 'lower', 'aspect': 'auto',
    'cmap': 'Spectral_r', 'vmax': 0.75, 'vmin': -0.75
}

# loop over columns
for i in range(2):

    # add panels
    X = left + i * dX
    Y1 = bottom + 2 * dY
    Y2 = bottom + 1 * dY
    Y3 = bottom
    ax1 = fig.add_axes([X, Y1, dX, dY])
    ax2 = fig.add_axes([X, Y2, dX, dY])
    ax3 = fig.add_axes([X, Y3, dX, dY])

    # plot
    f0 = [dfdx0, dfdvx0][i]
    f1 = [dfdx1, dfdvx1][i]
    res = [resdx, resdvx][i]
    im1 = ax1.imshow(f0.T, **imargs1)
    im2 = ax2.imshow(f1.T, **imargs1)
    im3 = ax3.imshow(res.T, **imargs2)

    # labels, ticks, etc.
    t = [r'$\partial F / \partial x$', r'$\partial F / \partial v_x$'][i]
    ax1.text(0.5, 1.1, t, ha='center', transform=ax1.transAxes)
    if i == 1:
        for j in range(3):
            ax = [ax1, ax2, ax3][j]
            bbox = dict(boxstyle='round', facecolor='white', alpha=1)
            ax.text(0, 0.925, labels[j], ha='center', va='top',
                    transform=ax.transAxes, bbox=bbox, zorder=100)
    if i == 0:
        ax2.set_ylabel(r'$v_x\ /\ v_\mathrm{esc}(r=0)$')
        ax2.yaxis.set_label_coords(-0.28, 0.5, transform=ax2.transAxes)
    if i == 1:
        ax3.set_xlabel(r'$x\ /\ a$')
        ax3.xaxis.set_label_coords(0, -0.13, transform=ax3.transAxes)
    for ax in [ax1, ax2, ax3]:
        ax.tick_params(right=True, top=True, direction='inout')
        ax.set_yticks([-1., -0.5, 0., 0.5, 1.])
        if i == 1:
            ax.tick_params(labelleft=False)

# colourbars
cax1 = fig.add_axes([left + 2 * dX, bottom + dY, CdX, 2 * dY])
cax2 = fig.add_axes([left + 2 * dX, bottom, CdX, dY])
cbar1 = plt.colorbar(im1, cax=cax1)
cbar2 = plt.colorbar(im3, cax=cax2)
ticks = [
    -1e+5, -1e+3, -1e+1, -1e-1, -1e-3, -1e-5,
    0,
    1e-5, 1e-3, 1e-1, 1e+1, 1e+3, 1e+5
]
cbar1.set_ticks(ticks)
cax1.set_ylabel(r"$\nabla F\ /\ \nabla F |_\mathrm{ref}$")
cax2.set_ylabel(r"Model / Exact - 1")

# save
fig.savefig("fig2_derivs.pdf")
