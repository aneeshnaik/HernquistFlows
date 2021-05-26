#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 3: Flow-reconstructed Hernquist accelerations.

Created: May 2021
Author: A. P. Naik
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from os.path import exists

sys.path.append("../src")
from cbe import calc_accel_CBE
from utils import diff_DF, sample_velocities
from constants import M_sun, kpc, G
from ml import load_flow, calc_DF_ensemble


if __name__ == "__main__":

    # check if plot data exists, otherwise generate
    dfile = "fig3_data.npz"
    if not exists(dfile):

        # hernquist params and scalings
        M = 1e+10 * M_sun
        a = 5 * kpc
        u_q = 10 * a
        u_p = np.sqrt(2 * G * M / a)

        # set up grid of (real space) points along x axis
        Nx = 50
        x = np.linspace(0, 20, Nx + 1)[1:] * a
        y = np.zeros_like(x)
        z = np.zeros_like(x)
        pos = np.stack((x, y, z), axis=-1)

        # exact acceleration
        a_exact = -(G * M / a**2) / (1 + x / a)**2

        # load flows
        n_flows = 20
        flows0 = []
        flows1 = []
        flows10 = []
        flowsaniso = []
        for i in range(n_flows):
            fname = f"../nflow_models/hq_iso_orig/{i}_best.pth"
            flows0.append(load_flow(fname, 6, 8, 64))
            fname = f"../nflow_models/hq_iso_1pc/{i % 10}_best.pth"
            flows1.append(load_flow(fname, 6, 8, 64))
            fname = f"../nflow_models/hq_iso_10pc/{i % 10}_best.pth"
            flows10.append(load_flow(fname, 6, 8, 64))
            fname = f"../nflow_models/hq_aniso_orig/{i}_best.pth"
            flowsaniso.append(load_flow(fname, 6, 8, 64))

        # loop over spatial points, get accel at each point
        a_iso0 = np.zeros_like(a_exact)
        a_iso1 = np.zeros_like(a_exact)
        a_iso10 = np.zeros_like(a_exact)
        a_aniso = np.zeros_like(a_exact)
        Nv = 1000
        v_esc = np.sqrt(2 * G * M / (x + a))
        for i in tqdm(range(Nx)):

            # set up grid of phase space points
            ve = v_esc[i]
            p = sample_velocities(Nv, v_max=0.9 * ve, pos=pos[i],
                                  vt_min=0.1 * ve, vx_min=0.2 * ve)
            q = np.tile(pos[i], [Nv, 1])

            # get acceleration from recon DF
            for j in range(4):
                model = [a_iso0, a_iso1, a_iso10, a_aniso][j]
                flows = [flows0, flows1, flows10, flowsaniso][j]
                df_args = {'u_q': u_q, 'u_p': u_p, 'flows': flows}
                f_ref = calc_DF_ensemble(q[0], p[0], **df_args)
                gradxf, gradvf = diff_DF(q, p, df_func=calc_DF_ensemble,
                                         df_args=df_args)
                gradxf /= f_ref
                gradvf /= f_ref
                model[i] = calc_accel_CBE(p, gradxf, gradvf)[0]

        # rescale data
        u = G * M / a**2
        x = x / a
        y0 = -a_exact / u
        y1 = -a_iso0 / u
        y2 = -a_iso1 / u
        y3 = -a_iso10 / u
        y4 = -a_aniso / u

        # save data file
        np.savez(dfile, x=x, y0=y0, y1=y1, y2=y2, y3=y3, y4=y4)

    else:
        # load data file
        data = np.load(dfile)
        x = data['x']
        y0 = data['y0']
        y1 = data['y1']
        y2 = data['y2']
        y3 = data['y3']
        y4 = data['y4']

    # set up figure
    fig = plt.figure(figsize=(3.3, 3.4), dpi=150)
    left = 0.16
    bottom = 0.11
    top = 0.99
    right = 0.99
    dX = (right - left)
    dY1 = (top - bottom) * 0.65
    dY2 = (top - bottom) * 0.35
    ax1 = fig.add_axes([left, bottom + dY2, dX, dY1])
    ax2 = fig.add_axes([left, bottom, dX, dY2])

    # plot settings
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 9
    plt.rcParams['ytick.labelsize'] = 8
    plt.rcParams['xtick.labelsize'] = 8
    sargs = {'zorder': 10, 's': 4}
    labels = [
        'Exact',
        r'Isotropic, $\sigma=0$',
        r'Isotropic, $\sigma = 1\%$',
        r'Isotropic, $\sigma = 10\%$',
        r'Anisotropic, $\sigma = 0$'
    ]

    # plot
    c = plt.cm.Spectral(np.linspace(0, 1, 10))
    c1 = c[0]
    c2 = c[2]
    c3 = c[7]
    c4 = c[9]
    ax1.plot(x, y0, label=labels[0], c='k', lw=2)
    ax1.scatter(x, y1, label=labels[1], c=c1[None], **sargs)
    ax1.scatter(x, y2, label=labels[2], c=c2[None], **sargs)
    ax1.scatter(x, y3, label=labels[3], c=c3[None], **sargs)
    ax1.scatter(x, y4, label=labels[4], c=c4[None], **sargs)
    ax2.plot(x, y1 / y0 - 1, c=c1, lw=2)
    ax2.plot(x, y2 / y0 - 1, c=c2, lw=2)
    ax2.plot(x, y3 / y0 - 1, c=c3, lw=2)
    ax2.plot(x, y4 / y0 - 1, c=c4, lw=2)
    xlim = ax2.get_xlim()
    ax2.plot([xlim[0], xlim[1]], [0, 0], c='k', ls='dotted')
    ax2.set_xlim(xlim)

    # scales, labels etc.
    ax1.set_yscale('log')
    ax2.set_xlabel(r"$x\ /\ a$")
    ax1.set_ylabel(r"$(\partial \Phi / \partial x)\ / \ (GM/a^2)$")
    ax2.set_ylabel(r"Model / Exact - 1")
    ax1.yaxis.set_label_coords(-0.135, 0.5)
    ax2.yaxis.set_label_coords(-0.135, 0.5)
    ax1.legend(frameon=False)
    ax1.tick_params(which='both', top=True, right=True, direction='inout')
    ax2.tick_params(which='both', top=True, right=True, direction='inout')
    ax2.set_ylim(-0.1, 0.1)

    # save figure
    fig.savefig("fig3_accs.pdf")
