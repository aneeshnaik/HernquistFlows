#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Various functions &c for machine learning.

Created: May 2021
Author: A. P. Naik
"""
import numpy as np
import copy
from time import perf_counter as time

import torch
from torch.utils.data import DataLoader as DL, TensorDataset as TDS
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau as ReduceLR

from nflows.flows import Flow
from nflows.distributions import StandardNormal
from nflows.transforms import CompositeTransform
from nflows.transforms import ReversePermutation
from nflows.transforms import MaskedAffineAutoregressiveTransform as MAAT


def setup_MAF(n_dim, n_layers, n_hidden):
    """Set up masked autoregressive flow."""
    base_dist = StandardNormal(shape=[n_dim])
    transforms = []
    for _ in range(n_layers):
        transforms.append(ReversePermutation(features=n_dim))
        transforms.append(MAAT(features=n_dim, hidden_features=n_hidden))
    transform = CompositeTransform(transforms)
    flow = Flow(transform, base_dist)
    return flow


def load_flow(state_dict, n_dim, n_layers, n_hidden):
    """Load flow from state dict."""
    flow = setup_MAF(n_dim=n_dim, n_layers=n_layers, n_hidden=n_hidden)
    flow.load_state_dict(torch.load(state_dict))
    flow.eval()
    return flow


def calc_total_loss(flow, data):
    """Compute total loss of model on data."""
    flow.eval()
    with torch.no_grad():
        loss = -flow.log_prob(inputs=data, context=None).mean()
    return loss


def train_epoch(flow, data, batch_size, optimiser, scheduler):
    """Train flow model on data for one epoch."""
    # set to training mode
    flow.train()

    # make data loader for training data
    loader = DL(TDS(data), batch_size=batch_size, shuffle=True)

    # loop over batches in data
    for batch_idx, batch in enumerate(loader):
        batch = batch[0]
        optimiser.zero_grad()
        loss = -flow.log_prob(inputs=batch, context=None).mean()
        loss.backward()
        optimiser.step()

    # compute total loss at end of epoch
    loss = calc_total_loss(flow, data)

    # step the lr scheduler
    scheduler.step(loss)

    return loss


def calc_DF_model(q, p, u_q, u_p, flow):
    """
    Evaluate DF from flow.

    Parameters
    ----------
    q : numpy array, shape (N, 3)
        DESCRIPTION.
    p : numpy array, shape (N, 3)
        DESCRIPTION.
    u_q : float
        DESCRIPTION.
    u_p : float
        DESCRIPTION.
    flow : TYPE
        DESCRIPTION.

    Returns
    -------
    f : TYPE
        DESCRIPTION.

    """
    # check shapes match
    assert q.shape == p.shape

    # check if inputs are 1D or 2D and whether np or torch
    oneD = False
    np_array = False
    if q.ndim == 1:
        oneD = True
        q = q[None]
        p = p[None]
    if type(q) == np.ndarray:
        np_array = True

    # convert inputs to torch tensors if nec.
    if np_array:
        q = torch.tensor(q)
        p = torch.tensor(p)

    # rescale units
    q = q / u_q
    p = p / u_p

    # concat
    eta = torch.cat((q, p), dim=-1).float()

    # eval f from flow
    f = flow.log_prob(eta).exp()

    # sort out format of output
    if oneD:
        f = f.item()
    elif np_array:
        f = f.detach().numpy()
    return f


def calc_DF_ensemble(q, p, u_q, u_p, flows):
    """
    Evaluate DF from ensemble of flows.

    Parameters
    ----------
    q : numpy array, shape (N, 3)
        DESCRIPTION.
    p : numpy array, shape (N, 3)
        DESCRIPTION.
    u_q : float
        DESCRIPTION.
    u_p : float
        DESCRIPTION.
    flow : TYPE
        DESCRIPTION.

    Returns
    -------
    f : TYPE
        DESCRIPTION.

    """
    # loop over flows
    N = len(flows)
    for i in range(N):
        if i == 0:
            f = calc_DF_model(q, p, u_q, u_p, flows[i]) / N
        else:
            f = f + calc_DF_model(q, p, u_q, u_p, flows[i]) / N
    return f


def train_flow(data, seed,
               n_dim=6, n_layers=8, n_hidden=64,
               lr=1e-3, lrgamma=0.5, lrpatience=5, lrmin=2e-6, lrthres=1e-6, lrcooldown=10,
               weight_decay=0, batch_size=10000, num_epochs=500,
               save_intermediate=False, save_interval=10,
               cut_early=True, patience=15):
    """
    Train normalising flow.

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    n_dim : TYPE, optional
        DESCRIPTION. The default is 6.
    n_layers : TYPE, optional
        DESCRIPTION. The default is 8.
    n_hidden : TYPE, optional
        DESCRIPTION. The default is 64.
    lr : TYPE, optional
        DESCRIPTION. The default is 2e-3.
    lrgamma : TYPE, optional
        DESCRIPTION. The default is 0.5.
    lrpatience : TYPE, optional
        DESCRIPTION. The default is 5.
    lrmin : TYPE, optional
        DESCRIPTION. The default is 1e-6.
    lrthres : TYPE, optional
        DESCRIPTION. The default is 1e-6.
    weight_decay : TYPE, optional
        DESCRIPTION. The default is 0.
    batch_size : TYPE, optional
        DESCRIPTION. The default is 10000.
    num_epochs : TYPE, optional
        DESCRIPTION. The default is 500.
    save_intermediate : TYPE, optional
        DESCRIPTION. The default is False.
    save_interval : TYPE, optional
        DESCRIPTION. The default is 10.
    cut_early : TYPE, optional
        DESCRIPTION. The default is True.
    patience : TYPE, optional
        DESCRIPTION. The default is 15.

    Returns
    -------
    None.

    """
    # set RNG seed
    torch.manual_seed(seed)

    # set up MAF
    flow = setup_MAF(n_dim=n_dim, n_layers=n_layers, n_hidden=n_hidden)

    # set up optimiser
    optimiser = Adam(flow.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLR(optimiser, factor=lrgamma, patience=lrpatience,
                         min_lr=lrmin, threshold=lrthres, cooldown=lrcooldown)

    # compute total loss pre-training
    losses = np.zeros(num_epochs + 1)
    loss = calc_total_loss(flow, data)
    losses[0] = loss
    print(f"Pre-training total loss={loss:.6e}")

    # train; loop over epochs
    best_epoch = -1
    best_loss = np.inf
    best_model = copy.deepcopy(flow)
    for epoch in range(num_epochs):

        # start stopclock
        t0 = time()

        # print start-of-epoch message
        lr = optimiser.param_groups[0]['lr']
        print(f'\nStarting epoch {epoch}; lr={lr:.4e}', flush=True)

        # train 1 epoch
        loss = train_epoch(flow, data, batch_size, optimiser, scheduler)
        losses[epoch + 1] = loss

        # if best so far, save model
        if loss < best_loss:
            best_epoch = epoch
            prevbest_loss = best_loss
            best_loss = loss
            best_model = copy.deepcopy(flow)

        # construct end-of-epoch message
        str = f"Finished epoch {epoch}; total loss={loss:.6e}"
        if epoch != 0:
            if best_epoch == epoch:
                f = (best_loss - prevbest_loss) / np.abs(prevbest_loss)
                str += f"\nBest epoch so far. Cf prev. best: dL/L={f:.2e}"
            else:
                f = (loss - best_loss) / np.abs(best_loss)
                str += f"\nBest epoch was {best_epoch}. Cf best: dL/L={f:.2e}"
        t1 = time()
        t = t1 - t0
        str += f"\nNum bad epochs: {scheduler.num_bad_epochs}"
        str += f"\nTime taken for epoch: {int(t)} seconds"
        print(str, flush=True)

        # save model if epoch = interval
        if save_intermediate and ((epoch + 1) % save_interval == 0):
            torch.save(flow.state_dict(), f'{seed}_{epoch+1}.pth')

        # stop loop if we've reached min learning rate
        if cut_early and lr <= lrmin:
            break

    # save best model and loss history
    torch.save(best_model.state_dict(), f'{seed}_best.pth')
    np.save(f'{seed}_losses', losses)
    return
