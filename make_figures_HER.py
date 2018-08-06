#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 13:13:17 2018

@author: eilers
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy.optimize as op
import time
import pickle
from astropy.table import Column, Table, join, vstack, hstack
import sys
from astropy.io import fits
from sklearn.decomposition import PCA
from astropy import units as u
from astropy.coordinates import SkyCoord
import astropy.coordinates as coord
from mpl_toolkits.mplot3d import Axes3D
import corner


# -------------------------------------------------------------------------------
# plotting settings
# -------------------------------------------------------------------------------

matplotlib.rcParams['ytick.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 14
matplotlib.rc('text', usetex=True)
fsize = 14
figsize = (8, 4.5)


# -------------------------------------------------------------------------------
# open inferred labels
# -------------------------------------------------------------------------------

N = 45787
Kfold = 2
lam = 30
name = 'N{0}_lam{1}_K{2}_parallax'.format(N, lam, Kfold)

print('loading new labels...')   
labels = Table.read('data/training_labels_new_{}.fits'.format(name), format = 'fits')    
labels.rename_column('ra_1', 'ra')
labels.rename_column('dec_1', 'dec')

# -------------------------------------------------------------------------------
# Figure 1
# -------------------------------------------------------------------------------

fig, ax = plt.subplots(1, 2, figsize = figsize)
ax[0].scatter(labels['bp_rp'], labels['J']-labels['K'], rasterized = True, s = 10, alpha = .1)
ax[1].scatter(labels['bp_rp'], labels['H']-labels['w2mpro'], rasterized = True, s = 10, alpha = .1)
ax[0].set_xlabel(r'$\rm B_P-R_p$', fontsize = fsize)
ax[1].set_xlabel(r'$\rm B_P-R_p$', fontsize = fsize)
ax[0].set_ylabel(r'$\rm J-K$', fontsize = fsize)
ax[1].set_ylabel(r'$\rm H-W_2$', fontsize = fsize)
ax[0].tick_params(axis=u'both', direction='in', which='both', right = 'on', top = 'on')
ax[1].tick_params(axis=u'both', direction='in', which='both', right = 'on', top = 'on')
#plt.subplots_adjust(wspace = .1)
plt.tight_layout()
plt.savefig('paper/parent_sample.pdf')

# -------------------------------------------------------------------------------
# Figure 2
# -------------------------------------------------------------------------------

cut_vis = labels['visibility_periods_used'] >= 8    
cut_par = labels['parallax_error'] < 0.1            
cut_cal = (labels['astrometric_chi2_al'] / np.sqrt(labels['astrometric_n_good_obs_al']-5)) <= 35         
train = cut_vis * cut_par * cut_cal  
best = train * (labels['parallax_over_error'] >= 20)

fig, ax = plt.subplots(1, 2, figsize = figsize)
ax[0].scatter(labels[train]['bp_rp'], labels[train]['J']-labels[train]['K'], rasterized = True, s = 10, alpha = .1)
ax[1].scatter(labels[train]['bp_rp'], labels[train]['H']-labels[train]['w2mpro'], rasterized = True, s = 10, alpha = .1)
ax[0].set_xlabel(r'$\rm B_P-R_p$', fontsize = fsize)
ax[1].set_xlabel(r'$\rm B_P-R_p$', fontsize = fsize)
ax[0].set_ylabel(r'$\rm J-K$', fontsize = fsize)
ax[1].set_ylabel(r'$\rm H-W_2$', fontsize = fsize)
ax[0].tick_params(axis=u'both', direction='in', which='both', right = 'on', top = 'on')
ax[1].tick_params(axis=u'both', direction='in', which='both', right = 'on', top = 'on')
#plt.subplots_adjust(wspace = .1)
plt.tight_layout()
plt.savefig('paper/training_sample.pdf')

# -------------------------------------------------------------------------------
# Figure 3 (parallax vs. parallax for training and Gaia excellent, colored by SNR)
# -------------------------------------------------------------------------------

fig, ax = plt.subplots(1, 2, figsize = figsize, sharey = True)
sc = ax[0].scatter(labels['parallax'][train], labels['spec_parallax'][train], c = labels['SNR'][train], cmap = 'viridis_r', s = 10, vmin = 50, vmax = 1000, rasterized = True)
ax[1].scatter(labels['parallax'][best], labels['spec_parallax'][best], c = labels['SNR'][best], cmap = 'viridis_r', s = 10, vmin = 50, vmax = 1000, rasterized = True)
ax[0].set_xlabel(r'$\varpi^{(a)}$', fontsize = fsize)
ax[1].set_xlabel(r'$\varpi^{(a)}$', fontsize = fsize)
ax[0].set_ylabel(r'$\varpi^{(sp)}$', fontsize = fsize)
ax[0].tick_params(axis=u'both', direction='in', which='both', right = 'on', top = 'on')
ax[1].tick_params(axis=u'both', direction='in', which='both', right = 'on', top = 'on')
ax[0].set_aspect('equal')
ax[1].set_aspect('equal')
ax[0].plot(np.arange(-1, 3), np.arange(-1, 3), color = '0.6', linestyle = ':')
ax[1].plot(np.arange(-1, 3), np.arange(-1, 3), color = '0.6', linestyle = ':')
ax[0].set_xlim(-.35, 2)
ax[1].set_xlim(-.35, 2)
ax[0].set_ylim(-.35, 2)
ax[1].set_ylim(-.35, 2)
ax[0].set_title('training set', fontsize = fsize)
ax[1].set_title(r'$\varpi^{(a)}/\sigma_{\varpi^{(a)}} \geq 20$', fontsize = fsize)
fig.subplots_adjust(right = 0.8)
cbar_ax = fig.add_axes([1, 0.1, 0.03, 0.8])
cb = fig.colorbar(sc, cax=cbar_ax)
cb.set_label(r'$\rm S/N$', fontsize = fsize)
plt.tight_layout()
plt.savefig('paper/residuals.pdf')

# -------------------------------------------------------------------------------
# Figure 4 (parallax vs. parallax colored by logg, Teff, [Fe/H], H-W2)
# -------------------------------------------------------------------------------

fig, ax = plt.subplots(1, 4, figsize = (16, 4.5))
sc = ax[0].scatter(labels['parallax'][train], labels['spec_parallax'][train], c = labels['TEFF'][train], cmap = 'viridis_r', s = 10, vmin = 3600, vmax = 4800, rasterized = True)
cb = fig.colorbar(sc, ax = ax[0], shrink = .8)
cb.set_label(r'$T_{\rm eff}$', fontsize = fsize)
sc = ax[1].scatter(labels['parallax'][train], labels['spec_parallax'][train], c = labels['LOGG'][train], cmap = 'viridis_r', s = 10, vmin = 0.1, vmax = 2.2, rasterized = True)
cb = fig.colorbar(sc, ax = ax[1], shrink = .8)
cb.set_label(r'$\log g$', fontsize = fsize)
sc = ax[2].scatter(labels['parallax'][train], labels['spec_parallax'][train], c = labels['FE_H'][train], cmap = 'viridis_r', s = 10, vmin = -1, vmax = .5, rasterized = True)
cb = fig.colorbar(sc, ax = ax[2], shrink = .8)
cb.set_label(r'$\rm [Fe/H]$', fontsize = fsize)
sc = ax[3].scatter(labels['parallax'][train], labels['spec_parallax'][train], c = labels['H'][train] - labels['w2mpro'][train], cmap = 'viridis_r', s = 10, vmin = 0., vmax = 0.5, rasterized = True)
cb = fig.colorbar(sc, ax = ax[3], shrink = .8)
cb.set_label(r'$\rm H-W_{2}$', fontsize = fsize)
ax[0].plot(np.arange(-1, 3), np.arange(-1, 3), color = '0.6', linestyle = ':')
ax[1].plot(np.arange(-1, 3), np.arange(-1, 3), color = '0.6', linestyle = ':')
ax[2].plot(np.arange(-1, 3), np.arange(-1, 3), color = '0.6', linestyle = ':')
ax[3].plot(np.arange(-1, 3), np.arange(-1, 3), color = '0.6', linestyle = ':')
ax[0].tick_params(axis=u'both', direction='in', which='both', right = 'on', top = 'on')
ax[1].tick_params(axis=u'both', direction='in', which='both', right = 'on', top = 'on')
ax[2].tick_params(axis=u'both', direction='in', which='both', right = 'on', top = 'on')
ax[3].tick_params(axis=u'both', direction='in', which='both', right = 'on', top = 'on')
ax[0].set_aspect('equal')
ax[1].set_aspect('equal')
ax[2].set_aspect('equal')
ax[3].set_aspect('equal')
ax[0].set_xlim(-.35, 2)
ax[1].set_xlim(-.35, 2)
ax[2].set_xlim(-.35, 2)
ax[3].set_xlim(-.35, 2)
ax[0].set_ylim(-.35, 2)
ax[1].set_ylim(-.35, 2)
ax[2].set_ylim(-.35, 2)
ax[3].set_ylim(-.35, 2)
plt.tight_layout()
plt.savefig('paper/residuals_training.pdf')

# -------------------------------------------------------------------------------
# Figure 5 (12 stellar clusters, fairly narrow bins)
# -------------------------------------------------------------------------------

# run test_distance.py

# -------------------------------------------------------------------------------
# Figure 6 (plot res.x vs. lambda for A and B model, linear (!) Cannon: logg output)
# -------------------------------------------------------------------------------

# linear Cannon with logg, Teff, [Fe/H] for A or B model --> overplot on res.x is some sensible way (partial wavelength range?)!

f = open('optimization/opt_results_0_{}.pickle'.format(name), 'r')
res1 = pickle.load(f1)
f1.close() 

fig, ax = plt.subplots(2, 1, figsize = figsize)

plt.savefig('paper/coefficients.pdf')


# -------------------------------------------------------------------------------
# Figure 7 (map of kinematics & abundances)
# -------------------------------------------------------------------------------

# 6 degree wegde in z
# patch colored by metallicty, opacity proportional to number of stars
# superimposed arrow color coded by v_z, opacity also proportional to number of stars

fig, ax = plt.subplots(1, 2, figsize = figsize)



plt.savefig('paper/map.pdf')


# -------------------------------------------------------------------------------'''






