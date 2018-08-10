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

cut_jk = (labels['J'] - labels['K']) < (0.4 + 0.45 * labels['bp_rp'])
cut_hw2 = (labels['H'] - labels['w2mpro']) > -0.05
labels = labels[cut_jk * cut_hw2]

# -------------------------------------------------------------------------------
# Figure 1
# -------------------------------------------------------------------------------

# line parameters
x = np.linspace(0, 8, 10)
y1 = 0.4 + 0.45 * x
y2 = -0.05 + 0. * x

bprplim = (0.5, 6.0)
jklim = (0.2, 2.7)
hw2lim = (-0.1, 1.3)

cm = 'viridis_r'
fig, ax = plt.subplots(1, 2, figsize = figsize)
ax[0].scatter(labels['bp_rp'], labels['J']-labels['K'], c = labels['LOGG'], cmap = cm, vmin = 0, vmax = 2.2, rasterized = True, s = 10, alpha = .5)
sc = ax[1].scatter(labels['bp_rp'], labels['H']-labels['w2mpro'], c = labels['LOGG'], cmap = cm, vmin = 0, vmax = 2.2, rasterized = True, s = 10, alpha = .5)
ax[0].set_xlabel(r'$\rm B_P-R_p$', fontsize = fsize)
ax[1].set_xlabel(r'$\rm B_P-R_p$', fontsize = fsize)
ax[0].set_ylabel(r'$\rm J-K$', fontsize = fsize)
ax[1].set_ylabel(r'$\rm H-W_2$', fontsize = fsize)
ax[0].tick_params(axis=u'both', direction='in', which='both', right = 'on', top = 'on')
ax[1].tick_params(axis=u'both', direction='in', which='both', right = 'on', top = 'on')
fig.subplots_adjust(right = 0.8)
cbar_ax = fig.add_axes([1, 0.15, 0.03, 0.82])
cb = fig.colorbar(sc, cax=cbar_ax)
cb.set_label(r'$\log g$', fontsize = fsize)
plt.tight_layout()
ax[0].set_xlim(bprplim)
ax[0].set_ylim(jklim)
ax[1].set_xlim(bprplim)
ax[1].set_ylim(hw2lim)
ax[0].plot(x, y1, linestyle= '-', color = '0.6')
ax[1].plot(x, y2, linestyle= '-', color = '0.6')
plt.savefig('paper/parent_sample.pdf', pad_inches=.2, bbox_inches = 'tight')

# -------------------------------------------------------------------------------
# Figure 2
# -------------------------------------------------------------------------------

cut_vis = labels['visibility_periods_used'] >= 8    
cut_par = labels['parallax_error'] < 0.1            
cut_cal = (labels['astrometric_chi2_al'] / np.sqrt(labels['astrometric_n_good_obs_al']-5)) <= 35         
train = cut_vis * cut_par * cut_cal  
best = train * (labels['parallax_over_error'] >= 20)

fig, ax = plt.subplots(1, 2, figsize = figsize)
ax[0].scatter(labels[train]['bp_rp'], labels[train]['J']-labels[train]['K'], c = labels[train]['LOGG'], cmap = cm, vmin = 0, vmax = 2.2, rasterized = True, s = 10, alpha = .5)
ax[1].scatter(labels[train]['bp_rp'], labels[train]['H']-labels[train]['w2mpro'], c = labels[train]['LOGG'], cmap = cm, vmin = 0, vmax = 2.2, rasterized = True, s = 10, alpha = .5)
ax[0].set_xlabel(r'$\rm B_P-R_p$', fontsize = fsize)
ax[1].set_xlabel(r'$\rm B_P-R_p$', fontsize = fsize)
ax[0].set_ylabel(r'$\rm J-K$', fontsize = fsize)
ax[1].set_ylabel(r'$\rm H-W_2$', fontsize = fsize)
ax[0].tick_params(axis=u'both', direction='in', which='both', right = 'on', top = 'on')
ax[1].tick_params(axis=u'both', direction='in', which='both', right = 'on', top = 'on')
fig.subplots_adjust(right = 0.8)
cbar_ax = fig.add_axes([1, 0.15, 0.03, 0.82])
cb = fig.colorbar(sc, cax=cbar_ax)
cb.set_label(r'$\log g$', fontsize = fsize)
plt.tight_layout()
ax[0].plot(x, y1, linestyle= '-', color = '0.6')
ax[1].plot(x, y2, linestyle= '-', color = '0.6')
ax[0].set_xlim(bprplim)
ax[0].set_ylim(jklim)
ax[1].set_xlim(bprplim)
ax[1].set_ylim(hw2lim)
plt.savefig('paper/training_sample.pdf', pad_inches=.2, bbox_inches = 'tight')

# -------------------------------------------------------------------------------
# Figure 3 (parallax vs. parallax for training and Gaia excellent, colored by SNR)
# -------------------------------------------------------------------------------

fig, ax = plt.subplots(1, 2, figsize = figsize, sharey = True)
sc = ax[0].scatter(labels['parallax'][train], labels['spec_parallax'][train], c = labels['SNR'][train], cmap = 'viridis_r', s = 10, vmin = 50, vmax = 1000, rasterized = True)
ax[1].scatter(labels['parallax'][best], labels['spec_parallax'][best], c = labels['SNR'][best], cmap = 'viridis_r', s = 10, vmin = 50, vmax = 1000, rasterized = True)
ax[0].set_xlabel(r'$\varpi^{\rm (a)}$', fontsize = fsize)
ax[1].set_xlabel(r'$\varpi^{\rm (a)}$', fontsize = fsize)
ax[0].set_ylabel(r'$\varpi^{\rm (sp)}$', fontsize = fsize)
ax[0].tick_params(axis=u'both', direction='in', which='both', right = 'on', top = 'on')
ax[1].tick_params(axis=u'both', direction='in', which='both', right = 'on', top = 'on')
#ax[0].set_aspect('equal')
#ax[1].set_aspect('equal')
ax[0].plot(np.arange(-1, 3), np.arange(-1, 3), color = '0.6', linestyle = ':')
ax[1].plot(np.arange(-1, 3), np.arange(-1, 3), color = '0.6', linestyle = ':')
ax[0].set_xlim(-.35, 2)
ax[1].set_xlim(-.35, 2)
ax[0].set_ylim(-.35, 2)
ax[1].set_ylim(-.35, 2)
ax[0].set_title('training set', fontsize = fsize)
ax[1].set_title(r'$\varpi^{\rm (a)}/\sigma_{\varpi^{\rm (a)}} \geq 20$', fontsize = fsize)
fig.subplots_adjust(right = 0.8)
cbar_ax = fig.add_axes([1, 0.15, 0.03, 0.75])
cb = fig.colorbar(sc, cax=cbar_ax)
cb.set_label(r'$\rm S/N$', fontsize = fsize)
plt.tight_layout()
plt.savefig('paper/residuals.pdf', pad_inches=.2, bbox_inches = 'tight')

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

f1 = open('optimization/opt_results_0_{}.pickle'.format(name), 'rb')
res1 = pickle.load(f1)
f1.close() 
f2 = open('optimization/opt_results_1_{}.pickle'.format(name), 'rb')
res2 = pickle.load(f2)
f2.close() 

hdulist = fits.open('./data/spectra/apStar-t9-2M00000002+7417074.fits')
header = hdulist[1].header
flux = hdulist[1].data[0]
start_wl = header['CRVAL1']
diff_wl = header['CDELT1']
val = diff_wl * (len(flux)) + start_wl
wl_full_log = np.arange(start_wl, val, diff_wl)
wl = [10**aval for aval in wl_full_log]

hdu = fits.open('data/all_flux_norm_parent.fits')
fluxes = hdu[0].data
gaps = (np.sum(fluxes.T, axis = 0)) == float(fluxes.T.shape[0])

fig, ax = plt.subplots(2, 1, figsize = figsize, sharex = True, sharey = True)
ax[0].plot(np.array(wl)[~gaps], res1.x[9:], drawstyle = 'steps-mid', lw = .8, color = 'k')
ax[1].plot(np.array(wl)[~gaps], res2.x[9:], drawstyle = 'steps-mid', lw = .8, color = 'k')
ax[1].set_xlabel(r'$\lambda~\rm[{\AA}]$', fontsize = 14)
ax[0].set_xlim(min(np.array(wl)[~gaps]), max(np.array(wl)[~gaps]))
ax[0].set_ylim(-0.8, 0.8)
plt.savefig('paper/coefficients.pdf')

# -------------------------------------------------------------------------------
# Figure 7 (map of kinematics & abundances)
# -------------------------------------------------------------------------------

# 6 degree wegde in z
# patch colored by metallicty, opacity proportional to number of stars
# superimposed arrow color coded by v_z, opacity also proportional to number of stars

# run rotation_curve_uncertainties.py


# -------------------------------------------------------------------------------'''

xx = res1.x
sigma = np.zeros_like(xx)
sigma[9:] = 0.01
sigma[1:9] = 0.05
ln_par_err = np.sqrt(np.sum(sigma * sigma * xx * xx))









