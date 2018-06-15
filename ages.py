#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 10:34:35 2018

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
fsize = 14

# -------------------------------------------------------------------------------
# open inferred labels
# -------------------------------------------------------------------------------

N = 45787
Kfold = 4
lam = 100
name = 'N{0}_lam{1}_K{2}'.format(N, lam, Kfold)

print('loading new labels...')   
labels = Table.read('data/training_labels_new_{}_2.fits'.format(name), format = 'fits')    
labels.rename_column('ra_1', 'ra')
labels.rename_column('dec_1', 'dec')

# -------------------------------------------------------------------------------
# new cross match with Melissa's ages! (or Sanders+ 2018)
# -------------------------------------------------------------------------------           

hdu = fits.open('data/labels_match_ages_melissa.fits')
melissa_labels = hdu[1].data
melissa_labels = Table(melissa_labels)
melissa_labels.rename_column('col2', 'ages')

# -------------------------------------------------------------------------------
# including ages
# -------------------------------------------------------------------------------           

'''spec_par = melissa_labels['spec_parallax'] * u.mas
distance = spec_par.to(u.parsec, equivalencies = u.parallax())

cs = coord.ICRS(ra = melissa_labels['ra_1'] * u.degree, 
                dec = melissa_labels['dec_1'] * u.degree, 
                distance = distance, 
                pm_ra_cosdec = melissa_labels['pmra_1a'] * u.mas/u.yr, 
                pm_dec = melissa_labels['pmdec_1a'] * u.mas/u.yr, 
                radial_velocity = melissa_labels['VHELIO_AVG'] *u.km/u.s)


gc = coord.Galactocentric(galcen_distance = X_GC_sun_kpc*u.kpc,
                          galcen_v_sun = coord.CartesianDifferential([-vX_GC_sun_kms, vY_GC_sun_kms, vZ_GC_sun_kms] * u.km/u.s),
                          z_sun = Z_GC_sun_kpc*u.kpc)

galcen = cs.transform_to(gc)
xs, ys, zs = galcen.x.to(u.kpc), galcen.y.to(u.kpc), galcen.z.to(u.kpc)
vxs, vys, vzs = galcen.v_x, galcen.v_y, galcen.v_z

XS = np.vstack([xs, ys, zs, vxs, vys, vzs]).T.value
Xlimits = [[-30, 10], [-10, 30], [-20, 20], 
           [-200, 200], [-200, 200], [-200, 200]]
Xlabels = ['$x$', '$y$', '$z$', r'$v_x$', r'$v_y$', r'$v_z$']

d2d = np.sqrt(XS[:, 0] ** 2 + XS[:, 1] ** 2)
units = XS[:, 0:2] / d2d[:, None]
perps = np.zeros_like(units)
perps[:, 0] = units[:, 1]
perps[:, 1] = -units[:, 0]
vtans = np.sum(perps * XS[:, 3:5], axis=1)
R = np.sqrt(XS[:, 0] ** 2 + XS[:, 1] ** 2) * np.sign(XS[:, 0])

lims = np.array([-35, 25, -20, 20])
x1m, x2m, y1m, y2m = lims[0], lims[1], lims[2], lims[3]
binnum = 50

H1, xedge1, yedge1 = np.histogram2d(R, XS[:, 2], bins = binnum, weights = melissa_labels['ages'], range = ((x1m,x2m), (y1m,y2m)))
H2, xedge2, yedge2 = np.histogram2d(R, XS[:, 2], bins = binnum, range = ((x1m,x2m), (y1m,y2m)))
H = H1/ H2
masked_H = np.ma.array(H, mask = np.isnan(H))

fig, ax = plt.subplots(1, 1, figsize = (12, 8))
plt.imshow(masked_H.T, interpolation="nearest", aspect = 'auto', origin = 'lower', extent = (x1m, x2m, y1m, y2m), cmap = 'RdYlBu_r', vmin = 3, vmax = 13)
cb = plt.colorbar(shrink = .8)
cb.set_label(r'Age [Gyr]', fontsize = 20)
plt.tick_params(axis=u'both', direction='in', which='both')
ax.set_xlabel(r"R$_{\rm{GAL}}$ (kpc) ", fontsize = 20, labelpad = 5)
ax.set_ylabel(r"Galactic height, $z$ (kpc)", fontsize = 20)
ax.set_aspect('equal')
plt.savefig('plots/ages.pdf')

vz_cut = np.logical_and(melissa_labels['ages'] < 4, abs(melissa_labels['b']) < 2)
fig, ax = plt.subplots(1, 1, figsize = (10, 10))
plt.scatter(XS[vz_cut, 0], XS[vz_cut, 1], c = melissa_labels['ages'][vz_cut], cmap = 'jet', vmin = 3, vmax = 13, s = 10, alpha = 0.2)
cb = plt.colorbar(shrink = .82)
cb.set_label(r'Age [Gyr]', fontsize = 15)
plt.xlim(Xlimits[0])
plt.ylim(Xlimits[1])
overplot_rings()
plt.tick_params(axis=u'both', direction='in', which='both')
plt.xlabel('$x$', fontsize = 14)
plt.ylabel('$y$', fontsize = 14)
ax.set_aspect('equal')
plt.savefig('plots/xy_young.pdf')

# -------------------------------------------------------------------------------
# mono-abundance rings?
# -------------------------------------------------------------------------------           

plt.scatter(labels['FE_H'][vz_cut], labels['MG_FE'][vz_cut], alpha = .2)
box_cut = (labels['MG_FE'] > -0.1) * (labels['MG_FE'] < 0.) * \
          (labels['FE_H'] > -0.5) * (labels['FE_H'] < -0.3) * \
          (labels['AL_FE'] > -0.1) * (labels['AL_FE'] < 0.05) * \
          (labels['CO_FE'] > -0.8) * (labels['CO_FE'] < 0.2) 

fig, ax = plt.subplots(1, 1, figsize = (12, 12))
plt.scatter(XS[~box_cut, 0], XS[~box_cut, 1], c = '0.7', alpha = .2, s = 10, zorder = 0)
plt.scatter(XS[box_cut, 0], XS[box_cut, 1], c = 'k', s = 10, zorder = 10) #labels['MG_FE'][box_cut], cmap = 'RdBu_r', vmin = -.1, vmax = .3, s = 10, zorder = 10)
plt.xlim(Xlimits[0])
plt.ylim(Xlimits[1])
overplot_rings()
plt.tick_params(axis=u'both', direction='in', which='both')
plt.xlabel('$x$', fontsize = 14)
plt.ylabel('$y$', fontsize = 14)
ax.set_aspect('equal')

plt.scatter(R[box_cut], labels['MG_FE'][box_cut])
plt.scatter(R[box_cut], labels['AL_FE'][box_cut])
plt.scatter(R[box_cut], labels['FE_H'][box_cut])


# -------------------------------------------------------------------------------
# predicting radius 
# -------------------------------------------------------------------------------           

labels = Table(labels)
missing = (labels['MG_FE'] > -100) * (labels['AL_FE'] > -100) * \
          (labels['FE_H'] > -100) * (labels['N_FE'] > -100) * \
          (labels['O_FE'] > -100) * (labels['CA_FE'] > -100) * \
          (labels['MN_FE'] > -100) * (labels['SI_FE'] > -100) * \
          (labels['NA_FE'] > -100) * (labels['C_FE'] > -100) * \
          (labels['CO_FE'] > -100) * (labels['NI_FE'] > -100)

abund = labels['random_index', 'MG_FE', 'AL_FE', 'FE_H', 'N_FE', 'O_FE', 'CA_FE', 'MN_FE', 'SI_FE', 'NA_FE', 'C_FE', 'CO_FE', 'NI_FE']
abund.add_column(Column(R), name = 'R')
abund = abund[missing]

def H_func(x, y, A, lam, ivar):    
    H = 0.5 * np.dot((y - np.dot(A, x)).T, ivar * (y - np.dot(A, x))) + lam * np.sum(np.abs(x))
    dHdx = -1. * np.dot(A.T, ivar * (y - np.dot(A, x))) + lam * np.sign(x)
    return H, dHdx

def check_H_func(x, y, A, lam, ivar):
    H0, dHdx0 = H_func(x, y, A, lam, ivar)
    dx = 0.001 # magic
    for i in range(len(x)):
        x1 = 1. * x
        x1[i] += dx
        H1, foo = H_func(x1, y, A, lam, ivar)
        dHdx1 = (H1 - H0) / dx
        print(i, x[i], dHdx0[i], dHdx1, (dHdx1 - dHdx0[i]) / dHdx0[i])
    return

# design matrix
y_all = abund['R']
#yerr_all = training_labels['Q_K_ERR'] #['Q_W2_ERR']
ivar_all = np.ones_like(y_all) # ** (-2)
AT_0 = np.vstack([np.ones_like(abund['R'])])
AT_linear = np.array([np.array(abund['MG_FE']), np.array(abund['AL_FE']), np.array(abund['FE_H']), np.array(abund['N_FE']), np.array(abund['O_FE']), np.array(abund['CA_FE']), np.array(abund['MN_FE']), np.array(abund['SI_FE']), np.array(abund['NA_FE']), np.array(abund['C_FE']), np.array(abund['CO_FE']), np.array(abund['NI_FE'])])
A_all = np.vstack([AT_0, AT_linear]).T

date = 'jun8'

# cross validation
Kfold = 2
y_pred_all = np.zeros_like(y_all)
lam = 1e-3

name = 'N{0}_lam{1}_K{2}'.format(len(y_all), lam, Kfold)

for k in range(Kfold):    
    
    # hold out data
    valid = abund['random_index'] % Kfold == k
    train = np.logical_not(valid)
    y = y_all[train]
    ivar = ivar_all[train]
    A = A_all[train, :]
    N, M = A.shape
    x0 = np.zeros((M,))
                 
    # optimize H_func
    res = op.minimize(H_func, x0, args=(y, A, lam, ivar), method='L-BFGS-B', jac=True, options={'maxfun':50000}) 
    print(res)  
    
    # prediction
    y_pred = np.dot(A_all[valid, :], res.x) 
    y_pred_all[valid] = y_pred

    plt.plot(res.x)
    plt.title(r'$\lambda = {}$'.format(lam))
    plt.savefig('plots/{0}/regularization_results_{1}.pdf'.format(date, name))
    plt.close()
    
plt.scatter(y_all, y_pred_all, alpha = .2)
plt.xlim(0, 30)
plt.ylim(0, 30)
plt.plot([0, 100], [0, 100], 'k-')
plt.xlabel(r'$R_{\rm true}$', fontsize = 14)
plt.ylabel(r'$R_{\rm pred}$', fontsize = 14)
plt.savefig('plots/{0}/radius_{1}.pdf'.format(date, name))

foo = (y_pred_all > 10) * (y_pred_all < 11)
R_cut = np.zeros_like(XS[:, 1], dtype=bool)
R_cut[missing] = foo

fig, ax = plt.subplots(1, 1, figsize = (12, 12))
plt.scatter(XS[:, 0], XS[:, 1], c = '0.7', alpha = .2, s = 10, zorder = 0)
plt.scatter(XS[R_cut, 0], XS[R_cut, 1], c = 'k', s = 10, zorder = 10) #labels['MG_FE'][box_cut], cmap = 'RdBu_r', vmin = -.1, vmax = .3, s = 10, zorder = 10)
plt.xlim(Xlimits[0])
plt.ylim(Xlimits[1])
overplot_rings()
plt.tick_params(axis=u'both', direction='in', which='both')
plt.xlabel('$x$', fontsize = 14)
plt.ylabel('$y$', fontsize = 14)
ax.set_aspect('equal')

# -------------------------------------------------------------------------------
# predicting radius 
# -------------------------------------------------------------------------------           

missing = (melissa_labels['FE_H'] > -100) * (melissa_labels['AL_FE'] > -100) * \
          (melissa_labels['MG_FE'] > -100) * (melissa_labels['N_FE'] > -100) * \
          (melissa_labels['O_FE'] > -100) * (melissa_labels['CA_FE'] > -100) * \
          (melissa_labels['MN_FE'] > -100) * (melissa_labels['SI_FE'] > -100) * \
          (melissa_labels['NA_FE'] > -100) * (melissa_labels['C_FE'] > -100) * \
          (melissa_labels['CO_FE'] > -100) * (melissa_labels['NI_FE'] > -100)

abund = melissa_labels['random_index', 'MG_FE', 'AL_FE', 'FE_H', 'N_FE', 'O_FE', 'CA_FE', 'MN_FE', 'SI_FE', 'NA_FE', 'C_FE', 'CO_FE', 'NI_FE']
abund.add_column(Column(R), name = 'R')
abund = abund[missing]

# design matrix
y_all = abund['R']
#yerr_all = training_labels['Q_K_ERR'] #['Q_W2_ERR']
ivar_all = np.ones_like(y_all) # ** (-2)
AT_0 = np.vstack([np.ones_like(abund['R'])])
AT_linear = np.array([np.array(abund['MG_FE']), np.array(abund['AL_FE']), np.array(abund['FE_H']), np.array(abund['N_FE']), np.array(abund['O_FE']), np.array(abund['CA_FE']), np.array(abund['MN_FE']), np.array(abund['SI_FE']), np.array(abund['NA_FE']), np.array(abund['C_FE']), np.array(abund['CO_FE']), np.array(abund['NI_FE'])])
A_all = np.vstack([AT_0, AT_linear]).T

date = 'jun8'

# cross validation
Kfold = 2
y_pred_all = np.zeros_like(y_all)
lam = 0

name = 'ages_N{0}_lam{1}_K{2}'.format(len(y_all), lam, Kfold)

for k in range(Kfold):    
    
    # hold out data
    valid = abund['random_index'] % Kfold == k
    train = np.logical_not(valid)
    y = y_all[train]
    ivar = ivar_all[train]
    A = A_all[train, :]
    N, M = A.shape
    x0 = np.zeros((M,))
                 
    # optimize H_func
    res = op.minimize(H_func, x0, args=(y, A, lam, ivar), method='L-BFGS-B', jac=True, options={'maxfun':50000}) 
    print(res)  
    
    # prediction
    y_pred = np.dot(A_all[valid, :], res.x) 
    y_pred_all[valid] = y_pred

    plt.plot(res.x)
    plt.title(r'$\lambda = {}$'.format(lam))
    plt.savefig('plots/{0}/regularization_results_{1}.pdf'.format(date, name))
    plt.close()
    
plt.scatter(y_all, y_pred_all, alpha = .2)
plt.xlim(0, 30)
plt.ylim(0, 30)
plt.plot([0, 100], [0, 100], 'k-')
plt.xlabel(r'$R_{\rm true}$', fontsize = 14)
plt.ylabel(r'$R_{\rm pred}$', fontsize = 14)
plt.savefig('plots/{0}/radius_{1}.pdf'.format(date, name))

foo = (y_pred_all > 10) * (y_pred_all < 11)
R_cut = np.zeros_like(XS[:, 1], dtype=bool)
R_cut[missing] = foo

fig, ax = plt.subplots(1, 1, figsize = (12, 12))
plt.scatter(XS[:, 0], XS[:, 1], c = '0.7', alpha = .2, s = 10, zorder = 0)
plt.scatter(XS[R_cut, 0], XS[R_cut, 1], c = 'k', s = 10, zorder = 10) #labels['MG_FE'][box_cut], cmap = 'RdBu_r', vmin = -.1, vmax = .3, s = 10, zorder = 10)
plt.xlim(Xlimits[0])
plt.ylim(Xlimits[1])
overplot_rings()
plt.tick_params(axis=u'both', direction='in', which='both')
plt.xlabel('$x$', fontsize = 14)
plt.ylabel('$y$', fontsize = 14)
ax.set_aspect('equal')'''