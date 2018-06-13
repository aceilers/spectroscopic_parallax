#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 08:02:06 2018

@author: eilers
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use('Agg')
#matplotlib.use('TkAgg')
import scipy.optimize as op
import time
import pickle
from astropy.table import Column, Table, join, vstack, hstack
import sys
from astropy.io import fits
from sklearn.decomposition import PCA
#from wpca import WPCA
from astropy import units as u
from astropy.coordinates import SkyCoord
import astropy.coordinates as coord
from mpl_toolkits.mplot3d import Axes3D
import corner

from normalize_all_spectra import LoadAndNormalizeData, NormalizeData

# -------------------------------------------------------------------------------
# plotting settings
# -------------------------------------------------------------------------------

matplotlib.rcParams['ytick.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 14
fsize = 14

# -------------------------------------------------------------------------------
# open training set
# -------------------------------------------------------------------------------

hdu = fits.open('data/training_labels_N40488_lam100_K2_part1_all.fits')
labels = hdu[1].data
hdu = fits.open('data/training_labels_N40488_lam100_K2_part2_all.fits')
labels2 = hdu[1].data
             
valid = labels['random_index'] % 2 == 1
labels['spec_parallax'][valid] = labels2['spec_parallax'][valid]
labels2 = 0 

#fits.writeto('data/training_labels_spec_parallax.fits', labels)
#labels = hdu[1].data

#hdu = fits.open('data/training_labels_match_ages.fits')
#labels = hdu[1].data
            
# -------------------------------------------------------------------------------
# plots
# -------------------------------------------------------------------------------           
            
plt.scatter(labels['parallax'], labels['spec_parallax'], c = labels['visibility_periods_used'], cmap = 'viridis_r', s = 10, vmin = 0, vmax = 50)
plt.colorbar()
plt.plot([-100, 100], [-100, 100], linestyle = '--', color = 'k')
plt.ylim(0, 2)
plt.xlim(0, 2)
#plt.plot([1e-5, 100], [1e-5, 100], 'k-')
#plt.ylim(1e-5, 5)
#plt.xlim(1e-5, 5)
#plt.xscale('log')
#plt.yscale('log')
plt.xlabel('Gaia parallax')
plt.ylabel('inferred parallax')
plt.savefig('plots/parallax_inferred.pdf')

# -------------------------------------------------------------------------------
# calculate cartesian coordinates
# -------------------------------------------------------------------------------           

spec_par = labels['spec_parallax'] * u.mas
distance = spec_par.to(u.parsec, equivalencies = u.parallax())

#cs = SkyCoord(ra = labels['ra'] * u.degree, dec = labels['dec'] * u.degree, distance = distance, pm_ra_cosdec = labels['pmra'] * u.mas / u.yr, pm_dec = labels['pmdec'] * u.mas / u.yr)
#xs = cs.cartesian.x
#ys = cs.cartesian.y
#zs = cs.cartesian.z

cs = coord.ICRS(ra = labels['ra'] * u.degree, 
                dec = labels['dec'] * u.degree, 
                distance = distance, 
                pm_ra_cosdec = labels['pmra'] * u.mas/u.yr, 
                pm_dec = labels['pmdec'] * u.mas/u.yr, 
                radial_velocity = labels['VHELIO_AVG'] *u.km/u.s)


#Galactocentric position of the Sun:
X_GC_sun_kpc = 8.    #[kpc]
Z_GC_sun_kpc = 0.025 #[kpc] (e.g. Juric et al. 2008)

#circular velocity of the Galactic potential at the radius of the Sun:
vcirc_kms = 220. #[km/s] (e.g. Bovy 2015)

#Velocity of the Sun w.r.t. the Local Standard of Rest (e.g. Schoenrich et al. 2009):
U_LSR_kms = 11.1  # [km/s]
V_LSR_kms = 12.24 # [km/s]
W_LSR_kms = 7.25  # [km/s]

#Galactocentric velocity of the Sun:
vX_GC_sun_kms = -U_LSR_kms           # = -U              [km/s]
vY_GC_sun_kms =  V_LSR_kms+vcirc_kms # = V+v_circ(R_Sun) [km/s]
vZ_GC_sun_kms =  W_LSR_kms           # = W               [km/s]

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
R = np.sqrt(XS[:, 0] ** 2 + XS[:, 1] ** 2) # + XS[:, 2] ** 2)

# -------------------------------------------------------------------------------
# 3D map
# -------------------------------------------------------------------------------           

fig = plt.figure(figsize = (10, 10))
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(xs, ys, zs, c = labels['FE_H'], cmap = 'viridis_r', vmin = -3, vmax = 1)
ax.set_xlabel('x', fontsize = 14)
ax.set_ylabel('y', fontsize = 14)
ax.set_zlabel('z', fontsize = 14)
ax.set_ylim(-20000, 20000)
ax.set_xlim(-20000, 20000)

# -------------------------------------------------------------------------------
# corner plot
# -------------------------------------------------------------------------------           

fig = corner.corner(XS, range = Xlimits, labels = Xlabels)
fig.savefig('plots/corner.pdf')

vz_cut = np.logical_and(abs(XS[:, 5]) < 5, abs(XS[:, 2]) < 1)
fig = corner.corner(XS[vz_cut, :], range = Xlimits, labels = Xlabels)
fig.savefig('plots/corner_vzcut.pdf')

def overplot_ring(r):
    tiny = 1e-4
    thetas = np.arange(0., 2*np.pi + tiny, 0.001 * np.pi)
    xs = r * np.cos(thetas)
    ys = r * np.sin(thetas)
    plt.plot(xs, ys, "k-", alpha=0.2, lw=1, zorder = -np.inf)
    return

def overplot_rings():
    for r in [5, 10, 15, 20, 25, 30]:
        overplot_ring(r)
    return

#vz_cut = abs(XS[:, 2]) < .3
vz_cut = np.logical_and(abs(labels['b']) < 2, XS[:, 5] < 20)
plt.scatter(XS[vz_cut, 0], XS[vz_cut, 1], c = XS[vz_cut, 3], cmap = 'RdBu', vmin = -200, vmax = 200, s = 10)
plt.colorbar(label = r'$v_x$')
plt.xlim(Xlimits[0])
plt.ylim(Xlimits[1])
plt.tick_params(axis=u'both', direction='in', which='both')
plt.xlabel('$x$', fontsize = 14)
plt.ylabel('$y$', fontsize = 14)
plt.savefig('plots/xy_vx.pdf', bbox_inches = 'tight')

plt.scatter(XS[vz_cut, 0], XS[vz_cut, 1], c = XS[vz_cut, 4], cmap = 'RdBu', vmin = -200, vmax = 200, s = 10)
plt.colorbar(label = r'$v_y$')
plt.xlim(Xlimits[0])
plt.ylim(Xlimits[1])
plt.tick_params(axis=u'both', direction='in', which='both')
plt.xlabel('$x$', fontsize = 14)
plt.ylabel('$y$', fontsize = 14)
plt.savefig('plots/xy_vy.pdf', bbox_inches = 'tight')

velocity = np.sqrt(XS[vz_cut, 3] ** 2 + XS[vz_cut, 4] ** 2)
plt.quiver(XS[vz_cut, 0], XS[vz_cut, 1], XS[vz_cut, 3], XS[vz_cut, 4], XS[vz_cut, 5])
#plt.colorbar(label = r'$v_z$')
plt.xlim(Xlimits[0])
plt.ylim(Xlimits[1])
plt.tick_params(axis=u'both', direction='in', which='both')
plt.xlabel('$x$', fontsize = 14)
plt.ylabel('$y$', fontsize = 14)
plt.savefig('plots/xy_arrows.pdf', bbox_inches = 'tight')

plt.scatter(XS[vz_cut, 0], XS[vz_cut, 1], c = vtans[vz_cut], cmap = 'RdBu', vmin = -200, vmax = 200, s = 10)
plt.colorbar(label = r'$v_{\rm tan}$')
plt.xlim(Xlimits[0])
plt.ylim(Xlimits[1])
plt.tick_params(axis=u'both', direction='in', which='both')
plt.xlabel('$x$', fontsize = 14)
plt.ylabel('$y$', fontsize = 14)
plt.savefig('plots/xy_vtan.pdf', bbox_inches = 'tight')

#c = labels['l'][vz_cut]
plt.scatter(R[vz_cut], vtans[vz_cut], c = np.arctan2(XS[vz_cut, 1], -XS[vz_cut, 0]), cmap = 'inferno', vmin = -1, vmax = 1, s = 10)
plt.colorbar(label = r'$\arctan2(y, -x)$')
plt.xlim(0, 30)
plt.ylim(-200, 500)
plt.tick_params(axis=u'both', direction='in', which='both')
plt.xlabel('$R$', fontsize = 14)
plt.ylabel(r'$v_{\rm tan}$', fontsize = 14)
plt.savefig('plots/vtanR.pdf', bbox_inches = 'tight')

fig, ax = plt.subplots(1, 1, figsize = (8, 8))
plt.scatter(XS[vz_cut, 0], XS[vz_cut, 1], c = np.arctan2(XS[vz_cut, 1], -XS[vz_cut, 0]), cmap = 'inferno', vmin = -1, vmax = 1, s = 10)
plt.colorbar(label = r'$\arctan2(y, -x)$', shrink = .8)
plt.xlim(Xlimits[0])
plt.ylim(Xlimits[1])
overplot_rings()
plt.tick_params(axis=u'both', direction='in', which='both')
plt.xlabel('$x$', fontsize = 14)
plt.ylabel(r'$y$', fontsize = 14)
ax.set_aspect('equal')
plt.savefig('plots/vtan_xy.pdf', bbox_inches = 'tight')

plt.scatter(XS[:, 0], XS[:, 2], c = vtans, cmap = 'RdBu', vmin = -200, vmax = 200, s = 10)
plt.colorbar(label = r'$v_{\rm tan}$')
plt.xlim(-30, 20)
plt.ylim(-30, 30)
plt.tick_params(axis=u'both', direction='in', which='both')
plt.xlabel('$x$', fontsize = 14)
plt.ylabel('$z$', fontsize = 14)
plt.savefig('plots/xz_vtan.pdf', bbox_inches = 'tight')

vz_cut = abs(XS[:, 5]) < 10

plt.scatter(XS[vz_cut, 0], XS[vz_cut, 1], c = XS[vz_cut, 2], cmap = 'RdBu', vmin = -20, vmax = 20, s = 10)
plt.colorbar(label = r'$v_{\rm tan}$')
plt.xlim(-30, 20)
plt.ylim(-30, 30)
plt.tick_params(axis=u'both', direction='in', which='both')
plt.xlabel('$x$', fontsize = 14)
plt.ylabel('$z$', fontsize = 14)


fig, ax = plt.subplots(1, 1, figsize = (12, 12))
plt.quiver(XS[vz_cut, 0], XS[vz_cut, 1], XS[vz_cut, 3], XS[vz_cut, 4], 
           np.clip(XS[vz_cut, 5], -10, 10), cmap = 'RdBu', scale_units='xy', 
           scale=200, alpha =.5, headwidth = 3, headlength = 5, width = 0.002)
cb = plt.colorbar(shrink = .85)
cb.set_label(r'$v_z$', fontsize = 15)
plt.xlim(Xlimits[0])
plt.ylim(Xlimits[1])
overplot_rings()
plt.tick_params(axis=u'both', direction='in', which='both')
plt.xlabel('$x$', fontsize = 14)
plt.ylabel('$y$', fontsize = 14)
ax.set_aspect('equal')
plt.savefig('plots/xz_arrow.pdf', bbox_inches = 'tight')


#vz_cut = np.logical_and(np.logical_and(abs(labels['b']) < 2, XS[:, 5] < 20), labels['FE_H'] > -1000)
vz_cut = np.logical_and(np.logical_and(abs(XS[:, 2]) < .3, labels['MG_FE'] > -1000), labels['FE_H'] > -1000)
fig, ax = plt.subplots(1, 1, figsize = (10, 10))
plt.scatter(XS[vz_cut, 0], XS[vz_cut, 1], c = labels['MG_FE'][vz_cut], cmap = 'RdBu_r', vmin = -.1, vmax = .3, s = 10)
cb = plt.colorbar(shrink = .82)
cb.set_label(r'$\rm [Mg/Fe]$', fontsize = 15)
plt.xlim(Xlimits[0])
plt.ylim(Xlimits[1])
overplot_rings()
plt.tick_params(axis=u'both', direction='in', which='both')
plt.xlabel('$x$', fontsize = 14)
plt.ylabel('$y$', fontsize = 14)
ax.set_aspect('equal')
plt.savefig('plots/xz_mgfe.pdf', bbox_inches = 'tight')

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
# including ages
# -------------------------------------------------------------------------------           

hdu = fits.open('data/labels_match_ages_melissa.fits')
melissa_labels = hdu[1].data
melissa_labels = Table(melissa_labels)

melissa_labels.rename_column('col2', 'ages')

spec_par = melissa_labels['spec_parallax'] * u.mas
distance = spec_par.to(u.parsec, equivalencies = u.parallax())

cs = coord.ICRS(ra = melissa_labels['ra_1'] * u.degree, 
                dec = melissa_labels['dec_1'] * u.degree, 
                distance = distance, 
                pm_ra_cosdec = melissa_labels['pmra_1a'] * u.mas/u.yr, 
                pm_dec = melissa_labels['pmdec_1a'] * u.mas/u.yr, 
                radial_velocity = melissa_labels['VHELIO_AVG'] *u.km/u.s)


## Payne ages
#hdu = fits.open('data/training_labels_match_ages.fits')
#labels = hdu[1].data
#labels = Table(labels)
#
#labels.rename_column('tRGBa', 'ages')
#
#spec_par = labels['spec_parallax'] * u.mas
#distance = spec_par.to(u.parsec, equivalencies = u.parallax())
#
#cs = coord.ICRS(ra = labels['ra_1'] * u.degree, 
#                dec = labels['dec_1'] * u.degree, 
#                distance = distance, 
#                pm_ra_cosdec = labels['pmra_2a'] * u.mas/u.yr, 
#                pm_dec = labels['pmdec_2a'] * u.mas/u.yr, 
#                radial_velocity = labels['VHELIO_AVG'] *u.km/u.s)


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
R = np.sqrt(XS[:, 0] ** 2 + XS[:, 1] ** 2) #* np.sign(XS[:, 0])

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
plt.savefig('plots/ages_payne.pdf')

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

plt.hist(R[vz_cut], 180)
plt.xlim(-20, 0)


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
ax.set_aspect('equal')

