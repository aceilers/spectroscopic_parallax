#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 09:43:21 2018

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
from astropy.coordinates import SkyCoord, Galactocentric
import astropy.coordinates as coord
from mpl_toolkits.mplot3d import Axes3D
import corner
from scipy.stats import binned_statistic_2d
from plotting_helpers import histcont


# -------------------------------------------------------------------------------
# plotting settings
# -------------------------------------------------------------------------------

matplotlib.rcParams['ytick.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 14
fsize = 14

def overplot_ring(r):
    tiny = 1e-4
    thetas = np.arange(0., 2*np.pi + tiny, 0.001 * np.pi)
    xs = r * np.cos(thetas)
    ys = r * np.sin(thetas)
    plt.plot(xs, ys, "k-", alpha=0.2, lw=1, zorder = -np.inf)
    plt.scatter(0, 0, s = 10, color = 'k', alpha=0.2)
    return

def overplot_rings():
    for r in [5, 10, 15, 20, 25, 30]:
        overplot_ring(r)
    return

# -------------------------------------------------------------------------------
# open inferred labels
# -------------------------------------------------------------------------------

N = 45787
Kfold = 2
lam = 30
name = 'N{0}_lam{1}_K{2}_mag_allcolors_offset_lams'.format(N, lam, Kfold)

print('loading new labels...')   
labels = Table.read('data/training_labels_new_{}_2.fits'.format(name), format = 'fits')    
labels.rename_column('ra_1', 'ra')
labels.rename_column('dec_1', 'dec')

# -------------------------------------------------------------------------------
# calculate cartesian coordinates
# -------------------------------------------------------------------------------           

spec_par = labels['spec_parallax'] * u.mas
distance = spec_par.to(u.parsec, equivalencies = u.parallax())

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

# keep proper motion of Sgr A* constant! 
vY_GC_sun_kms = X_GC_sun_kpc * vY_GC_sun_kms / 8.

gc = coord.Galactocentric(galcen_distance = X_GC_sun_kpc*u.kpc,
                          galcen_v_sun = coord.CartesianDifferential([-vX_GC_sun_kms, vY_GC_sun_kms, vZ_GC_sun_kms] * u.km/u.s),
                          z_sun = Z_GC_sun_kpc*u.kpc)

galcen = cs.transform_to(gc)
cyl = galcen.represent_as('cylindrical')

vr = cyl.differentials['s'].d_rho.to(u.km/u.s)
vphi = (cyl.rho * cyl.differentials['s'].d_phi).to(u.km/u.s, u.dimensionless_angles())
vz = cyl.differentials['s'].d_z.to(u.km/u.s)

rho = cyl.rho.to(u.kpc)
phi = cyl.phi
z = cyl.z

XS_cyl = np.vstack([rho, phi, z, vr, vphi, vz]).T.value

# -------------------------------------------------------------------------------
# Cartesian coordinates
# -------------------------------------------------------------------------------   

galcen = cs.transform_to(gc)
xs, ys, zs = galcen.x.to(u.kpc), galcen.y.to(u.kpc), galcen.z.to(u.kpc)
vxs, vys, vzs = galcen.v_x, galcen.v_y, galcen.v_z

XS_cart = np.vstack([xs, ys, zs, vxs, vys, vzs]).T.value
## eliminate np.nan
good = np.sum(np.isfinite(XS_cart), axis=1) == 6
XS_cart = XS_cart[good]
XS_cyl = XS_cyl[good]
labels = labels[good]

Xlimits = [[-30, 10], [-10, 30], [-20, 20], 
           [-200, 200], [-200, 200], [-200, 200]]
#Xlabels = ['$x$', '$y$', '$z$', r'$v_x$', r'$v_y$', r'$v_z$']
#
#d2d = np.sqrt(XS[:, 0] ** 2 + XS[:, 1] ** 2)
#units = XS[:, 0:2] / d2d[:, None]
#perps = np.zeros_like(units)
#perps[:, 0] = units[:, 1]
#perps[:, 1] = -units[:, 0]
#vtans = np.sum(perps * XS[:, 3:5], axis=1)
#R = np.sqrt(XS[:, 0] ** 2 + XS[:, 1] ** 2) # in cylindrical coordinates! # + XS[:, 2] ** 2)

# uncertainty on proper motion
mu_err = np.sqrt(labels['pmra_error']**2 + labels['pmdec_error']**2)

# -------------------------------------------------------------------------------
# divide Milky Way into patches
# -------------------------------------------------------------------------------   

box_size = .5               # that's just half of the box size
all_x = np.arange(-30., 30.01, box_size)
all_y = np.arange(-30., 30.01, box_size)
mean_XS = np.zeros((len(all_x), len(all_y), 6))
mean_XS_cart = np.zeros((len(all_x), len(all_y), 6))
var_XS = np.zeros((len(all_x), len(all_y), 3, 3))
N_stars = np.zeros((len(all_x), len(all_y)))
mean_HW2 = np.zeros((len(all_x), len(all_y)))
mean_mu_err = np.zeros((len(all_x), len(all_y)))

for i, box_center_x in enumerate(all_x):
    for j, box_center_y in enumerate(all_y):
        print(i, j)
        cut_patch = (abs(XS_cart[:, 2]) < box_size) * (abs(XS_cart[:, 0] - box_center_x) < box_size) * (abs(XS_cart[:, 1] - box_center_y) < box_size)
        N_stars[i, j] = np.sum(cut_patch)
        if N_stars[i, j] > 0:
            mean_XS[i, j, :] = np.nanmean(XS_cyl[cut_patch], axis = 0)
            mean_XS_cart[i, j, :] = np.nanmean(XS_cart[cut_patch], axis = 0)
            mean_HW2[i, j] = np.nanmean(labels['H'][cut_patch] - labels['w2mpro'][cut_patch])
            mean_mu_err[i, j] = np.nanmean(mu_err[cut_patch])
        if N_stars[i, j] > 7:
            dXS = XS_cyl[cut_patch] - mean_XS[i, j, :][None, :]
            var_XS[i, j, :, :] = np.dot(dXS[:, 3:].T, dXS[:, 3:]) / (N_stars[i, j] - 1.)
    
# -------------------------------------------------------------------------------
# plot
# -------------------------------------------------------------------------------        

# calculate to cartesian coordinates for plot
#x = mean_XS[:, :, 0] * np.cos(mean_XS[:, :, 1])
#y = mean_XS[:, :, 0] * np.sin(mean_XS[:, :, 1])
x = mean_XS_cart[:, :, 0]
y = mean_XS_cart[:, :, 1]
vx = mean_XS[:, :, 3] * np.cos(mean_XS[:, :, 1]) - mean_XS[:, :, 4] * np.sin(mean_XS[:, :, 1])
vy = mean_XS[:, :, 3] * np.sin(mean_XS[:, :, 1]) + mean_XS[:, :, 4] * np.cos(mean_XS[:, :, 1])

# plot v_phi vs. phi
plt.scatter(mean_XS[:, :, 1], mean_XS[:, :, 4])
plt.xlabel(r'$\varphi$', fontsize = fsize)
plt.ylabel(r'$v_{\varphi}$', fontsize = fsize)

        
fig, ax = plt.subplots(1, 1, figsize = (12, 12))        
plt.quiver(x, y, vx, vy, 
        np.clip(mean_XS[:, :, 5], -10, 10), cmap = 'RdYlBu', scale_units='xy', 
           scale=200, alpha =.8, headwidth = 3, headlength = 4, width = 0.002)
cb = plt.colorbar(shrink = .85)
cb.set_label(r'$v_z$', fontsize = 15)
plt.xlim(Xlimits[0])
plt.ylim(Xlimits[1])
overplot_rings()
plt.tick_params(axis=u'both', direction='in', which='both')
plt.xlabel('$x$', fontsize = fsize)
plt.ylabel('$y$', fontsize = fsize)
ax.set_aspect('equal')
plt.savefig('plots/rotation/xy_arrow_averaged_{}_cylindrical.pdf'.format(name), bbox_inches = 'tight')
plt.close()

# v_r, v_phi, v_z vs. phi at fixed R
fixed_R = 10
cut_R = abs(mean_XS[:, :, 0] - fixed_R) < 1.
fig, ax = plt.subplots(1, 3, figsize = (18, 5))        
ax[0].scatter(mean_XS[cut_R, 1], mean_XS[cut_R, 3])
ax[0].set_xlabel(r'$\varphi$', fontsize = 14)
ax[0].set_ylabel(r'$v_r$', fontsize = 14)
ax[1].scatter(mean_XS[cut_R, 1], mean_XS[cut_R, 4])
ax[1].set_xlabel(r'$\varphi$', fontsize = 14)
ax[1].set_ylabel(r'$v_{\varphi}$', fontsize = 14)
ax[2].scatter(mean_XS[cut_R, 1], mean_XS[cut_R, 5])
ax[2].set_xlabel(r'$\varphi$', fontsize = 14)
ax[2].set_ylabel(r'$v_z$', fontsize = 14)
ax[0].tick_params(axis=u'both', direction='in', which='both')
ax[1].tick_params(axis=u'both', direction='in', which='both')
ax[2].tick_params(axis=u'both', direction='in', which='both')
ax[0].set_ylim(-500, 500)
ax[1].set_ylim(-500, 500)
ax[2].set_ylim(-500, 500)
plt.tight_layout()
ax[1].set_title(r'$R={}\,\rm kpc$'.format(fixed_R), fontsize = 15)
plt.savefig('plots/rotation/all_v_R{0}_{1}_cylindrical.pdf'.format(fixed_R, name), bbox_inches = 'tight')
plt.close()

fig, ax = plt.subplots(1, 1, figsize = (12, 12))        
plt.quiver(x, y, vx, vy, 
        np.clip(mean_HW2, 0, 1.5), cmap = 'RdYlBu_r', scale_units='xy', 
           scale=200, alpha =.8, headwidth = 3, headlength = 4, width = 0.002)
cb = plt.colorbar(shrink = .85)
cb.set_label(r'H-W2', fontsize = 15)
plt.xlim(Xlimits[0])
plt.ylim(Xlimits[1])
overplot_rings()
plt.tick_params(axis=u'both', direction='in', which='both')
plt.xlabel('$x$', fontsize = fsize)
plt.ylabel('$y$', fontsize = fsize)
ax.set_aspect('equal')
plt.savefig('plots/rotation/xy_arrow_averaged_{}_HW2_withg.pdf'.format(name), bbox_inches = 'tight')
plt.close()

traceV = np.trace(var_XS, axis1=2, axis2=3)
plt.imshow(traceV.T, origin = (0, 0), cmap = 'viridis', vmin = 0, vmax = 10000)
plt.colorbar()

fig, ax = plt.subplots(1, 1, figsize = (12, 12))        
overplot_rings()
cm = plt.cm.get_cmap('viridis')
sc = plt.scatter(x.flatten(), y.flatten(), c = traceV.flatten(), vmin = 1000, vmax = 30000, s=60, cmap=cm)
cbar = plt.colorbar(sc, shrink = .85)
cbar.set_label('velocity dispersion', rotation=270, fontsize=14, labelpad=15)
plt.xlim(Xlimits[0])
plt.ylim(Xlimits[1])
plt.tick_params(axis=u'both', direction='in', which='both')
plt.xlabel('$x$', fontsize = fsize)
plt.ylabel('$y$', fontsize = fsize)
ax.set_aspect('equal')
plt.savefig('plots/rotation/xy_arrow_velocity_dispersion_{}_withg.pdf'.format(name), bbox_inches = 'tight')
plt.close()

fig, ax = plt.subplots(1, 1, figsize = (12, 12))        
overplot_rings()
cm = plt.cm.get_cmap('viridis')
sc = plt.scatter(x.flatten(), y.flatten(), c = var_XS[:, :, 0, 2].flatten(), vmin = -10000, vmax = 10000, s=60, cmap=cm)
cbar = plt.colorbar(sc, shrink = .85)
cbar.set_label('$v_R v_z$ covariance', rotation=270, fontsize=14, labelpad=15)
plt.xlim(Xlimits[0])
plt.ylim(Xlimits[1])
plt.tick_params(axis=u'both', direction='in', which='both')
plt.xlabel('$x$', fontsize = fsize)
plt.ylabel('$y$', fontsize = fsize)
ax.set_aspect('equal')
plt.savefig('plots/rotation/vrvz_{}.pdf'.format(name), bbox_inches = 'tight')
plt.close()

fig, ax = plt.subplots(1, 1, figsize = (12, 12))        
overplot_rings()
cm = plt.cm.get_cmap('viridis')
sc = plt.scatter(x.flatten(), y.flatten(), c = var_XS[:, :, 0, 1].flatten(), vmin = -10000, vmax = 10000, s=60, cmap=cm)
cbar = plt.colorbar(sc, shrink = .85)
cbar.set_label(r'$v_R v_{\varphi}$ covariance', rotation=270, fontsize=14, labelpad=15)
plt.xlim(Xlimits[0])
plt.ylim(Xlimits[1])
plt.tick_params(axis=u'both', direction='in', which='both')
plt.xlabel('$x$', fontsize = fsize)
plt.ylabel('$y$', fontsize = fsize)
ax.set_aspect('equal')
plt.savefig('plots/rotation/vrvphi_{}.pdf'.format(name), bbox_inches = 'tight')

# -------------------------------------------------------------------------------'''
