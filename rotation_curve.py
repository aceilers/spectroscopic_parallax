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
from astropy.coordinates import SkyCoord
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

# -------------------------------------------------------------------------------
# open inferred labels
# -------------------------------------------------------------------------------

N = 45787
Kfold = 2
lam = 100
name = 'N{0}_lam{1}_K{2}'.format(N, lam, Kfold)

print('loading new labels...')   
labels = Table.read('data/training_labels_new_{}_2.fits'.format(name), format = 'fits')    
labels.rename_column('ra_1', 'ra')
labels.rename_column('dec_1', 'dec')

# take only stars in mid-plane 
vz_cut = (abs(labels['b']) < 2) 
labels = labels[vz_cut]
N = len(labels)

# -------------------------------------------------------------------------------
# re-sampling of each star 
# -------------------------------------------------------------------------------           

N_samples = 10

# uncertainties in RA, DEC, parallax, proper motions in RA and DEC, position of the sun, velocity of the sun
# fix position of the sun at the moment! 

all_ra = np.zeros((N * N_samples, ))
all_dec = np.zeros((N * N_samples, ))
all_par = np.zeros((N * N_samples, ))
all_pm_ra = np.zeros((N * N_samples, ))
all_pm_dec = np.zeros((N * N_samples, ))
all_rv = np.zeros((N * N_samples, )) # NO UNCERTAINTIES FOR RADIAL VELOCITY?

np.random.seed(42)
for i in range(N):
    all_ra[i*N_samples : (i+1)*N_samples] = np.random.normal(loc = labels['ra'][i], scale = labels['ra_error_1'][i], size = (N_samples))
    all_dec[i*N_samples : (i+1)*N_samples] = np.random.normal(loc = labels['dec'][i], scale = labels['dec_error_1'][i], size = (N_samples))
    all_par[i*N_samples : (i+1)*N_samples] = np.random.normal(loc = labels['spec_parallax'][i], scale = 0.1, size = (N_samples))
    all_pm_ra[i*N_samples : (i+1)*N_samples] = np.random.normal(loc = labels['pmra'][i], scale = labels['pmra_error'][i], size = (N_samples))
    all_pm_dec[i*N_samples : (i+1)*N_samples] = np.random.normal(loc = labels['pmdec'][i], scale = labels['pmdec_error'][i], size = (N_samples))
    all_rv[i*N_samples : (i+1)*N_samples] = np.ones((N_samples)) * labels['VHELIO_AVG'][i]

# -------------------------------------------------------------------------------
# calculate cartesian coordinates
# -------------------------------------------------------------------------------           

spec_par = all_par * u.mas
distance = spec_par.to(u.parsec, equivalencies = u.parallax())

cs = coord.ICRS(ra = all_ra * u.degree, 
                dec = all_dec * u.degree, 
                distance = distance, 
                pm_ra_cosdec = all_pm_ra * u.mas/u.yr, 
                pm_dec = all_pm_dec * u.mas/u.yr, 
                radial_velocity = all_rv *u.km/u.s)

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
R = np.sqrt(XS[:, 0] ** 2 + XS[:, 1] ** 2) # in cylindrical coordinates! # + XS[:, 2] ** 2)

# -------------------------------------------------------------------------------
# corner plot
# -------------------------------------------------------------------------------           

fig = corner.corner(XS, range = Xlimits, labels = Xlabels)
fig.savefig('plots/corner.pdf')
plt.close()

# -------------------------------------------------------------------------------
# rings
# -------------------------------------------------------------------------------           

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


# -------------------------------------------------------------------------------
# theoretical rotation curves
# -------------------------------------------------------------------------------           

def KeplerianRotation(R):    
    v = 1./np.sqrt(R)    
    return v

# -------------------------------------------------------------------------------
# rotation curve
# -------------------------------------------------------------------------------           

# cylindrical coordinates 
phi = np.arctan2(XS[:, 1], -XS[:, 0]) - 60./360 * 2.*np.pi # rotate by 60 degrees
phi[phi < -np.pi] += 2. * np.pi
              
fig, ax = plt.subplots(1, 1, figsize = (8, 8))
sc = plt.scatter(XS[:, 0], XS[:, 1], c = phi, cmap = 'inferno', vmin = -2, vmax = 2, s = 10, rasterized = True)
cb = plt.colorbar(sc, shrink = 0.82)
cb.set_label(label = r'$\varphi$', fontsize = fsize)
plt.xlim(Xlimits[0])
plt.ylim(Xlimits[1])
overplot_rings()
plt.tick_params(axis=u'both', direction='in', which='both')
plt.xlabel('$x$', fontsize = fsize)
plt.ylabel(r'$y$', fontsize = fsize)
overplot_rings()
ax.set_aspect('equal')
plt.savefig('plots/rotation/xy_azimuth_samplingN{}.pdf'.format(N_samples), bbox_inches = 'tight', dpi = 120)
plt.close()

fig, ax = plt.subplots(1, 1, figsize = (10, 8))
sc = plt.scatter(R, vtans, c = phi, cmap = 'inferno', vmin = -2, vmax = 2, s = 10, rasterized = True)
cb = plt.colorbar(sc)
cb.set_label(label = r'$\varphi$', fontsize = fsize)
plt.xlim(0, 30)
plt.ylim(-200, 500)
plt.tick_params(axis=u'both', direction='in', which='both')
plt.xlabel('$R$', fontsize = fsize)
plt.ylabel(r'$v_{\rm tan}$', fontsize = fsize)
plt.savefig('plots/rotation/vtanR_sun{0}kpc_samplingN{1}.pdf'.format(X_GC_sun_kpc, N_samples), bbox_inches = 'tight', dpi = 120)
plt.close()

stats, x_edge, y_edge, bins = binned_statistic_2d(R, vtans, values = np.ones_like(R), statistic = 'count', bins = 100, range = [[0, 30], [-100, 500]])
stats[stats < 0.3 * N_samples] = np.nan
sc = plt.imshow(stats.T, cmap = 'viridis_r', origin = 'lower', extent = (0, 30, -100, 500), aspect = 'auto')
plt.tick_params(axis=u'both', direction='in', which='both')
plt.xlabel(r'$R\,\rm [kpc]$', fontsize = fsize)
plt.ylabel(r'$v_{\rm tan}\,\rm [km\,s^{-1}]$', fontsize = fsize)
r_kep = np.linspace(0, 30, 100)
plt.plot(r_kep, 200 * KeplerianRotation(r_kep) , color = 'k')
plt.savefig('plots/rotation/vtanR_density_sun{0}kpc_samplingN{1}.pdf'.format(X_GC_sun_kpc, N_samples), bbox_inches = 'tight')
plt.close()

# error ellipses...
#sigma_R = 1./(XS[vz_cut, 0]**2 + XS[vz_cut, 1]**2)
#sigma_vtan

# -------------------------------------------------------------------------------'''
