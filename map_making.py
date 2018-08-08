#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 08:02:06 2018

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
import astropy

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
lam = 30
#name = 'N{0}_lam{1}_K{2}'.format(N, lam, Kfold)
name = 'N{0}_lam{1}_K{2}_mag_allcolors_offset'.format(N, lam, Kfold)


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
X_GC_sun_kpc = 8.   #[kpc]
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
    G = astropy.constants.G
    M = 5.8e11 * astropy.constants.M_sun
    v = np.sqrt(G * M/(R * u.kpc))    
    return v.to(u.km / u.s).value

# -------------------------------------------------------------------------------
# rotation curve
# -------------------------------------------------------------------------------           

# take only stars in mid-plane 
vz_cut = (abs(labels['b']) < 2) # * (abs(XS[:, 5]) < 20)

# -------------------------------------------------------------------------------
# other plots
# -------------------------------------------------------------------------------
 
## test to rotate system
#x = (np.random.random(size = 10000) -0.5) * 100
#y = (np.random.random(size = 10000) -0.5) * 100
#phi = np.arctan2(y, -x) - 60./360 * 2 * np.pi
#phi[phi < -np.pi] += 2. * np.pi
#fig, ax = plt.subplots(1, 1, figsize = (8, 8))
#sc = plt.scatter(x, y, c = phi, cmap = 'inferno', vmin = -np.pi, vmax = np.pi, s = 10)
#cb = plt.colorbar(sc, shrink = 0.82)
#cb.set_label(label = r'$\arctan2(y, -x)$', fontsize = fsize)
#plt.xlim(-50, 50)
#plt.ylim(-50, 50)
#overplot_rings()
#plt.tick_params(axis=u'both', direction='in', which='both')
#plt.xlabel('$x$', fontsize = fsize)
#plt.ylabel(r'$y$', fontsize = fsize)
#ax.set_aspect('equal')

# cylindrical coordinates 
phi = np.arctan2(XS[vz_cut, 1], -XS[vz_cut, 0]) - 60./360 * 2.*np.pi # rotate by 60 degrees
phi[phi < -np.pi] += 2. * np.pi
              
fig, ax = plt.subplots(1, 1, figsize = (8, 8))
sc = plt.scatter(XS[vz_cut, 0], XS[vz_cut, 1], c = phi, cmap = 'inferno', vmin = -2, vmax = 2, s = 10, rasterized = True)
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
plt.savefig('plots/rotation/xy_azimuth.pdf', bbox_inches = 'tight', dpi = 120)
plt.close()

fig, ax = plt.subplots(1, 1, figsize = (10, 8))
sc = plt.scatter(R[vz_cut], vtans[vz_cut], c = phi, cmap = 'inferno', vmin = -2, vmax = 2, s = 10, rasterized = True)
cb = plt.colorbar(sc)
cb.set_label(label = r'$\varphi$', fontsize = fsize)
plt.xlim(0, 30)
plt.ylim(-200, 500)
plt.tick_params(axis=u'both', direction='in', which='both')
plt.xlabel('$R$', fontsize = fsize)
plt.ylabel(r'$v_{\rm tan}$', fontsize = fsize)
plt.savefig('plots/rotation/vtanR_sun{}kpc.pdf'.format(X_GC_sun_kpc), bbox_inches = 'tight', dpi = 120)
plt.close()

# cut in phi
cut_phi  = phi > -0.7
fig, ax = plt.subplots(1, 1, figsize = (10, 8))
sc = plt.scatter(R[vz_cut][cut_phi], vtans[vz_cut][cut_phi], c = phi[cut_phi], cmap = 'inferno', vmin = -2, vmax = 2, s = 10, rasterized = True)
cb = plt.colorbar(sc)
cb.set_label(label = r'$\varphi$', fontsize = fsize)
plt.xlim(0, 30)
plt.ylim(-200, 500)
plt.tick_params(axis=u'both', direction='in', which='both')
plt.xlabel('$R$', fontsize = fsize)
plt.ylabel(r'$v_{\rm tan}$', fontsize = fsize)
plt.title(r'$\langle \varphi \rangle = {}$'.format(round(np.median(phi[cut_phi]), 2)), fontsize = fsize)
plt.savefig('plots/rotation/vtanR_sun{}kpc_phicutbig.pdf'.format(X_GC_sun_kpc), bbox_inches = 'tight', dpi = 120)
plt.close()

cut_phi  = phi < -0.7
fig, ax = plt.subplots(1, 1, figsize = (10, 8))
sc = plt.scatter(R[vz_cut][cut_phi], vtans[vz_cut][cut_phi], c = phi[cut_phi], cmap = 'inferno', vmin = -2, vmax = 2, s = 10, rasterized = True)
cb = plt.colorbar(sc)
cb.set_label(label = r'$\varphi$', fontsize = fsize)
plt.xlim(0, 30)
plt.ylim(-200, 500)
plt.tick_params(axis=u'both', direction='in', which='both')
plt.xlabel('$R$', fontsize = fsize)
plt.ylabel(r'$v_{\rm tan}$', fontsize = fsize)
plt.title(r'$\langle \varphi \rangle = {}$'.format(round(np.median(phi[cut_phi]), 2)), fontsize = fsize)
plt.savefig('plots/rotation/vtanR_sun{}kpc_phicutsmall.pdf'.format(X_GC_sun_kpc), bbox_inches = 'tight', dpi = 120)
plt.close()

plt.hist(R[vz_cut], bins = np.linspace(0, 40, 100))
plt.tick_params(axis=u'both', direction='in', which='both')
plt.xlabel('R', fontsize = 14)
plt.savefig('plots/rotation/hist_R.pdf', bbox_inches = 'tight')
plt.close()

stats, x_edge, y_edge, bins = binned_statistic_2d(R, vtans, values = np.ones_like(R), statistic = 'count', bins = 100, range = [[0, 30], [-100, 500]])
stats[stats < 5] = np.nan
sc = plt.imshow(stats.T, cmap = 'viridis_r', origin = 'lower', extent = (0, 30, -100, 500), aspect = 'auto')
plt.tick_params(axis=u'both', direction='in', which='both')
plt.xlabel(r'$R\,\rm [kpc]$', fontsize = fsize)
plt.ylabel(r'$v_{\rm tan}\,\rm [km\,s^{-1}]$', fontsize = fsize)
r_kep = np.linspace(0, 27, 100)
plt.plot(r_kep, 0.1 * KeplerianRotation(r_kep) , color = 'k')
plt.savefig('plots/rotation/vtanR_density_sun{0}kpc.pdf'.format(X_GC_sun_kpc), bbox_inches = 'tight')
plt.close()

# -------------------------------------------------------------------------------
# maps
# -------------------------------------------------------------------------------           

fig, ax = plt.subplots(1, 1, figsize = (10, 10))
sc = plt.scatter(XS[vz_cut, 0], XS[vz_cut, 1], c = XS[vz_cut, 3], cmap = 'RdBu', vmin = -200, vmax = 200, s = 10, rasterized = True)
cb = plt.colorbar(sc, shrink = 0.82)
cb.set_label(label = r'$v_x$', fontsize = fsize)
plt.xlim(Xlimits[0])
plt.ylim(Xlimits[1])
plt.tick_params(axis=u'both', direction='in', which='both')
plt.xlabel('$x$', fontsize = fsize)
plt.ylabel('$y$', fontsize = fsize)
overplot_rings()
ax.set_aspect('equal')
plt.savefig('plots/maps/xy_vx.pdf', bbox_inches = 'tight', dpi = 120)
plt.close()

fig, ax = plt.subplots(1, 1, figsize = (10, 10))
sc = plt.scatter(XS[vz_cut, 0], XS[vz_cut, 1], c = XS[vz_cut, 4], cmap = 'RdBu', vmin = -200, vmax = 200, s = 10, rasterized = True)
cb = plt.colorbar(sc, shrink = 0.82)
cb.set_label(label = r'$v_y$', fontsize = fsize)
plt.xlim(Xlimits[0])
plt.ylim(Xlimits[1])
plt.tick_params(axis=u'both', direction='in', which='both')
plt.xlabel('$x$', fontsize = fsize)
plt.ylabel('$y$', fontsize = fsize)
overplot_rings()
ax.set_aspect('equal')
plt.savefig('plots/maps/xy_vy.pdf', bbox_inches = 'tight', dpi = 120)
plt.close()

fig, ax = plt.subplots(1, 1, figsize = (10, 10))
sc = plt.scatter(XS[vz_cut, 0], XS[vz_cut, 1], c = XS[vz_cut, 5], cmap = 'RdBu', vmin = -100, vmax = 100, s = 10, rasterized = True)
cb = plt.colorbar(sc, shrink = 0.82)
cb.set_label(label = r'$v_z$', fontsize = fsize)
plt.xlim(Xlimits[0])
plt.ylim(Xlimits[1])
plt.tick_params(axis=u'both', direction='in', which='both')
plt.xlabel('$x$', fontsize = fsize)
plt.ylabel('$y$', fontsize = fsize)
overplot_rings()
ax.set_aspect('equal')
plt.savefig('plots/maps/xy_vz.pdf', bbox_inches = 'tight', dpi = 120)
plt.close()

fig, ax = plt.subplots(1, 1, figsize = (10, 10))
sc = plt.scatter(XS[vz_cut, 0], XS[vz_cut, 1], c = labels['parallax_error'][vz_cut]**2 / (0.1)**2, cmap = 'RdBu', vmin = 0, vmax = 2, s = 10, rasterized = True)
cb = plt.colorbar(sc, shrink = 0.82)
cb.set_label(label = r'$\sigma^2_{\varpi,\,\rm Gaia}/\sigma^2_{\varpi,\,\rm inferred}$', fontsize = fsize)
plt.xlim(Xlimits[0])
plt.ylim(Xlimits[1])
plt.tick_params(axis=u'both', direction='in', which='both')
plt.xlabel('$x$', fontsize = fsize)
plt.ylabel('$y$', fontsize = fsize)
overplot_rings()
ax.set_aspect('equal')
plt.savefig('plots/maps/xy_parallax.pdf', bbox_inches = 'tight', dpi = 120)
plt.close()

fig, ax = plt.subplots(1, 1, figsize = (10, 10))
sc = plt.scatter(XS[vz_cut, 0], XS[vz_cut, 1], c = labels['parallax_over_error'][vz_cut], cmap = 'RdBu', vmin = 0, vmax = 20, s = 10, rasterized = True)
cb = plt.colorbar(sc, shrink = 0.82)
cb.set_label(label = r'$\varpi_{\rm Gaia}/\sigma_{\varpi, \rm Gaia}$', fontsize = fsize)
plt.xlim(Xlimits[0])
plt.ylim(Xlimits[1])
plt.tick_params(axis=u'both', direction='in', which='both')
plt.xlabel('$x$', fontsize = fsize)
plt.ylabel('$y$', fontsize = fsize)
overplot_rings()
ax.set_aspect('equal')
plt.savefig('plots/maps/xy_parallax_over_error.pdf', bbox_inches = 'tight', dpi = 120)
plt.close()

fig, ax = plt.subplots(1, 1, figsize = (12, 12))
plt.quiver(XS[vz_cut, 0], XS[vz_cut, 1], XS[vz_cut, 3], XS[vz_cut, 4], 
           np.clip(XS[vz_cut, 5], -10, 10), cmap = 'RdBu', scale_units='xy', 
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
plt.savefig('plots/maps/xy_arrow.pdf', bbox_inches = 'tight')
plt.close()

AKs = 0.918 * (labels['H'] - labels['w2mpro'] - 0.08)
HW2 = labels['H'] - labels['w2mpro']
fig, ax = plt.subplots(1, 1, figsize = (10, 10))
sc = plt.scatter(XS[vz_cut, 0], XS[vz_cut, 1], c = AKs[vz_cut], cmap = 'RdBu', vmin = 0, vmax = 1, s = 10, rasterized = True)
cb = plt.colorbar(sc, shrink = 0.82)
cb.set_label(label = r'$A(K_S) = 0.981(\rm H-W2 - 0.08)$', fontsize = fsize)
plt.xlim(Xlimits[0])
plt.ylim(Xlimits[1])
plt.tick_params(axis=u'both', direction='in', which='both')
plt.xlabel('$x$', fontsize = fsize)
plt.ylabel('$y$', fontsize = fsize)
overplot_rings()
ax.set_aspect('equal')
plt.savefig('plots/maps/xy_ext.pdf', bbox_inches = 'tight', dpi = 120)
plt.close()

fig, ax = plt.subplots(1, 1, figsize = (10, 10))
sc = plt.scatter(XS[vz_cut, 0], XS[vz_cut, 1], c = vtans[vz_cut], cmap = 'RdBu', vmin = -200, vmax = 200, s = 10, rasterized = True)
cb = plt.colorbar(sc, shrink = 0.82)
cb.set_label(label = r'$v_{\rm tan}$', fontsize = fsize)
plt.xlim(Xlimits[0])
plt.ylim(Xlimits[1])
plt.tick_params(axis=u'both', direction='in', which='both')
plt.xlabel('$x$', fontsize = fsize)
plt.ylabel('$y$', fontsize = fsize)
overplot_rings()
ax.set_aspect('equal')
plt.savefig('plots/maps/xy_vtan.pdf', bbox_inches = 'tight', dpi = 120)
plt.close()

fig, ax = plt.subplots(1, 1, figsize = (8, 8))
plt.scatter(XS[:, 0], XS[:, 2], c = vtans, cmap = 'RdBu', vmin = -200, vmax = 200, s = 10, alpha = .2, rasterized = True)
cb = plt.colorbar(sc, shrink = 0.82)
cb.set_label(label = r'$v_{\rm tan}$', fontsize = fsize)
plt.xlim(-30, 30)
plt.ylim(-30, 30)
plt.tick_params(axis=u'both', direction='in', which='both')
plt.xlabel('$x$', fontsize = fsize)
plt.ylabel('$z$', fontsize = fsize)
ax.set_aspect('equal')
plt.savefig('plots/maps/xz_vtan.pdf', bbox_inches = 'tight', dpi = 120)
plt.close()

# radial velocity!
fig, ax = plt.subplots(1, 1, figsize = (10, 10))
plt.scatter(XS[vz_cut, 0], XS[vz_cut, 1], c = labels['VHELIO_AVG'][vz_cut], cmap = 'RdBu_r', vmin = -150, vmax = 150, s = 10, rasterized = True)
cb = plt.colorbar(shrink = .82)
cb.set_label(r'$v_{\rm rad}$', fontsize = 15)
plt.xlim(Xlimits[0])
plt.ylim(Xlimits[1])
overplot_rings()
plt.tick_params(axis=u'both', direction='in', which='both')
plt.xlabel('$x$', fontsize = 14)
plt.ylabel('$y$', fontsize = 14)
ax.set_aspect('equal')
plt.savefig('plots/maps/xz_vrad.pdf', bbox_inches = 'tight', dpi = 120)
plt.close()

vz_cut_feh = vz_cut * (labels['FE_H'] > -10)
fig, ax = plt.subplots(1, 1, figsize = (10, 10))
plt.scatter(XS[vz_cut_feh, 0], XS[vz_cut_feh, 1], c = labels['FE_H'][vz_cut_feh], cmap = 'RdBu_r', vmin = -.5, vmax = .5, s = 10, rasterized = True)
cb = plt.colorbar(shrink = .82)
cb.set_label(r'$\rm [Fe/H]$', fontsize = 15)
plt.xlim(Xlimits[0])
plt.ylim(Xlimits[1])
overplot_rings()
plt.tick_params(axis=u'both', direction='in', which='both')
plt.xlabel('$x$', fontsize = 14)
plt.ylabel('$y$', fontsize = 14)
ax.set_aspect('equal')
plt.savefig('plots/maps/xz_feh.pdf', bbox_inches = 'tight', dpi = 120)
plt.close()

# -------------------------------------------------------------------------------
# 3D map
# -------------------------------------------------------------------------------           

#fig = plt.figure(figsize = (10, 10))
#ax = fig.add_subplot(111, projection = '3d')
#ax.scatter(xs, ys, zs, c = labels['FE_H'], cmap = 'viridis_r', vmin = -3, vmax = 1)
#ax.set_xlabel('x', fontsize = 14)
#ax.set_ylabel('y', fontsize = 14)
#ax.set_zlabel('z', fontsize = 14)
#ax.set_ylim(-20000, 20000)
#ax.set_xlim(-20000, 20000)

# ------------------------------------------------------------------------------- '''

