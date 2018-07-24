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
lam = 30
name = 'N{0}_lam{1}_K{2}_mag_allcolors_offset'.format(N, lam, Kfold)

print('loading new labels...')   
labels = Table.read('data/training_labels_new_{}_2.fits'.format(name), format = 'fits')    
labels.rename_column('ra_1', 'ra')
labels.rename_column('dec_1', 'dec')

# -------------------------------------------------------------------------------
# position of the sun
# -------------------------------------------------------------------------------           

# Galactocentric position of the Sun:
X_GC_sun_kpc = 8.3    #[kpc]
Z_GC_sun_kpc = 0.025 #[kpc] (e.g. Juric et al. 2008)

# circular velocity of the Galactic potential at the radius of the Sun:
vcirc_kms = 220. #[km/s] (e.g. Bovy 2015)

# Velocity of the Sun w.r.t. the Local Standard of Rest (e.g. Schoenrich et al. 2009):
U_LSR_kms = 11.1  # [km/s]
V_LSR_kms = 12.24 # [km/s]
W_LSR_kms = 7.25  # [km/s]

# Galactocentric velocity of the Sun:
vX_GC_sun_kms = -U_LSR_kms           # = -U              [km/s]
vY_GC_sun_kms =  V_LSR_kms+vcirc_kms # = V+v_circ(R_Sun) [km/s]
vZ_GC_sun_kms =  W_LSR_kms           # = W               [km/s]

# keep proper motion of Sgr A* constant! 
vY_GC_sun_kms = X_GC_sun_kpc * vY_GC_sun_kms / 8.

# -------------------------------------------------------------------------------
# re-sample each star
# -------------------------------------------------------------------------------           

#cuts = np.isfinite(labels['pmra']) * np.isfinite(labels['pmdec'])
#labels = labels[cuts]
N = len(labels)

N_sample = 256
np.random.seed(42)
mean_XS_cart_n = np.zeros((N, 6))
var_XS_cart_n = np.zeros((N, 3, 3))
mean_XS_cyl_n = np.zeros((N, 6))
var_XS_cyl_n = np.zeros((N, 3, 3))
XS_cart_true_n = np.zeros((N, 6))
XS_cyl_true_n = np.zeros((N, 6))

# RADIAL VELOCITY UNCERTAINTY?
floor_rv = 0.1 # km/s

fractional_parallax_error = 0.09

for i in range(N):
    
    if i%1000 == 0: print('working on star {0} out of {1}'.format(i, N))
    spec_par = np.random.normal(labels['spec_parallax'][i], scale = fractional_parallax_error * labels['spec_parallax'][i], size = N_sample) * u.mas
    assert np.all(spec_par > 0)
    distance = spec_par.to(u.parsec, equivalencies = u.parallax())
    distance_true = (labels['spec_parallax'][i] * u.mas).to(u.parsec, equivalencies = u.parallax())
    
    pmras = np.random.normal(labels['pmra'][i], scale = labels['pmra_error'][i], size = N_sample)
    pmdecs = np.random.normal(labels['pmdec'][i], scale = labels['pmdec_error'][i], size = N_sample)
    vrs = np.random.normal(labels['VHELIO_AVG'][i], scale = np.sqrt(floor_rv**2), size = N_sample)
                             
    # -------------------------------------------------------------------------------
    # calculate cartesian coordinates
    # -------------------------------------------------------------------------------           
    
    cs = coord.ICRS(ra = np.ones((N_sample)) * labels['ra'][i] * u.degree, 
                    dec = np.ones((N_sample)) * labels['dec'][i] * u.degree, 
                    distance = distance, 
                    pm_ra_cosdec = pmras * u.mas/u.yr, 
                    pm_dec = pmdecs * u.mas/u.yr, 
                    radial_velocity = vrs * u.km/u.s)
    
    cs_true = coord.ICRS(ra = labels['ra'][i] * u.degree, 
                dec = labels['dec'][i] * u.degree, 
                distance = distance_true, 
                pm_ra_cosdec = labels['pmra'][i] * u.mas/u.yr, 
                pm_dec = labels['pmdec'][i] * u.mas/u.yr, 
                radial_velocity = labels['VHELIO_AVG'][i] * u.km/u.s)

    gc = coord.Galactocentric(galcen_distance = X_GC_sun_kpc*u.kpc,
                          galcen_v_sun = coord.CartesianDifferential([-vX_GC_sun_kms, vY_GC_sun_kms, vZ_GC_sun_kms] * u.km/u.s),
                          z_sun = Z_GC_sun_kpc*u.kpc)

    galcen = cs.transform_to(gc)
    xs, ys, zs = galcen.x.to(u.kpc), galcen.y.to(u.kpc), galcen.z.to(u.kpc)
    vxs, vys, vzs = galcen.v_x, galcen.v_y, galcen.v_z
    XS_cart = np.vstack([xs, ys, zs, vxs, vys, vzs]).T.value
   
    cyl = galcen.represent_as('cylindrical')   
    vr = cyl.differentials['s'].d_rho.to(u.km/u.s)
    vphi = (cyl.rho * cyl.differentials['s'].d_phi).to(u.km/u.s, u.dimensionless_angles())
    vz = cyl.differentials['s'].d_z.to(u.km/u.s)
    rho = cyl.rho.to(u.kpc)
    phi = cyl.phi
    z = cyl.z    
    XS_cyl = np.vstack([rho, phi, z, vr, vphi, vz]).T.value
    
    galcen_true = cs_true.transform_to(gc)
    xs_true, ys_true, zs_true = galcen_true.x.to(u.kpc), galcen_true.y.to(u.kpc), galcen_true.z.to(u.kpc)
    vxs_true, vys_true, vzs_true = galcen_true.v_x, galcen_true.v_y, galcen_true.v_z
    XS_cart_true_n[i, :] = np.vstack([xs_true, ys_true, zs_true, vxs_true, vys_true, vzs_true]).T.value   
    cyl_true = galcen_true.represent_as('cylindrical')   
    vr_true = cyl_true.differentials['s'].d_rho.to(u.km/u.s)
    vphi_true = (cyl_true.rho * cyl_true.differentials['s'].d_phi).to(u.km/u.s, u.dimensionless_angles())
    vz_true = cyl_true.differentials['s'].d_z.to(u.km/u.s)    
    rho_true = cyl_true.rho.to(u.kpc)
    phi_true = cyl_true.phi
    z_true = cyl_true.z.to(u.kpc)  
    XS_cyl_true_n[i, :] = np.vstack([rho_true, phi_true, z_true, vr_true, vphi_true, vz_true]).T.value

    mean_XS_cart_n[i, :] = np.nanmean(XS_cart, axis = 0)
    dXS_cart = XS_cart - mean_XS_cart_n[i, :][None, :]
    var_XS_cart_n[i, :, :] = np.dot(dXS_cart[:, 3:].T, dXS_cart[:, 3:]) / (N_sample - 1.)
    
    mean_XS_cyl_n[i, :] = np.nanmean(XS_cyl, axis = 0)
    dXS_cyl = XS_cyl - mean_XS_cyl_n[i, :][None, :]
    var_XS_cyl_n[i, :, :] = np.dot(dXS_cyl[:, 3:].T, dXS_cyl[:, 3:]) / (N_sample - 1.)

hdu = fits.PrimaryHDU(mean_XS_cart_n)
hdu.writeto('data/mean_cart_n_{}.fits'.format(name), overwrite = True)
hdu = fits.PrimaryHDU(var_XS_cart_n)
hdu.writeto('data/var_cart_n_{}.fits'.format(name), overwrite = True)
hdu = fits.PrimaryHDU(mean_XS_cyl_n)
hdu.writeto('data/mean_cyl_n_{}.fits'.format(name), overwrite = True)
hdu = fits.PrimaryHDU(var_XS_cyl_n)
hdu.writeto('data/var_cyl_n_{}.fits'.format(name), overwrite = True)
hdu = fits.PrimaryHDU(XS_cyl_true_n)
hdu.writeto('data/true_cyl_n_{}.fits'.format(name), overwrite = True)
hdu = fits.PrimaryHDU(XS_cart_true_n)
hdu.writeto('data/true_cart_n_{}.fits'.format(name), overwrite = True)


# -------------------------------------------------------------------------------
# plots
# -------------------------------------------------------------------------------           

hdu = fits.open('data/mean_cart_n_{}.fits'.format(name))
mean_XS_cart_n = hdu[0].data
hdu = fits.open('data/var_cart_n_{}.fits'.format(name))
var_XS_cart_n = hdu[0].data
hdu = fits.open('data/mean_cyl_n_{}.fits'.format(name))
mean_XS_cyl_n = hdu[0].data
hdu = fits.open('data/var_cyl_n_{}.fits'.format(name))
var_XS_cyl_n = hdu[0].data
hdu = fits.open('data/true_cart_n_{}.fits'.format(name))
XS_cart_true_n = hdu[0].data
hdu = fits.open('data/true_cyl_n_{}.fits'.format(name))
XS_cyl_true_n = hdu[0].data

# cuts in metallicity
cut_high_feh = (labels['FE_H'] >= 0.0)
cut_solar_feh = (labels['FE_H'] >= -0.2) * (labels['FE_H'] < 0.0)
cut_low_feh = (labels['FE_H'] < -0.2) * (labels['FE_H'] >= -10) # remove [Fe/H] = -9999.0
cut_names = list(['lowFEH', 'solarFEH', 'hiFEH'])
cuts = list([cut_low_feh, cut_solar_feh, cut_high_feh])

cuts = np.isfinite(labels['pmra']) * np.isfinite(labels['pmdec']) * (labels['ALPHA_M'] < .12) #\
       #* cut_solar_feh
labels = labels[cuts]
mean_XS_cart_n = mean_XS_cart_n[cuts, :]
var_XS_cart_n = var_XS_cart_n[cuts, :, :]
mean_XS_cyl_n = mean_XS_cyl_n[cuts, :]
var_XS_cyl_n = var_XS_cyl_n[cuts, :, :]
XS_cart_true_n = XS_cart_true_n[cuts, :]
XS_cyl_true_n = XS_cyl_true_n[cuts, :]

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

Xlimits = [[-30, 10], [-10, 30], [-20, 20], 
           [-200, 200], [-200, 200], [-200, 200]]

#cut_z = abs(mean_XS_cart_n[:, 2]) < 0.5
#
#fig, ax = plt.subplots(1, 1, figsize = (12, 12))        
#overplot_rings()
#cm = plt.cm.get_cmap('viridis')
#sc = plt.scatter(mean_XS_cart_n[cut_z, 0], mean_XS_cart_n[cut_z, 1], c = var_XS_cyl_n[cut_z, 0, 0], vmin = 0, vmax = 100, s=20, cmap=cm, alpha = .8)
#cbar = plt.colorbar(sc, shrink = .85)
#cbar.set_label(r'$\sigma^2_{v_r}$', rotation=270, fontsize=14, labelpad=15)
#plt.xlim(Xlimits[0])
#plt.ylim(Xlimits[1])
#plt.tick_params(axis=u'both', direction='in', which='both')
#plt.xlabel('$x$', fontsize = fsize)
#plt.ylabel('$y$', fontsize = fsize)
#ax.set_aspect('equal')
#plt.savefig('plots/rotation/sigma_vr2_{}.pdf'.format(name), bbox_inches = 'tight')
#plt.close()
#
#fig, ax = plt.subplots(1, 1, figsize = (12, 12))        
#overplot_rings()
#cm = plt.cm.get_cmap('viridis')
#sc = plt.scatter(mean_XS_cart_n[cut_z, 0], mean_XS_cart_n[cut_z, 1], c = var_XS_cyl_n[cut_z, 1, 1], vmin = 0, vmax = 100, s=20, cmap=cm)
#cbar = plt.colorbar(sc, shrink = .85)
#cbar.set_label(r'$\sigma^2_{v_{\varphi}}$', rotation=270, fontsize=14, labelpad=15)
#plt.xlim(Xlimits[0])
#plt.ylim(Xlimits[1])
#plt.tick_params(axis=u'both', direction='in', which='both')
#plt.xlabel('$x$', fontsize = fsize)
#plt.ylabel('$y$', fontsize = fsize)
#ax.set_aspect('equal')
#plt.savefig('plots/rotation/sigma_vphi2_{}.pdf'.format(name), bbox_inches = 'tight')
#plt.close()
#
#fig, ax = plt.subplots(1, 1, figsize = (12, 12))        
#overplot_rings()
#cm = plt.cm.get_cmap('viridis')
#sc = plt.scatter(mean_XS_cart_n[cut_z, 0], mean_XS_cart_n[cut_z, 1], c = var_XS_cyl_n[cut_z, 2, 2], vmin = 0, vmax = 500, s=60, cmap=cm)
#cbar = plt.colorbar(sc, shrink = .85)
#cbar.set_label(r'$\sigma^2_{v_z}$', rotation=270, fontsize=14, labelpad=15)
#plt.xlim(Xlimits[0])
#plt.ylim(Xlimits[1])
#plt.tick_params(axis=u'both', direction='in', which='both')
#plt.xlabel('$x$', fontsize = fsize)
#plt.ylabel('$y$', fontsize = fsize)
#ax.set_aspect('equal')
#plt.savefig('plots/rotation/sigma_vz2_{}.pdf'.format(name), bbox_inches = 'tight')
#plt.close()
#
#fig, ax = plt.subplots(1, 1, figsize = (8, 8))        
#plt.scatter(mean_XS_cyl_n[cut_z, 0], labels['FE_H'][cut_z], s = 5, alpha = 0.1)
#plt.ylim(-1.9, 1)
#plt.xlim(0, 30)
#plt.xlabel(r'$\rm R_{GC}$', fontsize = fsize)
#plt.ylabel('[Fe/H]', fontsize = fsize)
#plt.tick_params(axis=u'both', direction='in', which='both')
#plt.savefig('plots/rotation/FEH_RGC_{}.pdf'.format(name), bbox_inches = 'tight')

# -------------------------------------------------------------------------------
# divide Milky Way into patches
# -------------------------------------------------------------------------------   

box_size = .5               # that's just half of the box size
all_x = np.arange(-30., 30.01, box_size)
all_y = np.arange(-30., 30.01, box_size)
mean_XS_cyl = np.zeros((len(all_x), len(all_y), 6))
mean_XS_cart = np.zeros((len(all_x), len(all_y), 6))
var_XS_cyl = np.zeros((len(all_x), len(all_y), 3, 3))
N_stars = np.zeros((len(all_x), len(all_y)))
error_var_XS_cyl = np.zeros((len(all_x), len(all_y), 3, 3))
vvT_cyl = np.zeros((len(all_x), len(all_y), 3, 3))

for i, box_center_x in enumerate(all_x):
    for j, box_center_y in enumerate(all_y):
        cut_patch = (abs(mean_XS_cart_n[:, 2]) < box_size) * (abs(mean_XS_cart_n[:, 0] - box_center_x) < box_size) * (abs(mean_XS_cart_n[:, 1] - box_center_y) < box_size)
        N_stars[i, j] = np.sum(cut_patch)
        #print(i, j, N_stars[i, j])        
        if N_stars[i, j] > 0:
            mean_XS_cyl[i, j, :] = np.nanmean(mean_XS_cyl_n[cut_patch], axis = 0)
            mean_XS_cart[i, j, :] = np.nanmean(mean_XS_cart_n[cut_patch], axis = 0)
        if N_stars[i, j] > 7:
            dXS = mean_XS_cyl_n[cut_patch] - mean_XS_cyl[i, j, :][None, :]
            var_XS_cyl[i, j, :, :] = np.dot(dXS[:, 3:].T, dXS[:, 3:]) / (N_stars[i, j] - 1.)
            error_var_XS_cyl[i, j, :, :] = np.nanmean(var_XS_cyl_n[cut_patch], axis=0)
            vvT_cyl[i, j, :, :] = np.dot(mean_XS_cyl_n[cut_patch, 3:].T, mean_XS_cyl_n[cut_patch, 3:]) / (N_stars[i, j] - 1.)

# loop over radial bins
        
# -------------------------------------------------------------------------------
# plot
# -------------------------------------------------------------------------------        

traceV = np.trace(var_XS_cyl, axis1=2, axis2=3)
traceC = np.trace(error_var_XS_cyl, axis1=2, axis2=3)

fig, ax = plt.subplots(1, 1, figsize = (12, 12))        
overplot_rings()
cm = plt.cm.get_cmap('viridis')
sc = plt.scatter(mean_XS_cart[:, :, 0].flatten(), mean_XS_cart[:, :, 1].flatten(), c = traceV.flatten(), vmin = 0, vmax = 10000, s=20, cmap=cm, alpha = .8)
cbar = plt.colorbar(sc, shrink = .85)
cbar.set_label(r'$\rm tr(\langle v v^T\rangle)$', rotation=270, fontsize=14, labelpad=15)
plt.xlim(Xlimits[0])
plt.ylim(Xlimits[1])
plt.tick_params(axis=u'both', direction='in', which='both')
plt.xlabel('$x$', fontsize = fsize)
plt.ylabel('$y$', fontsize = fsize)
ax.set_aspect('equal')

fig, ax = plt.subplots(1, 1, figsize = (12, 12))        
overplot_rings()
cm = plt.cm.get_cmap('viridis')
sc = plt.scatter(mean_XS_cart[:, :, 0].flatten(), mean_XS_cart[:, :, 1].flatten(), c = traceC.flatten(), vmin = 0, vmax = 10000, s=20, cmap=cm, alpha = .8)
cbar = plt.colorbar(sc, shrink = .85)
cbar.set_label(r'$\rm tr(C)$', rotation=270, fontsize=14, labelpad=15)
plt.xlim(Xlimits[0])
plt.ylim(Xlimits[1])
plt.tick_params(axis=u'both', direction='in', which='both')
plt.xlabel('$x$', fontsize = fsize)
plt.ylabel('$y$', fontsize = fsize)
ax.set_aspect('equal')

fig, ax = plt.subplots(1, 1, figsize = (12, 12))        
overplot_rings()
cm = plt.cm.get_cmap('viridis')
sc = plt.scatter(mean_XS_cart[:, :, 0].flatten(), mean_XS_cart[:, :, 1].flatten(), c = traceV.flatten() - traceC.flatten(), vmin = 0, vmax = 10000, s=20, cmap=cm, alpha = .8)
cbar = plt.colorbar(sc, shrink = .85)
cbar.set_label(r'$\rm tr(\langle v v^T\rangle - C)$', rotation=270, fontsize=14, labelpad=15)
plt.xlim(Xlimits[0])
plt.ylim(Xlimits[1])
plt.tick_params(axis=u'both', direction='in', which='both')
plt.xlabel('$x$', fontsize = fsize)
plt.ylabel('$y$', fontsize = fsize)
ax.set_aspect('equal')

# scale length: 3 kpc [for which metallicity?]
# MADE UP NUMBERS ABOUT 14!
# CHECK SIGNS!!
# HAVE TO BUILD DIFFERENT MODEL FOR ALL METALLICITIES

dlnrho_dlnR = (-mean_XS_cyl[:, :, 0]) / 3. #* (mean_XS_cyl[:, :, 0] > 11) + \
#              mean_XS_cyl[:, :, 0] / 3.5 * (mean_XS_cyl[:, :, 0] <= 11)
dlnvR2_dlnR = 5. * (mean_XS_cyl[:, :, 0] > 14)
HWRnumber = 1 + dlnrho_dlnR + dlnvR2_dlnR
#HWRnumber = 2.
plt.imshow(HWRnumber)
plt.colorbar()
vtilde = var_XS_cyl - error_var_XS_cyl + mean_XS_cyl[:, :, 3:, None] * mean_XS_cyl[:, :, None, 3:]
vc = np.sqrt(vtilde[:, :, 1, 1] - HWRnumber * vtilde[:, :, 0, 0])

fig, ax = plt.subplots(1, 1, figsize = (12, 12))        
overplot_rings()
cm = plt.cm.get_cmap('viridis')
sc = plt.scatter(mean_XS_cart[:, :, 0].flatten(), mean_XS_cart[:, :, 1].flatten(), c = vc.flatten(), vmin = 100, vmax = 300, s=20, cmap=cm, alpha = .8)
cbar = plt.colorbar(sc, shrink = .85)
#cbar.set_label(r'$\rm tr(\langle v v^T\rangle - C)$', rotation=270, fontsize=14, labelpad=15)
plt.xlim(Xlimits[0])
plt.ylim(Xlimits[1])
plt.tick_params(axis=u'both', direction='in', which='both')
plt.xlabel('$x$', fontsize = fsize)
plt.ylabel('$y$', fontsize = fsize)
ax.set_aspect('equal')
plt.savefig('plots/rotation/vc_{}.pdf'.format(name), bbox_inches = 'tight')

# 30 degree wedge
deg_wedge = 30.
wedge = (mean_XS_cart[:, :, 0] <= 0) * (abs(mean_XS_cart[:, :, 1]) <= ((-mean_XS_cart[:, :, 0]) * np.tan(deg_wedge/360. * 2*np.pi)))

# plot wedge only
fig, ax = plt.subplots(1, 1, figsize = (12, 12))        
overplot_rings()
cm = plt.cm.get_cmap('viridis')
sc = plt.scatter(mean_XS_cart[wedge, 0].flatten(), mean_XS_cart[wedge, 1].flatten(), c = vc[wedge].flatten(), vmin = 100, vmax = 300, s=20, cmap=cm)
cbar = plt.colorbar(sc, shrink = .85)
plt.scatter(mean_XS_cart[~wedge, 0].flatten(), mean_XS_cart[~wedge, 1].flatten(), c = '#929591', s = 20, alpha = .3) #np.ones_like(mean_XS_cart[~wedge, 1].flatten()) * 0.5)
#cbar.set_label(r'$\rm tr(\langle v v^T\rangle - C)$', rotation=270, fontsize=14, labelpad=15)
plt.xlim(Xlimits[0])
plt.ylim(Xlimits[1])
plt.tick_params(axis=u'both', direction='in', which='both')
plt.xlabel('$x$', fontsize = fsize)
plt.ylabel('$y$', fontsize = fsize)
ax.set_aspect('equal')
plt.savefig('plots/rotation/vc_wegde_{}.pdf'.format(name), bbox_inches = 'tight')

fig, ax = plt.subplots(1, 1, figsize = (8, 7))        
sc = plt.scatter(mean_XS_cyl[:, :, 0].flatten(), vc.flatten(), c = (mean_XS_cyl[:, :, 1].flatten() + .5) % 2*np.pi, s = 10, cmap = 'viridis_r')
cbar = plt.colorbar(sc)
cbar.set_label(r'$\varphi + 0.5$', rotation=270, fontsize=14, labelpad=15)
plt.ylim(0, 400)
plt.xlim(0, 30)
plt.xlabel(r'$R\,\rm [kpc]$', fontsize = fsize)
plt.ylabel(r'$v_c\,\rm [km\,s^{-1}]$', fontsize = fsize)
plt.tick_params(axis=u'both', direction='in', which='both')
plt.savefig('plots/rotation/vc_R_{}.pdf'.format(name), bbox_inches = 'tight')

def AB(coeff, R, feh):
    a0, a1, a2 = coeff
    # expect a0 ~ 1.5
    return a0 + a1 * (R - 5.) + a2 * feh

# make this plot for every star, rather than ensembles!
fig, ax = plt.subplots(1, 1, figsize = (8, 7))        
sc = plt.scatter(mean_XS_cyl[wedge, 0].flatten(), vc[wedge].flatten(), c = (mean_XS_cyl[wedge, 1].flatten() + .5) % 2*np.pi, s = 10, cmap = 'viridis_r')
cbar = plt.colorbar(sc)
cbar.set_label(r'$\varphi + 0.5$', rotation=270, fontsize=14, labelpad=15)
plt.ylim(50, 250)
plt.xlim(0, 30)
plt.axhline(220, linestyle = '--', color = '#929591')
plt.xlabel(r'$R\,\rm [kpc]$', fontsize = fsize)
plt.ylabel(r'$v_c\,\rm [km\,s^{-1}]$', fontsize = fsize)
plt.tick_params(axis=u'both', direction='in', which='both')
plt.savefig('plots/rotation/vc_R_wegde_{}_allFEH.pdf'.format(name), bbox_inches = 'tight')

# derivatives: dln(v_r^2)/dlnR
#for k, cut_name in enumerate(cut_names):
cut_name = 'allFEH'
deg_wedge = 10.
wedge = (abs(mean_XS_cart[:, :, 1]) <= ((-mean_XS_cart[:, :, 0]) * np.tan(deg_wedge/360. * 2*np.pi)))
fig, ax = plt.subplots(1, 1, figsize = (8, 7))        
#plt.scatter(np.log((mean_XS_cyl[wedge, 0].flatten())), np.log(mean_XS_cyl[wedge, 3].flatten() ** 2), s = 10, alpha = .5)
plt.scatter(np.log((mean_XS_cyl[wedge, 0].flatten())), np.log(vtilde[wedge, 0, 0].flatten() ** 2), s = 10, alpha = .5)
plt.xlabel(r'$\rm ln\, R$', fontsize = fsize)
plt.ylabel(r'$\rm ln\, v_R^2$', fontsize = fsize)
#plt.xscale('log')
#plt.yscale('log')
plt.ylim(-5, 10)
plt.xlim(np.log(5), np.log(25))
plt.axvline(np.log(14), linestyle = '--', color = '#929591')
plt.plot((np.log(14), np.log(25)), (3, 3+3*(np.log(25) - np.log(14))), color = 'r', lw = 2, label = r'$m = 3$')
plt.plot((np.log(14), np.log(25)), (3, 3+4*(np.log(25) - np.log(14))), color = 'g', lw = 2, label = r'$m = 4$')
plt.plot((np.log(14), np.log(25)), (3, 3+5*(np.log(25) - np.log(14))), color = 'b', lw = 2, label = r'$m = 5$')
plt.tick_params(axis=u'both', direction='in', which='both')
plt.legend(fontsize  = 14)
plt.savefig('plots/rotation/vR2_R_wegde_{0}_{1}.pdf'.format(name, cut_name), bbox_inches = 'tight')
plt.close()    

# -------------------------------------------------------------------------------'''
