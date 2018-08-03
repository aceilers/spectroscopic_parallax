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
name = 'N{0}_lam{1}_K{2}_parallax'.format(N, lam, Kfold)

print('loading new labels...')   
labels = Table.read('data/training_labels_new_{}.fits'.format(name), format = 'fits')    
labels.rename_column('ra_1', 'ra')
labels.rename_column('dec_1', 'dec')

# -------------------------------------------------------------------------------
# position of the sun
# -------------------------------------------------------------------------------           

# Galactocentric position of the Sun:
X_GC_sun_kpc = 8.122 #[kpc] # (Gravity collaboration 2018)
Z_GC_sun_kpc = 0.025 #[kpc] (e.g. Juric et al. 2008)

# Galactocentric velocity of the Sun:
vX_GC_sun_kms = -11.1 # [km/s]   (e.g. Schoenrich et al. 2009) 
vY_GC_sun_kms =  245.8 # [km/s]  (combined with Sgr A* proper motions from Reid & Brunnthaler 2004)
vZ_GC_sun_kms =  7.8 # [km/s]

# -------------------------------------------------------------------------------
# re-sample each star
# -------------------------------------------------------------------------------           

cuts_finite = np.isfinite(labels['pmra']) * np.isfinite(labels['pmdec']) * np.isfinite(labels['VHELIO_AVG'])
# remove high alpha elements
cuts = cuts_finite * (labels['ALPHA_M'] < .12)
labels = labels[cuts]
N = len(labels) # 33044 stars!

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
    vrs = np.random.normal(labels['VHELIO_AVG'][i], scale = np.sqrt(floor_rv**2 + labels['radial_velocity_error'][i]**2), size = N_sample)
                             
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
    z = cyl.z.to(u.kpc)
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

# -------------------------------------------------------------------------------'''
# load data
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

# -------------------------------------------------------------------------------
# cuts in metallicity (NOT NEEDED AT THE MOMENT)
# -------------------------------------------------------------------------------   

#cut_high_feh = (labels['FE_H'] >= 0.0)
#cut_solar_feh = (labels['FE_H'] >= -0.2) * (labels['FE_H'] < 0.0)
#cut_low_feh = (labels['FE_H'] < -0.2) * (labels['FE_H'] >= -10) # remove [Fe/H] = -9999.0
#cut_names = list(['lowFEH', 'solarFEH', 'hiFEH'])
#cuts = list([cut_low_feh, cut_solar_feh, cut_high_feh])
#labels = labels[cuts]
#mean_XS_cart_n = mean_XS_cart_n[cuts, :]
#var_XS_cart_n = var_XS_cart_n[cuts, :, :]
#mean_XS_cyl_n = mean_XS_cyl_n[cuts, :]
#var_XS_cyl_n = var_XS_cyl_n[cuts, :, :]
#XS_cart_true_n = XS_cart_true_n[cuts, :]
#XS_cyl_true_n = XS_cyl_true_n[cuts, :]

# -------------------------------------------------------------------------------
# for plotting
# -------------------------------------------------------------------------------   

def overplot_ring(r):
    tiny = 1e-4
    thetas = np.arange(0., 2*np.pi + tiny, 0.001 * np.pi)
    xs = r * np.cos(thetas)
    ys = r * np.sin(thetas)
    plt.plot(xs, ys, "k-", alpha=0.2, lw=1, zorder = -np.inf)
    plt.scatter(0, 0, s = 10, color = 'k', alpha=0.2)
    return

def overplot_ring_helio(r):
    tiny = 1e-4
    thetas = np.arange(0., 2*np.pi + tiny, 0.001 * np.pi)
    xs = r * np.cos(thetas) - X_GC_sun_kpc
    ys = r * np.sin(thetas)
    plt.plot(xs, ys, "k-", alpha=0.2, lw=1, zorder = -np.inf)
    # plt.scatter(-X_GC_sun_kpc, 0, s = 10, color = 'k', alpha=0.2)
    return

def overplot_rings():
    for r in [5, 10, 15, 20, 25, 30]:
        overplot_ring(r)
    return

def overplot_rings_helio():
    for r in [5, 10, 15, 20, 25, 30]:
        overplot_ring_helio(r)
    return

Xlimits = [[-30, 10], [-10, 30], [-20, 20], 
           [-200, 200], [-200, 200], [-200, 200]]

# plot [FE/H] vs. radius
cut_z = abs(mean_XS_cart_n[:, 2]) < 0.5
fig, ax = plt.subplots(1, 1, figsize = (8, 8))        
plt.scatter(mean_XS_cyl_n[cut_z, 0], labels['FE_H'][cut_z], s = 5, alpha = 0.5)
plt.ylim(-1.9, 1)
plt.xlim(0, 30)
plt.xlabel(r'$\rm R_{GC}$', fontsize = fsize)
plt.ylabel('[Fe/H]', fontsize = fsize)
plt.tick_params(axis=u'both', direction='in', which='both')
plt.savefig('plots/rotation/FEH_RGC_{}.pdf'.format(name), bbox_inches = 'tight')

# -------------------------------------------------------------------------------
# divide Milky Way into (x, y, z) patches
# -------------------------------------------------------------------------------   

box_size = .5               # that's just half of the box size
all_x = np.arange(-30., 30.01, box_size)
all_y = np.arange(-30., 30.01, box_size)
mean_XS_cyl = np.zeros((len(all_x), len(all_y), 6)) - np.inf
mean_XS_cart = np.zeros((len(all_x), len(all_y), 6)) - np.inf
N_stars = np.zeros((len(all_x), len(all_y)))
error_var_XS_cyl = np.zeros((len(all_x), len(all_y), 3, 3)) - np.inf
vvT_cyl = np.zeros((len(all_x), len(all_y), 3, 3)) - np.inf
mean_HW2 = np.zeros((len(all_x), len(all_y))) - np.inf
mean_feh = np.zeros((len(all_x), len(all_y))) - np.inf
cut_feh = labels['FE_H'] > -100
mean_sigma_mu = np.zeros((len(all_x), len(all_y))) - np.inf
mean_sigma_par = np.zeros((len(all_x), len(all_y))) - np.inf

for i, box_center_x in enumerate(all_x):
    for j, box_center_y in enumerate(all_y):
        cut_patch = (abs(mean_XS_cart_n[:, 2]) < box_size) * (abs(mean_XS_cart_n[:, 0] - box_center_x) < box_size) * (abs(mean_XS_cart_n[:, 1] - box_center_y) < box_size)
        N_stars[i, j] = np.sum(cut_patch)
        if N_stars[i, j] > 0:
            mean_XS_cyl[i, j, :] = np.nanmean(mean_XS_cyl_n[cut_patch], axis = 0) # NEVER USE MEAN PHI -- DOESN'T MAKE SENSE!
            mean_XS_cart[i, j, :] = np.nanmean(mean_XS_cart_n[cut_patch], axis = 0)
            mean_HW2[i, j] = np.nanmean(labels['H'][cut_patch] - labels['w2mpro'][cut_patch])
            mean_feh[i, j] = np.nanmean(labels['FE_H'][cut_patch * cut_feh])
            mean_sigma_mu[i, j] = np.nanmean(np.sqrt(labels['pmra_error'][cut_patch] ** 2 + labels['pmdec_error'][cut_patch] ** 2))
            mean_sigma_par[i, j] = np.nanmean(0.09 * labels['spec_parallax'][cut_patch])
            error_var_XS_cyl[i, j, :, :] = np.nanmean(var_XS_cyl_n[cut_patch], axis=0)
            vvT_cyl[i, j, :, :] = np.dot(XS_cyl_true_n[cut_patch, 3:].T, XS_cyl_true_n[cut_patch, 3:]) / N_stars[i, j]

# -------------------------------------------------------------------------------
# divide Milky Way into annuli of size dr
# -------------------------------------------------------------------------------   

# rotate phi component by 180 degrees (should be done above)
XS_cyl_true_n[:, 1] = XS_cyl_true_n[:, 1] - np.pi 
XS_cyl_true_n[:, 1][XS_cyl_true_n[:, 1] < -np.pi] += 2. * np.pi

# calculate annuli only in 30 degree wedge!  
deg_wedge = 30.
wedge = np.abs(XS_cyl_true_n[:, 1]) < (deg_wedge/360. * 2. * np.pi)

fig, ax = plt.subplots(1, 1, figsize = (12, 12))        
overplot_rings()
cm = plt.cm.get_cmap('viridis')
plt.scatter(XS_cart_true_n[~wedge, 0], XS_cart_true_n[~wedge, 1], c = '#929591', s = 20, alpha = .3) 
sc = plt.scatter(XS_cart_true_n[wedge, 0], XS_cart_true_n[wedge, 1], c = XS_cyl_true_n[wedge, 1], s=20, vmin = -np.pi, vmax = np.pi, cmap=cm, alpha = .8)
cbar = plt.colorbar(sc, shrink = .85)
plt.xlim(Xlimits[0])
plt.ylim(Xlimits[1])
plt.tick_params(axis=u'both', direction='in', which='both')
plt.xlabel('$x$', fontsize = fsize)
plt.ylabel('$y$', fontsize = fsize)
ax.set_aspect('equal')  
plt.savefig('plots/rotation/rotate_phi_wegde_{0}.pdf'.format(name), bbox_inches = 'tight')
plt.close()  

#dz = 1000. # pc
#dr = 1. # kpc
#bins_dr = np.arange(0, 40.01, dr/1.)
#N_stars_annulus = np.zeros_like(bins_dr)
#mean_XS_cyl_annulus = np.zeros((len(bins_dr), 6)) - np.inf
##mean_XS_cart_annulus = np.zeros((len(bins_dr), 6)) - np.inf
#var_XS_cyl_annulus = np.zeros((len(bins_dr), 3, 3)) - np.inf
#error_var_XS_cyl_annulus = np.zeros((len(bins_dr), 3, 3)) - np.inf
#vvT_cyl_annulus = np.zeros((len(bins_dr), 3, 3)) - np.inf
#
#for i, r_center in enumerate(bins_dr):
#    cut_annulus = wedge * (abs(mean_XS_cyl_n[:, 2]) < dz/2.) * (abs(mean_XS_cyl_n[:, 0] - r_center) < dr/2.)
#    N_stars_annulus[i] = np.sum(cut_annulus)
#    #print(i, N_stars_annulus[i])
#    if N_stars_annulus[i] > 0:
#        mean_XS_cyl_annulus[i, :] = np.nanmean(mean_XS_cyl_n[cut_annulus], axis = 0)
##        mean_XS_cart_annulus[i, :] = np.nanmean(mean_XS_cart_n[cut_annulus], axis = 0)
##    if N_stars_annulus[i] > 7:
##        dXS = mean_XS_cyl_n[cut_annulus] - mean_XS_cyl_annulus[i, :][None, :]
##        var_XS_cyl_annulus[i, :, :] = np.dot(dXS[:, 3:].T, dXS[:, 3:]) / (N_stars_annulus[i] - 1.)
#        error_var_XS_cyl_annulus[i, :, :] = np.nanmean(var_XS_cyl_n[cut_annulus], axis=0)
#        vvT_cyl_annulus[i, :, :] = np.dot(XS_cyl_true_n[cut_annulus, 3:].T, XS_cyl_true_n[cut_annulus, 3:]) / (N_stars_annulus[i] - 1.)

dz = 1000. # pc
dr = 1. # kpc
bins_start = np.array([0.])
bins_end = np.array([40.])

cut_z_wedge = wedge * (abs(mean_XS_cyl_n[:, 2]) < dz/2.)
foo = np.append(0., np.sort(mean_XS_cyl_n[cut_z_wedge, 0]))
bar = np.append(np.sort(mean_XS_cyl_n[cut_z_wedge, 0]), 100.)
stars_per_bin = 128
bin_start = 0.5 * (foo[::stars_per_bin] + bar[::stars_per_bin])
bin_end = bin_start[1:]
bin_start = bin_start[:-1]

N_stars_annulus = np.zeros_like(bin_start)
mean_XS_cyl_annulus = np.zeros((len(bin_start), 6)) - np.inf
#mean_XS_cart_annulus = np.zeros((len(bins_dr), 6)) - np.inf
var_XS_cyl_annulus = np.zeros((len(bin_start), 3, 3)) - np.inf
error_var_XS_cyl_annulus = np.zeros((len(bin_start), 3, 3)) - np.inf
vvT_cyl_annulus = np.zeros((len(bin_start), 3, 3)) - np.inf

for i, (r_start, r_end) in enumerate(zip(bin_start, bin_end)):
    cut_annulus = wedge * (abs(mean_XS_cyl_n[:, 2]) < dz/2.) * (mean_XS_cyl_n[:, 0] > r_start) * (mean_XS_cyl_n[:, 0] < r_end)
    N_stars_annulus[i] = np.sum(cut_annulus)
    if N_stars_annulus[i] > 0:
        mean_XS_cyl_annulus[i, :] = np.nanmean(mean_XS_cyl_n[cut_annulus], axis = 0)
        error_var_XS_cyl_annulus[i, :, :] = np.nanmean(var_XS_cyl_n[cut_annulus], axis=0)
        vvT_cyl_annulus[i, :, :] = np.dot(XS_cyl_true_n[cut_annulus, 3:].T, XS_cyl_true_n[cut_annulus, 3:]) / (N_stars_annulus[i] - 1.)

 
# -------------------------------------------------------------------------------
# calculate rotational velocity via Jeans equation
# -------------------------------------------------------------------------------        

vtilde = np.clip(vvT_cyl - error_var_XS_cyl, 0., np.Inf)
dlnrho_dlnR = (-mean_XS_cyl[:, :, 0]) / 3.
dlnvR2_dlnR = (-mean_XS_cyl[:, :, 0]) / 15. 
HWRnumber = 1 + dlnrho_dlnR + dlnvR2_dlnR
vc = np.sqrt(vtilde[:, :, 1, 1] - HWRnumber * vtilde[:, :, 0, 0])

## estimate dlnvR2/dlnR
#cut_name = '_allFEH'
deg_wedge = 10.
wedge10 = (abs(mean_XS_cart[:, :, 1]) <= ((-mean_XS_cart[:, :, 0]) * np.tan(deg_wedge/360. * 2*np.pi)))
fig, ax = plt.subplots(1, 1, figsize = (8, 7))        
plt.scatter(np.log((mean_XS_cyl[wedge10, 0].flatten())), np.log(vtilde[wedge10, 0, 0].flatten() ** 2), s = 10, alpha = .5)
plt.xlabel(r'$\rm ln\, R$', fontsize = fsize)
plt.ylabel(r'$\rm ln\, v_{rr}^2$', fontsize = fsize)
plt.ylim(-5, 10)
plt.xlim(np.log(5), np.log(25))
plt.axvline(np.log(14), linestyle = '--', color = '#929591')
#plt.plot((np.log(14), np.log(25)), (3, 3+3*(np.log(25) - np.log(14))), color = 'r', lw = 2, label = r'$m = 3$')
#plt.plot((np.log(14), np.log(25)), (3, 3+4*(np.log(25) - np.log(14))), color = 'g', lw = 2, label = r'$m = 4$')
#plt.plot((np.log(14), np.log(25)), (3, 3+5*(np.log(25) - np.log(14))), color = 'b', lw = 2, label = r'$m = 5$')
plt.tick_params(axis=u'both', direction='in', which='both')
plt.legend(fontsize  = 14)
plt.savefig('plots/rotation/vrr_wegde10_{0}.pdf'.format(name), bbox_inches = 'tight')
plt.close()    
         
# -------------------------------------------------------------------------------
# maps (x,y) for patches!
# -------------------------------------------------------------------------------        


# plot [FE/H] vs. radius
deg_wedge = 30.
wedge = np.abs(XS_cyl_true_n[:, 1]) < (deg_wedge/360. * 2. * np.pi)
cut_feh = labels['FE_H'] > -100
cut_z_wedge = wedge * (abs(XS_cyl_true_n[:, 2]) < dz/2.) * cut_feh
fig, ax = plt.subplots(1, 1, figsize = (8, 8))        
plt.scatter(XS_cyl_true_n[cut_z_wedge, 0], labels[cut_z_wedge]['FE_H'], s = 5, alpha = 0.5)
plt.ylim(-1.9, 1)
plt.xlim(0, 30)
plt.xlabel(r'$\rm R_{GC}$', fontsize = fsize)
plt.ylabel('[Fe/H]', fontsize = fsize)
plt.tick_params(axis=u'both', direction='in', which='both')
#plt.savefig('plots/rotation/FEH_RGC_{}.pdf'.format(name), bbox_inches = 'tight')

traceC = np.trace(error_var_XS_cyl, axis1=2, axis2=3)
traceVtilde = np.trace(vtilde, axis1=2, axis2=3)

fig, ax = plt.subplots(1, 1, figsize = (12, 12))        
overplot_rings()
cm = plt.cm.get_cmap('viridis')
sc = plt.scatter(mean_XS_cart[:, :, 0].flatten(), mean_XS_cart[:, :, 1].flatten(), c = traceC.flatten(), vmin = 0, vmax = 2000, s=20, cmap=cm, alpha = .8)
cbar = plt.colorbar(sc, shrink = .85)
cbar.set_label(r'$\rm tr(C)$', rotation=270, fontsize=14, labelpad=15)
plt.xlim(Xlimits[0])
plt.ylim(Xlimits[1])
plt.tick_params(axis=u'both', direction='in', which='both')
plt.xlabel('$x$', fontsize = fsize)
plt.ylabel('$y$', fontsize = fsize)
ax.set_aspect('equal')
plt.savefig('plots/rotation/traceC_{}.pdf'.format(name), bbox_inches = 'tight')
plt.close()

fig, ax = plt.subplots(1, 1, figsize = (12, 12))        
overplot_rings()
cm = plt.cm.get_cmap('viridis')
sc = plt.scatter(mean_XS_cart[:, :, 0].flatten(), mean_XS_cart[:, :, 1].flatten(), c = traceVtilde.flatten(), vmin = 0, vmax = 100000, s=20, cmap=cm, alpha = .8)
cbar = plt.colorbar(sc, shrink = .85)
cbar.set_label(r'$\rm tr(\tilde V)$', rotation=270, fontsize=14, labelpad=15)
plt.xlim(Xlimits[0])
plt.ylim(Xlimits[1])
plt.tick_params(axis=u'both', direction='in', which='both')
plt.xlabel('$x$', fontsize = fsize)
plt.ylabel('$y$', fontsize = fsize)
ax.set_aspect('equal')
plt.savefig('plots/rotation/traceVtilde_{}.pdf'.format(name), bbox_inches = 'tight')
plt.close()

fig, ax = plt.subplots(1, 1, figsize = (12, 12))        
overplot_rings()
cm = plt.cm.get_cmap('viridis')
sc = plt.scatter(mean_XS_cart[:, :, 0].flatten(), mean_XS_cart[:, :, 1].flatten(), c = vtilde[:, :, 0, 0].flatten(), vmin = 0, vmax = 8000, s=20, cmap=cm, alpha = .8)
cbar = plt.colorbar(sc, shrink = .85)
cbar.set_label(r'$\overline{v^2_{rr}}$', rotation=270, fontsize=14, labelpad=30)
plt.xlim(Xlimits[0])
plt.ylim(Xlimits[1])
plt.tick_params(axis=u'both', direction='in', which='both')
plt.xlabel('$x$', fontsize = fsize)
plt.ylabel('$y$', fontsize = fsize)
ax.set_aspect('equal')
plt.savefig('plots/rotation/Vtilde_rr_{}.pdf'.format(name), bbox_inches = 'tight')
plt.close()

fig, ax = plt.subplots(1, 1, figsize = (12, 12))        
overplot_rings()
cm = plt.cm.get_cmap('viridis')
sc = plt.scatter(mean_XS_cart[:, :, 0].flatten(), mean_XS_cart[:, :, 1].flatten(), c = vtilde[:, :, 1, 1].flatten(), vmin = 0, vmax = 100000, s=20, cmap=cm, alpha = .8)
cbar = plt.colorbar(sc, shrink = .85)
cbar.set_label(r'$\overline{v^2_{\varphi\varphi}}$', rotation=270, fontsize=14, labelpad=30)
plt.xlim(Xlimits[0])
plt.ylim(Xlimits[1])
plt.tick_params(axis=u'both', direction='in', which='both')
plt.xlabel('$x$', fontsize = fsize)
plt.ylabel('$y$', fontsize = fsize)
ax.set_aspect('equal')
plt.savefig('plots/rotation/Vtilde_pp_{}.pdf'.format(name), bbox_inches = 'tight')
plt.close()

fig, ax = plt.subplots(1, 1, figsize = (12, 12))        
overplot_rings()
cm = plt.cm.get_cmap('viridis')
sc = plt.scatter(mean_XS_cart[:, :, 0].flatten(), mean_XS_cart[:, :, 1].flatten(), c = vtilde[:, :, 2, 2].flatten(), vmin = 0, vmax = 8000, s=20, cmap=cm, alpha = .8)
cbar = plt.colorbar(sc, shrink = .85)
cbar.set_label(r'$\overline{v^2_{zz}}$', rotation=270, fontsize=14, labelpad=30)
plt.xlim(Xlimits[0])
plt.ylim(Xlimits[1])
plt.tick_params(axis=u'both', direction='in', which='both')
plt.xlabel('$x$', fontsize = fsize)
plt.ylabel('$y$', fontsize = fsize)
ax.set_aspect('equal')
plt.savefig('plots/rotation/Vtilde_zz_{}.pdf'.format(name), bbox_inches = 'tight')
plt.close()

fig, ax = plt.subplots(1, 1, figsize = (12, 12))        
overplot_rings()
cm = plt.cm.get_cmap('viridis')
sc = plt.scatter(mean_XS_cart[:, :, 0].flatten(), mean_XS_cart[:, :, 1].flatten(), c = vc.flatten(), vmin = 100, vmax = 300, s=20, cmap=cm, alpha = .8)
cbar = plt.colorbar(sc, shrink = .85)
cbar.set_label(r'$v_{\rm circ}$', rotation=270, fontsize=14, labelpad=15)
plt.xlim(Xlimits[0])
plt.ylim(Xlimits[1])
plt.tick_params(axis=u'both', direction='in', which='both')
plt.xlabel('$x$', fontsize = fsize)
plt.ylabel('$y$', fontsize = fsize)
ax.set_aspect('equal')
plt.savefig('plots/rotation/vc_{}.pdf'.format(name), bbox_inches = 'tight')
plt.close()

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
cbar.set_label(r'$v_{\rm circ}$', rotation=270, fontsize=14, labelpad=15)
plt.xlim(Xlimits[0])
plt.ylim(Xlimits[1])
plt.tick_params(axis=u'both', direction='in', which='both')
plt.xlabel('$x$', fontsize = fsize)
plt.ylabel('$y$', fontsize = fsize)
ax.set_aspect('equal')
plt.savefig('plots/rotation/vc_wegde_{}.pdf'.format(name), bbox_inches = 'tight')
plt.close()

r_rp = vtilde[:, :, 0, 1] / np.sqrt(vtilde[:, :, 0, 0] * vtilde[:, :, 1, 1])

fig, ax = plt.subplots(1, 1, figsize = (12, 12))        
overplot_rings()
cm = plt.cm.get_cmap('viridis')
sc = plt.scatter(mean_XS_cart[:, :, 0].flatten(), mean_XS_cart[:, :, 1].flatten(), c = r_rp.flatten(), vmin = 0., vmax = 0.5, s=20, cmap=cm)
cbar = plt.colorbar(sc, shrink = .85)
#plt.scatter(mean_XS_cart[~wedge, 0].flatten(), mean_XS_cart[~wedge, 1].flatten(), c = '#929591', s = 20, alpha = .3) #np.ones_like(mean_XS_cart[~wedge, 1].flatten()) * 0.5)
cbar.set_label(r'$\tilde{v}_{r\varphi}$', rotation=270, fontsize=14, labelpad=15)
plt.xlim(Xlimits[0])
plt.ylim(Xlimits[1])
plt.tick_params(axis=u'both', direction='in', which='both')
plt.xlabel('$x$', fontsize = fsize)
plt.ylabel('$y$', fontsize = fsize)
ax.set_aspect('equal')
plt.savefig('plots/rotation/vrvp_xy_{}.pdf'.format(name), bbox_inches = 'tight')
plt.close()

fig, ax = plt.subplots(1, 1, figsize = (12, 12))        
overplot_rings()
cm = plt.cm.get_cmap('viridis')
sc = plt.scatter(mean_XS_cart[:, :, 0].flatten(), mean_XS_cart[:, :, 1].flatten(), c = vtilde[:, :, 0, 2].flatten(), vmin = 0., vmax = 300, s=20, cmap=cm)
cbar = plt.colorbar(sc, shrink = .85)
#plt.scatter(mean_XS_cart[~wedge, 0].flatten(), mean_XS_cart[~wedge, 1].flatten(), c = '#929591', s = 20, alpha = .3) #np.ones_like(mean_XS_cart[~wedge, 1].flatten()) * 0.5)
cbar.set_label(r'$\tilde{v}_{rz}$', rotation=270, fontsize=14, labelpad=15)
plt.xlim(Xlimits[0])
plt.ylim(Xlimits[1])
plt.tick_params(axis=u'both', direction='in', which='both')
plt.xlabel('$x$', fontsize = fsize)
plt.ylabel('$y$', fontsize = fsize)
ax.set_aspect('equal')
plt.savefig('plots/rotation/vrvz_xy_{}.pdf'.format(name), bbox_inches = 'tight')
plt.close()

fig, ax = plt.subplots(1, 1, figsize = (12, 12))        
overplot_rings()
cm = plt.cm.get_cmap('viridis')
sc = plt.scatter(mean_XS_cart[wedge, 0].flatten(), mean_XS_cart[wedge, 1].flatten(), c = vtilde[wedge, 1, 2].flatten(), vmin = 0., vmax = 3000, s=20, cmap=cm)
cbar = plt.colorbar(sc, shrink = .85)
plt.scatter(mean_XS_cart[~wedge, 0].flatten(), mean_XS_cart[~wedge, 1].flatten(), c = '#929591', s = 20, alpha = .3) #np.ones_like(mean_XS_cart[~wedge, 1].flatten()) * 0.5)
cbar.set_label(r'$\tilde{v}_{\varphi z}$', rotation=270, fontsize=14, labelpad=15)
plt.xlim(Xlimits[0])
plt.ylim(Xlimits[1])
plt.tick_params(axis=u'both', direction='in', which='both')
plt.xlabel('$x$', fontsize = fsize)
plt.ylabel('$y$', fontsize = fsize)
ax.set_aspect('equal')
plt.savefig('plots/rotation/vpvz_xy_{}.pdf'.format(name), bbox_inches = 'tight')
plt.close()

fig, ax = plt.subplots(1, 1, figsize = (8, 6))        
sc = plt.scatter(mean_XS_cyl[wedge, 0].flatten(), vc[wedge].flatten(), c = (mean_XS_cyl[wedge, 1].flatten() + .5) % 2*np.pi, s = 10, cmap = 'viridis_r')
cbar = plt.colorbar(sc)
cbar.set_label(r'$\varphi + 0.5$', rotation=270, fontsize=14, labelpad=15)
plt.ylim(50, 250)
plt.xlim(0, 25)
plt.axhline(220, linestyle = '--', color = '#929591')
plt.xlabel(r'$R\,\rm [kpc]$', fontsize = fsize)
plt.ylabel(r'$v_c\,\rm [km\,s^{-1}]$', fontsize = fsize)
plt.tick_params(axis=u'both', direction='in', which='both')
plt.savefig('plots/rotation/vc_R_wegde_{}.pdf'.format(name), bbox_inches = 'tight')
plt.close()

# plot with arrows!
fig, ax = plt.subplots(1, 1, figsize = (12, 12))        
plt.quiver(mean_XS_cart[:, :, 0].flatten(), mean_XS_cart[:, :, 1].flatten(), mean_XS_cart[:, :, 3].flatten(), mean_XS_cart[:, :, 4].flatten(), \
        np.clip(mean_HW2.flatten(), 0, 1.5), cmap = 'RdYlBu_r', scale_units='xy', \
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
plt.savefig('plots/rotation/xy_arrow_averaged_{}_HW2.pdf'.format(name), bbox_inches = 'tight')
plt.close()

fig, ax = plt.subplots(1, 1, figsize = (12, 12))        
plt.quiver(mean_XS_cart[:, :, 0].flatten(), mean_XS_cart[:, :, 1].flatten(), mean_XS_cart[:, :, 3].flatten(), mean_XS_cart[:, :, 4].flatten(), \
        np.clip(mean_feh.flatten(), -.5, .3), cmap = 'RdBu_r', scale_units='xy', \
           scale=200, alpha =.8, headwidth = 3, headlength = 4, width = 0.002)
cb = plt.colorbar(shrink = .85)
cb.set_label(r'[Fe/H]', fontsize = 15)
plt.xlim(Xlimits[0])
plt.ylim(Xlimits[1])
overplot_rings()
plt.tick_params(axis=u'both', direction='in', which='both')
plt.xlabel('$x$', fontsize = fsize)
plt.ylabel('$y$', fontsize = fsize)
ax.set_aspect('equal')
plt.savefig('plots/rotation/xy_arrow_averaged_{}_feh.pdf'.format(name), bbox_inches = 'tight')
plt.close()

# plot with arrows!
mu_par = mean_sigma_mu / mean_sigma_par
fig, ax = plt.subplots(1, 1, figsize = (12, 12))        
plt.quiver(mean_XS_cart[:, :, 0].flatten(), mean_XS_cart[:, :, 1].flatten(), mean_XS_cart[:, :, 3].flatten(), mean_XS_cart[:, :, 4].flatten(), \
        np.clip(mean_sigma_mu.flatten(), 0, 2), cmap = 'RdYlBu_r', scale_units='xy', \
           scale=200, alpha =.8, headwidth = 3, headlength = 4, width = 0.002)
cb = plt.colorbar(shrink = .85)
cb.set_label(r'$\sigma_{\mu}$', fontsize = 15)
plt.xlim(Xlimits[0])
plt.ylim(Xlimits[1])
overplot_rings()
plt.tick_params(axis=u'both', direction='in', which='both')
plt.xlabel('$x$', fontsize = fsize)
plt.ylabel('$y$', fontsize = fsize)
ax.set_aspect('equal')
plt.savefig('plots/rotation/xy_arrow_averaged_{}_sigma_mu.pdf'.format(name), bbox_inches = 'tight')
plt.close()

# -------------------------------------------------------------------------------
# new plots vs. R_GC (in radial bins and individual stars)
# -------------------------------------------------------------------------------        

vtilde_annulus = vvT_cyl_annulus - error_var_XS_cyl_annulus
for i in range(3):
    vtilde_annulus[:, i, i] = np.clip(vtilde_annulus[:, i, i], 0., np.inf)
dlnrho_dlnR = (-mean_XS_cyl_annulus[:, 0]) / 4.
dlnvR2_dlnR = (-mean_XS_cyl_annulus[:, 0]) / 15.
HWRnumber = 1 + dlnrho_dlnR + dlnvR2_dlnR
vc_annulus = np.sqrt(vtilde_annulus[:, 1, 1] - HWRnumber * vtilde_annulus[:, 0, 0])

bins_dr = mean_XS_cyl_annulus[:, 0]
fig, ax = plt.subplots(1, 1, figsize = (8, 6))
plt.scatter(bins_dr, vc_annulus)
plt.tick_params(axis=u'both', direction='in', which='both')
plt.xlabel(r'$R_{\rm GC}\,\rm [kpc]$', fontsize = fsize)
plt.ylabel(r'$v_c\,\rm [km\,s^{-1}]$', fontsize = fsize)
plt.ylim(0, 300)
plt.xlim(0, 37)
plt.savefig('plots/rotation/vc_R_annuli_{}.pdf'.format(name), bbox_inches = 'tight')
plt.close()

#fig, ax = plt.subplots(1, 1, figsize = (8, 6))
#plt.scatter(bins_dr, vc)
#plt.tick_params(axis=u'both', direction='in', which='both')
#plt.xlabel(r'$R_{\rm GC}\,\rm [kpc]$', fontsize = fsize)
#plt.ylabel(r'$v_c\,\rm [km\,s^{-1}]$', fontsize = fsize)
#plt.ylim(0, 300)
#plt.xlim(0, 37)
#plt.savefig('plots/rotation/vc_R_annuli_{}.pdf'.format(name), bbox_inches = 'tight')
#plt.close()

fig, ax = plt.subplots(1, 1, figsize = (8, 6))
plt.scatter(bins_dr, vtilde_annulus[:, 0, 0])
plt.tick_params(axis=u'both', direction='in', which='both')
plt.xlabel(r'$R_{\rm GC}\,\rm [kpc]$', fontsize = fsize)
plt.ylabel(r'$\overline{v^2_{rr}}\,\rm [km\,s^{-1}]$', fontsize = fsize)
plt.ylim(0, 4000)
plt.xlim(0, 37)
plt.savefig('plots/rotation/vRvR_R_annuli_{}.pdf'.format(name), bbox_inches = 'tight')
plt.close()

fig, ax = plt.subplots(1, 1, figsize = (8, 6))
plt.scatter(bins_dr, vtilde_annulus[:, 1, 1])
plt.tick_params(axis=u'both', direction='in', which='both')
plt.xlabel(r'$R_{\rm GC}\,\rm [kpc]$', fontsize = fsize)
plt.ylabel(r'$\overline{v^2_{\varphi\varphi}}\,\rm [km\,s^{-1}]$', fontsize = fsize)
plt.ylim(0, 60000)
plt.xlim(0, 37)
plt.savefig('plots/rotation/vpvp_R_annuli_{}.pdf'.format(name), bbox_inches = 'tight')
plt.close()

fig, ax = plt.subplots(1, 1, figsize = (8, 6))
plt.scatter(bins_dr, vtilde_annulus[:, 2, 2])
plt.tick_params(axis=u'both', direction='in', which='both')
plt.xlabel(r'$R_{\rm GC}\,\rm [kpc]$', fontsize = fsize)
plt.ylabel(r'$\overline{v^2_{zz}}\,\rm [km\,s^{-1}]$', fontsize = fsize)
plt.ylim(0, 4000)
plt.xlim(0, 37)
plt.savefig('plots/rotation/vzvz_R_annuli_{}.pdf'.format(name), bbox_inches = 'tight')
plt.close()

fig, ax = plt.subplots(1, 1, figsize = (8, 6))
plt.scatter(bins_dr, vtilde_annulus[:, 0, 1])
plt.tick_params(axis=u'both', direction='in', which='both')
plt.xlabel(r'$R_{\rm GC}\,\rm [kpc]$', fontsize = fsize)
plt.ylabel(r'$\overline{v^2_{r\varphi}}\,\rm [km\,s^{-1}]$', fontsize = fsize)
plt.ylim(-4000, 4000)
plt.xlim(0, 37)
plt.savefig('plots/rotation/vrvp_R_annuli_{}.pdf'.format(name), bbox_inches = 'tight')
plt.close()

fig, ax = plt.subplots(1, 1, figsize = (8, 6))
plt.scatter(bins_dr, vtilde_annulus[:, 0, 2])
plt.tick_params(axis=u'both', direction='in', which='both')
plt.xlabel(r'$R_{\rm GC}\,\rm [kpc]$', fontsize = fsize)
plt.ylabel(r'$\overline{v^2_{rz}}\,\rm [km\,s^{-1}]$', fontsize = fsize)
plt.ylim(-500, 500)
plt.xlim(0, 37)
plt.savefig('plots/rotation/vrvz_R_annuli_{}.pdf'.format(name), bbox_inches = 'tight')
plt.close()

fig, ax = plt.subplots(1, 1, figsize = (8, 6))
plt.scatter(bins_dr, vtilde_annulus[:, 1, 2])
plt.tick_params(axis=u'both', direction='in', which='both')
plt.xlabel(r'$R_{\rm GC}\,\rm [kpc]$', fontsize = fsize)
plt.ylabel(r'$\overline{v^2_{\varphi z}}\,\rm [km\,s^{-1}]$', fontsize = fsize)
plt.ylim(-4000, 4000)
plt.xlim(0, 37)
plt.savefig('plots/rotation/vpvz_R_annuli_{}.pdf'.format(name), bbox_inches = 'tight')
plt.close()
        
# -------------------------------------------------------------------------------
# old plots
# -------------------------------------------------------------------------------        

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
