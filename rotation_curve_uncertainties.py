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
#from plotting_helpers import histcont
import matplotlib.gridspec as gridspec
import scipy.interpolate as interpol



# -------------------------------------------------------------------------------
# plotting settings
# -------------------------------------------------------------------------------

matplotlib.rcParams['ytick.labelsize'] = 18
matplotlib.rcParams['xtick.labelsize'] = 18
matplotlib.rc('text', usetex=True)
fsize = 18

# -------------------------------------------------------------------------------
# open inferred labels
# -------------------------------------------------------------------------------

N = 44784
Kfold = 2
lam = 30
name = 'N{0}_lam{1}_K{2}_offset0.0483_parallax'.format(N, lam, Kfold)

print('loading new labels...')   
labels = Table.read('data/training_labels_new_{}.fits'.format(name), format = 'fits')    
labels.rename_column('ra_1', 'ra')
labels.rename_column('dec_1', 'dec')

# -------------------------------------------------------------------------------
# position of the sun
# -------------------------------------------------------------------------------           

# Galactocentric position of the Sun:
X_GC_sun_kpc = 8.122 # [kpc] pm 0.031 kpc (Gravity collaboration 2018)
Z_GC_sun_kpc = 0.025 # [kpc] (e.g. Juric et al. 2008)

galcen_distance = X_GC_sun_kpc * u.kpc
pm_gal_sgrA = [-6.379, -0.202] * u.mas/u.yr # from Reid & Brunthaler 2004
vY_GC_sun_kms, vZ_GC_sun_kms = -(galcen_distance * pm_gal_sgrA).to(u.km/u.s, u.dimensionless_angles())

# Galactocentric velocity of the Sun:
vX_GC_sun_kms = -11.1 * u.km/u.s # [km/s]   (e.g. Schoenrich et al. 2009) 
#vY_GC_sun_kms =  245.8 # [km/s]  (combined with Sgr A* proper motions from Reid & Brunnthaler 2004)
#vZ_GC_sun_kms =  7.8 # [km/s]

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

# -------------------------------------------------------------------------------
# new plot
# -------------------------------------------------------------------------------           
#distance = (labels['spec_parallax'] * u.mas).to(u.parsec, equivalencies = u.parallax())
#
#fig, ax = plt.subplots(1, 1, figsize = (8, 4))        
#cm = plt.cm.get_cmap('RdBu_r')
#sc = ax.scatter(labels['phot_g_mean_mag'], distance/1000., c = np.log10(labels['parallax_error']/labels['spec_parallax_err']), s=5, cmap=cm, alpha = .8, vmin = -1, vmax = 1, rasterized = True)
#cb = fig.colorbar(sc)
#cb.set_label(r'$\log_{10}(\sigma_{\varpi}^{(\rm a)}/\sigma_{\varpi}^{(\rm sp)})$', rotation=90, fontsize=18, labelpad=30)
#plt.tight_layout()
#ax.set_xlabel(r'$G\,\rm [mag]$', fontsize = fsize)
#ax.set_ylabel(r'$\rm spectrophotometric\,distance\,[kpc]$', fontsize = fsize)
#ax.tick_params(axis=u'both', direction='in', which='both')
#ax.set_ylim(0, 20)
#plt.savefig('paper/precision.pdf', bbox_inches = 'tight', pad_inches=.2)
#
#dis = (distance/1000.).value
#precision = labels['parallax_error']/labels['spec_parallax_err']
#mag = (labels['phot_g_mean_mag'] > (14 - 0.5)) * (labels['phot_g_mean_mag'] <= (14 + 0.5))
#for i in range(20):    
#    row = (dis[mag] > (i - 0.5)) * (dis[mag] <= (i + 0.5))
#    prec = np.median(precision[mag][row])
#    print(i, prec)
#
# -------------------------------------------------------------------------------
# HRD plot
# -------------------------------------------------------------------------------           
#
#distance = (labels['spec_parallax'] * u.mas).to(u.parsec, equivalencies = u.parallax())
#MG = labels['phot_g_mean_mag'] - 5. * np.log10(distance.value) + 5
#sn = 10
#cut_snr = labels['parallax_over_error'] > sn
#res = labels['spec_parallax'] - labels['parallax']
#sc = plt.scatter(labels['bp_rp'][cut_snr], MG[cut_snr], c = res[cut_snr], vmin = -.5, vmax = .5, s = 8, alpha = 0.6)
#cbar = plt.colorbar(sc, shrink = .85)
#cbar.set_label(r'$\varpi^{\rm (sp)}-\varpi^{\rm (a)}$', rotation=270, fontsize=14, labelpad=30)
#plt.xlabel(r'$G_{\rm BP}-G_{\rm RP}$', fontsize=14)
#plt.ylabel(r'$G$', fontsize=14)
#plt.ylim(3, -3)
#plt.xlim(5, 1)
#plt.tick_params(axis=u'both', direction='in', which='both', right = 'on', top = 'on')
#plt.title(r'$\varpi^{{\rm (a)}}/\sigma_{{\varpi^{{\rm (a)}}}} > {}$'.format(sn), fontsize=14)
#plt.savefig('plots/HDR_{}_R08.pdf'.format(sn), bbox_inches = 'tight', pad_inches=.2)
#
#mean_XS_cart_n = np.zeros((N, 6))
#var_XS_cart_n = np.zeros((N, 3, 3))
#mean_XS_cyl_n = np.zeros((N, 6))
#var_XS_cyl_n = np.zeros((N, 3, 3))
#XS_cart_true_n = np.zeros((N, 6))
#XS_cyl_true_n = np.zeros((N, 6))
#
## RADIAL VELOCITY UNCERTAINTY?
#floor_rv = 0.1 # km/s
#
#for i in range(N): #23697
#    
#    if i%1000 == 0: print('working on star {0} out of {1}'.format(i, N))
#    spec_par = np.random.normal(labels['spec_parallax'][i], scale = labels['spec_parallax_err'][i], size = N_sample) * u.mas
#    #assert np.all(spec_par > 0)
#    distance = spec_par.to(u.parsec, equivalencies = u.parallax())
#    distance_true = (labels['spec_parallax'][i] * u.mas).to(u.parsec, equivalencies = u.parallax())
#    
#    pmras = np.random.normal(labels['pmra'][i], scale = labels['pmra_error'][i], size = N_sample)
#    pmdecs = np.random.normal(labels['pmdec'][i], scale = labels['pmdec_error'][i], size = N_sample)
#    vrs = np.random.normal(labels['VHELIO_AVG'][i], scale = np.sqrt(floor_rv**2), size = N_sample)
#                             
#    # -------------------------------------------------------------------------------
#    # calculate cartesian coordinates
#    # -------------------------------------------------------------------------------           
#    
#    cs = coord.ICRS(ra = np.ones((N_sample)) * labels['ra'][i] * u.degree, 
#                    dec = np.ones((N_sample)) * labels['dec'][i] * u.degree, 
#                    distance = distance, 
#                    pm_ra_cosdec = pmras * u.mas/u.yr, 
#                    pm_dec = pmdecs * u.mas/u.yr, 
#                    radial_velocity = vrs * u.km/u.s)
#    
#    cs_true = coord.ICRS(ra = labels['ra'][i] * u.degree, 
#                dec = labels['dec'][i] * u.degree, 
#                distance = distance_true, 
#                pm_ra_cosdec = labels['pmra'][i] * u.mas/u.yr, 
#                pm_dec = labels['pmdec'][i] * u.mas/u.yr, 
#                radial_velocity = labels['VHELIO_AVG'][i] * u.km/u.s)
#
#    gc = coord.Galactocentric(galcen_distance = X_GC_sun_kpc * u.kpc, 
#                          galcen_v_sun = coord.CartesianDifferential([-vX_GC_sun_kms, vY_GC_sun_kms, vZ_GC_sun_kms]), # * u.km/u.s,
#                          z_sun = Z_GC_sun_kpc*u.kpc)
#
#    galcen = cs.transform_to(gc)
#    xs, ys, zs = galcen.x.to(u.kpc), galcen.y.to(u.kpc), galcen.z.to(u.kpc)
#    vxs, vys, vzs = galcen.v_x, galcen.v_y, galcen.v_z
#    XS_cart = np.vstack([xs, ys, zs, vxs, vys, vzs]).T.value
#   
#    cyl = galcen.represent_as('cylindrical')   
#    vr = cyl.differentials['s'].d_rho.to(u.km/u.s)
#    vphi = (cyl.rho * cyl.differentials['s'].d_phi).to(u.km/u.s, u.dimensionless_angles())
#    vz = cyl.differentials['s'].d_z.to(u.km/u.s)
#    rho = cyl.rho.to(u.kpc)
#    phi = cyl.phi
#    z = cyl.z.to(u.kpc)
#    XS_cyl = np.vstack([rho, phi, z, vr, vphi, vz]).T.value
#    
#    galcen_true = cs_true.transform_to(gc)
#    xs_true, ys_true, zs_true = galcen_true.x.to(u.kpc), galcen_true.y.to(u.kpc), galcen_true.z.to(u.kpc)
#    vxs_true, vys_true, vzs_true = galcen_true.v_x, galcen_true.v_y, galcen_true.v_z
#    XS_cart_true_n[i, :] = np.vstack([xs_true, ys_true, zs_true, vxs_true, vys_true, vzs_true]).T.value   
#    cyl_true = galcen_true.represent_as('cylindrical')   
#    vr_true = cyl_true.differentials['s'].d_rho.to(u.km/u.s)
#    vphi_true = (cyl_true.rho * cyl_true.differentials['s'].d_phi).to(u.km/u.s, u.dimensionless_angles())
#    vz_true = cyl_true.differentials['s'].d_z.to(u.km/u.s)    
#    rho_true = cyl_true.rho.to(u.kpc)
#    phi_true = cyl_true.phi
#    z_true = cyl_true.z.to(u.kpc)  
#    XS_cyl_true_n[i, :] = np.vstack([rho_true, phi_true, z_true, vr_true, vphi_true, vz_true]).T.value
#
#    mean_XS_cart_n[i, :] = np.nanmean(XS_cart, axis = 0)
#    dXS_cart = XS_cart - mean_XS_cart_n[i, :][None, :]
#    var_XS_cart_n[i, :, :] = np.dot(dXS_cart[:, 3:].T, dXS_cart[:, 3:]) / (N_sample - 1.)
#    
#    mean_XS_cyl_n[i, :] = np.nanmean(XS_cyl, axis = 0)
#    dXS_cyl = XS_cyl - mean_XS_cyl_n[i, :][None, :]
#    var_XS_cyl_n[i, :, :] = np.dot(dXS_cyl[:, 3:].T, dXS_cyl[:, 3:]) / (N_sample - 1.)
#
#hdu = fits.PrimaryHDU(mean_XS_cart_n)
#hdu.writeto('data/mean_cart_n_{}.fits'.format(name), overwrite = True)
#hdu = fits.PrimaryHDU(var_XS_cart_n)
#hdu.writeto('data/var_cart_n_{}.fits'.format(name), overwrite = True)
#hdu = fits.PrimaryHDU(mean_XS_cyl_n)
#hdu.writeto('data/mean_cyl_n_{}.fits'.format(name), overwrite = True)
#hdu = fits.PrimaryHDU(var_XS_cyl_n)
#hdu.writeto('data/var_cyl_n_{}.fits'.format(name), overwrite = True)
#hdu = fits.PrimaryHDU(XS_cyl_true_n)
#hdu.writeto('data/true_cyl_n_{}.fits'.format(name), overwrite = True)
#hdu = fits.PrimaryHDU(XS_cart_true_n)
#hdu.writeto('data/true_cart_n_{}.fits'.format(name), overwrite = True)

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

def overplot_ring(r, ax = None):
    tiny = 1e-4
    thetas = np.arange(0., 2*np.pi + tiny, 0.001 * np.pi)
    xs = r * np.cos(thetas)
    ys = r * np.sin(thetas)
    if ax:
        ax.plot(xs, ys, "k-", alpha=0.2, lw=1, zorder = np.inf)
        ax.scatter(0, 0, s = 20, color = 'k', alpha=0.2, marker = 'x')
    else:
        plt.plot(xs, ys, "k-", alpha=0.2, lw=1, zorder = np.inf)
        plt.scatter(0, 0, s = 20, color = 'k', alpha=0.2, marker = 'x')
    return

def overplot_ring_helio(r, ax = None):
    tiny = 1e-4
    thetas = np.arange(0., 2*np.pi + tiny, 0.001 * np.pi)
    xs = r * np.cos(thetas) - X_GC_sun_kpc
    ys = r * np.sin(thetas)
    if ax:
        ax.plot(xs, ys, "k:", alpha=0.2, lw=1, zorder = -np.inf)
    else:
        plt.plot(xs, ys, "k:", alpha=0.2, lw=1, zorder = -np.inf)
    # plt.scatter(-X_GC_sun_kpc, 0, s = 10, color = 'k', alpha=0.2)
    return

def overplot_rings(ax = None):
    for r in [5, 10, 15, 20, 25, 30]:
        overplot_ring(r, ax = ax)
    return

def overplot_rings_helio(ax = None):
    for r in [5, 10, 15, 20, 25, 30]:
        overplot_ring_helio(r, ax = ax)
    return

Xlimits = [[-25.2, 10], [-12, 23], [-20, 20], 
           [-200, 200], [-200, 200], [-200, 200]]

# -------------------------------------------------------------------------------
# divide Milky Way into (x, y, z) patches
# ------------------------------------------------------------------------------- 

# take wegde in z 
deg_wedge_in_z = 6.
cut_vz = (XS_cart_true_n[:, 5] < 100)
wedge_z = (np.abs(mean_XS_cyl_n[:, 2])/(mean_XS_cyl_n[:, 0])) < np.tan(deg_wedge_in_z/360. * 2. * np.pi)
cut_z = np.logical_or(abs(mean_XS_cyl_n[:, 2]) < 0.5, wedge_z) * cut_vz

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
mean_rv = np.zeros((len(all_x), len(all_y))) - np.inf

for i, box_center_x in enumerate(all_x):
    for j, box_center_y in enumerate(all_y):
        #cut_patch = (abs(mean_XS_cart_n[:, 2]) < box_size) * (abs(mean_XS_cart_n[:, 0] - box_center_x) < box_size) * (abs(mean_XS_cart_n[:, 1] - box_center_y) < box_size)
        cut_patch = cut_z * (abs(mean_XS_cart_n[:, 0] - box_center_x) < box_size) * (abs(mean_XS_cart_n[:, 1] - box_center_y) < box_size)
        N_stars[i, j] = np.sum(cut_patch)
        if N_stars[i, j] > 0:
            mean_XS_cyl[i, j, :] = np.nanmean(mean_XS_cyl_n[cut_patch], axis = 0) # NEVER USE MEAN PHI -- DOESN'T MAKE SENSE!
            mean_XS_cart[i, j, :] = np.nanmean(mean_XS_cart_n[cut_patch], axis = 0)
            mean_HW2[i, j] = np.nanmean(labels['H'][cut_patch] - labels['w2mpro'][cut_patch])
            mean_feh[i, j] = np.nanmean(labels['FE_H'][cut_patch * cut_feh])
            mean_rv[i, j] = np.nanmean(labels['VHELIO_AVG'][cut_patch])
            mean_sigma_mu[i, j] = np.nanmean(np.sqrt(labels['pmra_error'][cut_patch] ** 2 + labels['pmdec_error'][cut_patch] ** 2))
            mean_sigma_par[i, j] = np.nanmean(0.09 * labels['spec_parallax'][cut_patch])
            error_var_XS_cyl[i, j, :, :] = np.nanmean(var_XS_cyl_n[cut_patch], axis=0)
            vvT_cyl[i, j, :, :] = np.dot(mean_XS_cyl_n[cut_patch, 3:].T, mean_XS_cyl_n[cut_patch, 3:]) / N_stars[i, j]

# -------------------------------------------------------------------------------
# Bootstrap uncertainties (not working yet!)
# -------------------------------------------------------------------------------        

def Bootstrap(indices, mean_XS_cyl_n, var_XS_cyl_n, nsample = 100):
        
    all_rr = np.zeros((nsample))
    all_pp = np.zeros((nsample))
    all_rr2 = np.zeros((nsample))
    all_pp2 = np.zeros((nsample))
#    all_vc = np.zeros((nsample))
    for i in range(nsample):
        ind_i = np.random.choice(indices, size=len(indices), replace=True)
        error_var_XS_cyl_annulus_i = np.nanmean(var_XS_cyl_n[ind_i], axis=0)
        vvT_cyl_annulus_i = np.dot(mean_XS_cyl_n[ind_i, 3:].T, mean_XS_cyl_n[ind_i, 3:]) / (len(indices))
        vt_ann_i = vvT_cyl_annulus_i - error_var_XS_cyl_annulus_i
        all_rr[i] = np.sqrt(vt_ann_i[0, 0])
        all_pp[i] = np.sqrt(vt_ann_i[1, 1])
        all_rr2[i] = vt_ann_i[0, 0]
        all_pp2[i] = vt_ann_i[1, 1]
#        all_vc[i] = np.sqrt(vt_ann_i[1, 1] - vt_ann_i[0, 0] * HWR)
    err_rr = np.nanpercentile(all_rr, (16, 50, 84))
    err_pp = np.nanpercentile(all_pp, (16, 50, 84))
    err_rr2 = np.nanpercentile(all_rr2, (16, 50, 84))
    err_pp2 = np.nanpercentile(all_pp2, (16, 50, 84))
#    err_vc = np.nanpercentile(all_pp, (16, 50, 84))
    return err_rr, err_pp, err_rr2, err_pp2

#def Bootstrap(XS, var_XS, n_sample):  
#    
#    vtilde_sample = np.zeros((n_sample, 3, 3))
#    vc_annulus_sample = np.zeros((n_sample))
#    
#    for i in range(n_sample):
#        ind = np.arange(len(XS))
#        ind_sample = np.random.choice(ind, size = len(XS))
#        XS_sample = XS[ind_sample, :]
#        var_XS_sample = var_XS[ind_sample, :]
#    
#        error_var_XS_sample = np.nanmean(var_XS_sample, axis=0)
#        vvT_sample = np.dot(XS_sample[:, 3:].T, XS_sample[:, 3:]) / (len(XS))    
#        vtilde_sample[i, :, :] = vvT_sample - error_var_XS_sample
#        HWRnumber = 1 + np.nanmean(XS_sample[:, 0]) / rho_R_exp + np.nanmean(XS_sample[:, 0]) / vrr_R_exp
#        vc_annulus_sample[i] = np.sqrt(vtilde_sample[i, 1, 1] - HWRnumber * vtilde_sample[i, 0, 0]) 
#
#    vtilde_annulus_vrr_err = np.nanpercentile(vtilde_sample[:, 0, 0], (16, 50, 84))
#    vtilde_annulus_vpp_err = np.nanpercentile(vtilde_sample[:, 1, 1], (16, 50, 84))
#    vc_annulus_err = np.nanpercentile(vc_annulus_sample, (16, 50, 84))
#    
#    return vtilde_annulus_vrr_err, vtilde_annulus_vpp_err, vc_annulus_err
#
#all_vtilde_annulus_vrr_err = np.zeros((len(bin_start), 3))
#all_vtilde_annulus_vpp_err = np.zeros((len(bin_start), 3))
#all_vc_annulus_err = np.zeros((len(bin_start), 3))
#for i, (r_start, r_end) in enumerate(zip(bin_start, bin_end)):
#    cut_annulus = cut_z_wedge * (mean_XS_cyl_n[:, 0] > r_start) * (mean_XS_cyl_n[:, 0] < r_end)    
#    all_vtilde_annulus_vrr_err[i, :], all_vtilde_annulus_vpp_err[i, :], all_vc_annulus_err[i, :] = Bootstrap(mean_XS_cyl_n[cut_annulus], var_XS_cyl_n[cut_annulus], 100)

# -------------------------------------------------------------------------------
# divide Milky Way into annuli of size dr
# -------------------------------------------------------------------------------   

# rotate phi component by 180 degrees (to avoid break in wedge)
XS_cyl_true_n[:, 1] = XS_cyl_true_n[:, 1] - np.pi 
XS_cyl_true_n[:, 1][XS_cyl_true_n[:, 1] < -np.pi] += 2. * np.pi
mean_XS_cyl_n[:, 1] = mean_XS_cyl_n[:, 1] - np.pi 
mean_XS_cyl_n[:, 1][mean_XS_cyl_n[:, 1] < -np.pi] += 2. * np.pi

# calculate annuli only in 60 degree wedge!  
deg_wedge = 30.
wedge1d = abs(mean_XS_cyl_n[:, 1]) < (deg_wedge/360. * 2. * np.pi) * cut_vz
#wedge1d = (mean_XS_cyl_n[:, 1] < (deg_wedge/360. * 2. * np.pi)) * (mean_XS_cyl_n[:, 1] > 0) * cut_vz
#wedge1d = (-mean_XS_cyl_n[:, 1] < (deg_wedge/360. * 2. * np.pi)) * (mean_XS_cyl_n[:, 1] < 0) * cut_vz


# -------------------------------------------------------------------------------
# test of vertical gradient
# -------------------------------------------------------------------------------   

cut_z_plus = mean_XS_cyl_n[:, 2] > 0
cut_z_minus = mean_XS_cyl_n[:, 2] < 0

# -------------------------------------------------------------------------------
# continue from above...
# -------------------------------------------------------------------------------   

dz = 1. # kpc
bins_start = np.array([1.])
bins_end = np.array([40.])

cut_z_wedge = wedge1d * cut_z #* cut_z_minus
foo = np.append(0., np.sort(mean_XS_cyl_n[cut_z_wedge, 0]))
bar = np.append(np.sort(mean_XS_cyl_n[cut_z_wedge, 0]), 100.)
min_stars_per_bin = 3
#bin_start = 0.5 * (foo[::stars_per_bin] + bar[::stars_per_bin])
#bin_end = bin_start[1:]
#bin_start = bin_start[:-1]
#
#new_bins = []
#d_max = 1.5 #kpc
#for i in range(len(bin_start)-1):
#    if (bin_start[i+1] - bin_start[i]) <= d_max:
#        new_bins.append(bin_start[i])
#    elif (bin_start[i+1] - bin_start[i]) > d_max:
#        new_bins.append(bin_start[i] + d_max)  
#    if i == (len(bin_start)-2):
#        while new_bins[-1] <= 25:
#            new_bins.append(new_bins[-1] + d_max)

#cut_within_sun = mean_XS_cyl_n[:, 0] < 8.122
#labels_within_sun = labels[cut_z_wedge * cut_within_sun]
#Table.write(labels_within_sun, 'data/spectrophotometric_parallax_HER2018_sun.fits', format = 'fits')

new_bins = []
dr = 0.5
bin_start = 0.5
new_bins.append(bin_start)
bin_end = bin_start + dr
while bin_end <= 25.:
    cut_annulus = (mean_XS_cyl_n[cut_z_wedge, 0] >= bin_start) * (mean_XS_cyl_n[cut_z_wedge, 0] < bin_end)
    if sum(cut_annulus) >= min_stars_per_bin:
        new_bins.append(bin_end)
        print('start: ', bin_start, 'end: ', bin_end, 'stars: ', sum(cut_annulus))
        bin_start = bin_end
        bin_end += dr
    else:
        foo_index = np.where(foo >= bin_end)[0][min_stars_per_bin - sum(cut_annulus)]
        bin_end = max(foo[foo_index], bin_start + dr)
        cut_annulus = (mean_XS_cyl_n[cut_z_wedge, 0] >= bin_start) * (mean_XS_cyl_n[cut_z_wedge, 0] < bin_end)
        new_bins.append(bin_end)
        print('start: ', bin_start, 'end: ', bin_end, 'stars: ', sum(cut_annulus))
        bin_start = bin_end
            
bin_end = new_bins[1:]
bin_start = new_bins[:-1]

N_stars_annulus = np.zeros_like(bin_start)
mean_XS_cyl_annulus = np.zeros((len(bin_start), 6)) - np.inf
median_XS_cyl_annulus = np.zeros((len(bin_start), 6)) - np.inf
error_var_XS_cyl_annulus = np.zeros((len(bin_start), 3, 3)) - np.inf
vvT_cyl_annulus = np.zeros((len(bin_start), 3, 3)) - np.inf
#mean_feh_annulus = np.zeros_like(bin_start)
vt_ann = np.zeros((len(bin_start), 3, 3)) - np.inf # copy of vtilde_annulus
vt_ann_err_rr = np.zeros((len(bin_start), 3))
vt_ann_err_pp = np.zeros((len(bin_start), 3))
vt_ann_err_rr2 = np.zeros((len(bin_start), 3))
vt_ann_err_pp2 = np.zeros((len(bin_start), 3))
indices = np.arange(mean_XS_cyl_n.shape[0])

for i, (r_start, r_end) in enumerate(zip(bin_start, bin_end)):
    #cut_annulus = wedge * (abs(mean_XS_cyl_n[:, 2]) < dz/2.) * (mean_XS_cyl_n[:, 0] > r_start) * (mean_XS_cyl_n[:, 0] < r_end)
    cut_annulus = cut_z_wedge * (mean_XS_cyl_n[:, 0] > r_start) * (mean_XS_cyl_n[:, 0] < r_end)
    N_stars_annulus[i] = np.sum(cut_annulus)
    if N_stars_annulus[i] > 0:
        mean_XS_cyl_annulus[i, :] = np.nanmean(mean_XS_cyl_n[cut_annulus], axis = 0)
        median_XS_cyl_annulus[i, :] = np.nanmedian(mean_XS_cyl_n[cut_annulus], axis = 0)
        error_var_XS_cyl_annulus[i, :, :] = np.nanmean(var_XS_cyl_n[cut_annulus], axis=0)
        vvT_cyl_annulus[i, :, :] = np.dot(mean_XS_cyl_n[cut_annulus, 3:].T, mean_XS_cyl_n[cut_annulus, 3:]) / (N_stars_annulus[i])
        vt_ann[i, :, :] = vvT_cyl_annulus[i, :, :] - error_var_XS_cyl_annulus[i, :, :]
        err_rr, err_pp, err_rr2, err_pp2 = Bootstrap(indices[cut_annulus], mean_XS_cyl_n, var_XS_cyl_n)
        vt_ann_err_rr[i, :] = err_rr
        vt_ann_err_pp[i, :] = err_pp 
        vt_ann_err_rr2[i, :] = err_rr2
        vt_ann_err_pp2[i, :] = err_pp2 
#        cut_feh = labels['FE_H'] > -100
#        mean_feh_annulus[i] = np.nanmean(labels['FE_H'][cut_annulus * cut_feh])  

       
               
# -------------------------------------------------------------------------------
# Figure 7 for HER18
# -------------------------------------------------------------------------------           

#cut_feh = abs(labels['FE_H']) < 10
#cuts = cut_z * cut_feh * cut_vz
#fig, ax = plt.subplots(1, 1, figsize = (8, 8))
#sc = plt.quiver(XS_cart_true_n[cuts, 0], XS_cart_true_n[cuts, 1], XS_cart_true_n[cuts, 3], XS_cart_true_n[cuts, 4], 
#           np.clip(labels['FE_H'][cuts], -.7, .7), cmap = 'RdBu_r', scale_units='xy', 
#           scale=200, alpha =.8, headwidth = 3, headlength = 5, width = 0.0015, rasterized = True)
##cb = plt.colorbar(shrink = .8)
##cb.set_label(r'$v_z$', fontsize = 15)
#plt.xlim(-25, 5)
#plt.ylim(-10, 20)
#overplot_rings()
#overplot_rings_helio()
#plt.tick_params(axis=u'both', direction='in', which='both')
#plt.xlabel('$x$ [kpc]', fontsize = fsize)
#plt.ylabel('$y$ [kpc]', fontsize = fsize)
#ax.set_aspect('equal')
#fig.subplots_adjust(right = 0.8)
#cbar_ax = fig.add_axes([1, 0.08, 0.03, .86])
#cb = fig.colorbar(sc, cax=cbar_ax)
#cb.set_label(r'[Fe/H]', fontsize = fsize)
#plt.tight_layout()
#plt.savefig('paper/map.pdf', bbox_inches = 'tight', pad_inches=.2, dpi=250)
#plt.close()
        
# -------------------------------------------------------------------------------
# fit dlnvRR/dlnR
# -------------------------------------------------------------------------------

bins_dr = mean_XS_cyl_annulus[:, 0]
idx5 = sum(bins_dr < 5)
vtilde_annulus = vvT_cyl_annulus - error_var_XS_cyl_annulus

def exp_fit(theta, R):
    fit = theta[0] * np.exp(-R / theta[1])
    return fit

def chi2(theta, R_obs, v_obs, sigma):
    fit = exp_fit(theta, R_obs)
    return np.nansum((fit - v_obs)**2 / sigma**2)

# exponential fit to v_RR  
sigmas_rr = 0.5 * (vt_ann_err_rr[:, 2] - vt_ann_err_rr[:, 0])
sigmas_pp = 0.5 * (vt_ann_err_pp[:, 2] - vt_ann_err_pp[:, 0])
x0 = np.array([50, 10])  
res = op.minimize(chi2, x0, args=(bins_dr[idx5:], np.sqrt(vtilde_annulus[idx5:, 0, 0]), sigmas_rr[idx5:]), method='L-BFGS-B', options={'maxfun':50000})
print(res)
theta_fit = res.x


rho_R_exp = 3. # kpc
vrr_R_exp = theta_fit[1] # kpc
print('vrr_R_exp = {}'.format(vrr_R_exp))

# -------------------------------------------------------------------------------
# calculate Jeans equation in annuli
# -------------------------------------------------------------------------------

#for i in range(3):
#    vtilde_annulus[:, i, i] = np.clip(vtilde_annulus[:, i, i], 0., np.inf)
dlnrho_dlnR = (-mean_XS_cyl_annulus[:, 0]) / rho_R_exp
dlnvR2_dlnR = (-mean_XS_cyl_annulus[:, 0]) / vrr_R_exp
HWRnumber = 1 + dlnrho_dlnR + dlnvR2_dlnR
vc_annulus = np.sqrt(vtilde_annulus[:, 1, 1] - HWRnumber * vtilde_annulus[:, 0, 0])
#sigmas_rr2 = 0.5 * (vt_ann_err_rr2[:, 2] - vt_ann_err_rr2[:, 0])
#sigmas_pp2 = 0.5 * (vt_ann_err_pp2[:, 2] - vt_ann_err_pp2[:, 0])
#vc_annulus_err_a = 0.5 / vc_annulus * np.sqrt(sigmas_pp2**2 + HWRnumber**2 * sigmas_rr2**2)
vc_annulus_err = 1. / vc_annulus * np.sqrt(mean_XS_cyl_annulus[:, 4]**2 * sigmas_pp**2 + HWRnumber**2 * mean_XS_cyl_annulus[:, 3]**2 * sigmas_rr**2)
vc_annulus_err_m = 1. / vc_annulus * np.sqrt(mean_XS_cyl_annulus[:, 4]**2 * (vt_ann_err_pp[:, 1] - vt_ann_err_pp[:, 0])**2 + HWRnumber**2 * mean_XS_cyl_annulus[:, 3]**2 * (vt_ann_err_rr[:, 1] - vt_ann_err_rr[:, 0])**2)
vc_annulus_err_p = 1. / vc_annulus * np.sqrt(mean_XS_cyl_annulus[:, 4]**2 * (vt_ann_err_pp[:, 2] - vt_ann_err_pp[:, 1])**2 + HWRnumber**2 * mean_XS_cyl_annulus[:, 3]**2 * (vt_ann_err_rr[:, 2] - vt_ann_err_rr[:, 1])**2)

t = Table()
rgc_table = Column(np.round(mean_XS_cyl_annulus[:, 0], 2), name = 'R')
vc_table = Column(np.round(vc_annulus, 2), name = 'v_{\rm c}')
vc_err_m_table = Column(np.round(vc_annulus_err_m, 2), name = '\sigma_{v^-_{\rm c}}')
vc_err_p_table = Column(np.round(vc_annulus_err_p, 2), name = '\sigma_{v^+_{\rm c}}')
t.add_column(rgc_table)
t.add_column(vc_table)
t.add_column(vc_err_m_table)
t.add_column(vc_err_p_table)
Table.write(t, 'paper_rotation_curve/table_vc.txt', format = 'latex')

# second version of v_circ
vc_annulus2 = np.sqrt(vtilde_annulus[:, 1, 1] - HWRnumber * (exp_fit(theta_fit, mean_XS_cyl_annulus[:, 0]))**2)
  
# -------------------------------------------------------------------------------
# errorbar on v_circ - v_phiphi
# -------------------------------------------------------------------------------

list_rho_R_exp = [2, 4]
vc_vpp = np.zeros((len(mean_XS_cyl_annulus[:, 0]), len(list_rho_R_exp)))
vc_rexp = np.zeros((len(mean_XS_cyl_annulus[:, 0]), len(list_rho_R_exp)))

for i, r_exp in enumerate(list(list_rho_R_exp)):
    dlnrho_dlnR = (-mean_XS_cyl_annulus[:, 0]) / r_exp
    HWRnumber = 1 + dlnrho_dlnR + dlnvR2_dlnR
    vc_ann_r = np.sqrt(vtilde_annulus[:, 1, 1] - HWRnumber * vtilde_annulus[:, 0, 0])
    vc_vpp[:, i] = vc_ann_r - np.sqrt(vtilde_annulus[:, 1, 1])
    vc_rexp[:, i] = vc_ann_r

#f = open('data/sys_vc_rexp.txt', 'w')
#np.savetxt(f, vc_rexp)
#f.close()

# -------------------------------------------------------------------------------
# calculate rotational velocity via Jeans equation (in patches)
# -------------------------------------------------------------------------------  

# velocity tensor!
vtilde = vvT_cyl - error_var_XS_cyl 
#for i in range(3):
#    vtilde[:, :, i, i] = np.clip(vtilde[:, :, i, i], 0., np.Inf) # uncomment??
dlnrho_dlnR = (-mean_XS_cyl[:, :, 0]) / rho_R_exp
dlnvR2_dlnR = (-mean_XS_cyl[:, :, 0]) / vrr_R_exp
HWRnumber = 1 + dlnrho_dlnR + dlnvR2_dlnR
vc = np.sqrt(vtilde[:, :, 1, 1] - HWRnumber * vtilde[:, :, 0, 0])          

# -------------------------------------------------------------------------------
# plots for paper
# -------------------------------------------------------------------------------  

# 30 degree wedge
wedge2d = (mean_XS_cart[:, :, 0] <= 0) * (abs(mean_XS_cart[:, :, 1]) <= ((-mean_XS_cart[:, :, 0]) * np.tan(deg_wedge/360. * 2*np.pi)))

fig, ax = plt.subplots(1, 3, figsize = (18, 6), sharex = True, sharey = True)        
cm = plt.cm.get_cmap('viridis')
sc = ax[0].scatter(mean_XS_cart[wedge2d, 0].flatten(), mean_XS_cart[wedge2d, 1].flatten(), c = np.sqrt(vtilde[wedge2d, 0, 0].flatten()), vmin = 50, vmax = 250, s=20, cmap=cm, alpha = .8)
ax[1].scatter(mean_XS_cart[wedge2d, 0].flatten(), mean_XS_cart[wedge2d, 1].flatten(), c = np.sqrt(vtilde[wedge2d, 1, 1].flatten()), vmin = 50, vmax = 250, s=20, cmap=cm, alpha = .8)
ax[2].scatter(mean_XS_cart[wedge2d, 0].flatten(), mean_XS_cart[wedge2d, 1].flatten(), c = vc[wedge2d].flatten(), vmin = 50, vmax = 250, s=20, cmap=cm, alpha = .8)
overplot_rings(ax[0])
#overplot_rings_helio(ax[0])
overplot_rings(ax[1])
#overplot_rings_helio(ax[1])
overplot_rings(ax[2])
#overplot_rings_helio(ax[2])
fig.subplots_adjust(right = 0.8)
cbar_ax = fig.add_axes([1, 0.05, 0.02, 0.92])
cb = fig.colorbar(sc, cax=cbar_ax)
cb.set_label(r'$v~\rm [km\, s^{-1}]$', rotation=270, fontsize=18, labelpad=30)
ax[0].scatter(mean_XS_cart[~wedge2d, 0].flatten(), mean_XS_cart[~wedge2d, 1].flatten(), c = '#929591', s = 20, alpha = .3) 
ax[1].scatter(mean_XS_cart[~wedge2d, 0].flatten(), mean_XS_cart[~wedge2d, 1].flatten(), c = '#929591', s = 20, alpha = .3) 
ax[2].scatter(mean_XS_cart[~wedge2d, 0].flatten(), mean_XS_cart[~wedge2d, 1].flatten(), c = '#929591', s = 20, alpha = .3) 
ax[0].set_xlim(Xlimits[0])
ax[0].set_ylim(Xlimits[1])
ax[0].set_xticks([-25, -20, -15, -10, -5, 0, 5, 10])
ax[0].tick_params(axis=u'both', direction='in', which='both')
ax[1].tick_params(axis=u'both', direction='in', which='both')
ax[2].tick_params(axis=u'both', direction='in', which='both')
ax[0].set_xlabel(r'$x\,\rm [kpc]$', fontsize = fsize)
ax[1].set_xlabel(r'$x\,\rm [kpc]$', fontsize = fsize)
ax[2].set_xlabel(r'$x\,\rm [kpc]$', fontsize = fsize)
ax[0].set_ylabel(r'$y\,\rm [kpc]$', fontsize = fsize)
ax[0].set_aspect('equal')
ax[1].set_aspect('equal')
ax[2].set_aspect('equal')
ax[0].annotate(r'$\sqrt{V_{RR}}$', (4., -10), fontsize = fsize, bbox=dict(boxstyle="square", fc="w"))
ax[1].annotate(r'$\sqrt{V_{\varphi\varphi}}$', (4., -10), fontsize = fsize, bbox=dict(boxstyle="square", fc="w"))
ax[2].annotate(r'$v_{\rm c}$', (6., -10), fontsize = fsize, bbox=dict(boxstyle="square", fc="w"))
plt.tight_layout()
plt.savefig('paper_rotation_curve/maps_wedge.pdf', bbox_inches = 'tight', pad_inches=.2)
plt.close()

fig, ax = plt.subplots(1, 1, figsize = (8, 8))        
cm = plt.cm.get_cmap('viridis')
sc = ax.scatter(mean_XS_cart[:, :, 0].flatten(), mean_XS_cart[:, :, 1].flatten(), c = (vc - np.sqrt(vtilde[:, :, 1, 1])).flatten(), vmin = 5, vmax = 10, s=20, cmap=cm, alpha = .8)
overplot_rings(ax)
fig.subplots_adjust(right = 0.8)
cbar_ax = fig.add_axes([1, 0.05, 0.02, 0.92])
cb = fig.colorbar(sc, cax=cbar_ax)
cb.set_label(r'$v~\rm [km\, s^{-1}]$', rotation=270, fontsize=18, labelpad=30)
ax.set_xlim(Xlimits[0])
ax.set_ylim(Xlimits[1])
ax.set_xticks([-25, -20, -15, -10, -5, 0, 5, 10])
ax.tick_params(axis=u'both', direction='in', which='both')
ax.set_xlabel(r'$x\,\rm [kpc]$', fontsize = fsize)
ax.set_ylabel(r'$y\,\rm [kpc]$', fontsize = fsize)
ax.set_aspect('equal')
ax.annotate(r'$v_c - \sqrt{V_{\varphi\varphi}}$', (4., -10), fontsize = fsize, bbox=dict(boxstyle="square", fc="w"))
plt.tight_layout()
plt.savefig('paper_rotation_curve/maps_vcmvp.pdf', bbox_inches = 'tight', pad_inches=.2)
plt.close()

vvT_cyl_n = np.zeros((mean_XS_cyl_n.shape[0], 3, 3))
for i in range(mean_XS_cyl_n.shape[0]):
    vvT_cyl_n[i, :, :] = np.outer(mean_XS_cyl_n[i, 3:], mean_XS_cyl_n[i, 3:])  # outer product!!
vtilde_n = vvT_cyl_n - var_XS_cyl_n
dlnrho_dlnR_n = (-mean_XS_cyl_n[:, 0]) / rho_R_exp
dlnvR2_dlnR_n = (-mean_XS_cyl_n[:, 0]) / vrr_R_exp
HWRnumber_n = 1 + dlnrho_dlnR_n + dlnvR2_dlnR_n
vc_n = np.sqrt(vtilde_n[:, 1, 1] - HWRnumber_n * vtilde_n[:, 0, 0])
vc_n2 = np.sqrt(vtilde_n[:, 1, 1] - HWRnumber_n * (exp_fit(theta_fit, mean_XS_cyl_n[:, 0]))**2) 

plot_R = np.arange(0, 30, .1)

f = open('data/rot_curve.txt', 'w')
np.savetxt(f, np.vstack([bins_dr, vc_annulus, vc_annulus_err_m, vc_annulus_err_p]))
f.close()

# plot with new subpanel
fig = plt.subplots(figsize = (18, 5), sharex = True) 
gs = gridspec.GridSpec(2, 3, height_ratios = (3, 1))
gs.update(wspace=0.25, hspace = 0.05)
ax0 = plt.subplot(gs[:, 0])
ax1 = plt.subplot(gs[:, 1])
ax2 = plt.subplot(gs[0, 2])
ax3 = plt.subplot(gs[1, 2])       
#ax2.scatter(bins_dr[:idx5], vc_annulus[:idx5], facecolors='none', edgecolors='#3778bf', zorder = 20)
ax2.errorbar(bins_dr[idx5:], vc_annulus[idx5:], yerr = [vc_annulus_err_m[idx5:], vc_annulus_err_p[idx5:]], fmt = 'o', markersize = 4, capsize=3, mfc='k', mec='k', ecolor = 'k', zorder = 30)
#ax0.scatter(bins_dr[:idx5], np.sqrt(vtilde_annulus[:idx5, 0, 0]), facecolors='none', edgecolors='#3778bf', zorder = 20)
#ax0.scatter(bins_dr[idx5:], np.sqrt(vtilde_annulus[idx5:, 0, 0]), facecolors='#3778bf', edgecolors='#3778bf', zorder = 30, alpha = .8)
#ax1.scatter(bins_dr[:idx5], np.sqrt(vtilde_annulus[:idx5, 1, 1]), facecolors='none', edgecolors='#3778bf', zorder = 20)
#ax1.scatter(bins_dr[idx5:], np.sqrt(vtilde_annulus[idx5:, 1, 1]), facecolors='#3778bf', edgecolors='#3778bf', zorder = 30, alpha = .8)

ax0.errorbar(bins_dr[:idx5], np.sqrt(vtilde_annulus[:idx5, 0, 0]), yerr = [vt_ann_err_rr[:idx5, 1]-vt_ann_err_rr[:idx5, 0], vt_ann_err_rr[:idx5, 2]-vt_ann_err_rr[:idx5, 1]], fmt = 'o', markersize = 4, capsize=3, mfc='w', mec='k', ecolor = 'k', zorder = 20)
ax0.errorbar(bins_dr[idx5:], np.sqrt(vtilde_annulus[idx5:, 0, 0]), yerr = [vt_ann_err_rr[idx5:, 1]-vt_ann_err_rr[idx5:, 0], vt_ann_err_rr[idx5:, 2]-vt_ann_err_rr[idx5:, 1]], fmt = 'o', markersize = 4, capsize=3, mfc='k', mec='k', ecolor = 'k', zorder = 30)
ax1.errorbar(bins_dr[:idx5], np.sqrt(vtilde_annulus[:idx5, 1, 1]), yerr = [vt_ann_err_pp[:idx5, 1]-vt_ann_err_pp[:idx5, 0], vt_ann_err_pp[:idx5, 2]-vt_ann_err_pp[:idx5, 1]], fmt = 'o', markersize = 4, capsize=3, mfc='w', mec='k', ecolor = 'k', zorder = 20)
ax1.errorbar(bins_dr[idx5:], np.sqrt(vtilde_annulus[idx5:, 1, 1]), yerr = [vt_ann_err_pp[idx5:, 1]-vt_ann_err_pp[idx5:, 0], vt_ann_err_pp[idx5:, 2]-vt_ann_err_pp[idx5:, 1]], fmt = 'o', markersize = 4, capsize=3, mfc='k', mec='k', ecolor = 'k', zorder = 30)

ax0.scatter(mean_XS_cyl_n[cut_z_wedge, 0], np.sqrt(vtilde_n[cut_z_wedge, 0, 0]), c = '#929591', s = 15, alpha = .05, zorder = -np.inf, rasterized = True)
ax1.scatter(mean_XS_cyl_n[cut_z_wedge, 0], np.sqrt(vtilde_n[cut_z_wedge, 1, 1]), c = '#929591', s = 15, alpha = .05, zorder = -np.inf, rasterized = True)
#ax2.scatter(mean_XS_cyl_n[cut_z_wedge, 0], vc_n[cut_z_wedge], c = '#929591', s = 15, alpha = .05, zorder = -np.inf, rasterized = True)
ax0.tick_params(axis=u'both', direction='in', which='both', right = 'on', top = 'on')
ax1.tick_params(axis=u'both', direction='in', which='both', right = 'on', top = 'on')
ax2.tick_params(axis=u'both', direction='in', which='both', right = 'on', top = 'on', labelbottom = 'off')
ax3.tick_params(axis=u'both', direction='in', which='both', right = 'on', top = 'on')
ax0.plot(plot_R, exp_fit(theta_fit, plot_R), color = '#feb308', linestyle='--', zorder = 40, label = r'$\sqrt{{V_{{RR}}}}(R)\propto \exp\left(-\frac{{R}}{{{1}\,\rm kpc}}\right)$'.format(round(theta_fit[0], 2), int(round(theta_fit[1])))) #'#c44240'
ax0.set_xlabel(r'$R\,\rm [kpc]$', fontsize = fsize)
ax1.set_xlabel(r'$R\,\rm [kpc]$', fontsize = fsize)
ax3.set_xlabel(r'$R\,\rm [kpc]$', fontsize = fsize)
ax2.set_ylabel(r'$v_{\rm c}~\rm [km\,s^{-1}]$', fontsize = fsize)
ax0.set_ylabel(r'$\sqrt{V_{RR}}~ \rm [km\,s^{-1}]$', fontsize = fsize)
ax1.set_ylabel(r'$\sqrt{V_{\varphi\varphi}} ~\rm [km\,s^{-1}]$', fontsize = fsize)
#ax3.scatter(bins_dr[idx5:], vc_annulus[idx5:] - np.sqrt(vtilde_annulus[idx5:, 1, 1]), facecolors='#3778bf', edgecolors='#3778bf', zorder = 30, alpha = .8)
# with errorbars
ax3.errorbar(bins_dr[idx5:], vc_annulus[idx5:] - np.sqrt(vtilde_annulus[idx5:, 1, 1]), yerr = [np.array(vc_vpp[idx5:, 0]), np.array(vc_vpp[idx5:, 1])], fmt = 'o', markersize = 4, capsize=3, mfc='k', mec='k', ecolor = 'k', zorder = 30)
ax3.set_ylabel(r'$v_{\rm c} - \sqrt{V_{\varphi\varphi}}$', fontsize = fsize) 
ax0.set_ylim(0, 325)
ax1.set_ylim(0, 325)
ax2.set_ylim(120, 250)
ax0.set_xlim(0, 25.2)
ax1.set_xlim(0, 25.2)
ax2.set_xlim(0, 25.2)
ax3.set_xlim(0, 25.2)
ax0.set_xticks([0, 5, 10, 15, 20, 25])
ax1.set_xticks([0, 5, 10, 15, 20, 25])
ax2.set_xticks([0, 5, 10, 15, 20, 25])
ax3.set_xticks([0, 5, 10, 15, 20, 25])
ax0.legend(fontsize = 16, frameon = True)
plt.savefig('paper_rotation_curve/radial_profile_sub.pdf', bbox_inches = 'tight', pad_inches=.2)


# -------------------------------------------------------------------------------'''
# systematics
# -------------------------------------------------------------------------------        

f = open('data/sys_vc_rexp.txt', 'r')
vc_rexp = np.loadtxt(f)
f.close()
dvc_rexp = 0.5 * abs(vc_rexp[idx5:, 0] - vc_rexp[idx5:, 1]) / vc_annulus[idx5:]

f = open('data/rot_curve_part1.txt', 'r')
vc_wed1 = np.loadtxt(f)
f.close()
vc_wed1_int = interpol.interp1d(vc_wed1[0, :], vc_wed1[1, :], kind = 'linear') 
vc_wed1_new = vc_wed1_int(bins_dr[idx5:])
f = open('data/rot_curve_part2.txt', 'r')
vc_wed2 = np.loadtxt(f)
f.close()
vc_wed2_int = interpol.interp1d(vc_wed2[0, :], vc_wed2[1, :], kind = 'linear') 
vc_wed2_new = vc_wed2_int(bins_dr[idx5:])
dvc_wedges = 0.5 * abs(vc_wed1_new - vc_wed2_new) / vc_annulus[idx5:]

f = open('data/rot_curve_zminus.txt', 'r')
vc_zm = np.loadtxt(f)
f.close()
vc_zm_int = interpol.interp1d(vc_zm[0, :], vc_zm[1, :], kind = 'linear') 
vc_zm_new = vc_zm_int(bins_dr[idx5:])
f = open('data/rot_curve_zplus.txt', 'r')
vc_zp = np.loadtxt(f)
f.close()
vc_zp_int = interpol.interp1d(vc_zp[0, :], vc_zp[1, :], kind = 'linear') 
vc_zp_new = vc_zp_int(bins_dr[idx5:])
dvc_z = 0.5 * abs(vc_zm_new - vc_zp_new) / vc_annulus[idx5:]

f = open('data/rot_curve_R08091.txt', 'r')
vc_Rm = np.loadtxt(f)
f.close()
f = open('data/rot_curve_R08153.txt', 'r')
vc_Rp = np.loadtxt(f)
f.close()
dvc_R0 = 0.5 * abs(vc_Rm[1][idx5:] - vc_Rp[1][idx5:]) / vc_annulus[idx5:]

f = open('data/rot_curve_mu_minus.txt', 'r')
vc_mu_minus = np.loadtxt(f)
f.close()
f = open('data/rot_curve_mu_plus.txt', 'r')
vc_mu_plus = np.loadtxt(f)
f.close()
dvc_mu = 0.5 * abs(vc_mu_minus[1][idx5:] - vc_mu_plus[1][idx5:]) / vc_annulus[idx5:]

fig, ax = plt.subplots(figsize = (8, 6)) 
ax.plot(bins_dr[idx5:], dvc_rexp, label = r'$R_{\rm exp} = 2\,{\rm or}\,4\,\rm kpc$')
#ax.plot(bins_dr[idx5:], dvc_z, label = r'above and below the plane')
ax.plot(bins_dr[idx5:], dvc_R0, label = r'$\pm 1 \sigma_{R_{\odot}} = \pm 31\,\rm pc$')
ax.plot(bins_dr[idx5:], dvc_mu, label = r'$\pm 1 \sigma_{\mu_{\rm Sgr\,A^{\star}}}$')
ax.plot(bins_dr[idx5:], dvc_wedges, label = r'two distinct wedges')
ax.set_xlabel(r'$R\,\rm [kpc]$', fontsize = fsize)
ax.set_ylabel(r'$\sigma_{\rm\,sys}$', fontsize = fsize)
ax.set_xlim(bins_dr[idx5], bins_dr[-1])
ax.set_ylim(0, 0.1)
plt.legend(fontsize = 16, frameon = True)
ax.tick_params(axis=u'both', direction='in', which='both', right = 'on', top = 'on')
plt.savefig('paper_rotation_curve/systematics.pdf', bbox_inches = 'tight', pad_inches=.2)


fig, ax = plt.subplots(figsize = (8, 6)) 
ax.fill_between(bins_dr[idx5:], dvc_rexp, label = r'$R_{\rm exp} = 2\,{\rm or}\,4\,\rm kpc$')
#ax.fill_between(bins_dr[idx5:], y1 = dvc_rexp+dvc_wedges+dvc_z, y2 = dvc_rexp+dvc_wedges, label = r'above and below the plane')
ax.fill_between(bins_dr[idx5:], y1 = dvc_R0+dvc_rexp, y2 = dvc_rexp, label = r'$\pm 1 \sigma_{R_{\odot}} = \pm 31\,\rm pc$')
ax.fill_between(bins_dr[idx5:], y1 = dvc_mu+dvc_R0+dvc_rexp, y2 = dvc_R0+dvc_rexp, label = r'$\pm 1 \sigma_{\mu_{\rm Sgr\,A^{\star}}}$')
ax.fill_between(bins_dr[idx5:], y1 = dvc_mu+dvc_R0+dvc_rexp+dvc_wedges, y2 = dvc_mu+dvc_R0+dvc_rexp, label = r'two distinct wedges')
ax.set_xlabel(r'$R\,\rm [kpc]$', fontsize = fsize)
ax.set_ylabel(r'$\sigma_{\rm\,sys}$', fontsize = fsize)
ax.set_xlim(bins_dr[idx5], bins_dr[-1])
ax.set_ylim(0, 0.2)
plt.legend(fontsize = 16, frameon = True, loc = 2)
ax.tick_params(axis=u'both', direction='in', which='both', right = 'on', top = 'on')
plt.savefig('paper_rotation_curve/systematics2.pdf', bbox_inches = 'tight', pad_inches=.2)

# -------------------------------------------------------------------------------'''
      
