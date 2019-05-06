#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 10:06:11 2019

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
from scipy.stats import binned_statistic_2d
#from plotting_helpers import histcont
import matplotlib.gridspec as gridspec
import scipy.interpolate as interpol

# -------------------------------------------------------------------------------
# colormaps
# -------------------------------------------------------------------------------

import matplotlib.colors as mcolors

c2 = plt.cm.Reds(np.linspace(0, 1, 32))
c1 = plt.cm.Blues_r(np.linspace(0, 1, 32))
c1 = np.vstack([c1, [1, 1, 1, 1]])
mymap = mcolors.LinearSegmentedColormap.from_list('eddymap', np.vstack((c1, c2)))


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

N = 44784 #66692
Kfold = 2
lam = 30
name = 'N{0}_lam{1}_K{2}_offset0.0483_parallax'.format(N, lam, Kfold)

print('loading new labels...')   
labels = Table.read('data/training_labels_new_{}.fits'.format(name), format = 'fits')    
labels.rename_column('ra_1', 'ra')
labels.rename_column('dec_1', 'dec')

#N = 66690 
#Kfold = 2
#lam = 30
#name = 'N{0}_lam{1}_K{2}_offset0.0483_parallax_apogeedr15'.format(N, lam, Kfold)
#
#print('loading new labels...')   
#labels = Table.read('data/training_labels_new_{}_try3.fits'.format(name), format = 'fits')    
#labels.rename_column('ra_1', 'ra')
#labels.rename_column('dec_1', 'dec')

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
# HRD plot
# -------------------------------------------------------------------------------           

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
# for plotting
# -------------------------------------------------------------------------------   

def overplot_ring(r, ax = None):
    tiny = 1e-4
    thetas = np.arange(0., 2*np.pi + tiny, 0.001 * np.pi)
    xs = r * np.cos(thetas)
    ys = r * np.sin(thetas)
    if ax:
        ax.plot(xs, ys, "k-", alpha=0.2, lw=1, zorder = np.inf)
        ax.scatter(0, 0, s = 30, color = 'k', alpha=0.2, marker = 'x')
    else:
        plt.plot(xs, ys, "k-", alpha=0.2, lw=1, zorder = np.inf)
        plt.scatter(0, 0, s = 30, color = 'k', alpha=0.2, marker = 'x')
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

Xlimits = [[-22, 2], [-10, 14], [-20, 20], 
           [-200, 200], [-200, 200], [-200, 200]]



# -------------------------------------------------------------------------------
# linear fit of circular velocity curve
# ------------------------------------------------------------------------------- 

def vc_lin(R):
    R_sun = 8.122 # kpc
    vc = 229.0 - 1.7 * (R - R_sun)
    return vc

vc_measured_table = Table.read('paper_rotation_curve/table_vc_measured.txt', format = 'ascii')
R_measured = vc_measured_table['col1']
vc_measured = vc_measured_table['col2']
# outside of ~25 kpc, take linear fit
vc_int = interpol.interp1d(R_measured, vc_measured, kind = 'linear')
        
# -------------------------------------------------------------------------------
# divide Milky Way into (x, y, z) patches
# ------------------------------------------------------------------------------- 

# take wegde in z 
deg_wedge_in_z = 6.
cut_vz = (XS_cart_true_n[:, 5] < 100)
wedge_z = (np.abs(mean_XS_cyl_n[:, 2])/(mean_XS_cyl_n[:, 0])) < np.tan(deg_wedge_in_z/360. * 2. * np.pi)
cut_z = np.logical_or(abs(mean_XS_cyl_n[:, 2]) < .5, wedge_z) * cut_vz

# CHANGED FROM dz = 1 kpc to 2 kpc!! (Feb 28, 2019)

box_size = 0.35               # that's just half of the box size
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
mean_vr = np.zeros((len(all_x), len(all_y))) - np.inf
vc_linear = np.zeros((len(all_x), len(all_y))) - np.inf
vc_measured_int = np.zeros((len(all_x), len(all_y))) - np.inf


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
            mean_vr[i, j] = np.nanmean(labels['VHELIO_AVG'][cut_patch])
            mean_sigma_mu[i, j] = np.nanmean(np.sqrt(labels['pmra_error'][cut_patch] ** 2 + labels['pmdec_error'][cut_patch] ** 2))
            mean_sigma_par[i, j] = np.nanmean(0.09 * labels['spec_parallax'][cut_patch])
            error_var_XS_cyl[i, j, :, :] = np.nanmean(var_XS_cyl_n[cut_patch], axis=0)
            vvT_cyl[i, j, :, :] = np.dot(mean_XS_cyl_n[cut_patch, 3:].T, mean_XS_cyl_n[cut_patch, 3:]) / N_stars[i, j]
            vc_linear[i, j] = vc_lin(np.nanmean(mean_XS_cyl_n[cut_patch, 0]))
            vc_measured_int[i, j] = vc_int(np.nanmean(mean_XS_cyl_n[cut_patch, 0]))


rho_R_exp = 3. # kpc
vrr_R_exp = 21.05382199334898 #theta_fit[1] # kpc
print('vrr_R_exp = {}'.format(vrr_R_exp))

vtilde_n = np.zeros((len(mean_XS_cyl_n), 3, 3))
for i in range(len(mean_XS_cyl_n)):
    vvT = np.outer(mean_XS_cyl_n[i, 3:].T, mean_XS_cyl_n[i, 3:])
    vtilde_n[i, :, :] = vvT - var_XS_cyl_n[i, :, :]

# -------------------------------------------------------------------------------
# calculate rotational velocity via Jeans equation (in patches)
# -------------------------------------------------------------------------------  

# velocity tensor!
vtilde = vvT_cyl - error_var_XS_cyl 
dlnrho_dlnR = (-mean_XS_cyl[:, :, 0]) / rho_R_exp
# power law: 
#dlnrho_dlnR = (-8.) * np.ones_like(mean_XS_cyl[:, :, 0])
dlnvR2_dlnR = (-mean_XS_cyl[:, :, 0]) / vrr_R_exp
HWRnumber = 1 + dlnrho_dlnR + dlnvR2_dlnR
vc = np.sqrt(vtilde[:, :, 1, 1] - HWRnumber * vtilde[:, :, 0, 0])          
vc_excess = vc - vc_linear
vc_excess2 = vc - vc_measured_int


fig, ax = plt.subplots(1, 1, figsize = (10, 8))        
cm = plt.cm.get_cmap('RdBu')
sc = plt.scatter(mean_XS_cart[:, :, 0].flatten(), mean_XS_cart[:, :, 1].flatten(), c = vc_excess.flatten(), vmin = -10, vmax = 10, s=np.sqrt(N_stars).flatten(), cmap=cm, alpha = .8)
overplot_rings(ax)
ax.set_xlim(Xlimits[0])
ax.set_ylim(Xlimits[1])
cb = plt.colorbar(sc)
cb.set_label(r'$v_{\rm c} - v_{\rm c,\,linear}(R)$', rotation=270, fontsize=18, labelpad=30)
ax.tick_params(axis=u'both', direction='in', which='both')
ax.set_aspect('equal')
ax.set_xlabel(r'$x\rm~[kpc]$', fontsize=18)
ax.set_ylabel(r'$y\rm~[kpc]$', fontsize=18)
plt.savefig('non_axisymmetries/vc_excess_{}.pdf'.format(name))

fig, ax = plt.subplots(1, 1, figsize = (10, 8))        
sc = plt.scatter(mean_XS_cart[:, :, 0].flatten(), mean_XS_cart[:, :, 1].flatten(), c = vc_excess2.flatten(), vmin = -5, vmax = 5, s=np.sqrt(N_stars).flatten(), cmap=cm, alpha = .8)
overplot_rings(ax)
ax.set_xlim(Xlimits[0])
ax.set_ylim(Xlimits[1])
cb = fig.colorbar(sc, label = r'$v_{\rm c} - v_{\rm c,\,measured}(R)$')
ax.tick_params(axis=u'both', direction='in', which='both')
ax.set_aspect('equal')
ax.set_xlabel(r'$x\rm~[kpc]$', fontsize=18)
ax.set_ylabel(r'$y\rm~[kpc]$', fontsize=18)
plt.savefig('non_axisymmetries/vc_excess2_{}.pdf'.format(name))


#fig, ax = plt.subplots(1, 1, figsize = (10, 8))        
#sc = plt.scatter(mean_XS_cart[:, :, 0].flatten(), mean_XS_cart[:, :, 1].flatten(), c = vtilde[:, :, 1, 0].flatten(), vmin = -1500, vmax = 1500, s=np.sqrt(N_stars).flatten(), cmap=cm, alpha = .8)
#overplot_rings(ax)
#ax.set_xlim(Xlimits[0])
#ax.set_ylim(Xlimits[1])
#cb = fig.colorbar(sc)
#cb.set_label(r'$V_{R\varphi}$', rotation=270, fontsize=18, labelpad=30)
#ax.tick_params(axis=u'both', direction='in', which='both')
#ax.set_aspect('equal')
#ax.set_xlabel(r'$x\rm~[kpc]$', fontsize=18)
#ax.set_ylabel(r'$y\rm~[kpc]$', fontsize=18)
#plt.savefig('non_axisymmetries/vrvp_{}.pdf'.format(name))

fig, ax = plt.subplots(1, 1, figsize = (10, 8))        
crp = vtilde[:, :, 1, 0] / np.sqrt(vtilde[:, :, 0, 0] * vtilde[:, :, 1, 1])
sc = plt.scatter(mean_XS_cart[:, :, 0].flatten(), mean_XS_cart[:, :, 1].flatten(), c = crp.flatten(), vmin = -0.5, vmax= 0.5, s=np.sqrt(N_stars).flatten(), cmap=cm, alpha = .8)
overplot_rings(ax)
ax.set_xlim(Xlimits[0])
ax.set_ylim(Xlimits[1])
cb = fig.colorbar(sc)
cb.set_label(r'correlation of $v_R$ and $v_{\varphi}$', rotation=270, fontsize=18, labelpad=30)
ax.tick_params(axis=u'both', direction='in', which='both')
ax.set_aspect('equal')
ax.set_xlabel(r'$x\rm~[kpc]$', fontsize=18)
ax.set_ylabel(r'$y\rm~[kpc]$', fontsize=18)
plt.savefig('non_axisymmetries/crp_{}.pdf'.format(name))

fig, ax = plt.subplots(1, 1, figsize = (10, 8))        
cm = plt.cm.get_cmap('RdBu')
crz = vtilde[:, :, 2, 0] / np.sqrt(vtilde[:, :, 0, 0] * vtilde[:, :, 2, 2])
sc = plt.scatter(mean_XS_cart[:, :, 0].flatten(), mean_XS_cart[:, :, 1].flatten(), c = crz.flatten(), vmin = -0.5, vmax= 0.5, s=np.sqrt(N_stars).flatten(), cmap=cm, alpha = .8)
overplot_rings(ax)
ax.set_xlim(Xlimits[0])
ax.set_ylim(Xlimits[1])
cb = fig.colorbar(sc)
cb.set_label(r'correlation of $v_R$ and $v_z$', rotation=270, fontsize=18, labelpad=30)
ax.tick_params(axis=u'both', direction='in', which='both')
ax.set_aspect('equal')
ax.set_xlabel(r'$x\rm~[kpc]$', fontsize=18)
ax.set_ylabel(r'$y\rm~[kpc]$', fontsize=18)
plt.savefig('non_axisymmetries/crz_{}.pdf'.format(name))

fig, ax = plt.subplots(1, 1, figsize = (10, 8))        
cm = plt.cm.get_cmap('RdBu')
cpz = vtilde[:, :, 2, 1] / np.sqrt(vtilde[:, :, 2, 2] * vtilde[:, :, 1, 1])
sc = plt.scatter(mean_XS_cart[:, :, 0].flatten(), mean_XS_cart[:, :, 1].flatten(), c = cpz.flatten(), vmin = -0.5, vmax= 0.5, s=np.sqrt(N_stars).flatten(), cmap=cm, alpha = .8)
overplot_rings(ax)
ax.set_xlim(Xlimits[0])
ax.set_ylim(Xlimits[1])
cb = fig.colorbar(sc)
cb.set_label(r'correlation of $v_{\varphi}$ and $v_z$', rotation=270, fontsize=18, labelpad=30)
ax.tick_params(axis=u'both', direction='in', which='both')
ax.set_aspect('equal')
ax.set_xlabel(r'$x\rm~[kpc]$', fontsize=18)
ax.set_ylabel(r'$y\rm~[kpc]$', fontsize=18)
plt.savefig('non_axisymmetries/cpz_{}.pdf'.format(name))


fig, ax = plt.subplots(1, 1, figsize = (10, 8))  
sc = plt.scatter(mean_XS_cart[:, :, 0].flatten(), mean_XS_cart[:, :, 1].flatten(), c = mean_XS_cyl[:, :, 5].flatten(), vmin = -8, vmax = 8, s=np.sqrt(N_stars).flatten(), cmap=cm, alpha = .8)
overplot_rings(ax)
ax.set_xlim(Xlimits[0])
ax.set_ylim(Xlimits[1])
cb = fig.colorbar(sc, label = r'$\langle v_z\rangle$')
cb.set_label(r'$\langle v_z\rangle$', rotation=270, fontsize=18, labelpad=30)
ax.tick_params(axis=u'both', direction='in', which='both')
ax.set_aspect('equal')
ax.set_xlabel(r'$x\rm~[kpc]$', fontsize=18)
ax.set_ylabel(r'$y\rm~[kpc]$', fontsize=18)
plt.savefig('non_axisymmetries/mean_vz_{}.pdf'.format(name))


fig, ax = plt.subplots(1, 1, figsize = (10, 8))  
vp_vc = mean_XS_cyl[:, :, 4] + vc_linear
sc = plt.scatter(mean_XS_cart[:, :, 0].flatten(), mean_XS_cart[:, :, 1].flatten(), c = vp_vc.flatten(), vmin = -10, vmax = 10, s=np.sqrt(N_stars).flatten(), cmap=cm, alpha = .8)
overplot_rings(ax)
ax.set_xlim(Xlimits[0])
ax.set_ylim(Xlimits[1])
cb = plt.colorbar(sc)
cb.set_label(r'$\langle v_{\varphi}\rangle - v_c$', rotation=270, fontsize=18, labelpad=30)
ax.tick_params(axis=u'both', direction='in', which='both')
ax.set_aspect('equal')
ax.set_xlabel(r'$x\rm~[kpc]$', fontsize=18)
ax.set_ylabel(r'$y\rm~[kpc]$', fontsize=18)
plt.savefig('non_axisymmetries/mean_vp_{}.pdf'.format(name))

fig, ax = plt.subplots(1, 1, figsize = (10, 8))        
sc = plt.scatter(mean_XS_cart[:, :, 0].flatten(), mean_XS_cart[:, :, 1].flatten(), c = mean_XS_cyl[:, :, 3].flatten(), vmin = -10, vmax = 10, s=np.sqrt(N_stars).flatten(), cmap=cm, alpha = .8)
overplot_rings(ax)
ax.set_xlim(Xlimits[0])
ax.set_ylim(Xlimits[1])
cb = fig.colorbar(sc)
cb.set_label(r'$\langle v_r\rangle$', rotation=270, fontsize=18, labelpad=30)
ax.tick_params(axis=u'both', direction='in', which='both')
ax.set_aspect('equal')
ax.set_xlabel(r'$x\rm~[kpc]$', fontsize=18)
ax.set_ylabel(r'$y\rm~[kpc]$', fontsize=18)
plt.savefig('non_axisymmetries/mean_vr_{}.pdf'.format(name))

# -------------------------------------------------------------------------------'''

#fig, ax = plt.subplots(1, 1, figsize = (10, 8))        
#sc = plt.scatter(mean_XS_cart_n[:, 2], mean_XS_cart_n[:, 5], c = mean_XS_cyl_n[:, 3], vmin = -10, vmax = 10, s=5, cmap=cm, alpha = .8)
#ax.set_xlim(-1, 1)
#ax.set_ylim(-60, 60)
#cb = fig.colorbar(sc)
#cb.set_label(r'$v_r~\rm [km/s]$', rotation=270, fontsize=18, labelpad=30)
#ax.set_ylabel(r'$v_z\rm~[km/s]$', fontsize=18)
#ax.set_xlabel(r'$z\rm~[kpc]$', fontsize=18)
#ax.tick_params(axis=u'both', direction='in', which='both')


# -------------------------------------------------------------------------------
# adaptive mesh
# -------------------------------------------------------------------------------

from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

# create grid
bin_width_start = .25
x = np.arange(-26, 10, bin_width_start)
y = np.arange(-16, 20, bin_width_start)
foo, bar = np.meshgrid(x, y)
bin_centers = np.array(np.vstack((foo.flatten(), bar.flatten())).T)
widths = np.ones((bin_centers.shape[0], 1)) * bin_width_start
numbers = np.zeros((bin_centers.shape[0], 1))
bins = np.hstack((bin_centers, widths, numbers))
N_max = 50000
velocity_directions = np.zeros((bins.shape[0], 2))

# count numbers in each bin again
for i in range(bins.shape[0]):
    cut_patch = cut_z * (abs(mean_XS_cart_n[:, 0] - bins[i, 0]) < .5 * bins[i, 2]) * (abs(mean_XS_cart_n[:, 1] - bins[i, 1]) < .5 * bins[i, 2])
    bins[i, 3] = np.sum(cut_patch)
    velocity_directions[i, 0] = np.nanmean(mean_XS_cart_n[cut_patch, 3])
    velocity_directions[i, 1] = np.nanmean(mean_XS_cart_n[cut_patch, 4])

for x in range(1):
    
    print('{0}th iteration: total number of points: {1}, N stars: {2}'.format(x, bins.shape[0], np.sum(bins[:, 3])))
    
    # make grid smoother for bins with many stars
    new_bins = []
    for i in range(bins.shape[0]):
        if bins[i, 3] < N_max:
            new_bins.append(bins[i, :])
        elif bins[i, 3] >= N_max:
            # replace with four new bins instead
            new_bins.append(np.array([bins[i, 0] + 0.25 * bins[i, 2], bins[i, 1] + 0.25 * bins[i, 2], 0.5 * bins[i, 2], -1.]))
            new_bins.append(np.array([bins[i, 0] + 0.25 * bins[i, 2], bins[i, 1] - 0.25 * bins[i, 2], 0.5 * bins[i, 2], -1.]))
            new_bins.append(np.array([bins[i, 0] - 0.25 * bins[i, 2], bins[i, 1] - 0.25 * bins[i, 2], 0.5 * bins[i, 2], -1.]))
            new_bins.append(np.array([bins[i, 0] - 0.25 * bins[i, 2], bins[i, 1] + 0.25 * bins[i, 2], 0.5 * bins[i, 2], -1.]))
     
    bins = np.array(new_bins)
    
    # count numbers in each bin again
    for i in range(bins.shape[0]):
        if bins[i, 3] < 0.:
            cut_patch = cut_z * (abs(mean_XS_cart_n[:, 0] - bins[i, 0]) < .5 * bins[i, 2]) * (abs(mean_XS_cart_n[:, 1] - bins[i, 1]) < .5 * bins[i, 2])
            bins[i, 3] = np.sum(cut_patch)
    velocity_directions[i, 0] = np.nanmean(mean_XS_cart_n[cut_patch, 3])
    velocity_directions[i, 1] = np.nanmean(mean_XS_cart_n[cut_patch, 4])
 

def average_in_bins(quantity, bins, mean_XS_cart_n, cut_z = True, expand_factor = 1, ind1 = None, ind2 = None):
   
#    assert len(quantity) == len(mean_XS_cart_n)
    
    n, d = bins.shape
    if d == 4:
        new_quantity = np.zeros((bins.shape[0], 1))
        bins = np.hstack((bins, new_quantity))
    n, d = bins.shape
    assert d == 5
    qsh = quantity.shape
        
    for i in range(n):
        cut_patch = cut_z * (abs(mean_XS_cart_n[:, 0] - bins[i, 0]) < .5 * expand_factor * bins[i, 2]) * (abs(mean_XS_cart_n[:, 1] - bins[i, 1]) < .5 * expand_factor * bins[i, 2])
        bins[i, 3] = np.sum(cut_patch)
        #bins[i, 4] = np.nanmean(quantity[cut_patch])
        if qsh == (32271, 3, 3): 
            vtilde_n = quantity
            numerator = np.nanmean(vtilde_n[cut_patch, ind1, ind2])
            denominator = np.sqrt(np.nanmean(vtilde_n[cut_patch, ind1, ind1]) * np.nanmean(vtilde_n[cut_patch, ind2, ind2]))
            bins[i, 4] = numerator / denominator
        else: 
            bins[i, 4] = np.nanmean(quantity[cut_patch])
        #bins[i, 0] = np.nanmean(mean_XS_cart_n[cut_patch, 0])
        #bins[i, 1] = np.nanmean(mean_XS_cart_n[cut_patch, 1])
    
    return bins


color_quantity = vtilde_n #mean_XS_cyl_n[:, 5] #vtilde_n[:, 2, 2]#mean_XS_cyl_n[:, 3] #vtilde_n#[:, 2, 2] #mean_XS_cyl_n[:, 2] * mean_XS_cyl_n[:, 5] #

expand_factor = 2.5
bins = average_in_bins(color_quantity, bins, mean_XS_cart_n, cut_z, expand_factor, 0, 1)

vmin, vmax = -.5, .5
name = r'$\langle v_{\rm R}v_{\rm \varphi}\rangle$'#'$\frac{\langle v_{\rm \varphi}v_{z}\rangle}{\sqrt{\langle v_{\rm \varphi}^2\rangle \langle v_{z}^2\rangle}}$'
name_plot = 'vrvp'
         
#  PLOT
cm = plt.cm.get_cmap('coolwarm_r')

# normalize chosen colormap
norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap=cm)

fig, ax = plt.subplots(1, 1, figsize = (10, 8))
sc = plt.scatter(bins[:, 0], bins[:, 1], c = bins[:, 4], cmap = cm, vmin=vmin, vmax=vmax, s = (4./expand_factor) * (bin_width_start/0.5) * np.sqrt(bins[:, 3]))
#sc = plt.quiver(bins[:, 0], bins[:, 1], velocity_directions[:, 0], velocity_directions[:, 1], 
#           np.clip(bins[:, 4], -0.5, 0.5), cmap = cm, scale_units='xy', 
#           scale=200, alpha =.8, headwidth = 3, headlength = 5, width = 0.0015, rasterized = True)
cbar = plt.colorbar(sc) 
cbar.set_label('{}'.format(name), rotation=270, fontsize=18, labelpad=30)
ax.set_xlim(-22, 2)
ax.set_ylim(-10, 14)
overplot_rings(ax)
ax.tick_params(axis=u'both', direction='in', which='both')
ax.set_aspect('equal')
ax.set_xlabel(r'$x\rm~[kpc]$', fontsize=18)
ax.set_ylabel(r'$y\rm~[kpc]$', fontsize=18)
plt.savefig('non_axisymmetries/mesh_vz_bc.pdf'.format(name_plot, name)) 
#plt.close()




fig, ax = plt.subplots(1, 3, figsize = (25, 10))
cm = plt.cm.get_cmap('viridis_r')

for i in range(3):
    color_quantity = vtilde_n[:, i, i]
    
    expand_factor = 2.5
    bins = average_in_bins(color_quantity, bins, mean_XS_cart_n, cut_z, expand_factor)
    
    vmin, vmax = 0, 5000
    if i == 1:
        vmax = 50000
    if i == 2:
        vmax = 4000
    name = [r'$\langle v_{\rm R}v_{\rm R}\rangle$', r'$\langle v_{\varphi}v_{\varphi}\rangle$', r'$\langle v_{z}v_{z}\rangle$']
    
    # normalize chosen colormap
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap=cm)
    
    sc = ax[i].scatter(bins[:, 0], bins[:, 1], c = bins[:, 4], cmap = cm, vmin=vmin, vmax=vmax, s = (4./expand_factor) * (bin_width_start/0.5) * np.sqrt(bins[:, 3]))
    cbar = plt.colorbar(sc, ax = ax[i]) 
    cbar.set_label('{}'.format(name[i]), rotation=270, fontsize=18, labelpad=30)
    ax[i].set_xlim(-22, 3)
    ax[i].set_ylim(-10, 15)
    overplot_rings(ax[i])
    ax[i].tick_params(axis=u'both', direction='in', which='both')
    ax[i].set_aspect('equal')
    ax[i].set_xlabel(r'$x\rm~[kpc]$', fontsize=18)
    ax[i].set_ylabel(r'$y\rm~[kpc]$', fontsize=18)
plt.savefig('non_axisymmetries/mesh_diagonal.png'.format(name_plot, name), bbox_inches = 'tight') 

fig, ax = plt.subplots(3, 1, figsize = (10, 25))
cm = plt.cm.get_cmap('RdBu_r')
for i in range(3):
    color_quantity = mean_XS_cyl_n[:, i+3]
    
    expand_factor = 2.5
    bins = average_in_bins(color_quantity, bins, mean_XS_cart_n, cut_z, expand_factor)
    
    vmin, vmax = -15, 15
    if i == 1:
        vmin, vmax = -220, 0
    name = [r'$\langle v_{\rm R}\rangle$', r'$\langle v_{\varphi}\rangle$', r'$\langle v_{z}\rangle$']
    
    # normalize chosen colormap
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap=cm)
    
    sc = ax[i].scatter(bins[:, 0], bins[:, 1], c = bins[:, 4], cmap = cm, vmin=vmin, vmax=vmax, s = (4./expand_factor) * (bin_width_start/0.5) * np.sqrt(bins[:, 3]))
    cbar = plt.colorbar(sc, ax = ax[i]) 
    cbar.set_label('{}'.format(name[i]), rotation=270, fontsize=18, labelpad=30)
    ax[i].set_xlim(-22, 3)
    ax[i].set_ylim(-10, 15)
    overplot_rings(ax[i])
    ax[i].tick_params(axis=u'both', direction='in', which='both')
    ax[i].set_aspect('equal')
    ax[i].set_xlabel(r'$x\rm~[kpc]$', fontsize=18)
    ax[i].set_ylabel(r'$y\rm~[kpc]$', fontsize=18)
plt.savefig('non_axisymmetries/mesh_vs.pdf'.format(name_plot, name), bbox_inches = 'tight') 


fig, ax = plt.subplots(3, 1, figsize = (10, 25))
cm = plt.cm.get_cmap('coolwarm_r')
ins = [(0, 1), (0, 2), (1, 2)]

for i in range(3):
    
    expand_factor = 2.5
    bins = average_in_bins(vtilde_n, bins, mean_XS_cart_n, cut_z, expand_factor, ins[i][0], ins[i][1])
    
    vmin, vmax = -0.5, 0.5
    name = [r'$\frac{\langle v_{\rm R}v_{\rm \varphi}\rangle}{\sqrt{\langle v_{\rm R}^2\rangle \langle v_{\varphi}^2\rangle}}$', r'$\frac{\langle v_{\rm R}v_{z}\rangle}{\sqrt{\langle v_{\rm R}^2\rangle \langle v_{z}^2\rangle}}$', r'$\frac{\langle v_{\rm \varphi}v_{z}\rangle}{\sqrt{\langle v_{\rm \varphi}^2\rangle \langle v_{z}^2\rangle}}$']
    
    # normalize chosen colormap
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap=cm)
    
    sc = ax[i].scatter(bins[:, 0], bins[:, 1], c = bins[:, 4], cmap = cm, vmin=vmin, vmax=vmax, s = (4./expand_factor) * (bin_width_start/0.5) * np.sqrt(bins[:, 3]))
    cbar = plt.colorbar(sc, ax = ax[i]) 
    cbar.set_label('{}'.format(name[i]), rotation=270, fontsize=18, labelpad=30)
    ax[i].set_xlim(-22, 3)
    ax[i].set_ylim(-10, 15)
    overplot_rings(ax[i])
    ax[i].tick_params(axis=u'both', direction='in', which='both')
    ax[i].set_aspect('equal')
    ax[i].set_xlabel(r'$x\rm~[kpc]$', fontsize=18)
    ax[i].set_ylabel(r'$y\rm~[kpc]$', fontsize=18)
plt.savefig('non_axisymmetries/mesh_off_diagonal.pdf'.format(name_plot, name), bbox_inches = 'tight') 


# -------------------------------------------------------------------------------
# log R vs. phi
# -------------------------------------------------------------------------------

#shift phi

bins = average_in_bins(vtilde_n[:, 2, 2], bins, mean_XS_cart_n, cut_z, expand_factor)
phi = np.arctan2(bins[:, 1], bins[:, 0])
phi[phi<-1] = phi[phi<-1] + 2 * np.pi
R = np.sqrt(bins[:, 0] ** 2 + bins[:, 1] ** 2)

fig, ax = plt.subplots(1, 1, figsize = (10, 8))
sc = ax.scatter(phi, np.log10(R), c = bins[:, 4], vmin = 0, vmax = 4000)
cbar = plt.colorbar(sc)
cbar.set_label(r'$v_r$', rotation=270, fontsize=18, labelpad=30)
ax.set_xlim(-1, 2.*np.pi-2)
ax.set_ylim(-0.5, 1.5)
ax.tick_params(axis=u'both', direction='in', which='both')
ax.set_xlabel(r'$\varphi$', fontsize=18)
ax.set_ylabel(r'$\log R$\rm ~[kpc]', fontsize=18)

# -------------------------------------------------------------------------------'''
# Bessel funtions
# -------------------------------------------------------------------------------

'''from scipy.special import j0


#fig, ax = plt.subplots(1, 1, figsize = (10, 8))
#sc = plt.scatter(xx[:, 0], xx[:, 1], c = xx[:, 2], cmap = cm, vmin=vmin, vmax=vmax, s = np.sqrt(bins[:, 3]))
#cbar = plt.colorbar(sc) 
#cbar.set_label('{}'.format(name), rotation=270, fontsize=18, labelpad=30)
#ax.set_xlim(-22, 2)
#ax.set_ylim(-10, 14)
#overplot_rings(ax)
#ax.tick_params(axis=u'both', direction='in', which='both')
#ax.set_aspect('equal')
#ax.set_xlabel(r'$x\rm~[kpc]$', fontsize=18)
#ax.set_ylabel(r'$y\rm~[kpc]$', fontsize=18)
#plt.savefig('non_axisymmetries/mesh_{}_{}.pdf'.format(name_plot, name)) 

#fig, ax = plt.subplots(1, 1, figsize = (10, 8))
#cbar = plt.colorbar(sc) 
#cbar.set_label('{}'.format(name), rotation=270, fontsize=18, labelpad=30)
#
#squares = []
#patch_colors = []
##alphas = []
#for i in range(bins.shape[0]):        
#    
#    if bins[i, 3] > 0:
#        rect = Rectangle((bins[i, 0] - 0.5 * bins[i, 2], bins[i, 1] - 0.5 * bins[i, 2]), bins[i, 2], bins[i, 2])
#        squares.append(rect)
#        patch_colors.append(bins[i, 4])
#        #alphas.append(np.float(0.1 * np.sqrt(bins[i, 3])))
#        
## Create patch collection with specified colour/alpha
#pc = PatchCollection(squares, facecolor=mapper.to_rgba(patch_colors), edgecolor='none')
#
#ax.add_collection(pc)        
#ax.set_xlim(-27, 9)
#ax.set_ylim(-17, 19)
#ax.tick_params(axis=u'both', direction='in', which='both')
#ax.set_aspect('equal')
#overplot_rings(ax)
#ax.set_xlabel(r'$x\rm~[kpc]$', fontsize=18)
#ax.set_ylabel(r'$y\rm~[kpc]$', fontsize=18)    
#plt.savefig('non_axisymmetries/squares_{}_{}.pdf'.format(name_plot, name))


# -------------------------------------------------------------------------------'''
# tesselation
# -------------------------------------------------------------------------------


'''from scipy.spatial import Voronoi, voronoi_plot_2d, KDTree

np.random.seed(42) 

# choose every Nth star randomly
disk_XS_cart = mean_XS_cart_n[cut_z, :]
disk_XS_cyl = mean_XS_cyl_n[cut_z, :]
quantity = disk_XS_cyl[:, 3] 
N = 50 
Nstars = len(quantity)
N_control_stars = int(Nstars/N)
indices_control_stars = np.random.choice(np.arange(Nstars), size = N_control_stars)
control_stars_XS = disk_XS_cart[indices_control_stars, :]
control_stars_bool = np.zeros((len(disk_XS_cart)), dtype = bool)
control_stars_bool[indices_control_stars] = True

# add set of stars
foo, bar = np.meshgrid(np.arange(-25, 11, 10), np.arange(-15, 26, 10))
xx = np.vstack((control_stars_XS[:, 0:2], np.vstack((foo.flatten(), bar.flatten())).T))

# create voronoi tesselation on control stars
vor = Voronoi(xx) #control_stars_XS[:, 0:2])

# for each star find nearest control star
kdt = KDTree(control_stars_XS[:, 0:2])
d, ins = kdt.query(disk_XS_cart[:, 0:2], k = 1)

# make average for every control star
name = r'$v_R$'
vmin, vmax = -10, 10
vs = np.array([np.nanmean(quantity[ins == i]) for i in range(N_control_stars)])
Ns = np.array([np.sum([ins == i]) for i in range(N_control_stars)])

# fake
cm = plt.cm.get_cmap('RdBu')
sc = plt.scatter(vs, vs, c = vs, cmap = cm, vmin=vmin, vmax=vmax)
plt.close()

# normalize chosen colormap
norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap=cm)

fig, ax = plt.subplots(1, 1, figsize = (10, 8))
cbar = plt.colorbar(sc) 

for r in range(len(vor.point_region)):
    region = vor.regions[vor.point_region[r]]
    if not -1 in region:
        polygon = [vor.vertices[i] for i in region]
        plt.fill(*zip(*polygon), color=mapper.to_rgba(vs[r]))
        
ax.set_xlim(Xlimits[0])
ax.set_ylim(Xlimits[1])
ax.tick_params(axis=u'both', direction='in', which='both')
ax.set_aspect('equal')
ax.set_xlabel(r'$x\rm~[kpc]$', fontsize=18)
ax.set_ylabel(r'$y\rm~[kpc]$', fontsize=18)

# -------------------------------------------------------------------------------'''