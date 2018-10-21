#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 11:10:48 2018

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

N = 44784
Kfold = 2
lam = 30
name = 'N{0}_lam{1}_K{2}_offset0.0483_parallax'.format(N, lam, Kfold)

print('loading new labels...')   
labels = Table.read('data/training_labels_new_{}.fits'.format(name), format = 'fits')    
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
# transformation
# -------------------------------------------------------------------------------           

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
# take open cluster
# -------------------------------------------------------------------------------

t = Table(names = ('cluster', 'RA', 'DEC', 'FE_H', 'distance', 'lat_name'), dtype=('S8', 'S8', 'S8', 'S16', 'f8', 'S8'))

# look up Melissa's paper: arxiv: 1701.07829

# M67
clus = 'm67'
lat = 'M 67'
ra_clus_hms = '08:51:18'
dec_clus_dms = '+11:48:00'
distance = 0.857 #kpc (from Yakut et al. 2009)
feh_clus = '+0.01\pm 0.05'
t.add_row([clus, ra_clus_hms, dec_clus_dms, feh_clus, distance, lat])

# NGC 6791
clus = 'ngc6791'
lat = 'NGC 6791'
ra_clus_hms = '19:20:53.0' 
dec_clus_dms = '+37:46:18'
distance = 4.078
feh_clus = '+0.47\pm 0.07'
t.add_row([clus, ra_clus_hms, dec_clus_dms, feh_clus, distance, lat])

# NGC 6819
clus = 'ngc6819'
lat = 'NGC 6819'
ra_clus_hms = '19:41:18.0' 
dec_clus_dms = '+40:11:12'
distance = 2.2 #kpc
feh_clus = '+0.09\pm 0.03'
t.add_row([clus, ra_clus_hms, dec_clus_dms, feh_clus, distance, lat])

# M35
clus = 'm35'
lat = 'M 35'
ra_clus_hms = '06 08 54.0' 
dec_clus_dms = '+24 20 00'
distance = 0.85 #kpc
feh_clus = '-0.21\pm 0.10'
t.add_row([clus, ra_clus_hms, dec_clus_dms, feh_clus, distance, lat])

# M92
clus = 'm92'
lat = 'M 92'
ra_clus_hms = '17 17 07.39' 
dec_clus_dms = '+43 08 09.4'
distance = 8.2 #kpc
feh_clus = '-2.35\pm 0.05'
t.add_row([clus, ra_clus_hms, dec_clus_dms, feh_clus, distance, lat])

# M15
clus = 'm15'
lat = 'M 15'
ra_clus_hms = '21 29 58.33' 
dec_clus_dms = '+12 10 01.2'
distance = 10.3 #kpc
feh_clus = '-2.33\pm 0.02'
t.add_row([clus, ra_clus_hms, dec_clus_dms, feh_clus, distance, lat])

# M53
clus = 'm53'
lat = 'M 53'
ra_clus_hms = '13 12 55.25' 
dec_clus_dms = '+18 10 05.4'
distance = 17.9 #kpc
feh_clus = '-2.06\pm 0.09'
t.add_row([clus, ra_clus_hms, dec_clus_dms, feh_clus, distance, lat])

# M13
clus = 'm13'
lat = 'M 13'
ra_clus_hms = '16 41 41.634' 
dec_clus_dms = '+36 27 40.75'
distance = 6.8 #kpc
feh_clus = '-1.58\pm 0.04'
t.add_row([clus, ra_clus_hms, dec_clus_dms, feh_clus, distance, lat])

# M2
clus = 'm2'
lat = 'M 2'
ra_clus_hms = '21 33 27.02' 
dec_clus_dms = '-00 49 23.7'
distance = 10 #kpc
feh_clus = '-1.66\pm 0.07'
t.add_row([clus, ra_clus_hms, dec_clus_dms, feh_clus, distance, lat])

# M3
clus = 'm3'
lat = 'M 3'
ra_clus_hms = '13 42 11.62' 
dec_clus_dms = '+28 22 38.2'
distance = 10.4 #kpc
feh_clus = '-1.50\pm 0.05'
t.add_row([clus, ra_clus_hms, dec_clus_dms, feh_clus, distance, lat])

# M5
clus = 'm5'
lat = 'M 5'
ra_clus_hms = '15 18 33.22' 
dec_clus_dms = '+02 04 51.7'
distance = 7.5 #kpc
feh_clus = '-1.33\pm 0.02'
t.add_row([clus, ra_clus_hms, dec_clus_dms, feh_clus, distance, lat])

# M71
clus = 'm71'
lat = 'M 71'
ra_clus_hms = '19 53 46.49' 
dec_clus_dms = '+18 46 45.1'
distance = 4. #kpc
feh_clus = '-1.33\pm 0.02'
t.add_row([clus, ra_clus_hms, dec_clus_dms, feh_clus, distance, lat])

# M107
clus = 'm107'
lat = 'M 107'
ra_clus_hms = '16 32 31.86' 
dec_clus_dms = '-13 03 13.6'
distance = 6.4 #kpc
feh_clus = '-1.03\pm 0.02'
t.add_row([clus, ra_clus_hms, dec_clus_dms, feh_clus, distance, lat])

# M45 = Pleiades (no stars found)
clus = 'm45'
lat = 'M 45'
ra_clus_hms = '03 47 00.0' 
dec_clus_dms = '+24 07 00'
distance = 0.136 #kpc
feh_clus = '+0.03\pm 0.02'
t.add_row([clus, ra_clus_hms, dec_clus_dms, feh_clus, distance, lat])

# NGC 2158
clus = 'ngc2158'
lat = 'NGC 2158'
ra_clus_hms = '06 07 25.0' 
dec_clus_dms = '+24 05 48'
distance = 3.37 #kpc
feh_clus = '-0.28\pm 0.05'
t.add_row([clus, ra_clus_hms, dec_clus_dms, feh_clus, distance, lat])

# NGC 188
clus = 'ngc188'
lat = 'NGC 188'
ra_clus_hms = '00 48 26.0' 
dec_clus_dms = '+85 15 18'
distance = 1.66 #kpc
feh_clus = '-0.03\pm 0.04'
t.add_row([clus, ra_clus_hms, dec_clus_dms, feh_clus, distance, lat])

# NGC 6819
clus = 'ngc6819'
lat = 'NGC 6819'
ra_clus_hms = '00 48 26.0' 
dec_clus_dms = '+85 15 18'
distance = 1.66 #kpc
feh_clus = '+0.09\pm 0.03'
t.add_row([clus, ra_clus_hms, dec_clus_dms, feh_clus, distance, lat])

# NGC 7789
clus = 'ngc7789'
lat = 'NGC 7789'
ra_clus_hms = '23 57 24.0' 
dec_clus_dms = '+56 42 30'
distance = 2.33 #kpc
feh_clus = '+0.02\pm 0.04'
t.add_row([clus, ra_clus_hms, dec_clus_dms, feh_clus, distance, lat])

# NGC 2420
clus = 'ngc2420'
lat = 'NGC 2420'
ra_clus_hms = '07 38 23.0' 
dec_clus_dms = '+21 34 24'
distance = 2.5 #kpc
feh_clus = '-0.20\pm 0.06'
t.add_row([clus, ra_clus_hms, dec_clus_dms, feh_clus, distance, lat])

c = SkyCoord('{0} {1}'.format(ra_clus_hms, dec_clus_dms), unit=(u.hourangle, u.deg))
ra_clus = c.ra.deg
dec_clus = c.dec.deg

cut_clus = (abs(labels['ra'] - ra_clus) < 1) \
          * (abs(labels['dec'] - dec_clus) < 1) # not isotropic --> rectangle!

XS_clus = np.vstack([labels['ra'][cut_clus], labels['dec'][cut_clus], labels['spec_parallax'][cut_clus]]).T
Xlabels_clus = ['RA', 'DEC', r'$\varpi$']
#fig = corner.corner(XS_clus, labels = Xlabels_clus, truths = [ra_clus, dec_clus, 1./distance])
#fig.savefig('plots/corner_{}_varpi.pdf'.format(clus))

# -------------------------------------------------------------------------------
# Sagittarius data
# -------------------------------------------------------------------------------

hdu = fits.open('data/Sgr_Candidate_in_Gaia.fits')
data_sgr = hdu[1].data

indices = np.arange(len(labels))     
xx = []              
for i in data_sgr['APOGEE_ID']:
    foo = labels['APOGEE_ID'] == i
    if np.sum(foo) == 1:
        xx.append(indices[foo][0])
xx = np.array(xx)

labels_sgr = labels[xx]

# -------------------------------------------------------------------------------
# plot for paper (3x2) histograms: Gaia vs. me (labeled by cluster name and metallicity)
# -------------------------------------------------------------------------------

cluster_list = ['m2', 'm3', 'm5', 'm13', 'm15', 'm53', 'm71', 'm92', 'm107', 'ngc6791', 'ngc6819']

fig, ax = plt.subplots(4, 3, figsize = (9, 12))
plt.subplots_adjust(wspace = 0.08, hspace = 0.25)
c, r = 0, 0
for i, clus in enumerate(cluster_list):       
    k = t['cluster'] == clus
    print(c, r, len(k))
    
    coord = SkyCoord('{0} {1}'.format(t['RA'][k][0], t['DEC'][k][0]), unit=(u.hourangle, u.deg))
    ra_clus = coord.ra.deg
    dec_clus = coord.dec.deg
    cut_clus = (abs(labels['ra'] - ra_clus) < .8) \
          * (abs(labels['dec'] - dec_clus) < .8) 
          
    bins = np.linspace(-0.1, .8, 40)
    ax[c, r].hist(labels['spec_parallax'][cut_clus], normed = True, bins = bins, histtype = 'step', lw = 3, color = 'k', label = r'$\varpi^{\rm (sp)}$')
    ax[c, r].hist(labels['parallax'][cut_clus], normed = True, bins = bins, histtype = 'step', lw = 1, color = 'k', label = r'$\varpi^{\rm (a)}$')
    ax[c, r].tick_params(axis=u'both', direction='in', which='both', right = 'on', top = 'on')
    
    ax[c, r].axvline(1./t['distance'][k][0], color = 'r', lw = 2)
    ax[c, r].set_title(r'{0}, $\rm [Fe/H] = {1}$'.format(t['lat_name'][k][0], t['FE_H'][k][0]), fontsize = 13)
    ax[c, r].set_xlabel(r'$\varpi$ [mas]', fontsize = 14)
    ax[c, r].set_ylabel('normalized counts', fontsize = 14)
    ax[c, r].set_xlim(-0.1, .8)
    if r == 2: 
        c += 1
        r = 0    
    else:
        r += 1
plt.tight_layout()
ax[0, 0].legend(frameon = True)
ax[3, 2].hist(labels_sgr['spec_parallax'], normed = True, bins = bins, histtype = 'step', lw=3, color='k', label = r'$\varpi^{(sp)}$')
ax[3, 2].hist(labels_sgr['parallax'], normed = True, bins = bins, histtype = 'step', lw=1, color='k', label = r'$\varpi^{(a)}$')
ax[3, 2].set_xlabel(r'$\varpi$ [mas]', fontsize = 14)
ax[3, 2].set_xlim(-0.1, .8)
ax[3, 2].axvline(1./20, color = 'r')#929591')
ax[3, 2].set_title('Sagittarius')
ax[3, 2].tick_params(axis=u'both', direction='in', which='both', right = 'on', top = 'on')
#plt.savefig('plots/open_clusters/test_open_clusters_{}.pdf'.format(name))
plt.savefig('paper/clusters.pdf')

# -------------------------------------------------------------------------------
# take members
# -------------------------------------------------------------------------------

hdu = fits.open('plots/open_clusters/M71members')
mem = Table(hdu[1].data)

offset = 0.0483
Kfold = 2
lam = 30                      # hyperparameter -- needs to be tuned! CROSS VALIDATED (10.08.2018)
N = 44784
name = 'N{0}_lam{1}_K{2}_offset{3}_parallax'.format(N, lam, Kfold, offset)

print('loading new labels...')   
labels = Table.read('data/training_labels_new_{}.fits'.format(name), format = 'fits') 

cut_jk = (labels['J'] - labels['K']) < (0.4 + 0.45 * labels['bp_rp'])
cut_hw2 = (labels['H'] - labels['w2mpro']) > -0.05
cut_finite = (labels['J'] > -100) * (labels['H'] > -100) * (labels['K'] > -100) *\
         (labels['J_ERR'] > 0) * (labels['H_ERR'] > 0) * (labels['K_ERR'] > 0) * \
         np.isfinite(labels['w1mpro']) * np.isfinite(labels['w2mpro']) * \
         (labels['w1mpro_error'] > 0) * (labels['w2mpro_error'] > 0)
labels = labels[cut_jk * cut_hw2 * cut_finite]

xx = join(mem, labels, join_type = 'inner', keys = 'APOGEE_ID')
plt.errorbar(xx['spec_parallax_2'], xx['LOGG_2'], xerr = xx['spec_parallax_err'], fmt = 'o', label = r'$\varpi^{\rm (sp)}$', zorder = 10)
plt.errorbar(xx['parallax_2'], xx['LOGG_2'], xerr = xx['parallax_error_2'], fmt = 'o', color = '#929591', label = r'$\varpi^{\rm (g)}$', zorder = 20)
avg = np.sum(xx['spec_parallax_2'] / xx['spec_parallax_err'] ** 2) / np.sum(1. / xx['spec_parallax_err'] **2)
avg_gaia = np.sum(xx['parallax_2'] / xx['parallax_error_2'] ** 2) / np.sum(1. / xx['parallax_error_2'] **2)
plt.axvline(avg_gaia, color='#929591', lw = 0.8, zorder = -10, linestyle = ':')
plt.axvline(avg, color='b', linestyle = ':', lw= 0.8, zorder = -10, label = 'avg')
distance = 4.
plt.axvline(1./distance, color = 'r', linestyle = ':', label = 'distance')
plt.legend()
plt.xlabel(r'$\varpi$ [mas]', fontsize = 14)
plt.ylabel(r'$\log g$', fontsize = 14)
plt.savefig('plots/open_clusters/m71_new.pdf')


cluster_list = ['M67', 'M71', 'M107', 'NGC2862']

fig, ax = plt.subplots(4, 1, figsize = (9, 9), sharex = True, sharey = True)
plt.subplots_adjust(wspace = 0.02, hspace = 0.02)
c, r = 0, 0
for i in range(4):
    hdu = fits.open('plots/open_clusters/{}members'.format(cluster_list[i]))
    mem = Table(hdu[1].data)

    xx = join(mem, labels, join_type = 'inner', keys = 'APOGEE_ID')
    ax[c].errorbar(xx['spec_parallax_2'], xx['LOGG_2'], xerr = xx['spec_parallax_err'], fmt = 'o', markersize = 4, color = "k", label = r'spectrophotometric parallax $\varpi^{\rm (sp)}$', zorder = 20)
    ax[c].errorbar(xx['parallax_2'], xx['LOGG_2'], xerr = xx['parallax_error_2'], fmt = 'o', markersize = 4, color = '#929591', label = r'Gaia DR2 parallax $\varpi^{\rm (a)}$', zorder = 10)
    avg = np.sum(xx['spec_parallax_2'] / xx['spec_parallax_err'] ** 2) / np.sum(1. / xx['spec_parallax_err'] **2)
    avg_gaia = np.sum(xx['parallax_2'] / xx['parallax_error_2'] ** 2) / np.sum(1. / xx['parallax_error_2'] **2)
    print(cluster_list[i], avg, avg_gaia)
    ax[c].axvline(avg, color="k", lw = 0.8, zorder = -10, linestyle = ':')
    ax[c].axvline(avg_gaia, color='#929591', lw = 0.8, zorder = -10, linestyle = ':')
    ax[c].tick_params(axis=u'both', direction='in', which='both', right = 'on', top = 'on')
    ax[c].annotate('{}'.format(cluster_list[i]), (1.25, 0.75), fontsize = 14) #, bbox = ('ec' = '0.5'))
    ax[c].set_ylabel(r'$\log g$', fontsize = 14)
    ax[c].set_ylim(0.6, 2.4)
    #distance = 6.4
    #plt.axvline(1./distance, color = 'r', linestyle = ':', label = 'distance')
#    if r == 1: 
#        c += 1
#        r = 0    
#    else:
#        r += 1
    c += 1
ax[0].legend(fontsize = 13)
ax[-1].set_xlabel(r'$\varpi$ [mas]', fontsize = 14)

plt.savefig('paper/clusters.pdf', bbox_inches = 'tight')


# -------------------------------------------------------------------------------
# take Sagittarius region...
# -------------------------------------------------------------------------------'''

'''XS_sgr = np.vstack([labels_sgr['ra'], labels_sgr['dec'], labels_sgr['spec_parallax']]).T
Xlabels_sgr = ['RA', 'DEC', r'$\varpi$']
fig = corner.corner(XS_sgr, labels = Xlabels_sgr, plot_datapoints = True)
fig.savefig('plots/open_clusters/corner_sgr_varpi_{}.pdf'.format(name)) 

fig, ax = plt.subplots(1, 1, figsize = (8, 8))
ax.scatter(labels_sgr['RA'], labels_sgr['spec_parallax'], zorder = 20, color = '#363737', s = 20, label = 'spectroscopic parallax')
ax.scatter(labels_sgr['RA'], labels_sgr['parallax'], zorder = 10, color = "#95d0fc", s = 15, label = 'Gaia parallax')
ax.set_xlabel('RA', fontsize = 14)
ax.legend(frameon = True, fontsize = 14)
ax.set_ylabel(r'$\varpi$', fontsize = 14)    
ax.tick_params(axis=u'both', direction='in', which='both')
ax.axhline(1./20, linestyle = '--', color = '#929591')
ax.set_title('Sagittarius', fontsize = 14)         
plt.savefig('plots/open_clusters/sagittarius_ra_varpi_{}.pdf'.format(name)) 
plt.close()

plt.hist(1./labels_sgr['spec_parallax'], bins = np.linspace(-1000, 1000, 60), histtype = 'step', lw=3, color='k', label = 'spec. parallax')
plt.hist(1./labels_sgr['parallax'], bins = np.linspace(-1000, 1000, 50), histtype = 'step', lw=1, color='k', label = 'Gaia parallax')
plt.xlabel(r'distance [kpc]', fontsize = 14)
plt.xlim(-500, 500)
plt.axvline(20, linestyle = '--', color = '#929591')
plt.tick_params(axis=u'both', direction='in', which='both')
plt.legend(frameon = True, fontsize = 12)
plt.savefig('plots/open_clusters/hist_sgr_dist_{}.pdf'.format(name)) 
plt.close()
                        
plt.hist(labels_sgr['spec_parallax'], bins = np.linspace(-1, 1, 100), histtype = 'step', lw=3, color='k', label = 'spec. parallax')
plt.hist(labels_sgr['parallax'], bins = np.linspace(-1, 1, 100), histtype = 'step', lw=1, color='k', label = 'Gaia parallax')
plt.xlabel(r'$\varpi$ [mas]', fontsize = 14)
plt.xlim(-0.2, 0.2)
plt.axvline(1./20, linestyle = '--', color = '#929591')
plt.tick_params(axis=u'both', direction='in', which='both')
plt.legend(frameon = True, fontsize = 11)
plt.savefig('plots/open_clusters/sagittarius_parallax_{}.pdf'.format(name))                        
plt.close()
# -------------------------------------------------------------------------------'''
