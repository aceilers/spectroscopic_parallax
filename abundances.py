#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 11:55:49 2018

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
matplotlib.rc('text', usetex=True)
fsize = 14

# -------------------------------------------------------------------------------
# open inferred labels
# -------------------------------------------------------------------------------

N = 45787
Kfold = 2
lam = 30
name = 'N{0}_lam{1}_K{2}_parallax_payne'.format(N, lam, Kfold)

print('loading new labels...')   
#labels = Table.read('data/training_labels_new_{}.fits'.format(name), format = 'fits')  
labels = Table.read('data/Eilers_Payne_Apogee.fits' , format = 'fits') 
labels.rename_column('pmra_2a', 'pmra')
labels.rename_column('pmdec_2a', 'pmdec') 
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

spec_par = labels['spec_parallax'] * u.mas
distance = spec_par.to(u.parsec, equivalencies = u.parallax())

cs = coord.ICRS(ra = labels['ra'] * u.degree, 
                dec = labels['dec'] * u.degree, 
                distance = distance, 
                pm_ra_cosdec = labels['pmra'] * u.mas/u.yr, 
                pm_dec = labels['pmdec'] * u.mas/u.yr, 
                radial_velocity = labels['VHELIO_AVG'] *u.km/u.s)

gc = coord.Galactocentric(galcen_distance = X_GC_sun_kpc*u.kpc,
                          galcen_v_sun = coord.CartesianDifferential([-vX_GC_sun_kms, vY_GC_sun_kms, vZ_GC_sun_kms] * u.km/u.s),
                          z_sun = Z_GC_sun_kpc*u.kpc)

galcen = cs.transform_to(gc)
xs, ys, zs = galcen.x.to(u.kpc), galcen.y.to(u.kpc), galcen.z.to(u.kpc)
vxs, vys, vzs = galcen.v_x, galcen.v_y, galcen.v_z
XS = np.vstack([xs, ys, zs, vxs, vys, vzs]).T.value
Rs = np.sqrt(XS[:, 0] ** 2 + XS[:, 1] ** 2)

# -------------------------------------------------------------------------------
# spatial cuts and bins in logg
# -------------------------------------------------------------------------------   

# calculate annuli only in 30 degree wedge!  
deg_wedge = 30.
wedge = np.abs(XS[:, 1]) < np.abs(XS[:, 0]) * np.tan(deg_wedge/360. * 2. * np.pi)

# take wegde in z 
deg_wedge_in_z = 6.
wedge_z = (np.abs(XS[:, 2])/Rs) < np.tan(deg_wedge_in_z/360. * 2. * np.pi)
cut_z = np.logical_or(abs(XS[:, 2]) < 0.5, wedge_z)

cuts = cut_z * wedge * (XS[:, 0] < 0)

labels = labels[cuts]
XS = XS[cuts]
Rs = Rs[cuts]

split_label, latex_label = 'TEFF', r'T_{\rm eff}' #'LOGG_2', r'\log g'
cut_logg = abs(labels[split_label]) < 8000
labels = labels[cut_logg]
XS = XS[cut_logg, :]
Rs = Rs[cut_logg]

percentiles = np.arange(0., 100.001, 100. / 8.)
print(percentiles)
logg_bins = np.percentile(labels[split_label], percentiles)
logg_bin_centers = 0.5 * (logg_bins[:-1] + logg_bins[1:])


# -------------------------------------------------------------------------------
# function
# -------------------------------------------------------------------------------   

elements = ['feh', 'ch', 'nh', 'oh', 'mgh', 'tih', 'sh', 'sih', 'kh', 'cah', 'crh', 'nih', 'mnh', 'alh', 'cuh']
elements_latex = ['[Fe/H]', '[C/Fe]', '[N/Fe]', '[O/Fe]', '[Mg/Fe]', '[Ti/Fe]', '[S/Fe]', '[Si/Fe]', '[K/Fe]', '[Ca/Fe]', '[Cr/Fe]', '[Ni/Fe]', '[Mn/Fe]', '[Al/Fe]', '[Cu/Fe]']
#elements = ['FE_H', 'ALPHA_M', 'M_H', 'C_FE', 'CI_FE', 'N_FE', 'O_FE', 'NA_FE', 'MG_FE', 'TI_FE', 'S_FE', 'SI_FE', 'P_FE', 'K_FE', 'CA_FE', 'TIII_FE', 'V_FE', 'CR_FE', 'CO_FE', 'NI_FE', 'MN_FE', 'AL_FE']
#elements_latex = ['[Fe/H]', '[$\\alpha$/M]', '[M/H]', '[C/Fe]', '[Ci/Fe]', '[N/Fe]', '[O/Fe]', '[Na/Fe]', '[Mg/Fe]', '[Ti/Fe]', '[S/Fe]', '[Si/Fe]', '[P/Fe]', '[K/Fe]', '[Ca/Fe]', '[TiII/Fe]', '[V/Fe]', '[Cr/Fe]', '[Co/Fe]', '[Ni/Fe]', '[Mn/Fe]', '[Al/Fe]']

stars_per_bin = 64
def abundance_bins(labels, Rs, elements):  
    
    foo = np.append(0., np.sort(Rs))
    bar = np.append(np.sort(Rs), 100.)
    bin_start = 0.5 * (foo[::stars_per_bin] + bar[::stars_per_bin])
    bin_end = bin_start[1:]
    bin_start = bin_start[:-1]
    n_bins = len(bin_start)
    n_elements = len(elements)
    
    bin_Rs = np.zeros((n_bins))
    bin_abundance = np.zeros((n_bins, n_elements))
    bin_errs = np.zeros((n_bins, n_elements))
    for j in range(len(bin_start)):
        inside = (Rs > bin_start[j]) * (Rs < bin_end[j])
        bin_Rs[j] = np.nanmean(Rs[inside])
        for k, el in enumerate(list(elements)):
            cut_el = abs(labels[el]) < 10
            if el != 'feh':
                cut_feh = abs(labels['feh']) < 10
                x_fe = labels[el][inside * cut_el * cut_feh] - labels['feh'][inside * cut_el * cut_feh]
                bin_abundance[j, k] = np.nanmean(x_fe)
                bin_errs[j, k] = np.sqrt(np.nanvar(x_fe) / np.sum(inside * cut_el * cut_feh))
            else:
                bin_abundance[j, k] = np.nanmean(labels[el][inside * cut_el])
                bin_errs[j, k] = np.sqrt(np.nanvar(labels[el][inside * cut_el]) / np.sum(inside * cut_el))            
    return bin_Rs, bin_abundance, bin_errs


logg_colors = [matplotlib.cm.viridis(x) for x in np.linspace(0., 1., 8)]

fig, ax = plt.subplots(5, 3, figsize = (18, 16), sharex = True)   

for i in range(len(logg_bins) - 1):
    
    inside = (labels[split_label] > logg_bins[i]) * (labels[split_label] < logg_bins[i+1])    
    rs_i, abundances_i, errs_i = abundance_bins(labels[inside], Rs[inside], elements)
    c, r = 0, 0
    for l, el in enumerate(list(elements)):
        ax[c, r].errorbar(rs_i, abundances_i[:, l], yerr = errs_i[:, l], c = logg_colors[i], fmt = 'o', markersize = 5, label = r'${0} \approx {1}$'.format(latex_label, round(logg_bin_centers[i], 3)))
    
        if i == 0:  
            ax[c, r].set_xlim(0, 25)
            #ax[c, r].set_ylim(np.percentile(labels[cut_z_wedge][el], 1), np.percentile(labels[cut_z_wedge][el], 99))
            ax[c, r].set_ylabel(r'{}'.format(elements_latex[l]), fontsize = fsize)
            ax[c, r].tick_params(axis=u'both', direction='in', which='both', right = 'on', top = 'on')
            ax[4, r].set_xlabel(r'$\rm R_{GC}$', fontsize = fsize)
        if r == 2: 
            c += 1
            r = 0    
        else:
            r += 1
ax[0, 0].legend(frameon = True)
fig.subplots_adjust(wspace = 0.0)
plt.tight_layout()
#fig.delaxes(ax[5, 2])
#fig.delaxes(ax[5, 3])
#plt.savefig('plots/rotation_curve/abundances/all_vs_R_annulus_{0}_{1}_bins_payne.pdf'.format(stars_per_bin, split_label), bbox_inches = 'tight', dpi = 200)

# -------------------------------------------------------------------------------
# v_z vs. z
# -------------------------------------------------------------------------------   

XS = np.vstack([xs, ys, zs, vxs, vys, vzs]).T.value

labels = Table.read('data/Eilers_Payne_Apogee.fits' , format = 'fits') 
labels.rename_column('pmra_2a', 'pmra')
labels.rename_column('pmdec_2a', 'pmdec') 
labels.rename_column('ra_1', 'ra')
labels.rename_column('dec_1', 'dec')
 
el = 'O_FE'
cmap = 'RdYlBu_r'
cut_x = (XS[:, 0] < -5) * (XS[:, 0] > -11)
fig, ax = plt.subplots(1, len(logg_bins)-1, figsize = (20, 5), sharex = True, sharey = True)   
for i in range(len(logg_bins) - 1):    
    inside = (labels[split_label] > logg_bins[i]) * (labels[split_label] < logg_bins[i+1])    
    ax[i].scatter(XS[inside * cut_x, 2], XS[inside * cut_x, 5], s = 2, c = labels[el][inside * cut_x], vmin = -.05, vmax = .3, cmap = cmap)
    ax[i].set_title(r'${0} \approx {1}$'.format(latex_label, round(logg_bin_centers[i], 3)))
    ax[i].set_xlabel(r'$z$ [kpc]', fontsize = 14)
ax[0].set_xlim(-2, 2)
ax[0].set_ylim(-75, 75)
ax[0].set_ylabel(r'$v_z$ [km/s]', fontsize = 14)
plt.tight_layout()

#plt.savefig('plots/rotation_curve/abundances/all_vs_R_annulus_{0}_{1}_bins_payne.pdf'.format(stars_per_bin, split_label), bbox_inches = 'tight', dpi = 200)
cmap = 'RdYlBu_r'
plt.scatter(XS[cut_x, 2], XS[cut_x, 5], s = 1, c = labels[el][cut_x], vmin = -.5, vmax = .3, cmap = cmap)
plt.xlim(-2, 2)
plt.ylim(-75, 75) 
plt.ylabel(r'$v_z$ [km/s]', fontsize = 14) 
plt.xlabel(r'$z$ [kpc]', fontsize = 14)
    
# -------------------------------------------------------------------------------'''
