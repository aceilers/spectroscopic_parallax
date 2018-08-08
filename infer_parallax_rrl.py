#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 15:15:21 2018

@author: eilers
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy.optimize as op
import pickle
from astropy.table import Column, Table, join, vstack, hstack
from astropy.io import fits
from scipy.stats import binned_statistic_2d
import matplotlib.gridspec as gridspec
from astropy import units as u

# -------------------------------------------------------------------------------
# plotting settings
# -------------------------------------------------------------------------------

matplotlib.rcParams['ytick.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 14
fsize = 14

# -------------------------------------------------------------------------------
# load spectra and labels
# -------------------------------------------------------------------------------

ps = True
print('loading labels...')
if ps: hdu = fits.open('data/RRL_PS1_Gaia_full.fits')
else: hdu = fits.open('data/RRL_GDR2_ALLWISE_full.fits')
labels = Table(hdu[1].data)

offset = 0.029 # mas as per Lindegren et al. 2018
labels['parallax'] += offset

# -------------------------------------------------------------------------------
# linear algebra
# -------------------------------------------------------------------------------

def H_func(x, y, A, lam1, lam2, ivar):
    foo = np.dot(A, x)
    y_model = np.exp(foo)
    dy = y - y_model
    H = 0.5 * np.dot(dy.T, ivar * dy) + np.sum(lam1 * np.abs(x)) + np.sum(lam2 * foo * foo)
    dHdx = -1. * np.dot(A.T * y_model[None, :], ivar * dy) + lam1 * np.sign(x) + 2. * lam2 * np.dot(A.T, foo)
    return H, dHdx

def check_H_func(x, y, A, lam1, lam2, ivar):
    H0, dHdx0 = H_func(x, y, A, lam1, lam2, ivar)
    dx = 1e-6 # magic
    for i in range(len(x)):
        x1 = 1. * x
        x1[i] += dx
        H1, foo = H_func(x1, y, A, lam1, lam2, ivar)
        dHdx1 = (H1 - H0) / dx
        print(i, x[i], H0, dHdx0[i], dHdx1, (dHdx1 - dHdx0[i]) / dHdx0[i])
    return

# -------------------------------------------------------------------------------
# predicting parallaxes
# -------------------------------------------------------------------------------

Kfold = 2
name = 'N{0}_K{1}_rrl'.format(len(labels), Kfold)
if ps: name += '_ps'   

if ps:
    cut_finite = np.isfinite(labels['phot_g_mean_mag']) * \
                 np.isfinite(labels['phot_bp_mean_mag']) * \
                 np.isfinite(labels['phot_rp_mean_mag']) * \
                 np.isfinite(labels['P']) * \
                 np.isfinite(labels['Ag']) * \
                 np.isfinite(labels['Ar']) * \
                 np.isfinite(labels['Ai']) * \
                 np.isfinite(labels['Az']) * \
                 np.isfinite(labels['gmag']) * \
                 np.isfinite(labels['rmag']) * \
                 np.isfinite(labels['imag']) * \
                 np.isfinite(labels['zmag']) 
else:             
    cut_finite = np.isfinite(labels['int_average_g']) * \
                 np.isfinite(labels['int_average_bp']) * \
                 np.isfinite(labels['int_average_rp']) * \
                 np.isfinite(labels['peak_to_peak_g']) * \
                 np.isfinite(labels['peak_to_peak_bp']) * \
                 np.isfinite(labels['peak_to_peak_rp']) * \
                 np.isfinite(labels['Jmag']) * \
                 np.isfinite(labels['Hmag']) * \
                 np.isfinite(labels['Kmag']) * \
                 np.isfinite(labels['W1mag']) * \
                 np.isfinite(labels['W2mag']) * \
                 np.isfinite(labels['pf']) * \
                 np.isfinite(labels['r21_g']) * \
                 np.isfinite(labels['r31_g']) 
if ps: 
    cut_ps = labels['RRab'] > .9
    cut_clean = True
else:
    cut_clean = labels['num_clean_epochs_g'] > 40.
    cut_ps = True
cut_mag = True #labels['W1mag'] < 100
cuts = cut_finite * cut_mag * cut_ps * cut_clean
labels = labels[cuts]
    
# data
y_all = labels['parallax']
yerr_all = labels['parallax_error']
ivar_all = yerr_all ** (-2)

# design matrix
AT_0 = np.vstack([np.ones_like(y_all)])
if ps:
    AT_linear = np.vstack([labels['gmag'], labels['rmag'], labels['imag'], labels['zmag'], \
#                       labels['Jmag'], labels['Hmag'], labels['Kmag'], \
                       labels['phot_g_mean_mag'], labels['phot_bp_mean_mag'], labels['phot_rp_mean_mag'], labels['P'], \
                       labels['Ag'], labels['Ar'], labels['Ai'], labels['Az']])
else:
    AT_linear = np.vstack([labels['int_average_g'], labels['int_average_bp'], labels['int_average_rp'], \
                       labels['peak_to_peak_g'], labels['peak_to_peak_bp'], labels['peak_to_peak_rp'], \
                       #labels['Jmag'], labels['Hmag'], labels['Kmag'], \
                       labels['W1mag'], labels['W2mag'], labels['pf'], \
                       labels['r21_g'], labels['r31_g']])
A_all = np.vstack([AT_0, AT_linear]).T
   
# split into training and validation set
y_pred_all = np.zeros_like(y_all)
y_pred_all_err = np.zeros_like(y_all)

# take care of inliers
labels.add_column(np.ones_like(y_all), name='inlier')
steps = 3
    
for k in range(Kfold):    
    
    valid = labels['random_index'] % Kfold == k
    train = np.logical_not(valid)
    print('k = {0}: # of stars for prediction: {1}'.format(k, sum(valid)))
    print('k = {0}: # of remaining of stars: {1}'.format(k, sum(train)))
        
    # -------------------------------------------------------------------------------
    # additional quality cuts for training set
    # -------------------------------------------------------------------------------
    
    print('more quality cuts for training sample...')
    
    # finite parallax required for training
    cut_parallax = np.isfinite(labels[train]['parallax'])
    
    # visibility periods used
    cut_vis = labels[train]['visibility_periods_used'] >= 8.
    
    # cut in parallax_error
    cut_par = labels[train]['parallax_error'] < 0.1       # this cut is not strictly required!
    cut_burnin = labels[train]['parallax_over_error'] > 15.
    cut_inlier = labels[train]['inlier'] > 0
    
    # cut in astrometric_gof_al (should be around unity...?) *Daniel Michalik's advice*
    #cut_gof = labels[train]['astrometric_gof_al'] < 5  
    # Coryn's advice!
    cut_cal = (labels[train]['astrometric_chi2_al'] / np.sqrt(labels[train]['astrometric_n_good_obs_al']-5)) <= 35         
    
    foo, M = A_all.shape
    x0 = np.zeros((M,)) + 0.001/M 
    x_new = None
    y = None
    for opt_step in range(steps):   
        if opt_step == 0:
            cut_all = cut_parallax * cut_vis * cut_par * cut_cal * cut_inlier * cut_burnin
            lam2 = 10.
        elif opt_step == 1:
            cut_all = cut_parallax * cut_vis * cut_par * cut_cal * cut_inlier
            x0 = x_new
            lam2 = 3.
        else:
            cut_all = cut_parallax * cut_vis * cut_par * cut_cal * cut_inlier
            dy = y - np.exp(np.dot(A_all[train, :][cut_all, :], x_new))
            cut_outlier = np.zeros_like(y).astype(bool) #abs(dy) > (10000. * y)
            print('# outliers = {}'.format(np.sum(cut_outlier)))
            labels[train][cut_all][cut_outlier]['inlier'] = 0
            cut_all[cut_all] *= np.logical_not(cut_outlier)
            x0 = x_new
            lam2 = 0.
            # adding quadratic terms: add lam1 for quadratic terms
    
        y = y_all[train][cut_all]
        ivar = ivar_all[train][cut_all]
        A = A_all[train, :][cut_all, :]

        print('k = {0}: # of stars in training set: {1}'.format(k, len(y)))    
                     
        # optimize H_func
        print('{} optimization...'.format(k+1))
        res = op.minimize(H_func, x0, args=(y, A, 0.0, lam2, ivar), method='L-BFGS-B', jac=True, options={'maxfun':50000}) 
        print(res)   
        x_new = res.x  
        assert res.success
                           
    # prediction
    y_pred = np.exp(np.dot(A_all[valid, :], x_new))
    y_pred_all[valid] = y_pred
    
#    plt.scatter(labels[valid]['parallax'], y_pred, alpha = .1)
#    plt.xlim(-1, 3)
#    plt.ylim(-1, 3)
                                       
    f = open('RRLyrae/opt_results_{0}_{1}.pickle'.format(k, name), 'wb')
    pickle.dump(res, f)
    f.close()   

spec_parallax = y_pred_all
labels.add_column(spec_parallax, name='spec_parallax')
spec_par = spec_parallax * u.mas
distance = spec_par.to(u.parsec, equivalencies = u.parallax())
labels.add_column(distance, name='distance')
Table.write(labels, 'RRLyrae/training_labels_new_{}.fits'.format(name), format = 'fits', overwrite = True)

# -------------------------------------------------------------------------------
# plots 
# -------------------------------------------------------------------------------
    
#print('loading new labels...')   
#labels = Table.read('data/training_labels_new_{}.fits'.format(name), format = 'fits')    


cut_vis = labels['visibility_periods_used'] >= 8    
cut_par = labels['parallax_error'] < 0.1           
cut_cal = (labels['astrometric_chi2_al'] / np.sqrt(labels['astrometric_n_good_obs_al']-5)) <= 35         
cut_inlier = labels['inlier'] > 0

# make plots for parent, valid, and best sample
valid = cut_vis * cut_par * cut_cal  * cut_inlier 
best = valid * (labels['parallax_over_error'] >= 15) #* (labels['astrometric_gof_al'] < 10)
parent = np.isfinite(labels['parallax'])
samples = [parent, valid, best]
samples_str = ['parent', 'validation', 'best']

# make this a density plot!
fig, ax = plt.subplots(1, 3, figsize = (17, 5))
for i, sam in enumerate(list(samples)):
    
    sam_i_str = samples_str[i]                        
    dy = (labels['spec_parallax'][sam] - labels['parallax'][sam]) / labels['parallax'][sam]
    s = 0.25 * (np.percentile(dy, 97.5) - np.percentile(dy, 2.5))
    print('1 sigma inferred parallax for {0} sample: {1}, {2}'.format(sam_i_str, 0.5 * (np.percentile(dy, 84) - np.percentile(dy, 16)), 0.25 * (np.percentile(dy, 97.5) - np.percentile(dy, 2.5))))

    sc = ax[i].scatter(labels['parallax'][sam], labels['spec_parallax'][sam], c = labels['visibility_periods_used'][sam], cmap = 'viridis_r', s = 10, vmin = 8, vmax = 20, label = r'$1\sigma={}$'.format(round(s, 3)), rasterized = True)
    if i == 0:
        cb = fig.colorbar(sc)
        cb.set_label(r'visibility periods used', fontsize = fsize)
    ax[i].set_title(r'{} sample'.format(sam_i_str), fontsize = fsize)
    if i == 2:
        ax[i].set_title(r'$\varpi/\sigma_{\varpi} \geq 15$', fontsize = fsize)
    ax[i].plot([-100, 100], [-100, 100], linestyle = '--', color = 'k')
    ax[i].set_ylim(-0.5, 1)
    ax[i].set_xlim(-0.5, 1)
    ax[i].legend(frameon = True, fontsize = fsize)
    if i == 0:
        ax[i].tick_params(axis=u'both', direction='in', which='both')
    else:
        ax[i].tick_params(axis=u'both', direction='in', which='both', labelleft = False)            
    ax[i].set_xlabel('Gaia parallax', fontsize = fsize)
ax[0].set_ylabel('inferred parallax', fontsize = fsize)
plt.subplots_adjust(wspace = 0.08)
plt.savefig('RRLyrae/parallax_inferred_{0}_vis.pdf'.format(name), dpi = 120)
#plt.close()

'''fig, ax = plt.subplots(1, 3, figsize = (17, 5))
for i, sam in enumerate(list(samples)):
    
    sam_i_str = samples_str[i]                        
    dy = (labels['spec_parallax'][sam] - labels['parallax'][sam]) / labels['parallax'][sam]
    s = 0.5 * (np.percentile(dy, 84) - np.percentile(dy, 16))

    sc = ax[i].scatter(labels['parallax'][sam], labels['spec_parallax'][sam], c = labels['parallax_error'][sam], cmap = 'viridis_r', s = 10, vmin = 0, vmax = 0.1, label = r'$1\sigma={}$'.format(round(s, 3)), rasterized = True)
    if i == 0:
        cb = fig.colorbar(sc)
        cb.set_label(r'$\sigma_{\varpi}$', fontsize = fsize)
    ax[i].set_title(r'{} sample'.format(sam_i_str), fontsize = fsize)
    if i == 2:
        ax[i].set_title(r'$\sigma_{\varpi} \geq 20$', fontsize = fsize)
    ax[i].plot([-100, 100], [-100, 100], linestyle = '--', color = 'k')
    ax[i].set_ylim(-0.5, 2)
    ax[i].set_xlim(-0.5, 2)
    ax[i].legend(frameon = True, fontsize = fsize)
    if i == 0:
        ax[i].tick_params(axis=u'both', direction='in', which='both')
    else:
        ax[i].tick_params(axis=u'both', direction='in', which='both', labelleft = False)            
    ax[i].set_xlabel('Gaia parallax', fontsize = fsize)
ax[0].set_ylabel('inferred parallax', fontsize = fsize)
plt.subplots_adjust(wspace = 0.08)
plt.savefig('RRLyrae/parallax_inferred_{0}_varpi.pdf'.format(name), dpi = 120)
plt.close()

fig = plt.subplots(1, 1, figsize = (8, 6))
plt.plot(res.x)
plt.tick_params(axis=u'both', direction='in', which='both')
plt.title(r'$N = {0}$'.format(len(labels)), fontsize = fsize)
plt.savefig('RRLyrae/optimization_results_0_{0}.pdf'.format(name))
f.close()

list_labels = ['W1mag', 'W2mag']
for lab in list_labels:
    fig, ax = plt.subplots(1, 3, figsize = (17, 5))
    for i, sam in enumerate(list(samples)):
        
        sam_i_str = samples_str[i]                        
        dy = (labels['spec_parallax'][sam] - labels['parallax'][sam]) / labels['parallax'][sam]
        s = 0.5 * (np.percentile(dy, 84) - np.percentile(dy, 16))
    
        sc = ax[i].scatter(labels['parallax'][sam], labels['spec_parallax'][sam], c = labels[lab][sam], cmap = 'viridis_r', s = 10, vmin = np.percentile(labels[lab][sam], 2.5), vmax = np.percentile(labels[lab][sam], 97.5), label = r'$1\sigma={}$'.format(round(s, 3)), rasterized = True)
        if i == 0:
            cb = fig.colorbar(sc)
            cb.set_label(r'{}'.format(lab), fontsize = fsize)
        ax[i].set_title(r'{} sample'.format(sam_i_str), fontsize = fsize)
        if i == 2:
            ax[i].set_title(r'$\varpi/\sigma_{\varpi} \geq 20$', fontsize = fsize)
        ax[i].plot([-100, 100], [-100, 100], linestyle = '--', color = 'k')
        ax[i].set_ylim(-0.5, 2)
        ax[i].set_xlim(-0.5, 2)
        ax[i].legend(frameon = True, fontsize = fsize)
        if i == 0:
            ax[i].tick_params(axis=u'both', direction='in', which='both')
        else:
            ax[i].tick_params(axis=u'both', direction='in', which='both', labelleft = False)            
        ax[i].set_xlabel('Gaia parallax', fontsize = fsize)
    ax[0].set_ylabel('inferred parallax', fontsize = fsize)
    plt.subplots_adjust(wspace = 0.08)
    plt.savefig('RRLyrae/parallax_inferred_{0}_{1}.pdf'.format(name, lab), dpi = 120)
    plt.close()

                
# -------------------------------------------------------------------------------'''
                      
                       










