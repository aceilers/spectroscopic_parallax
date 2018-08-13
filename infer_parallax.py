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

# -------------------------------------------------------------------------------
# plotting settings
# -------------------------------------------------------------------------------

matplotlib.rcParams['ytick.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 14
fsize = 14

# -------------------------------------------------------------------------------
# load spectra and labels
# -------------------------------------------------------------------------------

# make plots?
prediction = True

print('loading labels...')
hdu = fits.open('data/training_labels_parent.fits')
labels = Table(hdu[1].data)

offset = 0.0483 # mas our best fit (10.08.2018)! # NO LONGER: as per Lindegren et al. 2018
labels['parallax'] += offset


# NOTES:
# lam = 30, 50, 70 (take best one here and then vary offset)
# offset = 0.009, 0.029, 0.049, 0.069

# -------------------------------------------------------------------------------
# color_cuts
# -------------------------------------------------------------------------------

cut_jk = (labels['J'] - labels['K']) < (0.4 + 0.45 * labels['bp_rp'])
cut_hw2 = (labels['H'] - labels['w2mpro']) > -0.05
cut_finite = (labels['J'] > -100) * (labels['H'] > -100) * (labels['K'] > -100) *\
             (labels['J_ERR'] > 0) * (labels['H_ERR'] > 0) * (labels['K_ERR'] > 0) * \
             np.isfinite(labels['w1mpro']) * np.isfinite(labels['w2mpro']) * \
             (labels['w1mpro_error'] > 0) * (labels['w2mpro_error'] > 0)
labels = labels[cut_jk * cut_hw2 * cut_finite]

if prediction: 
    
    print('loading spectra...')

    hdu = fits.open('data/all_flux_sig_norm_parent.fits')
    fluxes = hdu[0].data[:, :, 0]
    sigmas = hdu[0].data[:, :, 1]
    fluxes = fluxes[:, cut_jk * cut_hw2 * cut_finite]
    sigmas = sigmas[:, cut_jk * cut_hw2 * cut_finite]
                          
# -------------------------------------------------------------------------------
# add pixel mask to remove gaps between chips! 
# -------------------------------------------------------------------------------

    print('removing chip gaps...')               
    gaps = (np.sum(fluxes.T, axis = 0)) == float(fluxes.T.shape[0])
    fluxes = fluxes[~gaps, :]
    sigmas = sigmas[~gaps, :]

# -------------------------------------------------------------------------------
# linear algebra
# -------------------------------------------------------------------------------

def H_func(x, y, A, lams, ivar):   
    y_model = np.exp(np.dot(A, x))
    dy = y - y_model
    H = 0.5 * np.dot(dy.T, ivar * dy) + np.sum(lams * np.abs(x))
    dHdx = -1. * np.dot(A.T * y_model[None, :], ivar * dy) + lams * np.sign(x)
    return H, dHdx

def check_H_func(x, y, A, lams, ivar):
    H0, dHdx0 = H_func(x, y, A, lams, ivar)
    dx = 1e-6 # magic
    for i in range(len(x)):
        x1 = 1. * x
        x1[i] += dx
        H1, foo = H_func(x1, y, A, lams, ivar)
        dHdx1 = (H1 - H0) / dx
        print(i, x[i], dHdx0[i], dHdx1, (dHdx1 - dHdx0[i]) / dHdx0[i])
    return

# -------------------------------------------------------------------------------
# predicting parallaxes
# -------------------------------------------------------------------------------

Kfold = 2
lam = 30                      # hyperparameter -- needs to be tuned! CROSS VALIDATED (10.08.2018)
name = 'N{0}_lam{1}_K{2}_offset{3}_parallax'.format(len(labels), lam, Kfold, offset)

# optimization schedule
# 1. photometry only
# 2. go down with lambda?

steps = 2

if prediction:        
    
    # data
    y_all = labels['parallax'] 
    yerr_all = labels['parallax_error'] 
    ivar_all = yerr_all ** (-2)
    
    phot_g_mean_mag_err = 1.09 * labels['phot_g_mean_flux_error'] / labels['phot_g_mean_flux']
    phot_bp_mean_mag_err = 1.09 * labels['phot_bp_mean_flux_error'] / labels['phot_bp_mean_flux']
    phot_rp_mean_mag_err = 1.09 * labels['phot_rp_mean_flux_error'] / labels['phot_rp_mean_flux']
    
    # design matrix
    AT_0 = np.vstack([np.ones_like(y_all)])
    ln_fluxes = np.log(np.clip(fluxes, 0.01, 1.2))
    ln_flux_err = np.clip(sigmas, 0, 0.05) / np.clip(fluxes, 0.01, 1.2) # 0.05 IS MAGIC NUMBER -- ERRORS DO DEPEND ON IT!
    AT_linear = np.vstack([labels['phot_g_mean_mag'], labels['phot_bp_mean_mag'], labels['phot_rp_mean_mag'], labels['J'], labels['H'], labels['K'], labels['w1mpro'], labels['w2mpro'], ln_fluxes])
    AT_linear_err = np.vstack([phot_g_mean_mag_err, phot_bp_mean_mag_err, phot_rp_mean_mag_err, labels['J_ERR'], labels['H_ERR'], labels['K_ERR'], labels['w1mpro_error'], labels['w2mpro_error'], ln_flux_err])
    A_all = np.vstack([AT_0, AT_linear]).T
    A_all_err = np.vstack([np.zeros_like(y_all), AT_linear_err]).T
    
    # fucking brittle
    lams = np.zeros_like(A_all[0])
    lams[-len(fluxes):] = lam
       
    # split into training and validation set
    y_pred_all = np.zeros_like(y_all)
    y_pred_all_err = np.zeros_like(y_all)
        
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
        cut_vis = labels[train]['visibility_periods_used'] >= 8
        
        # cut in parallax_error
        cut_par = labels[train]['parallax_error'] < 0.1       # this cut is not strictly required!
        cut_burnin = labels[train]['parallax_over_error'] > 20.
        
        # cut in astrometric_gof_al (should be around unity...?) *Daniel Michalik's advice*
        #cut_gof = labels[train]['astrometric_gof_al'] < 5  
        # Coryn's advice!
        cut_cal = (labels[train]['astrometric_chi2_al'] / np.sqrt(labels[train]['astrometric_n_good_obs_al']-5)) <= 35         
        
        # more cuts? e.g. astrometric_gof_al should be low!
        
        foo, M = A_all.shape
        x0 = np.zeros((M,)) + 0.001/M 
        x_new = None
        for opt_step in range(steps):   
            if opt_step == 0:
                cut_all = cut_parallax * cut_vis * cut_par * cut_cal * cut_burnin
            else:
                cut_all = cut_parallax * cut_vis * cut_par * cut_cal
                x0 = x_new
        
            y = y_all[train][cut_all]
            ivar = ivar_all[train][cut_all]
            A = A_all[train, :][cut_all, :]
    
            print('k = {0}: # of stars in training set: {1}'.format(k, len(y)))    
                         
            # optimize H_func
            print('{} optimization...'.format(k+1))
            res = op.minimize(H_func, x0, args=(y, A, lams, ivar), method='L-BFGS-B', jac=True, options={'maxfun':50000}) 
            print(res)   
            x_new = res.x  
            assert res.success
                               
        # prediction
        y_pred = np.exp(np.dot(A_all[valid, :], x_new))
        y_pred_err = y_pred * np.sqrt(np.dot(A_all_err[valid, :] ** 2, x_new ** 2)) # Hogg made this up

        y_pred_all[valid] = y_pred
        y_pred_all_err[valid] = y_pred_err
        
        plt.scatter(labels[valid]['parallax'], y_pred, alpha = .1)
        plt.xlim(-1, 3)
        plt.ylim(-1, 3)
                                           
        f = open('optimization/opt_results_{0}_{1}.pickle'.format(k, name), 'wb')
        pickle.dump(res, f)
        f.close()   
    
    spec_parallax = y_pred_all
    spec_parallax_err = y_pred_all_err
    labels.add_column(spec_parallax, name='spec_parallax')
    labels.add_column(spec_parallax_err, name='spec_parallax_err')
    Table.write(labels, 'data/training_labels_new_{}.fits'.format(name), format = 'fits', overwrite = True)
    
# -------------------------------------------------------------------------------
# plots 
# -------------------------------------------------------------------------------

if not prediction:
    
    print('loading new labels...')   
    labels = Table.read('data/training_labels_new_{}.fits'.format(name), format = 'fits') 

    cut_jk = (labels['J'] - labels['K']) < (0.4 + 0.45 * labels['bp_rp'])
    cut_hw2 = (labels['H'] - labels['w2mpro']) > -0.05
    cut_finite = (labels['J'] > -100) * (labels['H'] > -100) * (labels['K'] > -100) *\
             (labels['J_ERR'] > 0) * (labels['H_ERR'] > 0) * (labels['K_ERR'] > 0) * \
             np.isfinite(labels['w1mpro']) * np.isfinite(labels['w2mpro']) * \
             (labels['w1mpro_error'] > 0) * (labels['w2mpro_error'] > 0)
    labels = labels[cut_jk * cut_hw2 * cut_finite]
    
    hdu = fits.open('data/all_flux_sig_norm_parent.fits')
    fluxes = hdu[0].data[:, :, 0]
    sigmas = hdu[0].data[:, :, 1]
    fluxes = fluxes[:, cut_jk * cut_hw2 * cut_finite]
    sigmas = sigmas[:, cut_jk * cut_hw2 * cut_finite]
    print('removing chip gaps...')               
    gaps = (np.sum(fluxes.T, axis = 0)) == float(fluxes.T.shape[0])
    fluxes = fluxes[~gaps, :]
    sigmas = sigmas[~gaps, :]
    
    cut_vis = labels['visibility_periods_used'] >= 8    
    cut_par = labels['parallax_error'] < 0.1             
    cut_cal = (labels['astrometric_chi2_al'] / np.sqrt(labels['astrometric_n_good_obs_al']-5)) <= 35         
                                      
    # make plots for parent, valid, and best sample
    valid = cut_vis * cut_par * cut_cal  
    best = valid * (labels['parallax_over_error'] >= 20) #* (labels['astrometric_gof_al'] < 10)
    parent = np.isfinite(labels['parallax'])
    samples = [parent, valid, best]
    samples_str = ['parent', 'validation', 'best']
    
    # design matrix
    AT_0 = np.vstack([np.ones_like(labels['K'])])
    ln_fluxes = np.log(np.clip(fluxes, 0.01, 1.2))
    ln_flux_err = np.clip(sigmas, 0, .05) / np.clip(fluxes, 0.01, 1.2)
    #ln_flux_err1 = np.clip(sigmas, 0, .1) / np.clip(fluxes, 0.01, 1.2)
    AT_linear = np.vstack([labels['phot_g_mean_mag'], labels['phot_bp_mean_mag'], labels['phot_rp_mean_mag'], labels['J'], labels['H'], labels['K'], labels['w1mpro'], labels['w2mpro'], ln_fluxes])
    AT_linear_err = np.vstack([phot_g_mean_mag_err, phot_bp_mean_mag_err, phot_rp_mean_mag_err, labels['J_ERR'], labels['H_ERR'], labels['K_ERR'], labels['w1mpro_error'], labels['w2mpro_error'], ln_flux_err])
    #AT_linear_err1 = np.vstack([phot_g_mean_mag_err, phot_bp_mean_mag_err, phot_rp_mean_mag_err, labels['J_ERR'], labels['H_ERR'], labels['K_ERR'], labels['w1mpro_error'], labels['w2mpro_error'], ln_flux_err1])
    A_all = np.vstack([AT_0, AT_linear]).T
    A_all_err = np.vstack([np.zeros_like(labels['K']), AT_linear_err]).T
    #A_all_err1 = np.vstack([np.zeros_like(labels['K']), AT_linear_err1]).T
    
    # FLAG IF ERROR WITH SIGMA-CLIP = 0.1 and SIGMA_CLIP = 0.05 differs by X, raise flag
    A_all_err1 = 1. * A_all_err
    A_all_err1[:, :9] = 0
    A_all_err2 = 1. * A_all_err
    A_all_err2[:, 9:] = 0
    
    # cut down to valid!
    #valid = valid * (np.random.uniform(size = len(labels['K'])) < 0.05)
    #A_all = A_all[valid, :]
    #A_all_err = A_all_err[valid, :]
    #labels = labels[valid]
    #x_new = res.x
    #A_all_err = np.clip(A_all_err, 0, .05)
    
    y_pred = np.exp(np.dot(A_all, x_new))
    y_pred_err = y_pred * np.sqrt(np.dot(A_all_err ** 2, x_new ** 2)) # Hogg made this up
    y_pred_err1 = y_pred * np.sqrt(np.dot(A_all_err1 ** 2, x_new ** 2)) # Hogg made this up
    y_pred_err2 = y_pred * np.sqrt(np.dot(A_all_err2 ** 2, x_new ** 2)) # Hogg made this up

    # intrinsic scatter plot
    delta_par = np.array(labels['spec_parallax']) - np.array(labels['parallax'])    
    par_err = np.sqrt(np.array(y_pred_err) ** 2 + np.array(labels['parallax_error']) ** 2)
    par_err1 = np.sqrt(np.array(y_pred_err1) ** 2 + np.array(labels['parallax_error']) ** 2)
    chi = delta_par/par_err
    chi2 = np.median(delta_par ** 2 / par_err ** 2) * len(delta_par)
    chi21 = np.median(delta_par ** 2 / par_err1 ** 2) * len(delta_par)
    
    cmap = 'RdBu'
    fig, ax = plt.subplots(1, 1, figsize = (10, 5))
    sc = plt.scatter(labels['spec_parallax'], delta_par, c = (y_pred_err1 - y_pred_err2) /(y_pred_err1 + y_pred_err2) , vmin = -1, vmax = 1, cmap = cmap)
    plt.axhline(0, linestyle = ':', color = '#929591')
    plt.colorbar(sc)
    
    cmap = 'plasma_r'
    fig, ax = plt.subplots(1, 1, figsize = (10, 5))
    sc = plt.scatter(labels['spec_parallax'], delta_par, c = A_all_err[:, 1], vmin = 0, vmax = 0.05, cmap = cmap)
    plt.axhline(0, linestyle = ':', color = '#929591')
    plt.colorbar(sc)
    #plt.ylim(-.1, .1)
    plt.title(r'$N_{{\rm stars}} = {0}, \, \chi^2 = {1}$'.format(len(delta_par), chi2), fontsize = 14)

# -----------------------------------------------    
    # intrinsic scatter plot
    delta_par = np.array(labels[valid]['spec_parallax']) - np.array(labels[valid]['parallax'])    
    par_err = np.sqrt(np.array(labels['spec_parallax_err'][valid]) ** 2 + np.array(labels['parallax_error'][valid]) ** 2)
    par_err[~np.isfinite(par_err)] = np.inf
    labels['spec_parallax_err'][~np.isfinite(labels['spec_parallax_err'])] = 1e12
    chi = delta_par/par_err
    chi2 = np.sum(delta_par ** 2 / par_err ** 2)
    fig, ax = plt.subplots(1, 1, figsize = (10, 5))
    plt.scatter(labels[valid]['spec_parallax'], delta_par, c = chi)
    plt.axhline(0, linestyle = ':', color = '#929591')
    #plt.ylim(-.1, .1)
    plt.title(r'$N_{{\rm stars}} = {0}, \, \chi^2 = {1}$'.format(len(delta_par), chi2), fontsize = 14)
    plt.savefig('plots/parallax/intrinsic_scatter_{}.pdf'.format(name))

    # make this a density plot!
    fig, ax = plt.subplots(1, 3, figsize = (17, 5))
    for i, sam in enumerate(list(samples)):
        
        sam_i_str = samples_str[i]                        
        dy = (labels['spec_parallax'][sam] - labels['parallax'][sam]) / labels['parallax'][sam]
        s = 0.5 * (np.percentile(dy, 84) - np.percentile(dy, 16))
        print('1 sigma inferred parallax for {0} sample: {1}, {2}'.format(sam_i_str, 0.5 * (np.percentile(dy, 84) - np.percentile(dy, 16)), 0.25 * (np.percentile(dy, 97.5) - np.percentile(dy, 2.5))))
    
        sc = ax[i].scatter(labels['parallax'][sam], labels['spec_parallax'][sam], c = labels['visibility_periods_used'][sam], cmap = 'viridis_r', s = 10, vmin = 8, vmax = 20, label = r'$1\sigma={}$'.format(round(s, 3)), rasterized = True)
        if i == 0:
            cb = fig.colorbar(sc)
            cb.set_label(r'visibility periods used', fontsize = fsize)
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
    plt.savefig('plots/parallax/parallax_inferred_{0}_vis.pdf'.format(name), dpi = 120)
    plt.close()
    
    fig, ax = plt.subplots(1, 3, figsize = (17, 5))
    gs = gridspec.GridSpec(1, 3)
    gs.update(wspace=0.08)
    for i, sam in enumerate(list(samples)):        
        ax = plt.subplot(gs[i])
        sam_i_str = samples_str[i]                        
        dy = (labels['spec_parallax'][sam] - labels['parallax'][sam]) / labels['parallax'][sam]
        s = 0.5 * (np.percentile(dy, 84) - np.percentile(dy, 16))
    
        stats, x_edge, y_edge, bins = binned_statistic_2d(labels['spec_parallax'][sam], labels['parallax'][sam], values = labels['visibility_periods_used'][sam], bins = 50, range = [[-0.5, 2], [-0.5, 2]])
        sc = ax.imshow(stats, cmap = 'viridis_r', vmin = 8, vmax = 20, origin = 'lower', extent = (-0.5, 2, -0.5, 2))
        ax.set_title(r'{} sample'.format(sam_i_str), fontsize = fsize)
        if i == 2:
            ax.set_title(r'$\varpi/\sigma_{\varpi} \geq 20$', fontsize = fsize)
        ax.plot([-100, 100], [-100, 100], linestyle = '--', color = 'k')
        ax.set_ylim(-0.5, 2)
        ax.set_xlim(-0.5, 2)
        if i == 0:
            ax.tick_params(axis=u'both', direction='in', which='both')
            ax.set_ylabel('inferred parallax', fontsize = fsize)
        else:
            ax.tick_params(axis=u'both', direction='in', which='both', labelleft = False)            
        ax.set_xlabel('Gaia parallax', fontsize = fsize)
    cb = fig.colorbar(sc)
    cb.set_label(r'visibility periods used', fontsize = fsize)
    plt.savefig('plots/parallax/parallax_inferred_{0}_vis_density.pdf'.format(name))
    plt.close()
    
    fig, ax = plt.subplots(1, 3, figsize = (17, 5))
    gs = gridspec.GridSpec(1, 3)
    gs.update(wspace=0.08)
    for i, sam in enumerate(list(samples)):        
        ax = plt.subplot(gs[i])
        sam_i_str = samples_str[i]                        
        dy = (labels['spec_parallax'][sam] - labels['parallax'][sam]) / labels['parallax'][sam]
        s = 0.5 * (np.percentile(dy, 84) - np.percentile(dy, 16))
    
        stats, x_edge, y_edge, bins = binned_statistic_2d(labels['spec_parallax'][sam], labels['parallax'][sam], values = labels['parallax_error'][sam], bins = 50, range = [[-0.5, 2], [-0.5, 2]])
        sc = ax.imshow(stats, cmap = 'viridis_r', vmin = 0, vmax = 0.1, origin = 'lower', extent = (-0.5, 2, -0.5, 2))
        ax.set_title(r'{} sample'.format(sam_i_str), fontsize = fsize)
        if i == 2:
            ax.set_title(r'$\varpi/\sigma_{\varpi} \geq 20$', fontsize = fsize)
        ax.plot([-100, 100], [-100, 100], linestyle = '--', color = 'k')
        ax.set_ylim(-0.5, 2)
        ax.set_xlim(-0.5, 2)
        if i == 0:
            ax.tick_params(axis=u'both', direction='in', which='both')
            ax.set_ylabel('inferred parallax', fontsize = fsize)
        else:
            ax.tick_params(axis=u'both', direction='in', which='both', labelleft = False)            
        ax.set_xlabel('Gaia parallax', fontsize = fsize)
    cb = fig.colorbar(sc)
    cb.set_label(r'$\sigma_{\varpi}$', fontsize = fsize)
    plt.savefig('plots/parallax/parallax_inferred_{0}_varpi_density.pdf'.format(name))
    plt.close()
    
    fig, ax = plt.subplots(1, 3, figsize = (17, 5))
    gs = gridspec.GridSpec(1, 3)
    gs.update(wspace=0.08)
    for i, sam in enumerate(list(samples)):        
        ax = plt.subplot(gs[i])
        sam_i_str = samples_str[i]                        
        dy = (labels['spec_parallax'][sam] - labels['parallax'][sam]) / labels['parallax'][sam]
        s = 0.5 * (np.percentile(dy, 84) - np.percentile(dy, 16))
    
        stats, x_edge, y_edge, bins = binned_statistic_2d(labels['spec_parallax'][sam], labels['parallax'][sam], values = labels['parallax_over_error'][sam], bins = 50, range = [[-0.5, 2], [-0.5, 2]])
        sc = ax.imshow(stats, cmap = 'viridis_r', vmin = 0, vmax = 25, origin = 'lower', extent = (-0.5, 2, -0.5, 2))
        ax.set_title(r'{} sample'.format(sam_i_str), fontsize = fsize)
        if i == 2:
            ax.set_title(r'$\varpi/\sigma_{\varpi} \geq 20$', fontsize = fsize)
        ax.plot([-100, 100], [-100, 100], linestyle = '--', color = 'k')
        ax.set_ylim(-0.5, 2)
        ax.set_xlim(-0.5, 2)
        if i == 0:
            ax.tick_params(axis=u'both', direction='in', which='both')
            ax.set_ylabel('inferred parallax', fontsize = fsize)
        else:
            ax.tick_params(axis=u'both', direction='in', which='both', labelleft = False)            
        ax.set_xlabel('Gaia parallax', fontsize = fsize)
    cb = fig.colorbar(sc)
    cb.set_label(r'$\varpi/\sigma_{\varpi}$', fontsize = fsize)
    plt.savefig('plots/parallax/parallax_inferred_{0}_varpi_sigma_density.pdf'.format(name))
    plt.close()
    
    fig, ax = plt.subplots(1, 3, figsize = (17, 5))
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
    plt.savefig('plots/parallax/parallax_inferred_{0}_varpi.pdf'.format(name), dpi = 120)
    plt.close()
    
    fig = plt.subplots(1, 1, figsize = (8, 6))
    plt.plot(res.x)
    plt.tick_params(axis=u'both', direction='in', which='both')
    plt.title(r'$N = {0},\,\lambda = {1}$'.format(len(labels), lam), fontsize = fsize)
    plt.savefig('plots/optimization_results_0_{0}.pdf'.format(name))
    f.close()
    
    list_labels = ['TEFF', 'LOGG', 'FE_H', 'VMICRO', 'VMACRO', 'M_H', 'ALPHA_M', 'MG_FE', 'SNR', 'VSINI', 'w1mpro', 'w2mpro']
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
        plt.savefig('plots/parallax/parallax_inferred_{0}_{1}.pdf'.format(name, lab), dpi = 120)
        plt.close()

                
# -------------------------------------------------------------------------------'''
                      
                       










