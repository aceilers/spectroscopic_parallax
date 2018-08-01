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

offset = 0.029 # mas as per Lindegren et al. 2018
labels['parallax'] += offset

if prediction: 
    
    print('loading spectra...')

    hdu = fits.open('data/all_flux_norm_parent.fits')
    fluxes = hdu[0].data
                          
# -------------------------------------------------------------------------------
# add pixel mask to remove gaps between chips! 
# -------------------------------------------------------------------------------

    print('removing chip gaps...')               
    gaps = (np.sum(fluxes.T, axis = 0)) == float(fluxes.T.shape[0])
    fluxes = fluxes[~gaps, :]

# -------------------------------------------------------------------------------
# de-redden K band magnitudes!
# -------------------------------------------------------------------------------


# -------------------------------------------------------------------------------
# add absolute K magnitudes
# -------------------------------------------------------------------------------

    print('calculating Q_K...')
    m_K = labels['K']
    
    Q_factor = 10**(0.2 * m_K) / 100.                 # assumes parallaxes is in mas
    Q_K = labels['parallax'] * Q_factor
    Q_K_err = labels['parallax_error'] * Q_factor
    
    labels.add_column(Column(Q_K), name='Q_K')
    labels.add_column(Column(Q_K_err), name='Q_K_ERR')

# -------------------------------------------------------------------------------
# also add WISE Q (not needed at the moment)
# -------------------------------------------------------------------------------

    m_W1 = labels['w1mpro']
    m_W2 = labels['w2mpro']
    Q_W1 = 10**(0.2 * m_W1) * labels['parallax']/100.                    
    Q_W2 = 10**(0.2 * m_W2) * labels['parallax']/100.                    
    Q_W1_err = labels['parallax_error'] * 10**(0.2 * m_W1)/100.     
    Q_W2_err = labels['parallax_error'] * 10**(0.2 * m_W2)/100.     
    labels.add_column(Column(Q_W1), name='Q_W1')
    labels.add_column(Column(Q_W2), name='Q_W2')
    labels.add_column(Column(Q_W1_err), name='Q_W1_ERR')
    labels.add_column(Column(Q_W2_err), name='Q_W2_ERR')

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
    dx = 0.0001 # magic
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
lam = 10000                      # hyperparameter -- needs to be tuned!
name = 'N{0}_lam{1}_K{2}'.format(len(labels), lam, Kfold)

if prediction:        
    
    # data
    y_all = labels['Q_K'] 
    yerr_all = labels['Q_K_ERR'] 
    ivar_all = yerr_all ** (-2)
    
    JK = labels['J'] - labels['K']
    HW2 = labels['H'] - labels['w2mpro']
    JW1 = labels['J'] - labels['w1mpro']
    
    # design matrix
    AT_0 = np.vstack([np.ones_like(JK)])
    #AT_linear = np.vstack([JK, labels['bp_rp'], JW1, HW2, fluxes])
    AT_linear = np.vstack([labels['J'], labels['H'], labels['K'], labels['bp_rp'], labels['phot_g_mean_mag'], labels['w1mpro'], labels['w2mpro'], fluxes])
    A_all = np.vstack([AT_0, AT_linear]).T
    
    # fucking brittle
    lams = np.zeros_like(A_all[0])
    lams[-len(fluxes):] = lam
       
    # split into training and validation set
    y_pred_all = np.zeros_like(y_all)
        
    for k in range(Kfold):    
        
        valid = labels['random_index'] % Kfold == k
        train = np.logical_not(valid)
        print('k = {0}: # of stars for prediction: {1}'.format(k, sum(valid)))
        print('k = {0}: # of remaining of stars: {1}'.format(k, sum(train)))
            
        # -------------------------------------------------------------------------------
        # additional quality cuts for training set
        # -------------------------------------------------------------------------------
        
        print('more quality cuts for training sample...')
        
        # cuts in Q
        cut_Q = labels[train]['Q_K'] < 0.5 # necessary?
        
        # visibility periods used
        cut_vis = labels[train]['visibility_periods_used'] >= 8
        
        # cut in parallax_error
        cut_par = labels[train]['parallax_error'] < 0.1       # this cut is not strictly required!
        
        # cut in b (only necessary if infering extinction from WISE colors doesn't work...)
        bcut = 0
        cut_b = np.abs(labels[train]['b']) >= bcut
    
        # cut in astrometric_gof_al (should be around unity...?)
        cut_gof = labels[train]['astrometric_gof_al'] < 5                          
        
        # more cuts? e.g. astrometric_gof_al should be low!
                          
        cut_all = cut_Q * cut_vis * cut_par * cut_b * cut_gof      
        y = y_all[train][cut_all]
        ivar = ivar_all[train][cut_all]
        A = A_all[train, :][cut_all, :]
        N, M = A.shape
        x0 = np.zeros((M,)) # try ones
        print('k = {0}: # of stars in training set: {1}'.format(k, len(y)))    
                     
        # optimize H_func
        print('{} otimization...'.format(k+1))
        res = op.minimize(H_func, x0, args=(y, A, lams, ivar), method='L-BFGS-B', jac=True, options={'maxfun':50000}) 
        print(res)                       
                               
        # prediction
        y_pred = np.exp(np.dot(A_all[valid, :], res.x))
        y_pred_all[valid] = y_pred
                                           
        f = open('optimization/opt_results_{0}_{1}.pickle'.format(k, name), 'wb')
        pickle.dump(res, f)
        f.close()   
    
    spec_parallax = y_pred_all / Q_factor
    labels.add_column(spec_parallax, name='spec_parallax')
    labels.add_column(y_pred_all, name='Q_pred')
    Table.write(labels, 'data/training_labels_new_{}.fits'.format(name), format = 'fits', overwrite = True)
    

# -------------------------------------------------------------------------------
# plots 
# -------------------------------------------------------------------------------

if not prediction:
    
    print('loading new labels...')   
    labels = Table.read('data/training_labels_new_{}.fits'.format(name), format = 'fits')    
    
    # cuts in Q
    cut_Q = labels['Q_K'] < 0.5   
    # visibility periods used
    cut_vis = labels['visibility_periods_used'] >= 8    
    # cut in parallax_error
    cut_par = labels['parallax_error'] < 0.1           
    # cut in b 
    bcut = 0
    cut_b = np.abs(labels['b']) >= bcut  
    # cut in astrometric_gof_al
    cut_gof = labels['astrometric_gof_al'] < 5 
    # other cuts?                                      
    
    # make plots for parent, valid, and best sample
    valid = cut_Q * cut_vis * cut_par * cut_b * cut_gof  
    best = valid * (labels['parallax_over_error'] >= 20) #* (labels['astrometric_gof_al'] < 10)
    parent = np.isfinite(labels['parallax'])
    samples = [parent, valid, best]
    samples_str = ['parent', 'validation', 'best']

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
            
    fig, ax = plt.subplots(1, 3, figsize = (17, 5))
    for i, sam in enumerate(list(samples)):
        
        sam_i_str = samples_str[i]                        
        dy = (labels['Q_K'][sam] - labels['Q_pred'][sam]) / labels['Q_K'][sam]
        s = 0.5 * (np.percentile(dy, 84) - np.percentile(dy, 16))
    
        sc = ax[i].scatter(labels['Q_K'][sam], labels['Q_pred'][sam], c = labels['visibility_periods_used'][sam], cmap = 'viridis_r', s = 10, vmin = 8, vmax = 20, label = r'$1\sigma={}$'.format(round(s, 3)), rasterized = True)
        if i == 0:
            cb = fig.colorbar(sc)
            cb.set_label(r'visibility periods used', fontsize = fsize)
        ax[i].set_title(r'{} sample'.format(sam_i_str), fontsize = fsize)
        if i == 2:
            ax[i].set_title(r'$\varpi/\sigma_{\varpi} \geq 20$', fontsize = fsize)
        ax[i].plot([-100, 100], [-100, 100], linestyle = '--', color = 'k')
        ax[i].set_ylim(-0.1, 1)
        ax[i].set_xlim(-0.1, 1)
        ax[i].legend(frameon = True, fontsize = fsize)
        if i == 0:
            ax[i].tick_params(axis=u'both', direction='in', which='both')
        else:
            ax[i].tick_params(axis=u'both', direction='in', which='both', labelleft = False)            
        ax[i].set_xlabel(r'$Q_{K,\,\rm true}$', fontsize = fsize)
    ax[0].set_ylabel(r'$Q_{K,\,\rm predicted}$', fontsize = fsize)
    plt.subplots_adjust(wspace = 0.08)
    plt.savefig('plots/parallax/Q_inferred_{0}_vis.pdf'.format(name), dpi = 120)
    plt.close()
    
    f = open('optimization/opt_results_0_{}.pickle'.format(name), 'rb')
    res = pickle.load(f)
    f.close()
    
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
                      
                       










