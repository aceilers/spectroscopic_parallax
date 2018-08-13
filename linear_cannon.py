#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 12:59:13 2018

@author: eilers
"""

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

print('loading labels...')
hdu = fits.open('data/training_labels_parent.fits')
labels = Table(hdu[1].data)


hdulist = fits.open('./data/spectra/apStar-t9-2M00000002+7417074.fits')
header = hdulist[1].header
flux = hdulist[1].data[0]
start_wl = header['CRVAL1']
diff_wl = header['CDELT1']
val = diff_wl * (len(flux)) + start_wl
wl_full_log = np.arange(start_wl, val, diff_wl)
wl = np.array([10**aval for aval in wl_full_log])

# -------------------------------------------------------------------------------
# color_cuts
# -------------------------------------------------------------------------------

cut_jk = (labels['J'] - labels['K']) < (0.4 + 0.45 * labels['bp_rp'])
cut_hw2 = (labels['H'] - labels['w2mpro']) > -0.05
cuts = (abs(labels['TEFF']) < 8000) * (abs(labels['LOGG']) < 10) * (abs(labels['FE_H']) < 10)
more_cuts = (abs(labels['K']) < 100) * (labels['parallax_over_error'] > 15)
labels = labels[cut_jk * cut_hw2 * cuts * more_cuts]
    
print('loading spectra...')

hdu = fits.open('data/all_flux_sig_norm_parent.fits')
fluxes = hdu[0].data[:, :, 0]
sigmas = hdu[0].data[:, :, 1]
fluxes = fluxes[:, cut_jk * cut_hw2 * cuts * more_cuts]
sigmas = sigmas[:, cut_jk * cut_hw2 * cuts * more_cuts]
ivars = 1./(sigmas * sigmas)
                     
# -------------------------------------------------------------------------------
# add pixel mask to remove gaps between chips! 
# -------------------------------------------------------------------------------

teff = (labels['TEFF'] - np.mean(labels['TEFF'])) / np.sqrt(np.var(labels['TEFF']))
logg = (labels['LOGG'] - np.mean(labels['LOGG'])) / np.sqrt(np.var(labels['LOGG']))
feh = (labels['FE_H'] - np.mean(labels['FE_H'])) / np.sqrt(np.var(labels['FE_H']))
absM = labels['K'] + 5. * np.log10(labels['parallax'] / 100)
absM_scaled = (absM - np.mean(absM)) / np.sqrt(np.var(absM))

# design matrix
AT_0 = np.vstack([np.ones_like(teff)])    
AT_linear = np.vstack([absM_scaled])
A = np.vstack([AT_0, AT_linear]).T    
    
nwavelengths, nstars = fluxes.shape
foo, npars = A.shape
assert foo == nstars
parameters = np.zeros((nwavelengths, npars))

for l in range(nwavelengths):
    if l % 100 == 0:
        print(l)
    ATA = np.dot(A.T, ivars[l, :][:, None] * A)
    ATy = np.dot(A.T, ivars[l, :] * fluxes[l, :])
    parameters[l, :] = np.linalg.solve(ATA, ATy)
    
    
fits.writeto('data/linear_cannon.fits', np.array(parameters), overwrite = True)
   

hdu = fits.open('data/linear_cannon.fits')
parameters = hdu[0].data
                 
fig, ax = plt.subplots(4, 1, figsize = (10, 10), sharex = True)
for i in range(4):
    ax[i].plot(wl, parameters[:, i], drawstyle = 'steps-mid', lw = .8, color = 'k')
    ax[i].tick_params(axis=u'both', direction='in', which='both', right = 'on', top = 'on')
ax[-1].set_xlim(min(wl), max(wl))
ax[-1].set_xlabel(r'$\lambda~\rm[{\AA}]$', fontsize = 14)
plt.savefig('plots/linear_cannon.pdf')
      
# -------------------------------------------------------------------------------'''
    
    