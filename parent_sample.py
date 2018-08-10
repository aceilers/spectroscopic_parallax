#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 15:15:21 2018

@author: eilers
"""

import numpy as np
import pickle
from astropy.table import Column, Table, join, vstack, hstack
from astropy.io import fits

from normalize_all_spectra import LoadAndNormalizeData, NormalizeData

# -------------------------------------------------------------------------------
# load catalogs
# -------------------------------------------------------------------------------

print("opening Gaia DR2 and APOGEE cross match catalogue. ")
hdu = fits.open('data/Gaia_apogee_dr14.fits', ignore_missing_end=True)
xmatch = hdu[1].data
print("Gaia DR2 and APOGEE cross match catalogue: {} entries. ".format(len(xmatch))) 

apogee_table = fits.open('data/allStar-l31c.2.fits')
apogee_data = apogee_table[1].data
                          
# -------------------------------------------------------------------------------
# match the two catalogs
# -------------------------------------------------------------------------------

apogee_ids = '2M' + xmatch['apogee_id']

# apogee_all_oid = is index in APOGEE file
print('matching the catalogs...')
apogee_data_match = apogee_data[xmatch['apogee_all_oid']]

training_labels = hstack([Table(apogee_data_match), Table(xmatch)])

# -------------------------------------------------------------------------------
# cut in logg... take only RGB stars!
# -------------------------------------------------------------------------------

parent_logg_cut = 2.2
cut = np.logical_and(training_labels['LOGG'] <= parent_logg_cut, training_labels['LOGG'] > 0.)
training_labels = training_labels[cut]              
print('logg <= {0} cut: {1}'.format(parent_logg_cut, len(training_labels)))

# -------------------------------------------------------------------------------
# cut in Q_K (where training_labels['K'] was missing! (i.e. -9999))
# ------------------------------------------------------------------------------- 

cut = np.where(training_labels['K'] > 0)
training_labels = training_labels[cut]              
print('remove missing K: {}'.format(len(training_labels)))

# -------------------------------------------------------------------------------
# remove missing data
# ------------------------------------------------------------------------------- 

cut = np.where(np.isfinite(training_labels['bp_rp']))
training_labels = training_labels[cut]              
print('remove bad bp-rp: {}'.format(len(training_labels)))

# -------------------------------------------------------------------------------
# add WISE catalog
# -------------------------------------------------------------------------------

print('match to WISE...')
hdu = fits.open('data/gaia_wise.fits')
wise_data = hdu[1].data
training_labels = hstack([training_labels, Table(wise_data)])

# -------------------------------------------------------------------------------
# check for existing WISE colors
# -------------------------------------------------------------------------------

cut = np.isfinite(training_labels['w2mpro'])
training_labels = training_labels[cut] 
cut = np.isfinite(training_labels['w1mpro'])
training_labels = training_labels[cut] 
print('remove missing W1 and W2: {}'.format(len(training_labels))) 

# -------------------------------------------------------------------------------
# take only unique entries!
# -------------------------------------------------------------------------------

# might be better to make a quality cut here!
uni, uni_idx = np.unique(training_labels['apogee_id'], return_index = True)
training_labels = training_labels[uni_idx]
print('remove duplicats: {}'.format(len(training_labels)))

# -------------------------------------------------------------------------------
# remove variable stars (at least those marked by Gaia)
# ------------------------------------------------------------------------------- 

cut = training_labels['phot_variable_flag'] != 'VARIABLE'
training_labels = training_labels[cut]              
print('remove variable stars: {}'.format(len(training_labels)))
    
# -------------------------------------------------------------------------------
# get spectra
# -------------------------------------------------------------------------------

print('load and normalize spectra...')
file_name = 'all_flux_sig_norm_parent.fits'
data_norm, continuum, not_found = LoadAndNormalizeData(training_labels['FILE'], file_name, training_labels['LOCATION_ID'])
            
 # remove entries from training labels, where no spectrum was found (for whatever reason...)!
f = open('data/no_data_parent.pickle', 'rb') 
no_dat = pickle.load(f) 
f.close()
training_labels = training_labels[no_dat]  
print('remove stars with missing spectra: {}'.format(len(training_labels)))

# -------------------------------------------------------------------------------
# save training labels
# -------------------------------------------------------------------------------

print('save labels...')
fits.writeto('data/training_labels_parent.fits', np.array(training_labels), clobber = True)
                   
# -------------------------------------------------------------------------------'''


