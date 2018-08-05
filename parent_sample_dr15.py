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
import os.path
import subprocess

from normalize_all_spectra import LoadAndNormalizeData, NormalizeData

# -------------------------------------------------------------------------------
# load catalogs
# -------------------------------------------------------------------------------

print("opening Gaia DR2 and APOGEE cross match catalogue. ")

tbl1 = Table.read('data/gaiadr2_allStar-t9-l31c-58158.fits', 1)
tbl1.rename_column('apstar_id', 'APSTAR_ID')
tbl2 = Table.read('data/allStar-t9-l31c-58158.fits', 1)
training_lab = join(tbl1, tbl2, keys='APSTAR_ID')

print("Gaia DR2 and APOGEE cross match catalogue: {} entries. ".format(len(training_lab))) 

# -------------------------------------------------------------------------------
# cut in logg... take only RGB stars!
# -------------------------------------------------------------------------------

parent_logg_cut = 2.2
cut = np.logical_and(training_lab['LOGG'] <= parent_logg_cut, training_lab['LOGG'] > 0.)
training_lab = training_lab[cut]              
print('logg <= {0} cut: {1}'.format(parent_logg_cut, len(training_lab)))

# -------------------------------------------------------------------------------
# cut in Q_K (where training_labels['K'] was missing! (i.e. -9999))
# ------------------------------------------------------------------------------- 

cut = np.where(training_lab['K'] > 0)
training_lab = training_lab[cut]              
print('remove missing K: {}'.format(len(training_lab)))

# -------------------------------------------------------------------------------
# remove missing data
# ------------------------------------------------------------------------------- 

cut = np.where(np.isfinite(training_lab['bp_rp']))
training_lab = training_lab[cut]              
print('remove bad bp-rp: {}'.format(len(training_lab)))

# -------------------------------------------------------------------------------
# add WISE catalog -- NEEDS TO BE RE-DONE!!
# -------------------------------------------------------------------------------

print('match to WISE...')
hdu = fits.open('data/gaia_apogeedr15_wise.fits')
wise_data = hdu[1].data
training_labels = join(training_lab, wise_data, keys='source_id')

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
uni, uni_idx = np.unique(training_labels['REDUCTION_ID'], return_index = True)
training_labels = training_labels[uni_idx]
print('remove duplicats: {}'.format(len(training_labels)))

# -------------------------------------------------------------------------------
# remove variable stars (at least those marked by Gaia)
# ------------------------------------------------------------------------------- 

cut = training_labels['phot_variable_flag'] != 'VARIABLE'
training_labels = training_labels[cut]              
print('remove variable stars: {}'.format(len(training_labels)))

#Table.write(training_labels['source_id', 'RA', 'DEC'], 'data/gaiadr2_apogeedr15_IDs.txt', format = 'ascii', overwrite = True)
    
# -------------------------------------------------------------------------------
# download spectra
# -------------------------------------------------------------------------------

'''delete0 = 'find ./data/spectra/ -size 0c -delete'
subprocess.call(delete0, shell = True)

for i, (fn, tel, field) in enumerate(zip(training_labels['FILE'], training_labels['TELESCOPE'], training_labels['FIELD'])):
    
    fn2 = fn.replace('apStar', 'aspcapStar')
    destination = './data/spectra/' + fn.strip() 
    destination2 = './data/spectra/' + fn2.strip()    
    #print(tel, destination, os.path.isfile(destination))
    
    if not (os.path.isfile(destination) or os.path.isfile(destination2)):
        urlbase = 'https://data.sdss.org/sas/apogeework/apogee/spectro/redux/t9/stars/' \
        + str(tel).strip() + '/' + str(field).strip() + '/'
        url = urlbase + fn.strip()
        url2 = urlbase + fn2.strip()
        try:
            cmd = 'wget --user=sdss --password=2.5-meters ' + url + ' -O ' + destination
            print(cmd)
            subprocess.call(cmd, shell = True)
            subprocess.call(delete0, shell = True)
        except:
            try:
                cmd = 'wget --user=sdss --password=2.5-meters ' + url2 + ' -O ' + destination2
                print(cmd)
                subprocess.call(cmd, shell = True)
                subprocess.call(delete0, shell = True)                
            except:
                print(fn + " not found!")

# remove missing files 
found = np.ones_like(np.arange(len(training_labels)), dtype=bool)
destination = './data/spectra/'
for i in range(len(training_labels['FILE'])):
    entry = destination + (training_labels['FILE'][i]).strip()
    entry2 = entry.replace('apStar', 'aspcapStar').strip()
    #print(entry, entry2)
    try:
        hdulist = fits.open(entry)
    except:
        try:
            hdulist = fits.open(entry2)
        except:
            print(entry + " not found or corrupted; deleting!")
            cmd = 'rm -vf ' + entry 
            subprocess.call(cmd, shell = True)
            print(i, training_labels['FILE'][i], training_labels['FIELD'][i], training_labels['TELESCOPE'][i])
            found[i] = False   

training_labels = training_labels[found]
print('spectra found for: {}'.format(len(training_labels)))

# -------------------------------------------------------------------------------
# save training labels
# -------------------------------------------------------------------------------

print('save labels...')
#fits.writeto('data/training_labels_parent_apogeedr15.fits', np.array(training_labels), overwrite = True)
Table.write(training_labels, 'data/training_labels_parent_apogeedr15.fits', format = 'fits')

# -------------------------------------------------------------------------------'''
# normalize spectra
# -------------------------------------------------------------------------------

print('load and normalize spectra...')
file_name = 'all_flux_norm_parent_apogeedr15.fits'
data_norm, continuum, not_found = LoadAndNormalizeData(training_labels['FILE'][:10], file_name, training_labels['FIELD'][:10])
print('not found: {}'.format(np.sum(not_found)))       

## remove entries from training labels, where no spectrum was found (for whatever reason...)!
#f = open('data/no_data_parent.pickle', 'rb') 
#no_dat = pickle.load(f) 
#f.close()
#training_labels = training_labels[no_dat]  
#print('remove stars with missing spectra: {}'.format(len(training_labels)))
                
# -------------------------------------------------------------------------------'''



