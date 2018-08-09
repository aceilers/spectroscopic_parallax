#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 29 13:53:11 2018

@author: eilers
"""

import numpy as np
from astropy.table import Table, hstack, join, Column
import os.path
import subprocess
from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord
import pickle
import matplotlib.pyplot as plt
            
# -------------------------------------------------------------------------------
# download APOGEE catalogue
# -------------------------------------------------------------------------------

#delete0 = 'find ./data/ -size 0c -delete'
#substr = 'l31c'
#subsubstr = '2'
#fn = 'allStar-' + substr + '.' + subsubstr + '.fits'
#url = 'https://data.sdss.org/sas/dr14/apogee/spectro/redux/r8/stars/' + substr \
#    + '/' + substr + '.' + subsubstr + '/' + fn
#destination = './data/' + fn
#cmd = 'wget ' + url + ' -O ' + destination
#if not os.path.isfile(destination):
#    subprocess.call(cmd, shell = True) # warning: security
#    subprocess.call(delete0, shell = True)
#print("opening " + destination)
#apogee_table = fits.open(destination)
#apogee_data = apogee_table[1].data                         
##apogee_data = apogee_data[apogee_data['DEC'] > -90.0]
 
# -------------------------------------------------------------------------------
# normalize spectra: functions
# -------------------------------------------------------------------------------

def LoadAndNormalizeData(file_spectra, file_name, destinations, pca = False, X_mean = None):
    
    all_flux = np.zeros((len(file_spectra), 8575))
    all_sigma = np.zeros((len(file_spectra), 8575))
    all_wave = np.zeros((len(file_spectra), 8575))
    
    i=0
    no_data = []
    no_data_i = np.ones((len(file_spectra),), dtype = bool)
    for entry, destination_i in zip(file_spectra, destinations):
        print(i)
        destination = './data/spectra/' + str(destination_i) + '/'
        try:
            hdulist = fits.open(destination + entry.strip())
            if len(hdulist[1].data) < 8575: 
                flux = hdulist[1].data[0]
                sigma = hdulist[2].data[0]
#                print('something is weird...')
            else:
                flux = hdulist[1].data
                sigma = hdulist[2].data
            header = hdulist[1].header
            start_wl = header['CRVAL1']
            diff_wl = header['CDELT1']
            val = diff_wl * (len(flux)) + start_wl
            wl_full_log = np.arange(start_wl, val, diff_wl)
            wl_full = [10**aval for aval in wl_full_log]
            all_wave[i] = wl_full        
            all_flux[i] = flux
            all_sigma[i] = sigma
        except:
            entry = entry.replace('apStar-r8', 'aspcapStar-r8-l31c.2')
            try:
                hdulist = fits.open(destination + entry.strip())
                if len(hdulist[1].data) < 8575: 
                    flux = hdulist[1].data[0]
                    sigma = hdulist[2].data[0]
#                    print('something is weird...')
                else:
                    flux = hdulist[1].data
                    sigma = hdulist[2].data
                header = hdulist[1].header
                start_wl = header['CRVAL1']
                diff_wl = header['CDELT1']
                val = diff_wl * (len(flux)) + start_wl
                wl_full_log = np.arange(start_wl, val, diff_wl)
                wl_full = [10**aval for aval in wl_full_log]
                all_wave[i] = wl_full        
                all_flux[i] = flux
                all_sigma[i] = sigma
            except:
                no_data_i[i] = False
                no_data.append(entry)
        i += 1
        
    data = np.array([all_wave[no_data_i, :], all_flux[no_data_i, :], all_sigma[no_data_i, :]])
    print(data.shape, all_flux.shape, all_sigma.shape)
    data_norm, continuum = NormalizeData(data.T)
    
#    if pca != False:
#        data_coeff = pca.transform(data_norm[:, :, 1].T - X_mean)
#        f = open('data/data_coeff_color_cut.pickle' , 'wt')
#        pickle.dump(data_coeff, f)
#        f.close()     
    
#    f = open('data/no_data_parent.pickle' , 'wt')
#    pickle.dump(no_data_i, f)
#    f.close()
    
#    f = open('data/' + file_name, 'wt')
#    pickle.dump(data_norm, f)
#    f.close()
    fits.writeto('data/' + file_name, data_norm[:, :, 1:], overwrite = True)
    
    return data_norm, continuum, no_data_i

def NormalizeData(dataall):
        
    Nlambda, Nstar, foo = dataall.shape
    
    pixlist = np.loadtxt('data/pixtest8_dr13.txt', usecols = (0,), unpack = 1).astype(int)
    LARGE  = 3.0                          # magic LARGE sigma value
   
    continuum = np.zeros((Nlambda, Nstar))
    dataall_flat = np.ones((Nlambda, Nstar, 3))
    dataall_flat[:, :, 2] = LARGE
    for jj in range(Nstar):
        bad_a = np.logical_or(np.isnan(dataall[:, jj, 1]), np.isinf(dataall[:,jj, 1]))
        bad_b = np.logical_or(dataall[:, jj, 2] <= 0., np.isnan(dataall[:, jj, 2]))
        bad = np.logical_or(np.logical_or(bad_a, bad_b), np.isinf(dataall[:, jj, 2]))
        dataall[bad, jj, 1] = 1.
        dataall[bad, jj, 2] = LARGE
        var_array = LARGE**2 + np.zeros(len(dataall)) 
        var_array[pixlist] = 0.000
        
        take1 = np.logical_and(dataall[:,jj,0] > 15150, dataall[:,jj,0] < 15800)
        take2 = np.logical_and(dataall[:,jj,0] > 15890, dataall[:,jj,0] < 16430)
        take3 = np.logical_and(dataall[:,jj,0] > 16490, dataall[:,jj,0] < 16950)
        ivar = 1. / ((dataall[:, jj, 2] ** 2) + var_array) 
        fit1 = np.polynomial.chebyshev.Chebyshev.fit(x=dataall[take1,jj,0], y=dataall[take1,jj,1], w=ivar[take1], deg=2) # 2 or 3 is good for all, 2 only a few points better in temp 
        fit2 = np.polynomial.chebyshev.Chebyshev.fit(x=dataall[take2,jj,0], y=dataall[take2,jj,1], w=ivar[take2], deg=2)
        fit3 = np.polynomial.chebyshev.Chebyshev.fit(x=dataall[take3,jj,0], y=dataall[take3,jj,1], w=ivar[take3], deg=2)
        continuum[take1, jj] = fit1(dataall[take1, jj, 0])
        continuum[take2, jj] = fit2(dataall[take2, jj, 0])
        continuum[take3, jj] = fit3(dataall[take3, jj, 0])
        dataall_flat[:, jj, 0] = 1.0 * dataall[:, jj, 0]
        dataall_flat[take1, jj, 1] = dataall[take1,jj,1]/fit1(dataall[take1, 0, 0])
        dataall_flat[take2, jj, 1] = dataall[take2,jj,1]/fit2(dataall[take2, 0, 0]) 
        dataall_flat[take3, jj, 1] = dataall[take3,jj,1]/fit3(dataall[take3, 0, 0]) 
        dataall_flat[take1, jj, 2] = dataall[take1,jj,2]/fit1(dataall[take1, 0, 0]) 
        dataall_flat[take2, jj, 2] = dataall[take2,jj,2]/fit2(dataall[take2, 0, 0]) 
        dataall_flat[take3, jj, 2] = dataall[take3,jj,2]/fit3(dataall[take3, 0, 0]) 
        
        bad = dataall_flat[:, jj, 2] > 0.3 # MAGIC
        dataall_flat[bad, jj, 1] = 1.
        dataall_flat[bad, jj, 2] = LARGE
        
    for jj in range(Nstar):
        print("continuum_normalize_tcsh working on star", jj)
        bad_a = np.logical_not(np.isfinite(dataall_flat[:, jj, 1]))
        bad_a = np.logical_or(bad_a, dataall_flat[:, jj, 2] <= 0.)
        bad_a = np.logical_or(bad_a, np.logical_not(np.isfinite(dataall_flat[:, jj, 2])))
        bad_a = np.logical_or(bad_a, dataall_flat[:, jj, 2] > 1.)                    # magic 1.
        # grow the mask
        bad = np.logical_or(bad_a, np.insert(bad_a, 0, False, 0)[0:-1])
        bad = np.logical_or(bad, np.insert(bad_a, len(bad_a), False)[1:])
        dataall_flat[bad, jj, 1] = 1.
        dataall_flat[bad, jj, 2] = LARGE
            
    return dataall_flat, continuum

# -------------------------------------------------------------------------------
# normalize spectra
# -------------------------------------------------------------------------------

## remove all location_id = 1 (those spectra are not downloaded...)
#xx = apogee_data['LOCATION_ID'] == 1
#apogee_data = apogee_data[~xx]
#
#file_name = 'all_spectra_norm_5000.pickle'
#
#destination = './data/' + file_name
##if not os.path.isfile(destination):
#data_norm, continuum = LoadAndNormalizeData(apogee_data['FILE'][:5000], file_name, apogee_data['LOCATION_ID'][:5000])
#
#f = open('data/all_spectra_norm_5000.pickle', 'w')
#pickle.dump(data_norm, f)
#f.close()

