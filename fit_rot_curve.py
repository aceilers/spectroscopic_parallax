#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 10:17:39 2018

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
import astropy
import astropy.units as u
import astropy.cosmology as cosmo

planck = cosmo.Planck13

# -------------------------------------------------------------------------------
# plotting settings
# -------------------------------------------------------------------------------

matplotlib.rcParams['ytick.labelsize'] = 18
matplotlib.rcParams['xtick.labelsize'] = 18
matplotlib.rc('text', usetex=True)
fsize = 18
figsize = (8, 4.5)

# -------------------------------------------------------------------------------
# load rotation curve
# -------------------------------------------------------------------------------

f = open('data/rot_curve.txt', 'r')
xx = np.loadtxt(f)
f.close()

bins_dr = xx[0, :]
vc_annulus = xx[1, :]

idx5 = sum(bins_dr < 5)

# -------------------------------------------------------------------------------
# profiles
# -------------------------------------------------------------------------------

## Miyamoto - Nagai model (bulge, thin, thick disk)
#def vc_MN(R, a, b, M, z = 0):  
#    vc = R*u.kpc * np.sqrt(astropy.constants.G.to(u.kpc**3/u.M_sun/u.s**2) * M*u.M_sun / ((R*u.kpc)**2 + (a*u.kpc + np.sqrt((z*u.kpc)**2 + (b*u.kpc)**2))**2)**(3./2.))
#    return vc.to(u.km/u.s)
#
#def vc_total(theta, R):    
#    a1, M1, a2, M2, b3, M3 = theta
#    vc_thin_disk = vc_MN(R, a1, 0.25, M1*1e7) # b = 0: infinitely thin disk
#    vc_thick_disk = vc_MN(R, a2, 0.8, M2*1e7)
#    vc_halo = vc_MN(R, 0, b3, M3*1e7) # a = 0: spherical symmetry (Plummer's sphere)
#    vc_tot = np.sqrt(vc_thin_disk**2 + vc_thick_disk**2 + vc_halo**2)    
#    return vc_tot.value


X_GC_sun_kpc = 8.122 # kpc

def linear_fit(theta, R, X_GC_sun_kpc):
    fit = theta[0] * (R - X_GC_sun_kpc) + theta[1]
    return fit

def chi2(theta, R_obs, vc_obs, X_GC_sun_kpc):
    fit = linear_fit(theta, R_obs, X_GC_sun_kpc)
    return np.nansum((fit - vc_obs)**2)

# linear fit    
x0 = np.array([-1, 200])  
res = op.minimize(chi2, x0, args=(bins_dr[idx5:], vc_annulus[idx5:], X_GC_sun_kpc), method='L-BFGS-B', options={'maxfun':50000}) 
print(res)
theta_fit = res.x



## NFW profile (dark matter halo)
#def vc_NFW(R, a, rho0, z = 0):  
#    
#    rho_0 = rho0 * u.M_sun/u.kpc**3
#    rho_c = (3. * planck.H0**2 / (8. * np.pi * astropy.constants.G)).to(u.M_sun/u.kpc**3)
#    M = 4*np.pi*rho0*a**3*(np.log(1+R/a)-(R/a)/(1+R/a))
#    
#    R = R*u.kpc
#    a = a*u.kpc
#    R200 = R200*u.kpc
#    M200 = M200*u.M_sun
#    X200 = R200/a
#    rho_c = M200/200./4./np.pi*3./(R200**3)
#    factor = np.log(1. + X200) - X200 / (1. + X200)
#    rho0 = 200./3. * rho_c * X200**2 / factor
#    vc = R * np.sqrt((4*np.pi*astropy.constants.G.to(u.kpc**3/u.M_sun/u.s**2)*rho0*a**2)/((R+a)*R**2) - (4*np.pi*astropy.constants.G.to(u.kpc**3/u.M_sun/u.s**2)*rho0*a**3) * (np.log(R/a) + 1) / (R**(3./2.)))
#    return vc.to(u.km/u.s)



 
# -------------------------------------------------------------------------------
# plot
# -------------------------------------------------------------------------------

R = np.linspace(0, 30, 1000)

fig, ax = plt.subplots(1, 1, figsize = (8, 6), sharex = True)        
#plt.scatter(bins_dr[:idx5], vc_annulus[:idx5], facecolors='none', edgecolors='#3778bf', zorder = 20)
plt.scatter(bins_dr[idx5:], vc_annulus[idx5:], facecolors='#3778bf', edgecolors='#3778bf', zorder = 10, alpha = .8)
plt.ylim(0, 300)
plt.xlim(0, 25)
plt.tick_params(axis=u'both', direction='in', which='both', right = 'on', top = 'on')
plt.xlabel(r'$R_{\rm GC}\,\rm [kpc]$', fontsize = fsize)
plt.ylabel(r'$v_{\rm circ}~\rm [km\,s^{-1}]$', fontsize = fsize)
plt.plot(R, linear_fit(theta_fit, R, X_GC_sun_kpc), linestyle = ':', lw = 2, zorder = 20, color = '#929591', label=r'$y(R) = {0} {1}\cdot(R-R_{{\odot}})$'.format(round(theta_fit[1], 2), round(theta_fit[0], 2)))
#plt.plot(R, vc_tot, 'k', zorder = 100)
#plt.plot(R, vc_core, label = r'core')
#plt.plot(R, vc_bulge, label = r'bulge')
#plt.plot(R, vc_disk, label = r'disk')
#plt.plot(R, vc_halo, label = r'halo')
plt.legend(frameon = True, fontsize = 14)
plt.tight_layout()
plt.savefig('paper_rotation_curve/rotation_curve_fit.pdf', bbox_inches = 'tight')

# -------------------------------------------------------------------------------
# old plots
# -------------------------------------------------------------------------------

'''# Sofue (1996): Milky Way can be fit with 4 Miyamoto-Nagai models
vc_core = vc_MN(R, a = 0, b = 0.12, M = 0.05e11)
vc_bulge = vc_MN(R, a = 0., b = 0.75, M = 0.1e11)
vc_disk = vc_MN(R, a = 6., b = 0.5, M = 1.6e11)
vc_halo = vc_MN(R, a = 15., b = 15., M = 3.0e11)
vc_tot = np.sqrt(vc_core**2 + vc_bulge**2 + vc_disk**2 + vc_halo**2)

fig, ax = plt.subplots(1, 1, figsize = (9, 6), sharex = True)        
plt.scatter(bins_dr[:idx5], vc_annulus[:idx5], facecolors='none', edgecolors='#3778bf', zorder = 20)
plt.scatter(bins_dr[idx5:], vc_annulus[idx5:], facecolors='#3778bf', edgecolors='#3778bf', zorder = 30, alpha = .8)
plt.ylim(0, 300)
plt.xlim(0, 25)
plt.tick_params(axis=u'both', direction='in', which='both', right = 'on', top = 'on')
plt.xlabel(r'$R_{\rm GC}\,\rm [kpc]$', fontsize = fsize)
plt.ylabel(r'$v_{\rm circ}~\rm [km\,s^{-1}]$', fontsize = fsize)
plt.plot(R, vc_tot, 'k', zorder = 100)
plt.plot(R, vc_core, label = r'core')
plt.plot(R, vc_bulge, label = r'bulge')
plt.plot(R, vc_disk, label = r'disk')
plt.plot(R, vc_halo, label = r'halo')
plt.legend(frameon = True, fontsize = 14)
plt.title('Sofue et al. 1996', fontsize = 14)
plt.tight_layout()
plt.savefig('plots/rotation_curve/rotation_curve_fit_Sofue1996.pdf', bbox_inches = 'tight')


# Pouliasis (2016), model II
vc_thin = vc_MN(R, a = 4.8, b = 0.25, M = 1600*2.32e7)
vc_thick = vc_MN(R, a = 2.0, b = 0.8, M = 1700*2.32e7)
vc_halo = vc_MN(R, a = 14., b = 0., M = 9000*2.32e7)
vc_tot = np.sqrt(vc_thin**2 + vc_thick**2 + vc_halo**2)

fig, ax = plt.subplots(1, 1, figsize = (9, 6), sharex = True)        
plt.scatter(bins_dr[:idx5], vc_annulus[:idx5], facecolors='none', edgecolors='#3778bf', zorder = 20)
plt.scatter(bins_dr[idx5:], vc_annulus[idx5:], facecolors='#3778bf', edgecolors='#3778bf', zorder = 30, alpha = .8)
plt.ylim(0, 300)
plt.xlim(0, 25)
plt.tick_params(axis=u'both', direction='in', which='both', right = 'on', top = 'on')
plt.xlabel(r'$R_{\rm GC}\,\rm [kpc]$', fontsize = fsize)
plt.ylabel(r'$v_{\rm circ}~\rm [km\,s^{-1}]$', fontsize = fsize)
plt.plot(R, vc_tot, 'k', zorder = 100)
plt.plot(R, vc_thick, label = r'thick disk')
plt.plot(R, vc_thin, label = r'thin disk')
plt.plot(R, vc_halo, label = r'halo')
plt.legend(frameon = True, fontsize = 14)
plt.title('Pouliasis et al. 2016, model II', fontsize = 14)
plt.tight_layout()
plt.savefig('plots/rotation_curve/rotation_curve_fit_Pouliasis2016_II.pdf', bbox_inches = 'tight')


# Pouliasis (2016), model I
vc_bulge = vc_MN(R, a = 0., b = 0.3, M = 460*2.32e7)
vc_thin = vc_MN(R, a = 5.3, b = 0.25, M = 1600*2.32e7)
vc_thick = vc_MN(R, a = 2.6, b = 0.8, M = 1700*2.32e7)
vc_halo = vc_MN(R, a = 14., b = 0., M = 9000*2.32e7)
vc_tot = np.sqrt(vc_bulge**2 + vc_thin**2 + vc_thick**2 + vc_halo**2)

fig, ax = plt.subplots(1, 1, figsize = (9, 6), sharex = True)        
plt.scatter(bins_dr[:idx5], vc_annulus[:idx5], facecolors='none', edgecolors='#3778bf', zorder = 20)
plt.scatter(bins_dr[idx5:], vc_annulus[idx5:], facecolors='#3778bf', edgecolors='#3778bf', zorder = 30, alpha = .8)
plt.ylim(0, 300)
plt.xlim(0, 25)
plt.tick_params(axis=u'both', direction='in', which='both', right = 'on', top = 'on')
plt.xlabel(r'$R_{\rm GC}\,\rm [kpc]$', fontsize = fsize)
plt.ylabel(r'$v_{\rm circ}~\rm [km\,s^{-1}]$', fontsize = fsize)
plt.plot(R, vc_tot, 'k', zorder = 100)
plt.plot(R, vc_bulge, label = r'bulge')
plt.plot(R, vc_thick, label = r'thick disk')
plt.plot(R, vc_thin, label = r'thin disk')
plt.plot(R, vc_halo, label = r'halo')
plt.legend(frameon = True, fontsize = 14)
plt.title('Pouliasis et al. 2016, model I', fontsize = 14)
plt.tight_layout()
plt.savefig('plots/rotation_curve/rotation_curve_fit_Pouliasis2016_I.pdf', bbox_inches = 'tight')

# -------------------------------------------------------------------------------'''