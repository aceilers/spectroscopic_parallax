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
import emcee
import scipy
import matplotlib.patches as patches


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
vc_annulus_err_m = xx[2, :]
vc_annulus_err_p = xx[3, :]
sigmas = 0.5 * (vc_annulus_err_m + vc_annulus_err_p)

idx5 = sum(bins_dr < 5)

f = open('data/rot_curve_part1.txt', 'r')
xx1 = np.loadtxt(f)
f.close()
idx5a = sum(xx1[0, :] < 5)


f = open('data/rot_curve_part2.txt', 'r')
xx2 = np.loadtxt(f)
f.close()
idx5b = sum(xx2[0, :] < 5)


# -------------------------------------------------------------------------------
# profiles
# -------------------------------------------------------------------------------

# Miyamoto - Nagai model (bulge, thin, thick disk)
def vc_MN(R, a, b, M, z = 0):  
    vc = R*u.kpc * np.sqrt(astropy.constants.G.to(u.kpc**3/u.M_sun/u.s**2) * M*u.M_sun / ((R*u.kpc)**2 + (a*u.kpc + np.sqrt((z*u.kpc)**2 + (b*u.kpc)**2))**2)**(3./2.))
    return vc.to(u.km/u.s)

# NFW profile(R, z):
def vc_NFW(R, z, a, rho_0):
    # not sure everything is correct of z is not 0...
    rho_0 = rho_0 * u.M_sun / u.kpc**3
    a = a * u.kpc
    R = R * u.kpc
    z = z * u.kpc
    #term_mass = np.log(1 + R/a) - (R/a)/(1+R/a)
    #rho_0 = M / term_mass / (4. * np.pi * a**3)
    r = np.sqrt(R**2 + z**2)
    b = 4. * np.pi * astropy.constants.G * rho_0
    #M = b * term_mass
    term1 = b * a**3 * np.log(1 + r/a) / r**3
    term2 = b * a**3 / (r**2 * (r+a))
    vc = R * np.sqrt(term1 - term2)
    return vc.to(u.km/u.s)


rho_c = (3. * planck.H0**2 / (8. * np.pi * astropy.constants.G)).to(u.M_sun/(u.kpc)**3)

# with virial mass and concentration! 
def vc_NFW_old(R, z, M_vir, c, rho_c):
    
    M_vir = M_vir * u.M_sun
    R_vir = np.cbrt(M_vir / 200. / rho_c / (4*np.pi/3.))  
    a = R_vir / c
    term1 = np.log(1 + c) - c / (1 + c)
    rho_0 = M_vir / (4 * np.pi) / a**3 / term1
       
    # not sure everything is correct of z is not 0...
    R = R * u.kpc
    z = z * u.kpc

    r = np.sqrt(R**2 + z**2)
    b = 4. * np.pi * astropy.constants.G * rho_0
    term1 = b * a**3 * np.log(1 + r/a) / r**3
    term2 = b * a**3 / (r**2 * (r+a))
    vc = R * np.sqrt(term1 - term2)
    
    return vc.to(u.km/u.s)

def vc_plummer(R, b, M, z = 0):
    vc = R*u.kpc * np.sqrt(astropy.constants.G.to(u.kpc**3/u.M_sun/u.s**2) * M*u.M_sun / ((R*u.kpc)**2 + (b*u.kpc)**2)**(3./2.))
    return vc.to(u.km/u.s)

vc_bulge = vc_plummer(bins_dr[idx5:], 0.3, 460*2.32e7)
vc_thin_disk = vc_MN(bins_dr[idx5:], 5.3, 0.25, 1700*2.32e7) # b = 0: infinitely thin disk
vc_thick_disk = vc_MN(bins_dr[idx5:], 2.6, 0.8, 1700*2.32e7)
vc_stars = np.sqrt(vc_bulge**2 + vc_thin_disk**2 + vc_thick_disk**2)

def vc_total(theta, R, vc_stars):    
    a_halo, rho_halo = theta
    vc_halo = vc_NFW(R, 0, a_halo, rho_halo)
    vc_tot = np.sqrt(vc_stars**2 + vc_halo**2)    
    return vc_tot.value

def vc_total_old(theta, R, vc_stars, rho_c):    
    M_vir_halo, c_halo = theta
    vc_halo = vc_NFW_old(R, 0, M_vir_halo, c_halo, rho_c)
    vc_tot = np.sqrt(vc_stars**2 + vc_halo**2)    
    return vc_tot.value

X_GC_sun_kpc = 8.122 # kpc

def linear_fit(theta, R, X_GC_sun_kpc):
    fit = theta[0] * (R - X_GC_sun_kpc) + theta[1]
    return fit

#def chi2_lin(theta, R_obs, vc_obs, sigma_vc, X_GC_sun_kpc):
#    fit = linear_fit(theta, R_obs, X_GC_sun_kpc)
#    return np.nansum((fit - vc_obs)**2 / sigma_vc**2)
#
#def chi2_halo(theta, R_obs, vc_obs, sigma_vc, vc_stars):
#    fit = vc_total(theta, R_obs, vc_stars)
#    return np.nansum((fit - vc_obs)**2 / sigma_vc**2)

def lnlike(theta, R_obs, vc_obs, sigma_vc, vc_stars): 
    fit = vc_total(theta, R_obs, vc_stars)
    return np.nansum(-0.5 * (fit - vc_obs)**2 / sigma_vc**2)

def lnlike_old(theta, R_obs, vc_obs, sigma_vc, vc_stars, rho_c): 
    fit = vc_total_old(theta, R_obs, vc_stars, rho_c)
    return np.nansum(-0.5 * (fit - vc_obs)**2 / sigma_vc**2)

def lnprob(theta, R_obs, vc_obs, sigma_vc, vc_stars, coef_min, coef_max):
    lp = lnprior(theta, coef_min, coef_max)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, R_obs, vc_obs, sigma_vc, vc_stars) 

def lnprob_old(theta, R_obs, vc_obs, sigma_vc, vc_stars, coef_min, coef_max, rho_c):
    lp = lnprior(theta, coef_min, coef_max)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike_old(theta, R_obs, vc_obs, sigma_vc, vc_stars, rho_c) 

def lnlike_lin(theta, R_obs, vc_obs, sigma_vc, X_GC_sun_kpc): 
    fit = linear_fit(theta, R_obs, X_GC_sun_kpc)
    return np.nansum(-0.5 * (fit - vc_obs)**2 / sigma_vc**2)

def lnprob_lin(theta, R_obs, vc_obs, sigma_vc, X_GC_sun_kpc, coef_min, coef_max):
    lp = lnprior(theta, coef_min, coef_max)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike_lin(theta, R_obs, vc_obs, sigma_vc, X_GC_sun_kpc) 

def lnprior(theta, coef_min, coef_max):
    a, rho = theta
    if coef_min[0] < a < coef_max[0] and coef_min[1] < rho < coef_max[1]:
        return 0.0
    return -np.inf

def GetInitialPositions(nwalkers, coef_min, coef_max, ndim):
    np.random.seed(42)	
    p0 = np.random.uniform(size = (nwalkers, ndim))
    for i in range(0, ndim):
        p0[:, i] = coef_min[i] + p0[:, i] * (coef_max[i] - coef_min[i])
    return p0

def GetInitialPositionsBall(nwalkers, best_guess, ndim):
    np.random.seed(42)	
    p0 = np.random.normal(size = (nwalkers, ndim))
    for i in range(0, ndim):
        tiny = 0.1 * best_guess[i] # ball within 3% of initial guess
        p0[:, i] = best_guess[i] + p0[:, i] * tiny
    return p0

#coef_min = np.array([10, 1e5])
#coef_max = np.array([30, 1e10])
#ndim, nwalkers = 2, 100
#nsteps = 500
#np.random.seed(42)
#pos = GetInitialPositions(nwalkers, coef_min, coef_max, ndim)        
#sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(bins_dr[idx5:], vc_annulus[idx5:], sigmas[idx5:], vc_stars, coef_min, coef_max))
#sampler.run_mcmc(pos, nsteps)
#samples_halo = sampler.chain[:, 250:, :].reshape((-1, ndim))
#a_halo, rho_halo = np.median(samples_halo, axis = 0)
#sigma_a_halo, sigma_rho_halo = 0.5 * (np.percentile(samples_halo, 84, axis = 0) - np.percentile(samples_halo, 16, axis = 0))
#
#
#fig, axes = plt.subplots(ndim, 1, sharex=True, figsize=(8, 10))        
#for l in range(0, ndim):
#    axes[l].plot(sampler.chain[:, :, l].T, color="k", alpha=0.4)
#    axes[l].set_ylabel('$d_{}$'.format(l+1))
#    axes[l].tick_params(axis=u'both', direction='in', which='both') 
#axes[-1].set_yscale('log')              
#axes[-1].set_xlabel('step number')  

coef_min = np.array([1e10, 0])
coef_max = np.array([1e15, 50])
ndim, nwalkers = 2, 100
nsteps = 2000
np.random.seed(42)
best_guess = np.array([7.3e11, 12.75])
pos = GetInitialPositionsBall(nwalkers, best_guess, ndim)        
sampler_halo = emcee.EnsembleSampler(nwalkers, ndim, lnprob_old, args=(bins_dr[idx5:], vc_annulus[idx5:], sigmas[idx5:], vc_stars, coef_min, coef_max, rho_c))
sampler_halo.run_mcmc(pos, nsteps)
samples_halo = sampler_halo.chain[:, 1000:, :].reshape((-1, ndim))
M_vir_halo, c_halo = np.median(samples_halo, axis = 0)
sigma_M_vir_halo, sigma_c_halo = 0.5 * (np.percentile(samples_halo, 84, axis = 0) - np.percentile(samples_halo, 16, axis = 0))

fig, axes = plt.subplots(ndim, 1, sharex=True, figsize=(8, 10))        
for l in range(0, ndim):
    axes[l].plot(sampler_halo.chain[:, :, l].T, color="k", alpha=0.4)
    axes[l].set_ylabel('$d_{}$'.format(l+1))
    axes[l].tick_params(axis=u'both', direction='in', which='both') 
axes[0].set_yscale('log')              
axes[-1].set_xlabel('step number')  

#xx = sampler_halo.chain[:, 0, :].reshape((-1, ndim))
#yy = sampler_halo.chain[:, -1, :].reshape((-1, ndim))
#plt.hist(xx[:, 1])
#plt.hist(yy[:, 1], alpha = .6)
#plt.hist(xx[:, 0])
#plt.hist(yy[:, 0], alpha = .6)

## linear fit    
#x0 = np.array([-1, 200])  
#res = op.minimize(chi2_lin, x0, args=(bins_dr[idx5:], vc_annulus[idx5:], vc_annulus_err[idx5:], X_GC_sun_kpc), method='L-BFGS-B', options={'maxfun':50000}) 
#print(res)
#theta_fit = res.x

coef_min = np.array([-100, 100])
coef_max = np.array([0, 500])
ndim, nwalkers = 2, 100
nsteps = 200
pos = GetInitialPositions(nwalkers, coef_min, coef_max, ndim)        
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_lin, args=(bins_dr[idx5:], vc_annulus[idx5:], sigmas[idx5:], X_GC_sun_kpc, coef_min, coef_max))
sampler.run_mcmc(pos, nsteps)
samples = sampler.chain[:, 100:, :].reshape((-1, ndim))
m_lin, b_lin = np.median(samples, axis = 0)
sigma_m_lin, sigma_b_lin = 0.5 * (np.percentile(samples, 84, axis = 0) - np.percentile(samples, 16, axis = 0))

fig, axes = plt.subplots(ndim, 1, sharex=True, figsize=(8, 10))        
for l in range(0, ndim):
    axes[l].plot(sampler.chain[:, :, l].T, color="k", alpha=0.4)
    axes[l].set_ylabel('$d_{}$'.format(l+1))
    axes[l].tick_params(axis=u'both', direction='in', which='both') 
axes[-1].set_xlabel('step number')  

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
vc_bulge = vc_plummer(R, 0.3, 460*2.32e7)
vc_thin_disk = vc_MN(R, 5.3, 0.25, 1700*2.32e7) # b = 0: infinitely thin disk
vc_thick_disk = vc_MN(R, 2.6, 0.8, 1700*2.32e7)
vc_stars = np.sqrt(vc_bulge**2 + vc_thin_disk**2 + vc_thick_disk**2)
#vc_halo = vc_NFW(R, 0, a_halo, rho_halo)
vc_halo = vc_NFW_old(R, 0, M_vir_halo, c_halo, rho_c)
vc_tot = np.sqrt(vc_halo**2 + vc_stars**2)

#term_mass = np.log(1 + R/a_halo) - (R/a_halo)/(1+R/a_halo)
#b = 4. * np.pi * astropy.constants.G * rho_halo
#M = b * term_mass

#fig, ax = plt.subplots(1, 1, figsize = (8, 8), sharex = True)        
##plt.scatter(bins_dr[:idx5], vc_annulus[:idx5], facecolors='none', edgecolors='#3778bf', zorder = 20)
#plt.errorbar(bins_dr[idx5:], vc_annulus[idx5:], yerr = [vc_annulus_err_m[idx5:], vc_annulus_err_p[idx5:]], fmt = 'o', markersize = 6, capsize=4, mfc='#3778bf', mec='#3778bf', ecolor = '#3778bf', zorder = 30)
#plt.ylim(0, 400)
#plt.xlim(0, 25)
#plt.tick_params(axis=u'both', direction='in', which='both', right = 'on', top = 'on')
#plt.xlabel(r'$R_{\rm GC}\,\rm [kpc]$', fontsize = fsize)
#plt.ylabel(r'$v_{\rm circ}~\rm [km\,s^{-1}]$', fontsize = fsize)
#plt.plot(R, linear_fit([m_lin, b_lin], R, X_GC_sun_kpc), linestyle = ':', lw = 2, zorder = 20, color = '#929591', label=r'$v_{{\rm circ}}(R) = {0}\,{{\rm km\,s^{{-1}}}} {1}\,{{\rm km\,s^{{-1}}\,kpc^{{-1}}}}\cdot(R-R_{{\odot}})$'.format(round(b_lin, 2), round(m_lin, 2)))
#plt.plot(R, vc_tot, 'k', zorder = 100, label=r'total $v_{\rm circ}$ fit = all stars + halo')
#plt.plot(R, vc_halo, label = r'NFW profile for halo with $R_s = {0}\pm{1}{{\rm\,kpc}}$,'.format(round(a_halo, 2), round(sigma_a_halo, 2))+'\n'+r'$\rho_0=({0}\pm{1})\cdot 10^{2}\,{{\rm M_{{\odot}}\,kpc^{{-3}}}}$'.format(round(rho_halo/10**(int(np.log10(rho_halo))), 2), round(sigma_rho_halo/10**(int(np.log10(rho_halo))), 2), int(np.log10(rho_halo))))
#plt.plot(R, vc_bulge, label = r'bulge')
#plt.plot(R, vc_thin_disk, label = r'thin disk')
#plt.plot(R, vc_thick_disk, label = r'thick disk')
#plt.plot(R, vc_stars, label = r'all stars = bulge + thin disk + thick disk')
#plt.legend(frameon = True, fontsize = 14)
#plt.tight_layout()
#plt.savefig('paper_rotation_curve/rotation_curve_fit.pdf', bbox_inches = 'tight')

# pale red: #d9544d
# faded green: #7bb274
# amber: #feb308
# blue: #3778bf

#literature
huang = Table.read('data/Huang_2016.txt', format = 'ascii')
kafle = Table.read('data/data_kafle2012_cleaned.txt', format = 'ascii', data_start = 0)
lopez = Table.read('data/data_lopezcorredoira2014_cleaned.txt', format = 'ascii', data_start = 0)

draws = 100
np.random.seed(42)
# LINES IN DARK GREY/BLACK!
fig, ax = plt.subplots(1, 1, figsize = (9.3, 7.2), sharex = True)        
plt.errorbar(bins_dr[idx5:], vc_annulus[idx5:], yerr = [vc_annulus_err_m[idx5:], vc_annulus_err_p[idx5:]], fmt = 'o', markersize = 8, capsize=4, mfc='k', mec='k', ecolor = 'k', zorder = 300, label = r'Eilers et al.\ 2018 (this work)')
#plt.plot(xx1[0, idx5a:], xx1[1, idx5a:], 'o', markersize = 8, zorder = -300, color = '#D7D7D7')
#plt.plot(xx2[0, idx5b:], xx2[1, idx5b:], 'o', markersize = 8, zorder = -300, color = '#D7D7D7')
plt.ylim(0, 400) #310
plt.xlim(0, 25.2)
plt.xticks([0, 5, 10, 15, 20, 25])
plt.tick_params(axis=u'both', direction='in', which='both', right = 'on', top = 'on')
plt.xlabel(r'$R\,\rm [kpc]$', fontsize = fsize)
plt.ylabel(r'$v_{\rm c}~\rm [km\,s^{-1}]$', fontsize = fsize)
plt.plot(R, linear_fit([m_lin, b_lin], R, X_GC_sun_kpc), linestyle = ':', lw = 2, zorder = 20, color = '#0F1E66', label=r'$v_{{\rm c}}$: linear fit')
plt.plot(R, vc_halo, '#feb308', label = r'halo: NFW-profile fit', zorder = 40)
plt.plot(R, vc_tot, '#d9544d', zorder = 100, label=r'$v_{\rm c}$: all stellar components + halo')
for i in range(draws):
    d = np.random.choice(np.arange(len(samples_halo)))
    vc_halo_i = vc_NFW_old(R, 0, samples_halo[d, 0], samples_halo[d, 1], rho_c)
    vc_tot_i = np.sqrt(vc_halo_i**2 + vc_stars**2)
    plt.plot(R, vc_halo_i, color = '#feb308', lw = .6, alpha = .3, zorder = 30)
    plt.plot(R, vc_tot_i, color = '#d9544d', lw = .6, alpha = .3, zorder = 30)
plt.plot(R, vc_bulge, '#A8A8A8', linestyle=':', label = r'bulge')
plt.plot(R, vc_thin_disk, '#A8A8A8', linestyle = '-.', label = r'thin disk')
plt.plot(R, vc_thick_disk, '#A8A8A8', linestyle = '--', label = r'thick disk')
plt.plot(R, vc_stars, '#A8A8A8', label = r'all stellar components')
# literature        
plt.errorbar(huang['R'], huang['V_c'], yerr = huang['sigma_Vc'], fmt = 'v', markersize = 8, capsize=4, mfc='w', mec='#929591', ecolor = '#929591', zorder = 0, label = r'Huang et al.\ 2016')
plt.errorbar(kafle['col1'], kafle['col2'], yerr = kafle['col3'], fmt = 'D', markersize = 6, capsize=4, mfc='w', mec='#5e819d', ecolor = '#5e819d', zorder = 0, label = r'Kafle et al.\ 2012')
plt.errorbar(lopez['col1'], lopez['col2'], yerr = lopez['col3'], fmt = 'p', markersize = 6, capsize=4, mfc='w', mec='#82a67d', ecolor = '#82a67d', zorder = 0, label = r'Lopez-Corredoira et al.\ 2014')
ax.add_patch(patches.Rectangle((0, 0), 5, 400, facecolor = '#EFEFEF', edgecolor = '#EFEFEF', alpha = .8, zorder = -1000))        # (x,y), width, height       
handles, labels = ax.get_legend_handles_labels()
hand = [handles[7], handles[8], handles[10], handles[9], handles[0], handles[2], handles[1], handles[6], handles[3], handles[4], handles[5]]
lab = [labels[7], labels[8], labels[10], labels[9], labels[0], labels[2], labels[1], labels[6], labels[3], labels[4], labels[5]]
plt.legend(hand, lab, frameon = True, fontsize = 13, ncol = 3, loc = 1)
plt.tight_layout()
plt.savefig('paper_rotation_curve/rotation_curve_fit_paper_literature_part1.pdf', bbox_inches = 'tight')


# -------------------------------------------------------------------------------
# determine values
# -------------------------------------------------------------------------------

print('{:e} pm {:e}'.format(M_vir_halo, sigma_M_vir_halo))
print('{} pm {}'.format(c_halo, sigma_c_halo))
#print('{:e} pm {:e}'.format(rho_halo, sigma_rho_halo))
#print('{} pm {}'.format(a_halo, sigma_a_halo))
print(m_lin, b_lin, sigma_m_lin, sigma_b_lin)

rho_c = (3. * planck.H0**2 / (8. * np.pi * astropy.constants.G)).to(u.M_sun/(u.kpc)**3)
R_vir = np.cbrt(M_vir_halo * u.M_sun / rho_c / 200. / (4.*np.pi/3))
sigma_R_vir = (M_vir_halo * u.M_sun / 200./ rho_c / (4*np.pi/3)) ** (-2./3.) / 200. / rho_c / (4.*np.pi) * sigma_M_vir_halo * u.M_sun
a = R_vir / c_halo
sigma_a = np.sqrt((sigma_R_vir/c_halo)**2 + (R_vir*sigma_c_halo/c_halo**2)**2)
rho_0 = M_vir_halo * u.M_sun / (np.log(1+c_halo) - c_halo/(1+c_halo)) / (4.*np.pi*a**3)
term = np.log(1+c_halo) - c_halo / (1+c_halo)
d_rho_0_d_M_vir = 1. /term / (4.*np.pi*a**3)
d_rho_0_d_a = -3.*M_vir_halo * u.M_sun/term / (4.*np.pi*a**4)
d_rho_0_d_c = -1.*M_vir_halo * u.M_sun/term**2 / (4.*np.pi*a**3) * c_halo/(1+c_halo)**2
sigma_rho_0 = np.sqrt(d_rho_0_d_M_vir**2 * (sigma_M_vir_halo*u.M_sun)**2 + d_rho_0_d_a**2*sigma_a**2 + d_rho_0_d_c**2*sigma_c_halo**2)
R_sun = 8.122 * u.kpc
sigma_R_sun = 0.031 * u.kpc
c = astropy.constants.c
rho_local = rho_0 / (R_sun / a * (1 + R_sun / a)**2)
rho_local_energy = (rho_local * c**2).to(u.GeV / u.cm**3)

d_rho_local_d_rho_0 = 1. / (R_sun / a * (1 + R_sun / a)**2)
d_rho_local_d_R_sun = -1. * rho_0 / (R_sun / a * (1 + R_sun / a)**2)**2 * (1./a * (1+R_sun/a)**2 + 2*R_sun/a**2*(1+R_sun/a))
d_rho_local_d_a = -1.*rho_0 / (R_sun / a * (1 + R_sun / a)**2)**2 * (-1.*R_sun/a**2*(1+R_sun/a)**2 - 2.*R_sun**2/a**3*(1+R_sun/a))
sigma_rho_local = np.sqrt(d_rho_local_d_rho_0**2 * sigma_rho_0**2 + d_rho_local_d_R_sun**2 * sigma_R_sun**2 + d_rho_local_d_a**2 * sigma_a**2)
sigma_rho_local_energy = (sigma_rho_local * c**2).to(u.GeV / u.cm**3)
print('{:e} pm {:e}'.format(rho_0, sigma_rho_0))

# -------------------------------------------------------------------------------
# old plots
# -------------------------------------------------------------------------------'''

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