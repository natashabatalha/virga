

from __future__ import absolute_import, unicode_literals, print_function

import math, os

import time

import sys

import astropy.units as u

import virga.justdoit as jdi

#from bintools import binning

import pandas as pd

import dynesty

import concurrent.futures as cf

import pdb

import numpy as np

from numba import jit
import pickle


np.random.seed(0)

start_time=time.time()

########################

#VIRGA Setup

########################

metallicity = 1 #atmospheric metallicity relative to Solar
mean_molecular_weight = 2.2 # atmospheric mean molecular weight

# read in CARMA data
CARMA_files = '/Users/crooney/Documents/codes/all-data/CARMA/'

file = CARMA_files + 'pressure.csv'
df = pd.read_csv(file, sep=',',header=None)
pressure = df.values[0,:]/1e6

file = CARMA_files + 'temperature.csv'
df = pd.read_csv(file, sep=',',header=None)
temperature = df.values[0,:]

file = CARMA_files + 'kzz.csv'
df = pd.read_csv(file, sep=',',header=None)
kzz = df.values[0,:]

file = CARMA_files + 'opd.csv'
df = pd.read_csv(file, sep=',',header=None)
opd_carma = df.values[1:,:] # start at index 1 because carma gives opd at levels, virga gives opd at layers



# priors
def prior_transform(u):
    u[0] = 0.1 + 100*u[0] #alpha 
    u[1] = 0.1 + 5*u[1] #beta
    
    return u


#   log likelihood
def loglike(x):

    sum_planet = jdi.Atmosphere(['H2O'], fsed=x[0], mh=metallicity,
                         mmw = mean_molecular_weight,
                         b=x[1], param='exp')

    #set the planet gravity
    sum_planet.gravity(gravity=1e4, gravity_unit=u.Unit('cm/(s**2)'))

    #PT 
    sum_planet.ptk(df = pd.DataFrame({'pressure':pressure, 'temperature':temperature,
                           'kz':kzz}))

    #directory where mieff files are 
    mieff_dir = '/Users/crooney/Documents/codes/all-data/mieff_files'

    #get full dictionary output 
    all_out = sum_planet.compute(as_dict=True, directory=mieff_dir)

    avg_opd = np.mean(all_out['opd_per_layer'],1)
    y_mod = avg_opd

    loglikelihood=-0.5*np.sum((y_meas-y_mod)**2/err**2)
       
    return loglikelihood


y_meas = np.mean(opd_carma,1)
err = y_meas/100

########################

#MCMC output file name

########################

outname='VIRGA_RESULTS.pic'


#Dynesty Setup

########################

if __name__=='__main__':

    Nparam=2
    
    pool=cf.ProcessPoolExecutor()
    
    Nproc=2
    
    Nlive=1000
    
    dsampler = dynesty.NestedSampler(loglike, prior_transform, ndim=Nparam,
                    bound='multi', sample='auto', nlive=Nlive, update_interval=1.5, 
                    pool=pool, queue_size=Nproc)
    
    
    dsampler.run_nested(maxiter=9481)
    
    end_time=time.time()
    
    run_time=end_time-start_time
    
    pdf1=pd.DataFrame({'Time':[run_time]})
    
    pdf1.to_csv('runtime.csv',sep=' ',index=False)
    
    print('DONE')
    
    output=dsampler.results
    
    pickle.dump(output,open(outname,"wb"))


