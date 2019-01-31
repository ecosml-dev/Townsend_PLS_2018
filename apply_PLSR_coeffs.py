# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 14:29:10 2018

@author: zwang896

Applying ASD fresh spectra models and predict traits
"""

import numpy as np, os,pandas as pd,glob

#-----------------define inout and output
outdir = 'D:/test/output/'
plsrDir = 'D:/test/coefficients/'
specCSV = 'D:/test/test_spectra.csv'

plsrCSVs = glob.glob(plsrDir+'*.csv')
plsrCSVs = sorted(plsrCSVs)

df_spec = pd.read_csv(specCSV, sep=',')
spec = df_spec.iloc[:,0:2151].values

#---------------vector normalization
spec_len = np.tile(np.linalg.norm(spec, axis=1), (spec.shape[1], 1))
spec = spec/spec_len.T

#---------------5nm resampling
wl_step = 5
wl = np.arange(350,2501)
spec = spec[:,0::wl_step]
wl = wl[0::wl_step]

#----------------appyling PLSR coefficients
df_all=pd.DataFrame()
df_all=df_spec.iloc[:,2151:]

for plsrCSV in plsrCSVs:
    trait_model = pd.read_csv(plsrCSV, sep=',', index_col=0).values
    intercept = trait_model[:, 0]
    coefficients = np.array(trait_model[:, list(np.arange(1, trait_model.shape[1]))])
    
    traitPred = np.einsum('jl,ml->jm',spec,coefficients, optimize='optimal')
    traitPred = traitPred + intercept
    traitPred_mean = traitPred.mean(axis=1)
    traitPred_std = traitPred.std(axis=1,ddof=1)

    trait = os.path.basename(plsrCSV)[14:-4]
    df_all[trait+'_mean'] = traitPred_mean
    df_all[trait+'_std'] = traitPred_std

#----------------write predicted traits to csv
outfile = os.path.join(outdir,os.path.basename(specCSV)[:-4]+'_traits.csv')
df_all.to_csv(outfile)
