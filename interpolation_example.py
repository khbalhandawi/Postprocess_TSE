# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 18:33:01 2020

@author: Khalil
"""

import numpy as np
import itertools

mu_lob = np.array([0.15, 0.80, 0.80, 0.85])
mu_upb = np.array([0.85, 0.20, 0.20, 0.15])
 
Sigma_lob = np.array([0.1875, 0.125, 0.125, 0.1875]) # sigma^2
Sigma_upb = np.array([0.375 , 0.250, 0.250, 0.375]) # sigma^2

lob_req = np.append(mu_lob,Sigma_lob)
upb_req = np.append(mu_upb,Sigma_upb)

req_type_1 = 'uniform'
req_type_2 = 'guassian'

mu_1 = lob_req[:4]
mu_2 = upb_req[:4]

Sigma_1 = lob_req[4::] # Sigma^2
Sigma_2 = upb_req[4::] # Sigma^2

# linearly interpolate between two vectors
from scipy.interpolate import interp1d

linfit = interp1d([1,5], np.vstack([mu_1, mu_2]), axis=0)
mus = list(linfit([1,2,3,4,5]))
linfit = interp1d([1,5], np.vstack([Sigma_1, Sigma_2]), axis=0)
Sigmas = list(linfit([1,2,3,4,5]))

Sigma_2s = []
for Sigma in Sigmas:
    Sigma_2 = (Sigma/3)**2
    Sigma_2s += [Sigma_2]

req_list = [[req_type_1, req_type_2], mus, Sigmas ]
req_combinations = list(itertools.product(*req_list)) 