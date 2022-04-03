# -*- coding: utf-8 -*-
"""
Created on Sat May 15 12:27:54 2021

@author: mcalderonloor
"""
import numpy as np
import allocate_gurobi
from allocate_gurobi import gurobi_allocator
from allocate_maxprob import max_prob_allocator

import time


exx = np.load('E:/Marco/australia_lu_model/data/suitability/2010_suit_qld_0_100.npy')

import sys
sys.getsizeof(exx.astype('i4'))

# 0:4 ['latitude', 'longitude', 'sa2_id', 'prob_crop', 'prob_forest', 
# 5:11 'prob_grass', 'prob_urban', 'prob_urban39', 'prob_urban3927', 'prob_urban3', 'b2005', 'b2010']

ll = exx[:,0:2] * 10000000
ll = ll.astype('i4')
geoid = exx[:,2].astype('i4')
geoid = np.array([geoid]).transpose()
probs = exx[:,3:7] * 100000
probs = probs.astype('i4')
lc = exx[:,7:9].astype('i4')

#np.append(a, z, axis=1)
arr = np.concatenate([ll, geoid, probs, lc], axis=1)#.transpose()

#asas[0]
