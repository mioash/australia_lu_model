# -*- coding: utf-8 -*-
"""
Created on Mon May 31 12:26:09 2021

@author: mcalderonloor
"""

import numpy as np
#import allocate_gurobi
#from allocate_gurobi import gurobi_allocator
from allocate_maxprob import max_prob_allocator

import matplotlib as mpl
import matplotlib.pyplot as plt
from numpy_to_tiff import npy_long_to_wide
from numpy_to_tiff import npy_to_tif

from glob import glob

def read_all_files(stt):
   file_names = glob('E:/Marco/australia_lu_model/data/suitability/'+str(stt)+'/*')
   arrays = [np.load(f) for f in file_names]
   return np.concatenate(arrays,axis=0)    

tas_npy = read_all_files('tas1')

loc_past = 7 #7 for the previous, 8 for current
loc_future = 8 #8 for the previous, 7 for current

geoid = np.unique(tas_npy[:,2])
geoid_s = geoid[(geoid[:] != 0) & (geoid[:] != -9999)]

finn_o = np.zeros((0, 5))
nan_df_o = finn_o
acco = np.zeros((len(geoid_s),1))

for i in range(0,len(geoid_s),1):
    slap = tas_npy[tas_npy[:,2]==geoid_s[i],:]
    
    
    slap = slap[(slap[:, loc_past] != 4) &
                      (slap[:, loc_past] != 5) & 
                      (slap[:, loc_future] != 4) &
                      (slap[:, loc_future] != 5),:]
    
    nan_df = slap[(slap[:,3] < 0) & 
                  (slap[:,4] < 0) &
                  (slap[:,5] < 0) &
                  (slap[:,6] < 0),:]    
    
    nan_df = np.insert(nan_df,loc_future+1,-9999,axis=1) 
        
    nan_df_o =  nan_df[:,[0,1,loc_future,loc_future+1,2]]
    
    slap = slap[(slap[:,3] >= 0) & 
                  (slap[:,4] >= 0) &
                  (slap[:,5] >= 0) &
                  (slap[:,6] >= 0),:]
    
    dfmo, acco[i] = max_prob_allocator(slap,loc_future)
    print(np.sum(dfmo[:, 3] == 1) - np.sum(dfmo[:, 2] == 1))
    print(np.sum(dfmo[:, 3] == 2) - np.sum(dfmo[:, 2] == 2))
    print(np.sum(dfmo[:, 3] == 3) - np.sum(dfmo[:, 2] == 3))
    print(np.sum(dfmo[:, 3] == 0) - np.sum(dfmo[:, 2] == 0))
    finn_o = np.concatenate((dfmo,finn_o,nan_df_o),axis=0)


np.save('E:/Marco/australia_lu_model/data/allocation/tas1/2010_tas.npy', finn_o)
np.save('E:/Marco/australia_lu_model/data/allocation/tas1/2010_accuracy_tas.npy', acco)
np.save('E:/Marco/australia_lu_model/data/allocation/tas1/2010_geoid_order_tas.npy', geoid)

#[0,1,loc_future,loc_future,2]    
null_lat = tas_npy[tas_npy[:, 2] == -9999,0]
null_lon = tas_npy[tas_npy[:, 2] == -9999,1]
null_df = np.vstack([null_lat, null_lon]).transpose()
null_df = np.insert(null_df,2,tas_npy[tas_npy[:, 2] == -9999,loc_future],axis=1) 
null_df = np.insert(null_df,3,tas_npy[tas_npy[:, 2] == -9999,loc_future],axis=1) 
null_df = np.insert(null_df,4,tas_npy[tas_npy[:, 2] == -9999,2],axis=1) 

finn_o = np.concatenate((finn_o,null_df),axis=0)
finn_o[:,0] = finn_o[:,0] / 10000000
finn_o[:,1] = finn_o[:,1] / 10000000



npy_to_tif(finn_o,0,1,3,'E:/Marco/australia_lu_model/data/allocation/tas1/pred_2010_tas_v2.tiff')  #predicted is on number 3







