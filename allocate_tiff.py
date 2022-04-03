# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 14:45:28 2021

@author: mcalderonloor
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
from allocate_maxprob import max_prob_allocator
import gc
from numpy_to_tiff import npy_to_tif
from glob import glob
from joblib import Parallel, delayed

def read_all_files(stt):
   file_names = glob('E:/Marco/australia_lu_model/data/suitability/'+str(stt)+'/*')
   arrays = [np.load(f) for f in file_names]
   return np.concatenate(arrays,axis=0)    

def process(stt,df,geoid_s,loc_past,loc_future,i):
    slap = df[df[:,2]==geoid_s[i],:]
    nan_df = slap[(slap[:,3] < 0) & 
                  (slap[:,4] < 0) &
                  (slap[:,5] < 0) &
                  (slap[:,6] < 0),:]
    nan_df = np.insert(nan_df,loc_future+1,255,axis=1) 
    #nan_df =  nan_df[:,[0,1,loc_future,loc_future+1,2]]
    
    slap = slap[(slap[:,3] >= 0) & 
                  (slap[:,4] >= 0) &
                  (slap[:,5] >= 0) &
                  (slap[:,6] >= 0),:]
    
    dfmo, acco = max_prob_allocator(slap,loc_future)
    finn_a = np.concatenate((dfmo,nan_df[:,[0,1,loc_future,loc_future+1,2]]),axis=0)
    finn_a[:,0] = finn_a[:,0] / 10000000
    finn_a[:,1] = finn_a[:,1] / 10000000
    npy_to_tif(finn_a,0,1,3,'E:/Marco/australia_lu_model/data/allocation/'+str(stt)+'/pred_2010_'+str(geoid_s[i])+'_v1.tiff')  #predicted is on number 3
    return  acco #,finn_a

def allocate (stt):

    tas_npy = read_all_files(stt)
    
    geoid = np.unique(tas_npy[:,2])
    geoid = geoid[(geoid != 0) & (geoid != -9999)]
    loc_past = 7 #7 for the previous, 8 for current
    loc_future = 8 #8 for the previous, 7 for current
    
    tas_npy = tas_npy[(tas_npy[:, 2] != -9999) & 
                      (tas_npy[:, 2] != 0) &
                      (tas_npy[:, loc_past] != 4) &
                      (tas_npy[:, loc_past] != 5) & 
                      (tas_npy[:, loc_future] != 4) &
                      (tas_npy[:, loc_future] != 5),:]
         
    results = Parallel(n_jobs=6,prefer='threads')(delayed(process)(stt,tas_npy,geoid,loc_past,loc_future,i) for i in range(0,len(geoid)))
    
    del tas_npy
    gc.collect()
    
    #arr_tuple = [a_tuple[0] for a_tuple in results]
    #arr_tuple = np.vstack(arr_tuple)
    #acc_tuple = [a_tuple[0] for a_tuple in results]
    
    #np.save('E:/Marco/australia_lu_model/data/allocation/'+str(stt)+'/2010_'+str(stt)+'.npy', arr_tuple)
    np.save('E:/Marco/australia_lu_model/data/allocation/'+str(stt)+'/2010_accuracy_'+str(stt)+'.npy', np.c_[ results, geoid] )
    #np.save('E:/Marco/australia_lu_model/data/allocation/'+str(stt)+'/2010_geoid_order_'+str(stt)+'.npy', geoid)
