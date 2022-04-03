# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import allocate_gurobi
from allocate_gurobi import gurobi_allocator
from allocate_maxprob import max_prob_allocator
#import h5py 

import time
import gc
from numpy_to_tiff import npy_to_tif

#np.save('tas_npy.npy', tas_npy)
#tas_npya = np.load('C:/Users/mcalderonloor/Documents/model_lu_australia/australia_lu_model/data/tas_npy.npy')

# tas1 = np.load('E:/Marco/australia_lu_model/data/suitability/2010_suit_tas_0_100.npy')
# tas2 = np.load('E:/Marco/australia_lu_model/data/suitability/2010_suit_tas_100_200.npy')
# tas3 = np.load('E:/Marco/australia_lu_model/data/suitability/2010_suit_tas_200_289.npy')

# vic = np.load('E:/Marco/australia_lu_model/data/suitability/2010_suit_qld_0_100.npy')

from glob import glob

def read_all_files(stt):
   file_names = glob('E:/Marco/australia_lu_model/data/suitability/'+str(stt)+'/*')
   arrays = [np.load(f) for f in file_names]
   return np.concatenate(arrays,axis=0)    

stt = 'nt'
tas_npy = read_all_files(stt)

#702031057

# hf = h5py.File('E:/Marco/australia_lu_model/data/suitability/'+str(stt)+'/2010_'+str(stt)+'.h5', 'w')
geoid = np.unique(tas_npy[:,2])
geoid = geoid[(geoid != 0) & (geoid != -9999)]


# for i in range(0,len(geoid),1):
#     print (i)
#     hf.create_dataset(str(stt)+'_'+str(geoid[i]), data = tas_npy[tas_npy[:,2]==geoid[i],:])
# hf.close()




#tas_npyb = np.concatenate((tas1,tas2,tas3),axis=0)

#np.save('C:/Users/mcalderonloor/Documents/model_lu_australia/australia_lu_model/data/tas_npy_v2.npy', tas_npyb)
# 0:4 ['latitude', 'longitude', 'sa2_id', 'prob_crop', 'prob_forest', 
# 5:11 'prob_grass', 'prob_urban', 'prob_urban39', 'prob_urban3927', 'prob_urban3', 'b2005', 'b2010']

#tas_npy = tas_npya[:, [0,1,2,3,4,5,6,8,9]]  ##current
#tas
#tas_npy = tas_npyb[:, [0,1,2,3,4,5,6,7,8]]  ##current

#tas_npy = tc#vic[:, [0,1,2,3,4,5,6,7,8]]  ##current

#tas_npyb = np.load('australia_lu_model/data/tas_npy.npy')

loc_past = 7 #7 for the previous, 8 for current
loc_future = 8 #8 for the previous, 7 for current

# tas_discard = tas_npy[(tas_npy[:, 2] == -9999) | 
#                   (tas_npy[:, 3] == -9999) | 
#                   (tas_npy[:, 4] == -9999) | 
#                   (tas_npy[:, 5] == -9999) | 
#                   (tas_npy[:, 6] == -9999) | 
#                   (tas_npy[:, 7] == -9999) | 
#                   (tas_npy[:, 8] == -9999) | 
#                   (tas_npy[:, 2] == 0) |
#                   (tas_npy[:, loc_past] == 4) | 
#                   (tas_npy[:, loc_past] == 5) | 
#                   (tas_npy[:, loc_future] == 4) | 
#                   (tas_npy[:, loc_future] == 5),:]

tas_npy = tas_npy[(tas_npy[:, 2] != -9999) & 
                  # (tas_npy[:, 3] != -9999) &  
                  # (tas_npy[:, 4] != -9999) &  
                  # (tas_npy[:, 5] != -9999) & 
                  # (tas_npy[:, 6] != -9999) &  
                  # (tas_npy[:, 7] != -9999) &  
                  # (tas_npy[:, 8] != -9999) & 
                  (tas_npy[:, 2] != 0) &
                  (tas_npy[:, loc_past] != 4) &
                  (tas_npy[:, loc_past] != 5) & 
                  (tas_npy[:, loc_future] != 4) &
                  (tas_npy[:, loc_future] != 5),:]

#tas_npy[:3:7] = (tas_npy[:3:7] * 10000000).astype('i4')

# geoid = np.unique(tas_npy[:,2])

#geoid_s = geoid[geoid != 0]
# geoid_s = geoid#[geoid[:] == 201011008]
#geoid_s = geoid
# check area 201011002

# t1 = time.perf_counter()
# finn = np.zeros((0, 3))
# acc = np.zeros((len(geoid_s),1))

# for i in range(0,len(geoid_s),1):
#     slap = tas_npy[tas_npy[:,2]==geoid_s[i],:]
#     dfm, acc[i] = gurobi_allocator(slap,9)
#     finn = np.concatenate((dfm,finn),axis=0)

# finn_o = np.zeros((0, 5))
# nan_df_o = finn_o
# acco = np.zeros((len(geoid_s),1))
# #pos = np.zeros((len(geoid_s),1))
# #neg = np.zeros((len(geoid_s),1))

# t1 = time.perf_counter()    
# for i in range(0,len(geoid_s),1):
#     slap = tas_npy[tas_npy[:,2]==geoid_s[i],:]
#     nan_df = slap[(slap[:,3] < 0) & 
#                   (slap[:,4] < 0) &
#                   (slap[:,5] < 0) &
#                   (slap[:,6] < 0),:]
#     nan_df = np.insert(nan_df,loc_future+1,-9999,axis=1) 
    
#     #n_df = np.insert(slap, np.shape(slap)[1], max_index_row, axis=1)
#     #nan_df_i = np.zeros((0, 5))
#     nan_df_o =  nan_df[:,[0,1,loc_future,loc_future+1,2]]
#     #nan_df_i[:,2] =  nan_df[:,loc_future]
#     #nan_df_i[:,3] =  -9999
#     #nan_df_i[:,4] =  nan_df[:,2]
#     #nan_df_o = np.concatenate((nan_df_o,nan_df_i),axis=0)
#     slap = slap[(slap[:,3] >= 0) & 
#                   (slap[:,4] >= 0) &
#                   (slap[:,5] >= 0) &
#                   (slap[:,6] >= 0),:]
    
#     dfmo, acco[i] = max_prob_allocator(slap,loc_future)
#     print(np.sum(dfmo[:, 3] == 1) - np.sum(dfmo[:, 2] == 1))
#     print(np.sum(dfmo[:, 3] == 2) - np.sum(dfmo[:, 2] == 2))
#     print(np.sum(dfmo[:, 3] == 3) - np.sum(dfmo[:, 2] == 3))
#     print(np.sum(dfmo[:, 3] == 0) - np.sum(dfmo[:, 2] == 0))
#     # finn_o = np.concatenate((dfmo,finn_o,nan_df_o),axis=0)
#     finn_a = np.concatenate((dfmo,nan_df_o),axis=0)
#     finn_a[:,0] = finn_a[:,0] / 10000000
#     finn_a[:,1] = finn_a[:,1] / 10000000
#     npy_to_tif(finn_a,0,1,3,'E:/Marco/australia_lu_model/data/allocation/'+str(stt)+'/pred_2010_'+str(geoid[i])+'_v1.tiff')  #predicted is on number 3


################################

from joblib import Parallel, delayed

def process(df,geoid_s,loc_past,loc_future,i):
    slap = df[df[:,2]==geoid_s[i],:]
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
    
    dfmo, acco = max_prob_allocator(slap,loc_future)
    finn_a = np.concatenate((dfmo,nan_df_o),axis=0)
    finn_a[:,0] = finn_a[:,0] / 10000000
    finn_a[:,1] = finn_a[:,1] / 10000000
    npy_to_tif(finn_a,0,1,3,'E:/Marco/australia_lu_model/data/allocation/'+str(stt)+'/pred_2010_'+str(70333)+'_v1.tiff')  #predicted is on number 3a
    return finn_a, acco

t1 = time.perf_counter()        
results = Parallel(n_jobs=26,prefer='threads')(delayed(process)(tas_npy,geoid,loc_past,loc_future,i) for i in range(0,len(geoid)))
tf = time.perf_counter()
tt = tf-t1

#arr = np.vstack(results[][])
arr_tuple = [a_tuple[0] for a_tuple in results]
arr_tuple = np.vstack(arr_tuple)
acc_tuple = [a_tuple[1] for a_tuple in results]

 #range(0,len(geoid_s),1)
#results = Parallel(n_jobs=num_cores)(delayed(processInput)(i,j) for i,j in zip(a,b))

######################    
    
#acc = np.concatenate((acc,acc0))
#tf = time.perf_counter()
#tt = tf-t1
#del tc
del tas_npy
gc.collect()

import sklearn.metrics as metrics
# calculate the fpr and tpr for all thresholds of the classification
fpr, tpr, threshold = metrics.roc_curve(arr_tuple[:,3], arr_tuple[:,2])
roc_auc = metrics.auc(fpr, tpr)

# method I: plt
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# ac_tas_or= acco
# df_tas_or= finn_o

np.save('E:/Marco/australia_lu_model/data/allocation/'+str(stt)+'/2010_'+str(stt)+'.npy', arr_tuple)
np.save('E:/Marco/australia_lu_model/data/allocation/'+str(stt)+'/2010_accuracy_'+str(stt)+'.npy', acc_tuple)
np.save('E:/Marco/australia_lu_model/data/allocation/'+str(stt)+'/2010_geoid_order_'+str(stt)+'.npy', geoid)


########################################
########################################
#from numpy_to_tiff import npy_long_to_wide



#dfs = np.load('E:/Marco/australia_lu_model/data/allocation/tas1/2010_tas.npy')
#finn_o[:,0] = finn_o[:,0] / 10000000
#finn_o[:,1] = finn_o[:,1] / 10000000

npy_to_tif(arr_tuple,0,1,3,'E:/Marco/australia_lu_model/data/allocation/'+str(stt)+'/pred_2010_'+str(stt)+'_v1.tiff')  #predicted is on number 3


