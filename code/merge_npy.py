# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import allocate_gurobi
from allocate_gurobi import gurobi_allocator
from allocate_maxprob import max_prob_allocator

import time


#np.save('tas_npy.npy', tas_npy)
tas_npya = np.load('C:/Users/mcalderonloor/Documents/model_lu_australia/australia_lu_model/data/tas_npy.npy')

tas1 = np.load('C:/Users/mcalderonloor/Documents/model_lu_australia/australia_lu_model/data/tas_0_99_v1.npy')
tas2 = np.load('C:/Users/mcalderonloor/Documents/model_lu_australia/australia_lu_model/data/tas_100_199_v1.npy')
tas3 = np.load('C:/Users/mcalderonloor/Documents/model_lu_australia/australia_lu_model/data/tas_200_289_v1.npy')

tas_npyb = np.concatenate((tas1,tas2,tas3),axis=0)

#np.save('C:/Users/mcalderonloor/Documents/model_lu_australia/australia_lu_model/data/tas_npy_v2.npy', tas_npyb)
# 0:4 ['latitude', 'longitude', 'sa2_id', 'prob_crop', 'prob_forest', 
# 5:11 'prob_grass', 'prob_urban', 'prob_urban39', 'prob_urban3927', 'prob_urban3', 'b2005', 'b2010']

#tas_npy = tas_npya[:, [0,1,2,3,4,5,6,8,9]]  ##current
tas_npy = tas_npyb[:, [0,1,2,3,4,5,9,10,11]]  ##current


#tas_npyb = np.load('australia_lu_model/data/tas_npy.npy')

loc_past = 7 #7 for the previous, 8 for current
loc_future = 8 #8 for the previous, 7 for current

tas_discard = tas_npy[(tas_npy[:, 2] == -9999) | 
                  (tas_npy[:, 3] == -9999) | 
                  (tas_npy[:, 4] == -9999) | 
                  (tas_npy[:, 5] == -9999) | 
                  (tas_npy[:, 6] == -9999) | 
                  (tas_npy[:, 7] == -9999) | 
                  (tas_npy[:, 8] == -9999) | 
                  (tas_npy[:, 2] == 0) |
                  (tas_npy[:, loc_past] == 4) | 
                  (tas_npy[:, loc_past] == 5) | 
                  (tas_npy[:, loc_future] == 4) | 
                  (tas_npy[:, loc_future] == 5),:]

tas_npy = tas_npy[(tas_npy[:, 2] != -9999) & 
                  (tas_npy[:, 3] != -9999) & 
                  (tas_npy[:, 4] != -9999) & 
                  (tas_npy[:, 5] != -9999) & 
                  (tas_npy[:, 6] != -9999) & 
                  (tas_npy[:, 7] != -9999) & 
                  (tas_npy[:, 8] != -9999) & 
                  (tas_npy[:, 2] != 0) & 
                  (tas_npy[:, loc_past] != 4) & 
                  (tas_npy[:, loc_past] != 5) & 
                  (tas_npy[:, loc_future] != 4) & 
                  (tas_npy[:, loc_future] != 5),:]


geoid = np.unique(tas_npy[:,2])

geoid_s = geoid#[0:25]
#geoid_s = geoid[geoid[:] == 601011003]

# t1 = time.perf_counter()
# finn = np.zeros((0, 3))
# acc = np.zeros((len(geoid_s),1))

# for i in range(0,len(geoid_s),1):
#     slap = tas_npy[tas_npy[:,2]==geoid_s[i],:]
#     dfm, acc[i] = gurobi_allocator(slap,9)
#     finn = np.concatenate((dfm,finn),axis=0)

finn_o = np.zeros((0, 5))
acco = np.zeros((len(geoid_s),1))
#pos = np.zeros((len(geoid_s),1))
#neg = np.zeros((len(geoid_s),1))

t1 = time.perf_counter()    
for i in range(0,len(geoid_s),1):
    slap = tas_npy[tas_npy[:,2]==geoid_s[i],:]
    dfmo, acco[i] = max_prob_allocator(slap,loc_future)
    print(np.sum(dfmo[:, 3] == 1) - np.sum(dfmo[:, 2] == 1))
    print(np.sum(dfmo[:, 3] == 2) - np.sum(dfmo[:, 2] == 2))
    print(np.sum(dfmo[:, 3] == 3) - np.sum(dfmo[:, 2] == 3))
    print(np.sum(dfmo[:, 3] == 0) - np.sum(dfmo[:, 2] == 0))
    finn_o = np.concatenate((dfmo,finn_o),axis=0)
    
#acc = np.concatenate((acc,acc0))
tf = time.perf_counter()
tt = tf-t1

ac_new_all_urban3 = acco
df_new_all_urban3 = finn_o

np.save('C:/Users/mcalderonloor/Documents/model_lu_australia/australia_lu_model/data/df_new_all_urban.npy', df_new_all_urban)
#np.save('C:/Users/mcalderonloor/Documents/model_lu_australia/australia_lu_model/data/df_new_all_urban.npy', df_new_all_urban)
np.save('C:/Users/mcalderonloor/Documents/model_lu_australia/australia_lu_model/data/accuracy_maxprob_new.npy', acco)
np.save('C:/Users/mcalderonloor/Documents/model_lu_australia/australia_lu_model/data/geoid_order_maxprob_new.npy', geoid)
np.save('C:/Users/mcalderonloor/Documents/model_lu_australia/australia_lu_model/data/accuracy_maxprob_old.npy', ac_prev)

# lc_past = np.unique(tas_npy[:,8])
# lc_future = np.unique(tas_npy[:,9])



# LU0_targ = np.sum(slap[:, 9] == 0) # 1334
# LU1_targ = np.sum(slap[:, 9] == 1) # 40083
# LU2_targ = np.sum(slap[:, 9] == 2) # 5116
# LU3_targ = np.sum(slap[:, 9] == 3) # 6661
# LU4_targ = np.sum(slap[:, 9] == 4) # 6661
# LU4_targ = np.sum(slap[:, 9] == 5) # 6661
