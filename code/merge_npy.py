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

#tas1 = np.load('tas_0_99.npy')
#tas2 = np.load('tas_100_199.npy')
#tas3 = np.load('tas_200_289.npy')

#tas_npy = np.concatenate((tas1,tas2,tas3),axis=0)

#np.save('tas_npy.npy', tas_npy)
tas_npy = np.load('tas_npy.npy')

tas_npy = tas_npy[(tas_npy[:, 2] != -9999) & 
                  (tas_npy[:, 2] != 0) & 
                  (tas_npy[:, 8] != 4) & 
                  (tas_npy[:, 8] != 5) & 
                  (tas_npy[:, 9] != 4) & 
                  (tas_npy[:, 9] != 5),:]

geoid = np.unique(tas_npy[:,2])

geoid_s = geoid[0:15]
#geoid_s = geoid[geoid[:] == 601011003]

t1 = time.perf_counter()
finn = np.zeros((0, 3))
acc = np.zeros((len(geoid_s),1))

for i in range(0,len(geoid_s),1):
    slap = tas_npy[tas_npy[:,2]==geoid_s[i],:]
    dfm, acc[i] = gurobi_allocator(slap,9)
    finn = np.concatenate((dfm,finn),axis=0)

finn_o = np.zeros((0, 5))
acco = np.zeros((len(geoid_s),1))
#pos = np.zeros((len(geoid_s),1))
#neg = np.zeros((len(geoid_s),1))

t1 = time.perf_counter()    
for i in range(0,len(geoid_s),1):
    slap = tas_npy[tas_npy[:,2]==geoid_s[i],:]
    dfmo, acco[i] = max_prob_allocator(slap,9)
    print(np.sum(dfmo[:, 3] == 1) - np.sum(dfmo[:, 2] == 1))
    print(np.sum(dfmo[:, 3] == 2) - np.sum(dfmo[:, 2] == 2))
    print(np.sum(dfmo[:, 3] == 3) - np.sum(dfmo[:, 2] == 3))
    print(np.sum(dfmo[:, 3] == 0) - np.sum(dfmo[:, 2] == 0))
    finn_o = np.concatenate((dfmo,finn_o),axis=0)
    
#acc = np.concatenate((acc,acc0))
tf = time.perf_counter()
tt = tf-t1


np.save('tas_maxprob_2010_output.npy', finn_o)
np.save('accuracy_maxprob.npy', acco)
np.save('geoid_order_maxprob.npy', geoid)


# lc_past = np.unique(tas_npy[:,8])
# lc_future = np.unique(tas_npy[:,9])



# LU0_targ = np.sum(slap[:, 9] == 0) # 1334
# LU1_targ = np.sum(slap[:, 9] == 1) # 40083
# LU2_targ = np.sum(slap[:, 9] == 2) # 5116
# LU3_targ = np.sum(slap[:, 9] == 3) # 6661
# LU4_targ = np.sum(slap[:, 9] == 4) # 6661
# LU4_targ = np.sum(slap[:, 9] == 5) # 6661
