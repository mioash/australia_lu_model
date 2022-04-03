# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 11:22:17 2021

@author: mcalderonloor
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from numpy_to_tiff import npy_long_to_wide
from numpy_to_tiff import npy_to_tif

#finn_o = np.load('tas_maxprob_2010_output.npy')

# tas_prev = np.load('C:/Users/mcalderonloor/Documents/model_lu_australia/australia_lu_model/data/tas_maxprob_2010_output.npy')
# tas_new = np.load('C:/Users/mcalderonloor/Documents/model_lu_australia/australia_lu_model/data/df_new_all_urban.npy')
# tas_rec = np.concatenate((tas_new,tas_discard[1:,[0,1,7,7,2]]),axis=0)

# acc_prev = np.load('C:/Users/mcalderonloor/Documents/model_lu_australia/australia_lu_model/data/accuracy_maxprob_old.npy')
# acc_new = np.load('C:/Users/mcalderonloor/Documents/model_lu_australia/australia_lu_model/data/accuracy_maxprob_new.npy')
# diffsa = acc_new - acc_prev

dfs = np.load('E:/Marco/australia_lu_model/data/allocation/tas1/2010_tas.npy')
dfs[:,0] = dfs[:,0] / 10000000
dfs[:,1] = dfs[:,1] / 10000000

geoid = np.load('E:/Marco/australia_lu_model/data/allocation/vic1/2010_geoid_order_vic.npy')

cmapa = mpl.colors.ListedColormap(['yellow','green','orange', 'red','blue','gray'])
#pv_arr = npy_long_to_wide(tas_npy,0,1,10)

#gdfm, acc = gurobi_allocator(slap,9)

xx = npy_long_to_wide(tas_new[tas_new[:,4] == geoid[ind_geo]],0,1,3)

ind_geo = 4
plt.imshow(npy_long_to_wide(tas_new[tas_new[:,4] == geoid[ind_geo]],0,1,3),vmin=0,vmax=5,cmap=cmapa)
plt.imshow(npy_long_to_wide(tas_prev[tas_prev[:,4] == geoid[ind_geo]],0,1,2),vmin=0,vmax=5,cmap=cmapa)

plt.imshow(npy_long_to_wide(df_end,0,1,10),vmin=0,vmax=5,cmap=cmapa)
plt.imshow(npy_long_to_wide(df_end,0,1,9),vmin=0,vmax=5,cmap=cmapa)
plt.imshow(npy_long_to_wide(slap,0,1,7),vmin=0,vmax=5,cmap=cmapa)

plt.imshow(npy_long_to_wide(dfmo,0,1,2),vmin=0,vmax=5,cmap=cmapa)


#accs_g = np.load('accuracy.npy')
#gorder = np.load('geoid_order.npy')
#finn_o = np.load('tas_maxprob_2010_output.npy')
npy_to_tif(dfs,0,1,3,'E:/Marco/australia_lu_model/data/allocation/tas1/pred_2010_tas_v1.tiff')  #predicted is on number 3



finn = np.load('tas_gurobi_2010_output.npy')
npy_to_tif(finn,0,1,2,'tas_gurobi_2010_tas.tif')

ac1 = np.load('accuracy.npy')
ac2 = np.load('accuracy_maxprob.npy')
