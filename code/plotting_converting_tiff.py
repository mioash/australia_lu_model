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

finn_o = np.load('tas_maxprob_2010_output.npy')


cmapa = mpl.colors.ListedColormap(['yellow','green','orange', 'red','blue','gray'])
pv_arr = npy_long_to_wide(df_end,0,1,10)

gdfm, acc = gurobi_allocator(slap,9)


plt.imshow(npy_long_to_wide(finn,0,1,2),vmin=0,vmax=5,cmap=cmapa)
plt.imshow(npy_long_to_wide(df_end,0,1,10),vmin=0,vmax=5,cmap=cmapa)
plt.imshow(npy_long_to_wide(df_end,0,1,9),vmin=0,vmax=5,cmap=cmapa)
plt.imshow(npy_long_to_wide(finn_o,0,1,2),vmin=0,vmax=5,cmap=cmapa)

plt.imshow(npy_long_to_wide(dfmo,0,1,2),vmin=0,vmax=5,cmap=cmapa)


#accs_g = np.load('accuracy.npy')
#gorder = np.load('geoid_order.npy')

finn_o = np.load('tas_maxprob_2010_output.npy')
npy_to_tif(finn_o,0,1,2,'gt_2010_tas')

finn = np.load('tas_gurobi_2010_output.npy')
npy_to_tif(finn,0,1,2,'tas_gurobi_2010_tas.tif')

ac1 = np.load('accuracy.npy')
ac2 = np.load('accuracy_maxprob.npy')
