# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 08:26:52 2021

@author: mcalderonloor
"""

import gurobipy as gp  # Gurobi optimization package
from gurobipy import GRB
import glob as glob
import numpy as np   # for matrices
import os, time, shutil
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Latitude, Longitude, Probability of being Crop (Pcrop), Pforest, Pgrassland, Purban, 
# Pother, LU_2005, LU_2010
#datafile = np.load('C:/Users/mcalderonloor/Downloads/allcoation/sla_601021010.npy')

# Get rid of nodata rows and rows where LU2015 = water
def gurobi_allocator (datafile,column_future):
    
        
    # SA2_MAIN11	 crop_next	forest_next	  grass_next   	urban_next	water_next	other_next
    # 601021010	 1368	    40132	      5124	         6702	  403  	        10
    
    LU0_targ = np.sum(datafile[:, column_future] == 0) # 1334
    LU1_targ = np.sum(datafile[:, column_future] == 1) # 40083
    LU2_targ = np.sum(datafile[:, column_future] == 2) # 5116
    LU3_targ = np.sum(datafile[:, column_future] == 3) # 6661
    LU4_targ = np.sum(datafile[:, column_future] == 4) # 6661
    
    s = LU0_targ + LU1_targ + LU2_targ + LU3_targ
    
    numcells = datafile.shape[0]
    
    try:
        
        # Create a new model
        t = gp.Model("mip1")
            
        # Create decision variables
        x0 = t.addMVar(numcells, vtype=gp.GRB.BINARY, name="LU0") 
        x1 = t.addMVar(numcells, vtype=gp.GRB.BINARY, name="LU1") 
        x2 = t.addMVar(numcells, vtype=gp.GRB.BINARY, name="LU2") 
        x3 = t.addMVar(numcells, vtype=gp.GRB.BINARY, name="LU3") 
        
        # Set up linear expressions (matrix multiplications equivalent to np.sum(a * b))
        LE0 = datafile[:, 3] @ x0
        LE1 = datafile[:, 4] @ x1
        LE2 = datafile[:, 5] @ x2
        LE3 = datafile[:, 6] @ x3
        
        # Set objective function   
        t.setObjective(LE0 + LE1 + LE2 + LE3, gp.GRB.MAXIMIZE) # Maximise probability of land-use occurence
        
        t.addConstr(x0.sum() <= LU0_targ, "L01_target")
        t.addConstr(x1.sum() <= LU1_targ, "LU1_target")
        t.addConstr(x2.sum() <= LU2_targ, "LU2_target")
        t.addConstr(x3.sum() <= LU3_targ, "LU3_target")
    
        t.addConstr(x0 + x1 + x2 + x3 <= np.repeat(1, numcells), "cell area") # check syntax
        
        #8: Optimize model
        t.optimize()
     
    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ': ' + str(e))
    
    
    except AttributeError:
        print('Encountered an attribute error')
    
    
    # Get results
    result = np.array([i.x for i in t.getVars()], dtype=np.bool).reshape(4, numcells).transpose()
    
    # Format to 1D array LU 0, 1, 2, 3
    x = np.nonzero(result)[1]
    acc = np.sum(x == datafile[:, 8]) / numcells
    # Calculate simple % accuracy
    print('Accuracy = ', acc)
    
    # Save out array with x, y, LU_2010
    outarray = np.zeros((numcells, 3))
    outarray[:, 0:2] = datafile[:, 0:2]
    outarray[:, 2] = x
    #outarray[:, 3] = datafile[:,]
    return outarray, acc
    #np.save('C:/Users/mcalderonloor/Downloads/allcoation/gurobi_output.npy', outarray)

