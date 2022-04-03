# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 13:02:28 2021

@author: mcalderonloor
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import asarray
from numpy import savetxt
import timer
import pandas as pd

def greedy_prob_allocator (datafile,column_future):
    
    slap = datafile
    ##calculate demand based on future data // when it is available
    LU0_targ = np.sum(slap[:, column_future] == 0) 
    LU1_targ = np.sum(slap[:, column_future] == 1) 
    LU2_targ = np.sum(slap[:, column_future] == 2) 
    LU3_targ = np.sum(slap[:, column_future] == 3)
    
    LU_targs = np.array([LU0_targ,LU1_targ,LU2_targ,LU3_targ])
    #LUs_targ = np.array([LU0_targ,LU1_targ,LU2_targ,LU3_targ])
    #LU4_targ = np.sum(datafile[:, column_future] == 4) # 6661
    
    #max_index_col = np.argmax(slap[5430:5438,3:7], axis=0)
    
    ##calculate the winner on the columns of interest
    max_index_row = np.argmax(slap[:,3:7], axis=1)
    
    ##append the initial allocated pixels to the array
    n_df = np.insert(slap, np.shape(slap)[1], max_index_row, axis=1)
    
    bad_df = n_df[(n_df[:,column_future+1] == -9999) | (n_df[:,column_future] == -9999),:]
    bad_df[:,column_future+1] = -9999
    n_df = n_df[n_df[:,column_future+1] != -9999,:]
    n_df = n_df[n_df[:,column_future] != -9999,:]
    #n_df[n_df[:,column_future] == -9999,column_future] = n_df[n_df[:,column_future] == -9999,column_future-1]
    
    ##calculate the amount allocated pixels
    LU0_ini = np.sum(n_df[:, column_future+1] == 0) 
    LU1_ini = np.sum(n_df[:, column_future+1] == 1) 
    LU2_ini = np.sum(n_df[:, column_future+1] == 2) 
    LU3_ini = np.sum(n_df[:, column_future+1] == 3)
    #LUs_ini= np.array([LU0_ini,LU1_ini,LU2_ini,LU3_ini])
    
    LU_inis = np.array([LU0_ini,LU1_ini,LU2_ini,LU3_ini])
    #calculate differences between target and initial allocation
    diffs = np.array([LU_inis[0]-LU_targs[0],
                      LU_inis[1]-LU_targs[1],
                      LU_inis[2]-LU_targs[2],
                      LU_inis[3]-LU_targs[3]])
    #print(diffs)
    ##positive
    #iord= np.argsort(diffs)
    
    ##divide overestimation and underestimation
    pos_d = np.where(diffs >= 0)[0]
    neg_d = np.where(diffs < 0)[0]
    
    finn = np.full((0, 10),-9999)
    discard = np.full((0, 10),-9999)
    to_append =np.full((0, 10),-9999)
    
    ## drop the pixels in excess (where the differences were positive) based on the pixels with the highest probability
       
    for i in range(0,len(pos_d)):
        y1 = n_df[n_df[:,column_future+1] == pos_d[i],:]
        y1= y1[y1[:,3+pos_d[i]].argsort()]
        nc1 = y1[0:diffs[pos_d[i]]]
        ##replace prob with 0
        #nc1[:,3+pos_d[i]] = 0
        yc1 = y1[diffs[pos_d[i]]:len(y1)]
        ##pixels correctly allocated
        finn = np.concatenate((yc1,finn),axis=0)
        ## discarded pixels
        discard = np.concatenate((nc1,discard),axis=0)
       
    if len(pos_d) == 1:
        discard[:,3+pos_d[0]] = 0
        df_lc_pos1 = finn[finn[:,column_future+1] == pos_d[0],:]
        df_lc_neg1 = n_df[n_df[:,column_future+1] == neg_d[0],:]
        df_lc_neg1[:,3+pos_d[0]] = 0
        df_lc_neg2 = n_df[n_df[:,column_future+1] == neg_d[1],:]
        df_lc_neg2[:,3+pos_d[0]] = 0
        df_lc_neg3 = n_df[n_df[:,column_future+1] == neg_d[2],:]
        df_lc_neg3[:,3+pos_d[0]] = 0
    elif len(pos_d) == 2:
        discard[:,3+pos_d[0]] = 0
        discard[:,3+pos_d[1]] = 0
        df_lc_pos1 = finn[finn[:,column_future+1] == pos_d[0],:]
        df_lc_pos2 = finn[finn[:,column_future+1] == pos_d[1],:]
        df_lc_neg1 = n_df[n_df[:,column_future+1] == neg_d[0],:]
        df_lc_neg1[:,3+pos_d[0]] = 0
        df_lc_neg1[:,3+pos_d[1]] = 0
        df_lc_neg2 = n_df[n_df[:,column_future+1] == neg_d[1],:]
        df_lc_neg2[:,3+pos_d[0]] = 0
        df_lc_neg2[:,3+pos_d[1]] = 0
    elif len(pos_d) == 3:
        discard[:,3+pos_d[0]] = 0
        discard[:,3+pos_d[1]] = 0
        discard[:,3+pos_d[2]] = 0
        df_lc_pos1 = finn[finn[:,column_future+1] == pos_d[0],:]
        df_lc_pos2 = finn[finn[:,column_future+1] == pos_d[1],:]
        df_lc_pos3 = finn[finn[:,column_future+1] == pos_d[2],:]
        df_lc_neg1 = n_df[n_df[:,column_future+1] == neg_d[0],:]
        df_lc_neg1[:,3+pos_d[0]] = 0
        df_lc_neg1[:,3+pos_d[1]] = 0
        df_lc_neg1[:,3+pos_d[2]] = 0
        
        
    ##complete the classes that were underestimated
        
    if len(neg_d)==0:
        disc = np.full((0, 10),-9999)
    ##if only one class was underestimated
    elif len(neg_d)==1:
        mdisc = discard[discard[:,column_future+1]!=-9999,:]
        ##assign the lc_class to all the discarded pixels
        mdisc[:,column_future+1] = neg_d[0]
        # ##extract pixels belonging to lc_class that were correctly allocated in the 1st round
        # taped = n_df[n_df[:,column_future+1] == neg_d[0],:]
        # to_append = np.concatenate((taped,to_append),axis=0)
        
        disc = mdisc[mdisc[:,column_future+1]!=-9999,:]
        df_out = np.concatenate((disc,df_lc_pos1,df_lc_pos2,df_lc_pos3,df_lc_neg1,bad_df),axis=0)
        
    elif len(neg_d)==2:
        mdisc = discard[discard[:,column_future+1]!=-9999,:] #discard contains the pixels in excess
        max_index_row_1 = np.argmax(mdisc[:,3:7], axis=1) ## calculate the maximum prob among the pixels that still need to be allocated
        #mdisc = np.insert(slap, np.shape(slap)[1], max_index_row, axis=1)
        mdisc[:,column_future+1] = max_index_row_1 ##assign the winner lc class
                
        #this one contains all the pixels for the 2 underestimated LC
        temp_neg_df = np.concatenate((mdisc, df_lc_neg1, df_lc_neg2),axis=0) 
                
        #calculate differences between allocation and target 
        diffs1 = np.array([np.sum(temp_neg_df[:, column_future+1] == neg_d[0]) - np.sum(slap[:, column_future] == neg_d[0]),
                      np.sum(temp_neg_df[:, column_future+1] == neg_d[1]) - np.sum(slap[:, column_future] == neg_d[1])])
        
        new_diff = np.array([0,0,0,0])
        new_diff[pos_d[0]] = 0
        new_diff[pos_d[1]] = 0
        new_diff[neg_d[0]] = diffs1[0]
        new_diff[neg_d[1]] = diffs1[1]
        
        ##divide overestimation and underestimation
        pos_d1 = np.where(new_diff > 0)[0]
        neg_d1 = np.where(new_diff < 0)[0]
        
        if len(neg_d1)==1:
            
            last_partition  = temp_neg_df[temp_neg_df[:,column_future+1] == pos_d1[0],]
            last_partition = np.array(sorted(last_partition, key= lambda x: -x[3+pos_d1[0]]))#slap[slap[:,flu+3].argsort()]
            df_lc_pos3 = last_partition[0:LU_targs[pos_d1[0]],:]
            df_lc_pos4 = last_partition[LU_targs[pos_d1[0]]:len(last_partition),:]
            df_lc_pos4 [:,column_future+1] = neg_d1[0]
            df_lc_pos4 = np.concatenate((df_lc_pos4,
                                         temp_neg_df[temp_neg_df[:,column_future+1] == neg_d1[0],]),axis=0)
        df_out = np.concatenate((df_lc_pos1,df_lc_pos2,df_lc_pos3,df_lc_pos4,bad_df),axis=0)
        
        
    elif len(neg_d)==3:
        
        mdisc = discard[discard[:,column_future+1]!=-9999,:] #discard contains the pixels in excess
        max_index_row_1 = np.argmax(mdisc[:,3:7], axis=1) ## calculate the maximum prob among the pixels that still need to be allocated
        #mdisc = np.insert(slap, np.shape(slap)[1], max_index_row, axis=1)
        mdisc[:,column_future+1] = max_index_row_1 ##assign the winner lc class
                
        #this one contains all the pixels for the 3 underestimated LC
        temp_neg_df = np.concatenate((mdisc, df_lc_neg1, df_lc_neg2, df_lc_neg3),axis=0) 
        
        #calculate differences between allocation and target 
        diffs1 = np.array([np.sum(temp_neg_df[:, column_future+1] == neg_d[0]) - np.sum(slap[:, column_future] == neg_d[0]),
                      np.sum(temp_neg_df[:, column_future+1] == neg_d[1]) - np.sum(slap[:, column_future] == neg_d[1]),
                      np.sum(temp_neg_df[:, column_future+1] == neg_d[2]) - np.sum(slap[:, column_future] == neg_d[2])])
               
        new_diff = np.array([0,0,0,0])
        new_diff[pos_d[0]] = 0
        new_diff[neg_d[0]] = diffs1[0]
        new_diff[neg_d[1]] = diffs1[1]
        new_diff[neg_d[2]] = diffs1[2]
        
        ##divide overestimation and underestimation
        pos_d1 = np.where(new_diff >= 0)[0]
        neg_d1 = np.where(new_diff < 0)[0]
        
        if len(neg_d1)==1:            
            last_partition  = temp_neg_df[temp_neg_df[:,column_future+1] == pos_d1[0],]
            last_partition1 = temp_neg_df[temp_neg_df[:,column_future+1] == pos_d1[1],]
            #last_partition = np.concatenate((last_partition,last_partition1),axis=0)
            
            last_partition = np.array(sorted(last_partition, key= lambda x: -x[3+pos_d1[0]]))
            last_partition1 = np.array(sorted(last_partition1, key= lambda x: -x[3+pos_d1[1]]))
            
            df_lc_pos2 = last_partition[0:LU_targs[pos_d1[0]],:]
            df_lc_pos3 = last_partition1[0:LU_targs[pos_d1[1]],:]
            
            df_lc_pos4 = last_partition[LU_targs[pos_d1[0]]:len(last_partition),:]
            df_lc_pos4 [:,column_future+1] = neg_d1[0]
            df_lc_pos4a = last_partition1[LU_targs[pos_d1[1]]:len(last_partition1),:]
            df_lc_pos4a [:,column_future+1] = neg_d1[0]
            
            df_lc_pos4 = np.concatenate((df_lc_pos4,df_lc_pos4a,
                                         temp_neg_df[temp_neg_df[:,column_future+1] == neg_d1[0],]),axis=0)
            
        elif len(neg_d1)==2:
            mdisca = mdisc
            mdisca[pos_d1[0]] = 0
            mdisca[pos_d1[1]] = 0
            #mdisc = discard[discard[:,column_future+1]!=-9999,:] #discard contains the pixels in excess
            max_index_row_2 = np.argmax(mdisca[:,3:7], axis=1) ## calculate the maximum prob among the pixels that still need to be allocated
            #mdisc = np.insert(slap, np.shape(slap)[1], max_index_row, axis=1)
            mdisca[:,column_future+1] = max_index_row_2 ##assign the winner lc class
                    
            #this one contains all the pixels for the 2 underestimated LC
            temp_neg_df1 = np.concatenate((mdisca, df_lc_neg1, df_lc_neg2),axis=0) 
                    
            #calculate differences between allocation and target 
            diffs2 = np.array([np.sum(temp_neg_df1[:, column_future+1] == neg_d1[0]) - np.sum(slap[:, column_future] == neg_d1[0]),
                          np.sum(temp_neg_df1[:, column_future+1] == neg_d1[1]) - np.sum(slap[:, column_future] == neg_d1[1])])
            
            new_diff1 = np.array([0,0,0,0])
            new_diff1[pos_d1[0]] = 0
            new_diff1[pos_d1[1]] = 0
            new_diff1[neg_d1[0]] = diffs2[0]
            new_diff1[neg_d1[1]] = diffs2[1]
            
            ##divide overestimation and underestimation
            pos_d3 = np.where(new_diff1 > 0)[0]
            neg_d3 = np.where(new_diff1 < 0)[0]
            
            if len(neg_d3)==1:
                
                last_partition  = temp_neg_df1[temp_neg_df1[:,column_future+1] == pos_d3[0],]
                last_partition = np.array(sorted(last_partition, key= lambda x: -x[3+pos_d3[0]]))#slap[slap[:,flu+3].argsort()]
                df_lc_pos3 = last_partition[0:LU_targs[pos_d3[0]],:]
                df_lc_pos4 = last_partition[LU_targs[pos_d3[0]]:len(last_partition),:]
                df_lc_pos4 [:,column_future+1] = neg_d3[0]
                df_lc_pos4 = np.concatenate((df_lc_pos4,
                                             temp_neg_df1[temp_neg_df1[:,column_future+1] == neg_d3[0],]),axis=0)
            
        df_out = np.concatenate((df_lc_pos1,df_lc_pos2,df_lc_pos3,df_lc_pos4,bad_df),axis=0)
        
        
 
    df_end = df_out
    
    acc = np.sum(df_end[:, 8] == df_end[:, 9]) / df_end.shape[0]
    #print('Accuracy = ', acc)
    #bad_df = n_df[n_df[:,column_future+1] == -9999,:]
    df_confusion = pd.crosstab(pd.Series(df_end[df_end[:,column_future] != -9999, column_future], name="Actual"), pd.Series(df_end[df_end[:,column_future] != -9999, column_future+1], name="Predicted"))
    #print (df_confusion)
    
    # Save out array with x, y, LU_2010
    outarray = np.zeros((len(df_end), 5))
    outarray[:, 0:2] = df_end[:, 0:2] ##lat, lon
    outarray[:, 2] = df_end[:, column_future] #ground truth
    outarray[:, 3] = df_end[:,column_future+1] #predicted
    outarray[:, 4] = df_end[:,2] #geoid
    #outarray[:, 3] = datafile[:,]
    return outarray, acc, df_confusion#, len(pos_d),len(neg_d)
    
    