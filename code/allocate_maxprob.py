# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 14:24:59 2021

@author: mcalderonloor
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import asarray
from numpy import savetxt
import timer

def max_prob_allocator (datafile,column_future):
    
    slap = datafile
    ##calculate demand based on future data // when it is available
    LU0_targ = np.sum(slap[:, column_future] == 0) 
    LU1_targ = np.sum(slap[:, column_future] == 1) 
    LU2_targ = np.sum(slap[:, column_future] == 2) 
    LU3_targ = np.sum(slap[:, column_future] == 3)
    #LUs_targ = np.array([LU0_targ,LU1_targ,LU2_targ,LU3_targ])
    #LU4_targ = np.sum(datafile[:, column_future] == 4) # 6661
    
    #max_index_col = np.argmax(slap[5430:5438,3:7], axis=0)
    
    ##calculate the winner on the columns of interest
    max_index_row = np.argmax(slap[:,3:6], axis=1)
    
    ##append the initial allocated pixels to the array
    n_df = np.insert(slap, np.shape(slap)[1], max_index_row, axis=1)
    
    ##calculate the amount allocated pixels
    LU0_ini = np.sum(n_df[:, column_future+1] == 0) 
    LU1_ini = np.sum(n_df[:, column_future+1] == 1) 
    LU2_ini = np.sum(n_df[:, column_future+1] == 2) 
    LU3_ini = np.sum(n_df[:, column_future+1] == 3)
    #LUs_ini= np.array([LU0_ini,LU1_ini,LU2_ini,LU3_ini])
    
    #calculate differences between target and initial allocation
    diffs = np.array([LU0_ini-LU0_targ,LU1_ini-LU1_targ,LU2_ini-LU2_targ,LU3_ini-LU3_targ])
    #print(diffs)
    ##positive
    iord= np.argsort(diffs)
    
    ##divide overestimation and underestimation
    pos_d = np.where(diffs >= 0)[0]
    neg_d = np.where(diffs < 0)[0]

    
    finn = np.full((0, 10),-9999)
    discard = np.full((0, 10),-9999)
    to_append =np.full((0, 10),-9999)
    
    
    # for i in range(0,len(pos_d)):
    #     y1 = n_df[n_df[:,10] == pos_d[i],:]
    #     y1= y1[y1[:,3+pos_d[i]].argsort()]
    #     nc1 = y1[0:diffs[pos_d[i]]]
    #     yc1 = y1[diffs[pos_d[i]]:len(y1)]
    #     finn = np.concatenate((yc1,finn),axis=0)
    #     discard = np.concatenate((nc1,discard),axis=0)
    
    # for i in range(0, len(neg_d)):
    #     if i ==0:
    #         mdisc = discard[discard[:,3+neg_d[i]].argsort()]
    #         mdisc[len(mdisc)-abs(diffs[neg_d[i]]):len(mdisc),10] = neg_d[i]
    #         mdisc[0:len(mdisc)-abs(diffs[neg_d[i]])-1,10] = -9999
        
    #     taped = n_df[n_df[:,column_future+1] == neg_d[i],:]
    #     to_append = np.concatenate((taped,to_append),axis=0)
    #     disc = mdisc[mdisc[:,10]!=-9999,:]
        
    #     if(len(neg_d)>1):
    #         mdisc1 = mdisc[mdisc[:,10]==-9999,:]
    #         mdisc1 = mdisc1[mdisc1[:,3+neg_d[i]].argsort()]
    #         mdisc1[len(mdisc1)-abs(diffs[i]):len(mdisc1),10] = neg_d[i]
    #         disc= np.concatenate((disc,mdisc1),axis=0)
    #breakpoint()
    ## drop the pixels in excess (where the differences were positive) based on the pixels with the highest probability
    for i in range(0,len(pos_d)):
        y1 = n_df[n_df[:,column_future+1] == pos_d[i],:]
        y1= y1[y1[:,3+pos_d[i]].argsort()]
        nc1 = y1[0:diffs[pos_d[i]]]
        yc1 = y1[diffs[pos_d[i]]:len(y1)]
        ##pixels correctly allocated
        finn = np.concatenate((yc1,finn),axis=0)
        ## discarded pixels
        discard = np.concatenate((nc1,discard),axis=0)
    ##complete the classes that were underestimated
    ##if only one class was underestimated
    if len(neg_d)==1:
        mdisc = discard[discard[:,column_future+1]!=-9999,:]
        ##assign the lc_class to all the discarded pixels
        mdisc[:,column_future+1] = neg_d[0]
        
        ##extract pixels belonging to lc_class that were correctly allocated in the 1st round
        taped = n_df[n_df[:,column_future+1] == neg_d[0],:]
        to_append = np.concatenate((taped,to_append),axis=0)
        
        disc = mdisc[mdisc[:,column_future+1]!=-9999,:]
    ##if two classes was underestimated
    elif len(neg_d)==2:
        ##choose the first underestimated lc_class and sort it (from min to max)
        mdisc = discard[discard[:,3+neg_d[0]].argsort()]
        ##pick the pixels with the highest prob of being this lc_class (from the end because is sorted in ascendet order)
        mdisc[len(mdisc)-abs(diffs[neg_d[0]]):len(mdisc),column_future+1] = neg_d[0]
        ##set aside the pixels that were allocaed for the first underestimated lc_class
        mdisc0 = mdisc[mdisc[:,column_future+1] == neg_d[0],:]
        ##fill the rest of the pixels with -9999
        mdisc[0:len(mdisc)-abs(diffs[neg_d[0]]),column_future+1] = -9999
        
        ##set aside pixels that still need to be re-allocated
        mdisc1 = mdisc[mdisc[:,column_future+1]==-9999,:]
        ##fill all these pixels with value for lc_class 2
        mdisc1[:,column_future+1] = neg_d[1]
        
        disc = np.concatenate((mdisc0,mdisc1),axis=0)
        disc = disc[disc[:,column_future+1]!=-9999,:]
        #print('len discarcded',len(disc))
        #print('neg1',neg_d[0])
        #print('neg 2',neg_d[1])
        taped = n_df[n_df[:,column_future+1] == neg_d[0],:]
        taped1 = n_df[n_df[:,column_future+1] == neg_d[1],:]
        to_append = np.concatenate((taped,taped1,to_append),axis=0)
        #print('len to_append',len(taped))
        #print('len to_append',len(taped1))
        
    elif len(neg_d)==3:
        mdisc = discard[discard[:,3+neg_d[0]].argsort()]
        mdisc[len(mdisc)-abs(diffs[neg_d[0]]):len(mdisc),column_future+1] = neg_d[0]
        mdisc[0:len(mdisc)-abs(diffs[neg_d[0]]),column_future+1] = -9999
        mdisc0 = mdisc[mdisc[:,column_future+1] == neg_d[0],:]
        
        mdisc1 = mdisc[mdisc[:,column_future+1]==-9999,:]
        mdisc1 = mdisc1[mdisc1[:,3+neg_d[1]].argsort()]
        mdisc1[len(mdisc1)-abs(diffs[neg_d[1]]):len(mdisc1),column_future+1] = neg_d[1]
        mdisc1[0:len(mdisc1)-abs(diffs[neg_d[1]]),column_future+1] = -9999
        mdisc1a = mdisc1[mdisc1[:,column_future+1] == neg_d[1],:]
        
        mdisc2 = mdisc1[mdisc1[:,column_future+1]==-9999,:]
        mdisc2[:,column_future+1] = neg_d[2]
                     
        disc = np.concatenate((mdisc0,mdisc1a,mdisc2),axis=0)
        disc = disc[disc[:,column_future+1]!=-9999,:]
        
        taped = n_df[n_df[:,column_future+1] == neg_d[0],:]
        taped1 = n_df[n_df[:,column_future+1] == neg_d[1],:]
        taped2 = n_df[n_df[:,column_future+1] == neg_d[2],:]
        to_append = np.concatenate((taped,taped1,taped2,to_append),axis=0)
        
        # disc = np.full((0, 11),-9999)
        # for i in range(0, len(neg_d)):
        #     if i ==0:
        #         mdisc = discard[discard[:,3+neg_d[i]].argsort()]
        #         mdisc[len(mdisc)-abs(diffs[neg_d[i]]):len(mdisc),column_future+1] = neg_d[i]
        #         mdisc[0:len(mdisc)-abs(diffs[neg_d[i]])-1,10] = -9999
        #     else:
        #         mdisc = mdisc[mdisc[:,10]==-9999,:]
        #         mdisc = mdisc[mdisc[:,3+neg_d[i]].argsort()]
        #         mdisc[len(mdisc)-abs(diffs[neg_d[i]]):len(mdisc),10] = neg_d[i]
        #     disc= np.concatenate((disc,mdisc),axis=0)
        #     taped = n_df[n_df[:,column_future+1] == neg_d[i],:]
        #     to_append = np.concatenate((taped,to_append),axis=0)    
    ##concatenate all the pervious arrays to build the final df
    df_end= np.concatenate((finn,disc,to_append),axis=0)
    
    acc = np.sum(df_end[:, 8] == df_end[:, 9]) / df_end.shape[0]
    
    print('Accuracy = ', acc)
    
    # Save out array with x, y, LU_2010
    outarray = np.zeros((len(df_end), 5))
    outarray[:, 0:2] = df_end[:, 0:2] ##lat, lon
    outarray[:, 2] = df_end[:, column_future] #ground truth
    outarray[:, 3] = df_end[:,column_future+1] #predicted
    outarray[:, 4] = df_end[:,2] #geoid
    #outarray[:, 3] = datafile[:,]
    return outarray, acc#, len(pos_d),len(neg_d)
    
    