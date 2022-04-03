# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 14:45:28 2021

@author: mcalderonloor
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

"""
import os
from change_credentials import change_credentials

uuser = 'mioash'
STTs = ['Queensland']
project = True

year = 2020
change_credentials(uuser)
os.chdir('C:/Users/mcalderonloor/Documents/lc_model/australia_lu_model/code')
import numpy as np
import pandas as pd
from allocate_maxprob import max_prob_allocator
import gc
from numpy_to_tiff import npy_to_tif
from numpy_to_tiff import npy_long_to_wide
from glob import glob
from joblib import Parallel, delayed
from joblib.externals.loky import set_loky_pickler
from joblib import parallel_backend
#from joblib import Parallel, delayed
from joblib import wrap_non_picklable_objects

import ee
ee.Initialize()
import numpy as np

#from dwn_main_func import build_df
#from dwn_main_func import dwn_save
#from dwn_main_func import LatLonImg
from dwn_main_func_float import build_df
from dwn_main_func_float import dwn_save

# var imc = ee.ImageCollection([image3,image4])
# imc = imc.reduce(ee.Reducer.max())


two_lc=True
demand = pd.read_csv('E:/Marco/australia_lu_model/data/demand_qld_v0.csv')
#ext_sla=True
sla_ids = pd.read_csv('E:/Marco/australia_lu_model/data/sla_ids.csv')
if uuser=='mioash':
    suit_crop = ee.Image('users/'+str(uuser)+'/Calderon_etal_Australian_land-cover/'+str(year)+'_SUIT_AUS_v3_25k_150trees_cropland')
    #users/mioash/Calderon_etal_Australian_land-cover/2010_SUIT_AUS_v25k_forest
    suit_forest = ee.Image('users/'+str(uuser)+'/Calderon_etal_Australian_land-cover/'+str(year)+'_SUIT_AUS_v3_25k_150trees_forest')
    suit_grass = ee.Image('users/'+str(uuser)+'/Calderon_etal_Australian_land-cover/'+str(year)+'_SUIT_AUS_v3_25k_150trees_grassland')
    suit_urb = ee.Image('users/'+str(uuser)+'/Calderon_etal_Australian_land-cover/'+str(year)+'_SUIT_AUS_v3_25k_150trees_urban')
else:
    suit_crop = ee.Image('users/'+str(uuser)+'/'+str(year)+'_SUIT_AUS_v3_25k_150trees_cropland')
    #users/mioash/Calderon_etal_Australian_land-cover/2010_SUIT_AUS_v25k_forest
    suit_forest = ee.Image('users/'+str(uuser)+'/'+str(year)+'_SUIT_AUS_v3_25k_150trees_forest')
    suit_grass = ee.Image('users/'+str(uuser)+'/'+str(year)+'_SUIT_AUS_v3_25k_150trees_grassland')
    suit_urb = ee.Image('users/'+str(uuser)+'/'+str(year)+'_SUIT_AUS_v3_25k_150trees_urban')

lc_aus = ee.Image("users/mioash/Calderon_etal_Australian_land-cover/Lc_Aus_1985_2015_v1")   
 
#lc_prev = ee.Image('users/mioash/ALUM/pred_aus_2010_v1')
lc_prev = lc_aus.select('b2015')

lc_next = lc_aus.select('b2015')
#i = 0

loc_past = 7 #7 for the previous, 8 for current
loc_future = 8 #8 for the previous, 7 for current

def main_process (stt,slas,loc_past,loc_future,demand,iq,ind_grid):
    
    if ind_grid:
        sla_id = int(slas[slas == ind_grid])
        #sla_id = int(slos[0])#int(slos[slos == 601061035])
    else:
        sla_id = int(slas[iq])
        
    print('sla_id:',sla_id)
    if project:        
        demand = demand.loc[demand.period==year]
        demand = demand.loc[demand.SA2_MAIN11==sla_id]
        
    #build_df (year,STT,filt_sla,suit_crop,suit_forest,suit_grass,suit_urb,two_lc,lc_prev,lc_next):
    probs, sgrid, sgridsize = build_df(year,STT,sla_id,suit_crop,suit_forest,suit_grass,suit_urb,two_lc,lc_prev,lc_next)
    print('grid_size:',sla_id,sgridsize)
    llist = list(range(1,sgridsize+1))
    if ind_grid:
        #(stt,i_ini,i_end,sgrid,llist,probs,year,id_filt,save,two_lc)
        ##if name == main, then
        slap = Parallel(n_jobs=30,prefer='threads')(delayed(dwn_save)(stt,iq,iq+1,sgrid,llist,probs,year,sla_id,False,two_lc) for iq in range(0,sgridsize))
        #slap = Parallel(n_jobs=-1)(delayed(dwn_save)(stt,iq,iq+1,sgrid,llist,probs,year,sla_id,False,two_lc) for iq in range(0,sgridsize))
        slap = np.vstack(slap)
    else:
        slap = dwn_save (stt,0,sgridsize,sgrid,llist,probs,year,sla_id,False,two_lc)
    
    slap= slap[(slap[:, 2] != 0) &
                          (slap[:, loc_past] != 4) &
                          (slap[:, loc_past] != 5) & 
                          (slap[:, loc_future] != 4) &
                          (slap[:, loc_future] != 5),:]
    ################
    
    #sla1 = np.load('E:/Marco/australia_lu_model/data/suitability/nt/2010_suit_nt_702031057.npy')
    
    nan_df = slap[(slap[:,3] < 0) & 
                  (slap[:,4] < 0) &
                  (slap[:,5] < 0) &
                  (slap[:,6] < 0),:]
    nan_df = np.insert(nan_df,loc_future+1,255,axis=1) 
    
    slap = slap[(slap[:,3] >= 0) & 
                  (slap[:,4] >= 0) &
                  (slap[:,5] >= 0) &
                  (slap[:,6] >= 0),:]
    
    #dfmo, acco, conf = max_prob_allocator(slap,loc_future)
    
    from allocate_maxprob_cummulative import max_prob_allocator_v1
    if project:
        dfmo1 = max_prob_allocator_v1(slap,loc_future,True,demand)
    else:
        dfmo1, acco1, conf1 = max_prob_allocator_v1(slap,loc_future,True,demand)
    
#    from allocate_maxprob_cum2 import max_prob_allocator_v2
    #datafile,flu,slu,tlu,clu,column_future
    # import matplotlib as mpl
    # import matplotlib.pyplot as plt
    # cmapa = mpl.colors.ListedColormap(['yellow','green','orange', 'red','blue','gray'])    
    
    # from allocate_greedyprob import greedy_prob_allocator
    # dfmo3, acco3, conf3 = greedy_prob_allocator(slap,loc_future)
    # plt.imshow(npy_long_to_wide(dfmo, 0, 1, 3),vmin=0,vmax=5,cmap=cmapa)
    # plt.imshow(npy_long_to_wide(dfmo1, 0, 1, 3),vmin=0,vmax=5,cmap=cmapa)
    # plt.imshow(npy_long_to_wide(dfmo3, 0, 1, 3),vmin=0,vmax=5,cmap=cmapa)

    
    #pivoted_arr3 = npy_long_to_wide(dfmo, 0, 1, 3)
    
#    import matplotlib as mpl
#    import matplotlib.pyplot as plt
    
    #plt.imshow(npy_long_to_wide(dfmo, 0, 1, 3),vmin=0,vmax=5,cmap=cmapa)
    #plt.imshow(npy_long_to_wide(dfmo1, 0, 1, 3),vmin=0,vmax=5,cmap=cmapa)
    
    
    # finn_a = np.concatenate((dfmo,nan_df[:,[0,1,loc_future,loc_future+1,2]]),axis=0)
    # finn_a[:,0] = finn_a[:,0] #/ 10000000
    # finn_a[:,1] = finn_a[:,1] #/ 10000000
    # npy_to_tif(finn_a,0,1,3,'E:/Marco/australia_lu_model/data/allocation/'+str(stt)+'/'+str(year)+'/pred_'+str(year)+'_'+str(sla_id)+'_v3.tiff')  #predicted is on number 3
    
    finn_a1 = np.concatenate((dfmo1,nan_df[:,[0,1,loc_future,loc_future+1,2]]),axis=0)
    finn_a1[:,0] = finn_a1[:,0] #/ 10000000
    finn_a1[:,1] = finn_a1[:,1] #/ 10000000
    #npy_to_tif(finn_a1,0,1,3,'E:/Marco/australia_lu_model/data/allocation/'+str(stt)+'/'+str(year)+'/pred_'+str(year)+'_'+str(sla_id)+'_v3_test_stable.tiff')  #predicted is on number 3
    npy_to_tif(finn_a1,0,1,3,'Z:/Marco-Rodrigo/allocation/'+str(stt)+'/'+str(year)+'/pred_'+str(year)+'_'+str(sla_id)+'_v3_test_stable.tiff')  #predicted is on number 3
    
    # finn_a2 = np.concatenate((dfmo3,nan_df[:,[0,1,loc_future,loc_futu re+1,2]]),axis=0)
    # finn_a2[:,0] = finn_a2[:,0] #/ 10000000
    # finn_a2[:,1] = finn_a2[:,1] #/ 10000000
    # npy_to_tif(finn_a2,0,1,3,'E:/Marco/australia_lu_model/data/allocation/'+str(stt)+'/'+str(year)+'b/pred_'+str(year)+'_'+str(sla_id)+'_v2_test_greedy.tiff')  #predicted is on number 3
    
    diffs1 = np.array([np.sum(dfmo1[:, 3] == 0) - np.sum(slap[:, loc_future] == 0),
                      np.sum(dfmo1[:, 3] == 1) - np.sum(slap[:, loc_future] == 1),
                      np.sum(dfmo1[:, 3] == 2) - np.sum(slap[:, loc_future] == 2),
                      np.sum(dfmo1[:, 3] == 3) - np.sum(slap[:, loc_future] == 3)])
    # print('Acc_norm',sla_id, acco)
    if project:
        diffs1 = np.array([np.sum(dfmo1[:, 3] == 0) - demand.crop,
                      np.sum(dfmo1[:, 3] == 1) - demand.forest,
                      np.sum(dfmo1[:, 3] == 2) - demand.grass,
                      np.sum(dfmo1[:, 3] == 3) - demand.urban])
        print('stable',sla_id,diffs1)
    else:
        print('Acc_stable',sla_id, acco1)
        # print('Acc_greedy',sla_id, acco3)
        print('stable',sla_id,diffs1)
        #conf1.to_csv('E:/Marco/australia_lu_model/data/allocation/'+str(stt)+'/'+str(year)+'a/'+str(year)+'_cfa_'+str(sla_id)+'.csv')
        conf1.to_csv('Z:/Marco-Rodrigo/allocation/'+str(stt)+'/'+str(year)+'a/'+str(year)+'_cfa_'+str(sla_id)+'.csv')    
    
    # diffs2 = np.array([np.sum(dfmo3[:, 3] == 0) - np.sum(slap[:, loc_future] == 0),
    #                   np.sum(dfmo3[:, 3] == 1) - np.sum(slap[:, loc_future] == 1),
    #                   np.sum(dfmo3[:, 3] == 2) - np.sum(slap[:, loc_future] == 2),
    #                   np.sum(dfmo3[:, 3] == 3) - np.sum(slap[:, loc_future] == 3)])
    # print('greedy',sla_id,diffs2)
    
    # del finn_a
    # del dfmo
    del nan_df
    del slap
    
    gc.collect()
    # conf.to_csv('E:/Marco/australia_lu_model/data/allocation/'+str(stt)+'/'+str(year)+'/'+str(year)+'_cf_'+str(sla_id)+'.csv')    
    
    # conf3.to_csv('E:/Marco/australia_lu_model/data/allocation/'+str(stt)+'/'+str(year)+'b/'+str(year)+'_cfb_'+str(sla_id)+'.csv')    

for ist in range(0,len(STTs)):
    STT = STTs[0]
    slos = sla_ids.loc[sla_ids['STE_NAME11'] == STT,"SA2_MAIN11"]
    slos = np.array(slos)    
    slas1 = slos#[1:20]
    if STT == 'New South Wales':        
        stt = 'nsw'
        rem_sla = [105011092,105011093,105011094,105011095,105011096,105021098,109021177,109021178,109021179,110031196,113011257,
                   103021062,103011061,101031015,101011006,105031099,105031105,106041129,108011152]
        for i in range(0,len(rem_sla)):    
            slas1 =slas1[slas1 != rem_sla[i]]
        slas1 = slas1[slas1 != 108031161]
        ext_sla = 1
    elif STT == 'Victoria':
        stt = 'vic'
        rem_sla = [205021080,204031072,216021412,215021398,215011390,205021085,205021082,215031400,202031033,215011394,217011420,
                   204031072,215021398,215011390,205021085,205021082]
        for i in range(0,len(rem_sla)):    
            slas1 =slas1[slas1 != rem_sla[i]]
        ext_sla = 1
    elif STT == 'Queensland':
        stt = 'qld'
        rem_sla = [315021404,315031411,315031410,315021406,315021407,318011463,315031408,315011396,315011397,307011171,307011177,307011178,
                   308011190,308011190,308021194,308031219,312011338]    #315011397
        for i in range(0,len(rem_sla)):    
            slas1 =slas1[slas1 != rem_sla[i]]
        ext_sla = 1
    elif STT == 'South Australia':
        stt = 'sa'
        rem_sla = [406021140,406021141,406011135,406021138,407031161,406011134,406011131,406011130,406011132]    ##faltan todos (aumentar una grande de ~73 grids)
        for i in range(0,len(rem_sla)):            
            slas1 =slas1[slas1 != rem_sla[i]]
        ext_sla = 1
    elif STT == 'Western Australia':
        stt = 'wa'
        rem_sla = [508031203,508051215,508061219,508041208,508041207,508021197,508031202,509031247,508051217,508041209,509021242,
                   508011195,508061218,508041206] ##faltan los 4 primeros
        for i in range(0,len(rem_sla)):    
            slas1 =slas1[slas1 != rem_sla[i]]
        ext_sla = 1
    elif STT == 'Tasmania':
        stt = 'tas'
        rem_sla = [602031061, 602031063, 602031064, 603011065, 603011068,603031074, 604031094,604031098,604031097,602031061,602031063,603011065,603011068]
        for i in range(0,len(rem_sla)):    
            slas1 =slas1[slas1 != rem_sla[i]]
        ext_sla = 1
    elif STT == 'Northern Territory':
        stt = 'nt'
        rem_sla = [702011050,702011053,702021055,702011052,702011054,702051068,702051065,702051066,702051067,702041063, #702051064,
                   702031058,702031057,702031061,702031060] ##add one with 74, 58 and 64 grids (702051067, 702051064,702051066,702051066,702051066)
        for i in range(0,len(rem_sla)):    
            slas1 =slas1[slas1 != rem_sla[i]]
        ext_sla = 1
    elif STT == 'Australian Capital Territory':
        stt = 'act'
        ext_sla = 0
    
    try:
        Parallel(n_jobs=30,prefer='threads')(delayed(main_process)(stt,slas1,loc_past,loc_future,demand,iq,False) for iq in range(0,len(slas1)))#range(0,len(slas1)))
    # if __name__ == '__main__':
    #     Parallel(n_jobs=-1)(delayed(main_process)(stt,slas1,loc_past,loc_future,iq,False) for iq in range(0,len(slas1)))
    #rem_sla = [127011505]
    except Exception as error:
            if str(error) == 'User memory limit exceeded.' or str(error) == 'Computation timed out.' or str(error) == 'An internal error has occurred.':
                print(error,f'going to sleep for 2 second')                
                for qqj in range(0,len(slas1)):
                    if os.path.isfile('Z:/Marco-Rodrigo/allocation/'+str(stt)+'/'+str(year)+'/pred_'+str(year)+'_'+str(slas1[qqj])+'_v3_test_stable.tiff'):
                        a=1
                        #print('exists')#main_process(stt,slas1,loc_past,loc_future,0,slas1[qqj])
                    else:
                        #print(qqj,'to_dwn')
                        main_process(stt,slas1,loc_past,loc_future,demand,0,slas1[qqj])
                #rem_sla = [113011257]            
                if ext_sla == 1:
                    for j in range(0,len(rem_sla)):#len(rem_sla)
                        if os.path.isfile('Z:/Marco-Rodrigo/allocation/'+str(stt)+'/'+str(year)+'/pred_'+str(year)+'_'+str(rem_sla[j])+'_v3_test_stable.tiff'):
                            a=1
                        else:
                            main_process(stt,slos,loc_past,loc_future,demand,0,rem_sla[j])

    
        ##to download now:
#315021404,315031411



##separately
#big (drop from main list)
#wa = [508031203,508051215,508061219,508041208,508041207,508021197,508031202]
#sa = [406021141,406011135,406021138]
#nt = [702011050,702011053,702021055,702011052,702011054,702051068,702051065,702051066]
#qld = [315021404,315031411,315031410,315021406,315021407,318011463,315031408,315011396]
#nsw = [105011092,105011093,105011094,105011095,105011096,105021098,109021177,109021178,109021179,110031196,113011257]
#vic = [205021080,204031072,216021412,215021398,215011390,205021085,205021082,215031400]
# dwn_nsw = [127031522,127031524,128021532,128021533,128021534,128021535,128021537,128021538,102011028,102011029,102011030,102011031,102011032,102011033,102011034,
#            102011035,102011036,102011037,102011038,102011039,102011040,102011041,102011042,102011043,102021044,102021047,102021051,117011325,117021328,119011353,
#            119011356,119011358,119011359,119011360,119021362,119021363,119021364,119021365,119021366,119021367,119031372,119041375,119041377,121021404,122021420,
#            122021421,122031432,123021439,123021442,105011094,105031099,106041129,108021156,108021157,108021158,108021159,111021218]
