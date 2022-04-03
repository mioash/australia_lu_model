# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 09:36:39 2021

@author: mcalderonloor
"""
from main_suitability import suitability
import ee 
#from ee import batch

lc_all= ee.Image("users/mioash/Calderon_etal_Australian_land-cover/Lc_Aus_1985_2015_v1")

###2010
lc_t1 = lc_all.select('b2015')
lc_t2 = lc_all.select('b2010')
lc_t3 = lc_all.select('b2005')
lc_t4 = lc_all.select('b2000')
lc_t5 = lc_all.select('b1995')

# ##future
# lc_t1 =  ee.Image('users/mioash/ALUM/pred_aus_2010_v1')
# #lc_t1 = lc_all.select('b2005')
# lc_t2 = lc_all.select('b2005')
# lc_t3 = lc_all.select('b2000')
# lc_t4 = lc_all.select('b1995')
# lc_t5 = lc_all.select('b1990')


#neig_suit = ee.Image('users/mioash/drivers_lcaus/Distances_nigh_national_2010_byte')
#neig_suit = ee.Image('users/mioash/drivers_lcaus/Distances_nigh_national_2005_byte')
neig_suit = ee.Image('users/mioash/drivers_lcaus/Only_neigh_national_gt_2010for_pred_2015')
#train_year,pred_year,lc_class,lc_t1,lc_t2,lc_t3,lc_t4,lc_t5,neigh_suit,neigh_img,exp_ptos):

##distances - for predicting 2010 I need distances in 2005, for 2015 in 2010 (pred_year =2015), for 2020 in 2015    
##ONly for neigh
#suitability(2010,2020,'cropland',lc_t1,lc_t2,lc_t3,lc_t4,lc_t5,'neigh',neig_suit,False)

suitability(2010,2020,'cropland',lc_t1,lc_t2,lc_t3,lc_t4,lc_t5,'suitability',neig_suit,False)
suitability(2010,2020,'forest',lc_t1,lc_t2,lc_t3,lc_t4,lc_t5,'suitability',neig_suit,False)
suitability(2010,2020,'grassland',lc_t1,lc_t2,lc_t3,lc_t4,lc_t5,'suitability',neig_suit,False)
suitability(2010,2020,'urban',lc_t1,lc_t2,lc_t3,lc_t4,lc_t5,'suitability',neig_suit,False)
