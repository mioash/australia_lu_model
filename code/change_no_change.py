# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 12:13:33 2020

@author: mcalderonloor
"""

import ee 
from stratitication import entropy

#lc_all= ee.Image("users/mioash/Calderon_etal_Australian_land-cover/Lc_Aus_1985_2015_v1")
#ent_all = ee.Image('users/mioash/Calderon_etal_Australian_land-cover/Australia_ent_v1')
#mov_all = ee.Image('users/mioash/Calderon_etal_Australian_land-cover/margin_of_victory')


def suit_layer(lc_img,mode,c_by_c,lc_class):
    
    if mode=='two_steps':
        temporal_entropy = entropy(lc_img.select(['lc_dep','lc_t1'])).rename('entropy')
        cnc = temporal_entropy.where(temporal_entropy.neq(0),ee.Image(1))
        return cnc.rename('lc_dep')
        #cnc = lc_img.select('lc_dep').subtract(lc_img.select('lc_t1'))
        #cnc = cnc.where(cnc.neq(0),1).rename('lc_change')#.clip(geom)
    elif mode=='mult_steps':
        temporal_entropy = entropy(lc_img).rename('entropy')
        #temporal_entropy = temporal_entropy.add(ee.Image(10))
        #strata = ee.Image(0).updateMask(temporal_entropy)
        cnc = temporal_entropy.where(temporal_entropy.neq(0),ee.Image(1))
        return cnc.rename('lc_dep')
    elif mode=='binary':
        cnc = lc_img
    
    if c_by_c==True:
        
        lc_bands = lc_img.bandNames().getInfo()
        #print(len(lc_bands))
        lc_img0 = lc_img.select('lc_dep').add(ee.Image(1))        
        lc_img0 = lc_img0.where(lc_img0.neq(lc_class+1),0)
        lc_img0 = lc_img0.where(lc_img0.eq(lc_class+1),1)
        if mode=='binary':
            for i in range(1,len(lc_bands)):
                lc_img1 = lc_img.select(lc_bands[i]).add(ee.Image(1))        
                lc_img1 = lc_img1.where(lc_img1.neq(lc_class+1),0)
                lc_img1 = lc_img1.where(lc_img1.eq(lc_class+1),1)
                lc_img0 = lc_img0.addBands(lc_img1)
                #print(i)
                #print(lc_img0.bandNames().getInfo())
        
        return lc_img0
        #cnc = lc_img0
            
    #if c_by_c==True:
        
    #elif c_by_c==False & mode!= 'binary':   
        #return cnc.rename('lc_dep')
    