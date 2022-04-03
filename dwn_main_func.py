# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 08:45:43 2021

@author: mcalderonloor
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 09:59:27 2021

@author: mcalderonloor
"""
import ee
ee.Initialize()
import numpy as np
import matplotlib.pyplot as plt
from numpy import asarray
from numpy import savetxt
import pandas as pd

lc_aus = ee.Image("users/mioash/Calderon_etal_Australian_land-cover/Lc_Aus_1985_2015_v1")
tasgrid = ee.FeatureCollection("users/mioash/drivers_lcaus/boundaries/tasmania_grid_v2302")
vicgrid = ee.FeatureCollection("users/mioash/drivers_lcaus/boundaries/vic_grid_v0506")
nswgrid = ee.FeatureCollection("users/mioash/drivers_lcaus/boundaries/nsw_grid_v0506")
qldgrid = ee.FeatureCollection("users/mioash/drivers_lcaus/boundaries/qld_grid_v0506")
sagrid = ee.FeatureCollection("users/mioash/drivers_lcaus/boundaries/sa_grid_v0506")
wagrid = ee.FeatureCollection("users/mioash/drivers_lcaus/boundaries/wa_grid_v0506")
ntgrid = ee.FeatureCollection("users/mioash/drivers_lcaus/boundaries/nt_grid_v0506")
actgrid = ee.FeatureCollection("users/mioash/drivers_lcaus/boundaries/act_grid_v0506")

sa2 = ee.FeatureCollection("users/mioash/drivers_lcaus/boundaries/SA2")

def build_df (year,STT,filt_sla,suit_crop,suit_forest,suit_grass,suit_urb,two_lc,lc_prev,lc_next):

    #crop = ee.Image('users/mioash/Calderon_etal_Australian_land-cover/2010_SUIT_AUS_cropland').rename('prob_crop')
    #forest = ee.Image('users/mioash/Calderon_etal_Australian_land-cover/2010_SUIT_AUS_forest').rename('prob_forest')
    #grass = ee.Image('users/mioash/Calderon_etal_Australian_land-cover/2010_SUIT_AUS_grassland').rename('prob_grass')
    #urb = ee.Image('users/mioash/Calderon_etal_Australian_land-cover/2010_SUIT_AUS_urban').rename('prob_urban')
    crop = suit_crop.rename('prob_crop')
    forest = suit_forest.rename('prob_forest')
    grass = suit_grass.rename('prob_grass')
    urb = suit_urb.rename('prob_urban')
    
    
    ## 'New South Wales': 1
    ## 'Victoria': 2
    ## 'Queensland' : 3
    ## 'South Australia' : 4
    ## 'Western Australia' : 5
    ## "Tasmania': 6 
    ## 'Northern Territory' : 7
    ## 'Australian Capital Territory' : 8
    sa2_tas = sa2.filterMetadata('STE_NAME11','equals',STT)
    
    if STT == 'New South Wales':        
        sgrid = nswgrid
    elif STT == 'Victoria':
        sgrid = vicgrid
    elif STT == 'Queensland':
        sgrid = qldgrid
    elif STT == 'South Australia':
        sgrid = sagrid
    elif STT == 'Western Australia':
        sgrid = wagrid
    elif STT == 'Tasmania':
        sgrid = tasgrid
    elif STT == 'Northern Territory':
        sgrid = ntgrid
    elif STT == 'Australian Capital Territory':
        sgrid = actgrid
    
        
    probs = crop.addBands(forest).addBands(grass).addBands(urb)
    if two_lc:
        #probs = probs.addBands(lc_aus.select(['b2010','b2005']))
        probs = probs.addBands(lc_prev.rename('lc_prev'))
        probs = probs.addBands(lc_next.rename('lc_next'))
       
    id_look = 'SA2_MAIN11'
    
    def id_parse (FCC):
         return ee.Feature(FCC).set(id_look, ee.Number.parse(FCC.get(id_look)))
    
    sa2_tas = sa2_tas.map(id_parse)
    
    if filt_sla:
        sa2_tas = sa2_tas.filterMetadata(id_look,'equals',filt_sla)
        sgrid = sgrid.filterBounds(sa2_tas)
        indexes = ee.List(sgrid.aggregate_array('system:index'))
        ids = ee.List.sequence(1, indexes.size())
        idByIndex = ee.Dictionary.fromLists(indexes, ids)
        def add_id (feature):
            return feature.set('id_filt', idByIndex.get(feature.get('system:index')))
        sgrid= sgrid.map(add_id)

    sgridsize = sgrid.size().getInfo()
    
    sa2_painted = ee.Image(-9999).paint(sa2_tas,id_look).rename('sa2_id')
    sa2_painted = sa2_painted.clipToCollection(sgrid)
    
    probs = probs.clipToCollection(sa2_tas)
    return probs.unmask(-9999,False).addBands(sa2_painted), sgrid, sgridsize
    
def LatLonImg(img,region,two_lc):
   
    img = img.addBands(ee.Image.pixelLonLat())
    
    img = img.reduceRegion(reducer=ee.Reducer.toList(),\
                                        geometry=region,\
                                        maxPixels=1e13,\
                                        scale=30,\
                                        tileScale= 2);
 
    sa2id= np.array((ee.Array(img.get("sa2_id")).getInfo()))#.astype('i4')
    if np.max(sa2id) != -9999:        
        lats = np.array((ee.Array(img.get("latitude")).getInfo()))
        lats = (lats * 10000000).astype('i4')
        lons = np.array((ee.Array(img.get("longitude")).getInfo()))#.astype('i4')
        lons = (lons * 10000000).astype('i4')
        cr = np.array((ee.Array(img.get("prob_crop")).getInfo()))
        cr = (cr * 1000000).astype('i4')
        fr = np.array((ee.Array(img.get("prob_forest")).getInfo()))#.astype('i4')
        fr = (fr * 1000000).astype('i4')
        gr= np.array((ee.Array(img.get("prob_grass")).getInfo()))#.astype('i4')
        gr = (gr * 1000000).astype('i4')
        ur= np.array((ee.Array(img.get("prob_urban")).getInfo()))#.astype('i4')
        ur = (ur * 1000000).astype('i4')
        if two_lc:
            lprev= np.array((ee.Array(img.get("lc_prev")).getInfo())).astype('i4')
            lpred= np.array((ee.Array(img.get("lc_next")).getInfo())).astype('i4')
            arr = np.vstack([lats, lons, sa2id, cr, fr, gr, ur ,lprev, lpred]).transpose()
        else:
            arr = np.vstack([lats, lons, sa2id, cr, fr, gr, ur]).transpose()
        return arr.astype('i4')
    else:
        lats = np.array((ee.Array(img.get("latitude")).getInfo()))
        lats = (lats * 10000000).astype('i4')
        lons = np.array((ee.Array(img.get("longitude")).getInfo()))#.astype('i4')
        lons = (lons * 10000000).astype('i4')
        x = np.arange(len(lons), dtype='i4')
        null_arr = np.full_like(x, -9999)
        if two_lc:
            arr = np.vstack([lats, lons, sa2id, null_arr, null_arr, null_arr, null_arr ,null_arr, null_arr]).transpose()
        else:
            arr = np.vstack([lats, lons, sa2id, null_arr, null_arr, null_arr, null_arr ]).transpose()        
        return arr.astype('i4')

def dwn_save (stt,i_ini,i_end,sgrid,llist,probs,year,id_filt,save,two_lc):
    if two_lc:
        appended = np.zeros([1,9]).astype('i4')
    else:
        appended = np.zeros([1,7]).astype('i4')
    for i in range(i_ini,i_end,1):
        print('grid:',llist[i], 'of:', len(llist))
        if id_filt:
            sgrid1 = sgrid.filterMetadata('id_filt', 'equals',llist[i])
        else:
            sgrid1 = sgrid.filterMetadata('id_grid', 'equals',llist[i])
        probs1 = probs.clip(sgrid1)
        limg = LatLonImg(probs1,sgrid1,two_lc)
        if limg is None:
            print('Bad:',i)
        else:
            appended = np.concatenate((appended,limg),0)
    appended = appended.astype('i4')
    if save:
        if id_filt:
            np.save('E:/Marco/australia_lu_model/data/suitability/'+str(stt)+'/'+str(year)+'_suit_'+str(stt)+'_'+str(id_filt),appended)
        else:
            np.save('E:/Marco/australia_lu_model/data/suitability/'+str(stt)+'/'+str(year)+'_suit_'+str(stt)+'_'+str(i_ini)+'_'+str(i_end),appended)
    else:
        return appended