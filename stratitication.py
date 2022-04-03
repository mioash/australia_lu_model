# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 06:40:27 2020

@author: mcalderonloor
"""
import ee 

def entropy(img1,lclasses):
    modee = ee.Image(img1).reduce(ee.Reducer.mode())
    #// Map.addLayer(imc.reduce('variance'))
    total = ee.Image(img1).reduce('count')
    #// Class and counts
    fr = ee.Image(img1).reduce(ee.Reducer.autoHistogram(lclasses, 1))
    #// Just get the counts
    p = fr.arraySlice(1, 1).divide(total)
    ct=p.neq(0)
    #mask array
    p = p.arrayMask(ct)
    #// Log with custom base
    def log_b(x,base):
        return x.log().divide(ee.Number(base).log())
    H = log_b(p, 2).multiply(p).arrayReduce('sum', [0]).arrayFlatten([['ent'], ['']]).multiply(-1)
    return H#.clip(area_country)

def make_strata (img,lcl,binary):
    temporal_entropy = entropy(img,lcl).rename('entropy')
    temporal_entropy = temporal_entropy.add(ee.Image(10))
    
    strata = ee.Image(0).updateMask(temporal_entropy)
    
    if binary==False:
        strata = temporal_entropy.where(temporal_entropy.eq(10),ee.Image(100))
        strata = strata.where(temporal_entropy.gt(10).And(temporal_entropy.lt(11.1)),ee.Image(200))
        strata = strata.where(temporal_entropy.gt(11.2).And(temporal_entropy.lt(11.7)),ee.Image(300))
        strata = strata.where(temporal_entropy.gt(11.7),ee.Image(400))
    else:
        strata = temporal_entropy.where(temporal_entropy.eq(10),ee.Image(100))
        strata = strata.where(temporal_entropy.gt(10).And(temporal_entropy.lt(11)),ee.Image(200))
        strata = strata.where(temporal_entropy.gte(11),ee.Image(300))
    
    im_lc = img.select('lc_dep')
    strata = strata.add(im_lc).rename('strata')
           
    # strata = strata.add(im_lc.eq(0).add(9))
    # strata = strata.add(im_lc.eq(1).add(10))
    # strata = strata.add(im_lc.eq(2).add(11))
    # strata = strata.add(im_lc.eq(3).add(12))
    # strata = strata.add(im_lc.eq(4).add(13))
    # strata = strata.add(im_lc.eq(5).add(14)).rename('strata')
    return strata
    
def prop_allocation (img,reg,npoints):
    SAMPLE = img.stratifiedSample(
        numPoints= npoints, #// 0 points for pixel values not in 'allocation'
        classBand= 'strata', # class band name
        #classValues= ee.List([100,101,102,103,104,105,200,201,202,203,204,205,300,301,302,303,304,305,400,401,402,403,404,405]), #// pixel values
        #classPoints= [], # sample allocation 
        region=reg, 
        scale= 30, # Landsat spatial resolution
        geometries= True)
    return SAMPLE
    