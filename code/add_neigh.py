# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 08:21:37 2020

@author: mcalderonloor
"""
import ee

SCALE = 30

def weight_maker (size,size_centre,blank):
    list_1 = ee.List.repeat(1,size)
    if blank==True:
        list_centre = ee.List.repeat(0,size_centre)
        list_centre = ee.List.repeat(1,ee.Number((size - size_centre)/2)).cat(list_centre).cat(ee.List.repeat(1,ee.Number((size - size_centre)/2)))
        list_up = ee.List.repeat(list_1,(size-size_centre)/2)
        list_cen = ee.List.repeat(list_centre,size_centre)
        list_all = list_up.cat(list_cen).cat(list_up)
    else:
        list_all = ee.List.repeat(list_1,size)
    return list_all
      
def add_neigh(image,size,size_centre,blank,suffix):
    kernel = ee.Kernel.fixed(size, size, weight_maker(size,size_centre,blank), (2-size), (2-size), False)
    neigh_bands = image.neighborhoodToBands(kernel).reproject('EPSG:4326', None, SCALE)
    img_entropy = image.entropy(kernel).rename('entropy'+str(size))
    counts = neigh_bands.reduce(ee.Reducer.fixedHistogram(0,6,6)) 
    frequency = counts.arraySlice(1, 1).arrayReduce('sum', [1]).arrayFlatten([['c_'+str(size)+str(suffix), 'f_'+str(size)+str(suffix), 'g_'+str(size)+str(suffix), 'b_'+str(size)+str(suffix), 'w_'+str(size)+str(suffix), 
                                                                               'o_'+str(size)+str(suffix)],['']],'').divide((size*size) - (size_centre*size_centre)).multiply(100)
    #return frequency.addBands(img_entropy).reproject('EPSG:4326', None, SCALE)
    return frequency.reproject('EPSG:4326', None, SCALE)