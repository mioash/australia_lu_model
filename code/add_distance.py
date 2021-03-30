# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 10:11:40 2020

@author: mcalderonloor
"""

"""
Created on Thu Sep 24 06:40:27 2020

@author: mcalderonloor
"""
import ee 

def add_distance(im,lcclass,bband,to_clip,cpixel):
    nn = 0+lcclass
    imf = im.updateMask(im.select(bband).eq(nn))
    imf = imf.toInt()
    patchsize = imf.connectedPixelCount(100, False)
    imf =  imf.updateMask(patchsize.gt(cpixel))
    dist = imf.fastDistanceTransform(1000).sqrt().multiply(ee.Image.pixelArea().sqrt()).clip(to_clip)
    dist = dist.select('distance').rename('distto_'+str(nn))
    return im.addBands(dist)