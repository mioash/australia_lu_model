# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 08:36:23 2021

@author: mcalderonloor
"""

import ee
ee.Initialize()
import numpy as np
import matplotlib.pyplot as plt
from numpy import asarray
from numpy import savetxt
import osgeo
from osgeo import gdal
from osgeo import gdal_array
from osgeo import osr

fil = np.load('sla_601021010.npy')

#convert the lat, lon and array into an image
def toImage(dato):
    # get the unique coordinates
    uniqueLats = np.unique(dato[:,0])
    uniqueLons = np.unique(dato[:,1])
    data = dato[:,8]
    lats = dato[:,0]
    lons = dato[:,1]
    
    #print(type(data))
    # get number of columns and rows from coordinates
    ncols = len(uniqueLons)
    nrows = len(uniqueLats)
 
    # determine pixelsizes
    ys = uniqueLats[1] - uniqueLats[0]
    xs = uniqueLons[1] - uniqueLons[0]
 
    # create an array with dimensions of image
    arr = np.zeros([nrows, ncols], np.float32) #-9999
 
    # fill the array with values
    counter = 0
    #nnn = 0
    
    # for i in range(0,ncols,1):
    #     for j in range(0,nrows,1):
    #         while uniqueLats[j]  != lats []
    
    for y in range(0,len(arr),1):
        for x in range(0,len(arr[0]),1):
            if lats[counter] == uniqueLats[y] and lons[counter] == uniqueLons[x] and counter < len(lats)-1:
                arr[len(uniqueLats)-1-y,x] = data[counter] # we start from lower left corner
                counter+=1
                #print(counter)
    return arr, counter
 
image, ct  = toImage(fil)


data = fil[:,8]
lats = fil[:,0]
lons = fil[:,1]
    
xmin,ymin,xmax,ymax = [lons.min(),lats.min(),lons.max(),lats.max()]

uniqueLats = np.unique(lats)
uniqueLons = np.unique(lons)
ncols = len(uniqueLons)
nrows = len(uniqueLats)

xres = (xmax-xmin)/float(ncols)
yres = (ymax-ymin)/float(nrows)

geotransform=(xmin,xres,0,ymax,0, -yres)

output_raster = gdal.GetDriverByName('GTiff').Create('C:\Users\MyName\Temp\inv_masked.tif',ncols, nrows, 1 ,gdal.GDT_Float32)
#writting output raster
output_raster.GetRasterBand(1).WriteArray( array ) 
output_raster.SetGeoTransform(geotransform)
srs = osr.SpatialReference()
srs.ImportFromEPSG(2346) #British National Grid OSGB1936
output_raster.SetProjection(srs.ExportToWkt())
output_raster = None
