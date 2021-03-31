# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 03:52:46 2021

@author: mcalderonloor
"""

#import ee
#ee.Initialize()
import numpy as np
import matplotlib.pyplot as plt
from numpy import asarray
from numpy import savetxt
import osgeo
from osgeo import gdal
from osgeo import gdal_array
from osgeo import osr

import rasterio
from rasterio.transform import from_origin

#arr= np.load('sla_601021010.npy')

#convert the lat, lon and array into an image

def npy_long_to_wide (arr,lats_id,lons_id,vals_id):
    
    lats = arr[:,lats_id]
    lons = arr[:,lons_id]
     
    rows, row_pos = np.unique(lats, return_inverse=True)
    cols, col_pos = np.unique(lons, return_inverse=True)
    
    #uniqueLats = np.unique(lats)
    #uniqueLons = np.unique(lons)
    ncols = len(cols)
    nrows = len(rows)
    
    #pivoted_arr3 = np.zeros((len(rows), len(cols)))
    pivoted_arr3 = np.full((len(rows), len(cols)), -9999, dtype=int)
    
    pivoted_arr3[row_pos, col_pos] = arr[:, vals_id]
    
    pivoted_arr3 = np.flipud(pivoted_arr3)
    return pivoted_arr3
    
    
def npy_to_tif (arr,lats_id,lons_id,vals_id,out_name):
    
    lats = arr[:,lats_id]
    lons = arr[:,lons_id]
     
    rows, row_pos = np.unique(lats, return_inverse=True)
    cols, col_pos = np.unique(lons, return_inverse=True)
    
    pivoted_arr3 = npy_long_to_wide(arr, lats_id, lons_id, vals_id)
    
    xmin,ymin,xmax,ymax = [lons.min(),lats.min(),lons.max(),lats.max()]
    
    xres = (xmax-xmin)/float(len(cols))
    yres = (ymax-ymin)/float(len(rows))
    
    #geotransform=(xmin,xres,0,ymax,0, -yres)
    
#data = arr[:,8]
    
    
    #cmapa = mpl.colors.ListedColormap(['yellow','green','orange', 'red','blue','gray'])
    #plt.imshow(,vmin=0,vmax=5,cmap=cmapa)
    
    # srs = osr.SpatialReference()
    # srs.ImportFromEPSG(2346) #British National Grid OSGB1936
    
    # output_raster = gdal.GetDriverByName('GTiff').Create('sla_tif.tif',ncols, nrows, 1 ,gdal.GDT_Float32)
    # output_raster.SetMetadata( {'prediction_year': '2010'} )
    # output_raster.SetGeoTransform(geotransform)
    # output_raster.SetProjection(srs.ExportToWkt())
    # output_raster.GetRasterBand(1).WriteArray(pivoted_arr3) 
    # output_raster.GetRasterBand(1).SetNoDataValue(-9999) 
    # #writting output raster
    # output_raster = None
    
    #img=gdal.Open('sla_tif.tif')
    #dataset = gdal.Open('sla_tif.tif', gdal.GA_ReadOnly) # Note GetRasterBand() takes band no. starting from 1 not 0
    
    #band = dataset.GetRasterBand(1)
    #arr1 = band.ReadAsArray()
    #plt.imshow(arr1)
    
    
    
    # array of shape (h, w)
    #dst_shape = pivoted_arr3.shape
    
    # bbox is the bounding box BBox instance used in Wms/WcsRequest 
    #dst_transform = rasterio.transform.from_bounds(xmax,ymin,xmin,ymax,
    #                                               width=dst_shape[1], height=dst_shape[0])
    #dst_crs = {'init': CRS.ogc_string(bbox.crs)}
    
    # Write it out to a file.
    
     
    transform = from_origin(xmin, ymax, xres, yres)
    #arr = pivoted_arr3
    
    new_dataset = rasterio.open(out_name, 'w', driver='GTiff',
                                height = pivoted_arr3.shape[0], width = pivoted_arr3.shape[1],
                                count=1, dtype=str(pivoted_arr3.dtype),
                                crs='+proj=latlong',
                                transform=transform)
    
    new_dataset.write(pivoted_arr3, 1)
    new_dataset.close()
