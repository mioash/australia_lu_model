# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 14:56:25 2021

@author: mcalderonloor
"""


import gc
from allocate_tiff import allocate
import glob 
from osgeo import gdal
import subprocess

# allocate('act')
# allocate('sa')

# allocate('nt')
# gc.collect()

# list all files in directory that match pattern


#def read_all_files(stt):
#   file_names = glob('E:/Marco/australia_lu_model/data/suitability/'+str(stt)+'/*')
#   arrays = [np.load(f) for f in file_names]
#   return np.concatenate(arrays,axis=0)    

stt = 'tas'
year =2020
demList = glob.glob('E:/Marco/australia_lu_model/data/allocation/'+str(stt)+'/'+str(year)+'/pred_'+str(year)+'_*')

#demList = demList[0:110]
# gdal_merge
# cmd = "python.exe gdal_merge.py -o mergedALL.tif"
# subprocess.call(cmd.split()+demList)

# build virtual raster and convert to geotiff
vrt = gdal.BuildVRT("merged.vrt", demList)


translate_options = gdal.TranslateOptions(format = 'GTiff',  noData=255,
                                          creationOptions = ['TFW=YES', 'COMPRESS=LZW'])

gdal.Translate("E:/Marco/australia_lu_model/data/allocation/"+str(stt)+"/"+str(year)+"/pred_"+str(year)+"_merged_"+str(stt)+"_v3.tif", vrt, options=translate_options)
vrt = None

# demLista = glob.glob('E:/Marco/australia_lu_model/data/allocation/'+str(stt)+'/'+str(year)+'a/pred_'+str(year)+'_*')
# vrta = gdal.BuildVRT("mergeda.vrt", demLista)
# gdal.Translate("E:/Marco/australia_lu_model/data/allocation/"+str(stt)+"/"+str(year)+"a/pred_"+str(year)+"_merged_stable_"+str(stt)+"_v3.tif", vrta, options=translate_options)
# vrta = None
#subprocess.call(['python.exe', 'hello.py', 'htmlfilename.htm'])

# allocate('wa')
# gc.collect()

# allocate('qld')

#allocate('vic')
#allocate('nsw')
#allocate('tas')