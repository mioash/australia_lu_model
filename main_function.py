# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 08:31:32 2020

@author: mcalderonloor
"""
import ee 
from ee import batch

# Initialize the Earth Engine object, using the authentication credentials.
ee.Initialize()

# In[3]:

from make_clim import make_climate
#import sys
# insert at position 1 in the path, as 0 is the path of this file.
#sys.path.insert(1, 'C:/Users/mcalderonloor/OneDrive - Deakin University/PHD/development/Work/2nd paper/code/stack')

from stack_images_qgis import stacking
from mask_image import define_image
from stratitication import make_strata
import geemap

# In[3]:

# tmin = ee.List([ee.Image('users/mioash/drivers_lcaus/climate/tmin/tmin1982')])
# tmax = ee.List([ee.Image('users/mioash/drivers_lcaus/climate/tmax/tmax1982')])
# rain = ee.List([ee.Image('users/mioash/drivers_lcaus/climate/rainfall/rain1982')])

# for l in range(1983,2019):
#     tmin = tmin.cat(ee.List([ee.Image('users/mioash/drivers_lcaus/climate/tmin/tmin'+str(l))]))
#     tmax = tmax.cat(ee.List([ee.Image('users/mioash/drivers_lcaus/climate/tmax/tmax'+str(l))]))
#     rain = rain.cat(ee.List([ee.Image('users/mioash/drivers_lcaus/climate/rainfall/rain'+str(l))]))

# tmean = ee.List([ee.Image(tmin.get(0)).add(ee.Image(tmax.get(0))).divide(2)])
# for r in range (1,38):
#     tmean = tmean.cat(ee.List([ee.Image(tmin.get(r)).add(ee.Image(tmax.get(r))).divide(2)]))
# In[4]:
lc_img = define_image(40,0,'mov',2010,5)

fst_img = stacking(2010,2014,'mean',False)

clm_mean = stacking(2010,2014,'mean',True)
clm_lterm = stacking(1983,2014,'mean',True)

clm_anomaly = clm_mean.subtract(clm_lterm)
bn = clm_anomaly.bandNames().getInfo()
bn = ["{}{}".format(i,str('_anom')) for i in bn]

clm_anomaly = clm_anomaly.rename(bn) 
#tmean_anomaly = tmean_anomaly.rename(['anom_tmean_djf_2010','anom_tmean_mam_2010','anom_tmean_jja_2010','anom_tmean_son_2010','anom_tmean_year_2010'])
strat = make_strata(lc_img)

#print(clm_anomaly.bandNames().getInfo())

# In[5]:

Map = geemap.Map(center=[-37.74,144.93], zoom=5)
Map.add_basemap("SATELLITE")

Map.addLayer(ee.Image(strat), {'bands':['strata'],'min': 100, 'max': 450, 'opacity':1}, 'strat')
#Map.addLayer(ee.Image(imc_filled_ent.toList(7).get(0)), {'bands':['ent_'],'min': 0, 'max': 3, 'opacity':1}, 'Ent')
Map


