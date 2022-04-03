# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 10:02:05 2021

@author: mcalderonloor
"""

import ee 
from ee import batch

# Initialize the Earth Engine object, using the authentication credentials.
ee.Initialize()
from make_clim import make_climate

from stack_images_qgis import stacking
from mask_image import define_image
from stratitication import make_strata
from stratitication import prop_allocation
from change_no_change import suit_layer 

from add_neigh import add_neigh
from add_distance import add_distance



val_year = 2015
train_year = 2010
i_year = 1983
#lc_img = define_image(50,0,'tmov',train_year,5) #temporal mov threshold >= 40, 
#lc_img_classify = define_image(0,0,'none',train_year,5)

#lc_img_for_masking = define_image(50,0,'tmov',train_year,5)

lim_temp_f = ee.FeatureCollection('users/mioash/aust_cd66states')

geom = lim_temp_f


fst_img = stacking(train_year-10,train_year-5,'mean',False,False) # initial year, end year, reducer for climate, get climate, get precipitation

#img_pred = ee.Image('users/mioash/ALUM/pred_aus_2010_v1').rename('lc_t1')
img_pred = ee.Image("users/mioash/Calderon_etal_Australian_land-cover/Lc_Aus_1985_2015_v1").select('b2015').rename('lc_t1')
#proj = lc_img_classify.projection()
#################CHANGE ACCESSIBILITY


lcclasses = [0,1,2,3,4]
cpixels = [15,15,15,15,15]

#imgn = img_for_train
idist = img_pred.select('lc_t1').rename('lc_temp')

for i in range(0,len(lcclasses)):
    idist = add_distance(idist,lcclasses[i],'lc_temp',geom,cpixels[i])

idist = idist.select(idist.bandNames().remove(str('lc_t1')).getInfo())
idist = idist.select(idist.bandNames().remove(str('lc_temp')).getInfo())

# nat_npa = ee.Image("users/mioash/drivers_lcaus/natural_areas/nat_npa_ready")
# nat_npa = nat_npa.select('npa')

ineigh = ee.Image(1).addBands(add_neigh(img_pred.select('lc_t1'),3,1,False,'_nb')) #-> r=1
ineigh = ineigh.addBands(add_neigh(img_pred.select('lc_t1'),9,3,True,'_3b')) #-> r=4

ineigh = ineigh.addBands(add_neigh(img_pred.select('lc_t1'),27,9,True,'_9b')) #-> r=13

ineigh = ineigh.addBands(add_neigh(img_pred.select('lc_t1'),81,27,True,'_27b')) #-> r=33

#distto = distn.select(['distto_0','distto_1','distto_2','distto_3','distto_4','distto_npa','dist_roads','dist_wways'])
idist= idist.unitScale(0,1372959).multiply(255).round().toByte()
 
#neigh = distn.select(nneigh).rename(nneigh1)
ineigh = ineigh.unitScale(0,100).multiply(255).round().toByte()
im_to_exp = idist.addBands(ineigh)

im_to_exp  = im_to_exp.toByte()
print(im_to_exp.getInfo())

# Export image
llx = 108.76 
lly = -44 
urx = 155 
ury = -10  #australia
geometri = [[llx,lly], [llx,ury], [urx,ury], [urx,lly]]

ee.batch.Export.image.toAsset(image=im_to_exp,
                              description='Distances_neigh_national_2015_byte',
                              assetId='users/mioash/drivers_lcaus/Distances_nigh_national_2015_byte',scale=30,
                             region = geometri,maxPixels=1e13,crs='EPSG:4326').start()
