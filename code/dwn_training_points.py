# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 08:38:29 2021

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
#from change_no_change import suit_layer 
import geemap

import numpy as np
import matplotlib.pyplot as plt
from numpy import asarray
from numpy import savetxt
import pandas as pd

#lc_all = define_image(0,0,'none',2010,5)

#parameters to change
# tmov threshold - 40, 50, 60 

val_year = 2015
train_year = 2010
i_year = 1983
lc_img = define_image(50,0,'tmov',train_year,5) #temporal mov threshold >= 40, 
lc_img_classify = define_image(0,0,'none',train_year,5)

lim_temp_f = ee.FeatureCollection('users/mioash/aust_cd66states')
ucl = ee.FeatureCollection('users/mioash/SUA_2016_AUST')
vic = lim_temp_f.filterMetadata('STE', 'equals', 6)
geom = vic

roads = ee.FeatureCollection('users/verdemuskuna/aus_roads/gis_osm_roads_free_1')
wways = ee.FeatureCollection('users/verdemuskuna/aus_roads/gis_osm_waterways_free_1')
pop = ee.ImageCollection("projects/sat-io/open-datasets/hrslpop").median().clip(geom)
nat_npa = ee.Image("users/mioash/drivers_lcaus/natural_areas/nat_npa_ready")
tas_dist_neigh = ee.Image("users/mioash/Calderon_etal_Australian_land-cover/Tas_neigh_dists_c2005")

proj = lc_img.projection()
#################CHANGE ACCESSIBILITY

fst_img = stacking(train_year-10,train_year-5,'mean',False,False) # initial year, end year, reducer for climate, get climate, get precipitation
bfst = fst_img.bandNames().getInfo()

#clm_mean = stacking(train_year-10,train_year-5,'p50',True,True)
# clm_lterm = stacking(i_year,train_year-5,'p50',True,True)
# clm_anomaly = clm_mean.subtract(clm_lterm)

# bn = clm_anomaly.bandNames().getInfo()
# bn = ["{}{}".format(i,str('_anom')) for i in bn]
# clm_anomaly = clm_anomaly.rename(bn) 


roads = roads.filter(ee.Filter.inList('fclass', ['motorway','trunk','primary','secondary','unclassified','tertiary','residential','living_street']))
rd = ee.Image(0).paint(roads,'code').updateMask(fst_img.select('cindex'))
rd = rd.updateMask(rd.neq(0))
dist_roads = rd.select('constant').fastDistanceTransform(1000).sqrt().multiply(ee.Image.pixelArea().sqrt())#//.divide(1000000)//.clip(to_clip)
dist_roads = dist_roads.rename('dist_roads').updateMask(fst_img.select('cindex').add(100))

#dist_roads = dist_roads.expression(
#  '(1 + (1/dd))',{
#  'dd': dist_roads.select(['dist_roads'])
#}).rename('dist_roads')


wways = wways.filter(ee.Filter.inList('fclass', ['river']))
ww = ee.Image(0).paint(wways,'code').updateMask(fst_img.select('cindex'))
ww = ww.updateMask(ww.neq(0))
dist_wways = ww.select('constant').fastDistanceTransform(1000).sqrt().multiply(ee.Image.pixelArea().sqrt())#//.divide(1000000)//.clip(to_clip)
dist_wways = dist_wways.rename('dist_wways').updateMask(fst_img.select('cindex').add(100))

pop = pop.unmask(0).reproject(proj.atScale(30))#.clip(geom)
pop = pop.select('b1').rename('pop')

img_for_train = lc_img.addBands(fst_img).addBands(dist_roads).addBands(dist_wways).addBands(pop)


lcclasses = [0,1,2,3,4]
cpixels = [15,15,15,15,15]

#imgn = img_for_train
idist = img_for_train.select('lc_t1').rename('lc_temp')

for i in range(0,len(lcclasses)):
    idist = add_distance(idist,lcclasses[i],'lc_temp',geom,cpixels[i])

idist = idist.select(idist.bandNames().remove(str('lc_t1')).getInfo())
idist = idist.select(idist.bandNames().remove(str('lc_temp')).getInfo())

nat_npa = nat_npa.select('npa')
to_test = ee.Image(0).clip(geom).where(nat_npa,1)
to_test = to_test.rename('npa')

distto_npa = add_distance(to_test,1,'npa',geom,1)
distto_npa = distto_npa.rename(['npa','distto_npa'])


img_for_train = img_for_train.addBands(distto_npa.select('distto_npa')).addBands(idist)
#add_neigh(image,size,size_centre,blank):

ineigh = ee.Image(1).addBands(add_neigh(lc_img.select('lc_t1'),3,1,False,'_nb')) #-> r=1
ineigh = ineigh.addBands(add_neigh(lc_img.select('lc_t1'),9,3,True,'_3b')) #-> r=4
#ineigh = ineigh.addBands(add_neigh(lc_img.select('lc_t1'),9,1,False,'_nb')) #-> r=4

ineigh = ineigh.addBands(add_neigh(lc_img.select('lc_t1'),27,9,True,'_9b')) #-> r=13
#ineigh = ineigh.addBands(add_neigh(lc_img.select('lc_t1'),27,1,False,'_nb')) #-> r=13

ineigh = ineigh.addBands(add_neigh(lc_img.select('lc_t1'),67,27,True,'_27b')) #-> r=33
#ineigh = ineigh.addBands(add_neigh(lc_img.select('lc_t1'),67,1,False,'_nb')) #-> r=33

img_for_train = img_for_train.addBands(ineigh)
# img_for_train = img_for_train.addBands(add_neigh(lc_img.select('lc_t1'),3,1,False,'_nb')) #-> r=1

# img_for_train = img_for_train.addBands(add_neigh(lc_img.select('lc_t1'),9,3,True,'_3b')) #-> r=4
# img_for_train = img_for_train.addBands(add_neigh(lc_img.select('lc_t1'),9,1,False,'_nb')) #-> r=4

# img_for_train = img_for_train.addBands(add_neigh(lc_img.select('lc_t1'),27,9,True,'_9b')) #-> r=13
# img_for_train = img_for_train.addBands(add_neigh(lc_img.select('lc_t1'),27,1,False,'_nb')) #-> r=13

# img_for_train = img_for_train.addBands(add_neigh(lc_img.select('lc_t1'),67,27,True,'_27b')) #-> r=33
# img_for_train = img_for_train.addBands(add_neigh(lc_img.select('lc_t1'),67,1,False,'_nb')) #-> r=33


im_to_exp = idist.addBands(distto_npa.select('distto_npa')).addBands(ineigh).addBands(dist_roads).addBands(dist_wways)

img_ready = lc_img.addBands(fst_img).addBands(tas_dist_neigh).addBands(pop)
img_ready = img_ready.unmask(-9999,False).updateMask(lc_img_classify.select('lc_t1'))
#print(img_ready.bandNames().getInfo())
#print(im_to_exp.bandNames().getInfo())
bbands = img_ready.bandNames().getInfo()
#print(bbands)

def myproperties(feature):
  feature=ee.Feature(feature).setGeometry(None)
  return feature

def LatLonImg(img,bands_to_get,npts):
   
    img = img.addBands(ee.Image.pixelLonLat())
    strat = make_strata(img,6,False)
    ssamples = prop_allocation(ee.Image(strat).toInt(),geom,npts)
    # ssamples = img.sampleRegions(collection= ssamples, 
    #                              scale= 30,
    #                              tileScale=2)
    
    ssamples = img.reduceRegion(reducer=ee.Reducer.toList(),\
                                geometry = ssamples, \
                                 scale= 30,\
                                 tileScale=2)
    
    #tvals = tvals.map(myproperties)
    #ssamples = ssamples.map(myproperties)
    #print(ssamples.getInfo())
    # img = img.reduceRegion(reducer=ee.Reducer.toList(),\
    #                                     geometry=region,\
    #                                     maxPixels=1e13,\
    #                                     scale=30,\
    #                                     tileScale= 2);
 
    #sa2id = np.array((ee.Array(img.get("sa2_id")).getInfo()))#.astype('i4')
    lats = np.array((ee.Array(ssamples.get("latitude")).getInfo()))#.astype('i4')
    lons = np.array((ee.Array(ssamples.get("longitude")).getInfo()))#.astype('i4')
    arr = np.vstack([lats, lons])
    
    for i in range(0,len(bands_to_get)):
         bx = np.array((ee.Array(ssamples.get(str(bands_to_get[i]))).getInfo()))
         arr = np.vstack([arr,bx])
    #arr = arr
    # cr = np.array((ee.Array(img.get("prob_crop")).getInfo()))#.astype('i4')
    # fr = np.array((ee.Array(img.get("prob_forest")).getInfo()))#.astype('i4')
    # gr= np.array((ee.Array(img.get("prob_grass")).getInfo()))#.astype('i4')
    # ur= np.array((ee.Array(img.get("prob_urban")).getInfo()))#.astype('i4')
    # ot= np.array((ee.Array(img.get("prob_other")).getInfo()))#.astype('i4')
    # lprev= np.array((ee.Array(img.get("b2005")).getInfo()))#.astype('i4')
    # lpred= np.array((ee.Array(img.get("b2010")).getInfo()))#.astype('i4')
    #arr = np.vstack([lats, lons, sa2id, cr, fr, gr, ur, ot, lprev, lpred]).transpose()
    return arr.transpose()

pts = LatLonImg(img_ready, bbands,5000)
pts = pts[pts[:,2] != -9999,:]
#print(pts)
nbands = ['latitude','longitude']+bbands#,'lc_dep','lc_t1','lc_t2','lc_t3','lc_t4','lc_t5','constant']#+bbands

pts_mx = pd.DataFrame(data=pts, columns=nbands)
#pts_mx.to_csv(path_or_buf='tas_1st_test.csv',index=False)

labels = np.array(pts_mx['lc_dep'])

# Remove the labels from the features
# axis 1 refers to the columns
features= pts_mx.drop(['lc_dep','constant','latitude','longitude'], axis = 1)

# Saving feature names for later use
feature_list = list(features.columns)

# Convert to numpy array
features = np.array(features)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.2,
                                                                            random_state = 55)

# Instantiate model 
rf = RandomForestClassifier(n_estimators= 20, random_state=55)

# Train the model on training data
rf.fit(train_features, train_labels)

y_pred = rf.predict(test_features)

print(confusion_matrix(test_labels,y_pred))
print(classification_report(test_labels,y_pred))
print(accuracy_score(test_labels, y_pred))

# Get numerical feature importances
importances = list(rf.feature_importances_)

# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]

# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];


# Import matplotlib for plotting and use magic command for Jupyter Notebooks
import matplotlib.pyplot as plt
#%matplotlib inline
# Set the style

plt.style.use('fivethirtyeight')# list of x locations for plotting
x_values = list(range(len(importances)))# Make a bar chart
plt.bar(x_values, importances, orientation = 'vertical')# Tick labels for x axis
plt.xticks(x_values, feature_list, rotation='vertical')# Axis labels and title
plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances');
# for i in range(0,len(bbands)):
#     tvals = lc_change.select(bbands[i]).addBands(ee.Image.pixelLonLat().float()).sampleRegions(
#                             collection= ssamples, 
#                             scale= 30,
#                             tileScale=16)
#     tvals = tvals.map(myproperties)

## not now -- later when image available
# img_for_extract = img_for_train.addBands(ee.Image.pixelLonLat())
# strat = make_strata(img_for_extract ,6,False)
# ssamples = prop_allocation(ee.Image(strat).toInt(),geom,10)
# ee.batch.Export.table.toDrive(collection=ssamples,description='tas_10p').start()

#import geemap
#geemap.update_package()
from geemap import ml

trees =  ml.rf_to_strings(rf,feature_list)

user_id = geemap.ee_user_id()
user_id

ml.export_trees_to_fc(trees,user_id + "/20_trees_tas_test")


################
llx = 143.537
lly =  -43.89
urx = 148.965
ury = -39.42  #tas
#geometri = [[llx,lly], [llx,ury], [urx,ury], [urx,lly]]
geometri = ee.Geometry.Rectangle([143.537,-43.89,148.965,-39.42]) #//#tas

# ee.batch.Export.image.toAsset(image=im_to_exp,
#                               description='Tas_neigh_dists_c2005',
#                               assetId='users/mioash/Calderon_etal_Australian_land-cover/Tas_neigh_dists_c2005',scale=30,
#                              region = geometri,maxPixels=1e13,crs='EPSG:4326').start()
