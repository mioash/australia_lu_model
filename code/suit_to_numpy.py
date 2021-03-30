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
# define data
#data = asarray([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
# save to csv file


#from osgeo import gdal
#from osgeo import osr
#import time

crop = ee.Image('users/mioash/Calderon_etal_Australian_land-cover/Tas_lc2010_anuclim_nneigh_ndist_naccess_tenure_cropland').rename('prob_crop')
forest = ee.Image('users/mioash/Calderon_etal_Australian_land-cover/Tas_lc2010_anuclim_nneigh_ndist_naccess_tenure_forest').rename('prob_forest')
grass = ee.Image('users/mioash/Calderon_etal_Australian_land-cover/Tas_lc2010_anuclim_nneigh_ndist_naccess_tenure_grassland').rename('prob_grass')
urb = ee.Image('users/mioash/Calderon_etal_Australian_land-cover/Tas_lc2010_anuclim_nneigh_ndist_naccess_tenure_urban').rename('prob_urban')
other = ee.Image('users/mioash/Calderon_etal_Australian_land-cover/Tas_lc2010_anuclim_nneigh_ndist_naccess_tenure_other').rename('prob_other')
lc_aus = ee.Image("users/mioash/Calderon_etal_Australian_land-cover/Lc_Aus_1985_2015_v1")
sgrid = ee.FeatureCollection("users/mioash/drivers_lcaus/boundaries/tasmania_grid_v2302")

probs = crop.addBands(forest).addBands(grass).addBands(urb).addBands(other)


#probs= probs.multiply(10000000000000000)
probs = probs.addBands(lc_aus.select(['b2010','b2005']))
#urb = urb.multiply(10000000000000000)

# tas_sa2 = ee.FeatureCollection("users/mioash/transitions/tas_sa2_pixels")
# tas_sa2_summary = ee.FeatureCollection("users/mioash/transitions/tas_summary_sa2")
sa2 = ee.FeatureCollection("users/mioash/drivers_lcaus/boundaries/SA2")
sa2_tas = sa2.filterMetadata('STE_CODE11','equals','6')
id_look = 'SA2_MAIN11'
#sa2_id = [601021010, 601021005]  #601021011
# sa2_id = 601021010 #603011065 good accuracy

def id_parse (FCC):
     return ee.Feature(FCC).set(id_look, ee.Number.parse(FCC.get(id_look)))

sa2_tas = sa2_tas.map(id_parse)


#tas_hex = ee.FeatureCollection('users/mioash/tas_hex_clipped2')
#th = tas_hex.filterMetadata('Id_hex_new','equals',13)

sa2_painted = ee.Image(-9999).paint(sa2_tas,id_look).rename('sa2_id')
sa2_painted = sa2_painted.clipToCollection(sgrid)

#sa_test = sa2_tas.filterMetadata(id_look,'equals',sa2_id)#filter(ee.Filter.inList(id_look, sa2_id))#.filterMetadata(id_look,'equals',sa2_id)

probs = probs.unmask(-9999,False).addBands(sa2_painted)

#probs = probs#.toInt8()

def LatLonImg(img,region):
   
    img = img.addBands(ee.Image.pixelLonLat())
    
    img = img.reduceRegion(reducer=ee.Reducer.toList(),\
                                        geometry=region,\
                                        maxPixels=1e13,\
                                        scale=30,\
                                        tileScale= 2);
 
    sa2id= np.array((ee.Array(img.get("sa2_id")).getInfo()))#.astype('i4')
    if np.max(sa2id) != -9999:        
        lats = np.array((ee.Array(img.get("latitude")).getInfo()))#.astype('i4')
        lons = np.array((ee.Array(img.get("longitude")).getInfo()))#.astype('i4')
        cr = np.array((ee.Array(img.get("prob_crop")).getInfo()))#.astype('i4')
        fr = np.array((ee.Array(img.get("prob_forest")).getInfo()))#.astype('i4')
        gr= np.array((ee.Array(img.get("prob_grass")).getInfo()))#.astype('i4')
        ur= np.array((ee.Array(img.get("prob_urban")).getInfo()))#.astype('i4')
        ot= np.array((ee.Array(img.get("prob_other")).getInfo()))#.astype('i4')
        lprev= np.array((ee.Array(img.get("b2005")).getInfo()))#.astype('i4')
        lpred= np.array((ee.Array(img.get("b2010")).getInfo()))#.astype('i4')
        
        arr = np.vstack([lats, lons, sa2id, cr, fr, gr, ur, ot, lprev, lpred]).transpose()
        return arr[arr[:,3] != -9999,:]
     #lol[lol[:,9] == 13,:]
        
    # try:
    #     data = np.array((ee.Array(img.get("classification")).getInfo()))
    # except:
    #     data= np.full_like(lats, np.nan,dtype=np.float32)
    
    #return lats, lons, cr, fr, gr, ur, ot, lprev, lpred

llist = list(range(0,289))
#llist = [60,61]
#sgrid1 = sgrid.filterMetadata('id_grid','equals',143) #143 yes, 163:water

appended = np.zeros([1,10])

for i in range(200,289,1):
    sgrid1 = sgrid.filterMetadata('id_grid', 'equals',llist[i])
    probs1 = probs.clip(sgrid1)
    limg = LatLonImg(probs1,sgrid1)
    if limg is None:
        a =1
    else:
        appended = np.concatenate((appended,limg),0)

np.save('tas_200_289',appended)

limga = np.around(limg,6)

lol = limg[limg[:,2].argsort()[::-1]]
ncol = np.zeros([len(limg),1])
lol = np.append(lol,ncol,1)
#lol = np.append(lol,ncol,1)

#lol[0:1368,9] = 0

for i in range(0,len(lol)-1,1):
    idex =  np.where(lol[i,:] == np.amax(lol[i,2:6]))[0][0] 
    lol[i,9] = idex-2+10
    
icr = np.count_nonzero(lol[:,9]==10) #1368 --calc 1672
ifo = np.count_nonzero(lol[:,9]==11) #40132 -- 32708
igr = np.count_nonzero(lol[:,9]==12) #5124 -- 7224
iur = np.count_nonzero(lol[:,9]==13) #6702 -- 12169  
iot = np.count_nonzero(lol[:,9]==14) #10 -- 0  

diffs = np.array([icr-1368,ifo-40132,igr-5124,iur-6702,iot-10])

#nn = np.where(diffs == np.amax(diffs))[0][0]
dur = lol[lol[:,9] == 13,:]
dur = dur[dur[:,5].argsort()]
nur = dur[0:5466,]
dur = dur[5467:len(dur),]

dgr = lol[lol[:,9] == 12,:]
dgr = dgr[dgr[:,4].argsort()]
ngr = dgr[0:2099,]
dgr = dgr[2100:len(dgr),]

dcr = lol[lol[:,9] == 10,:]
dcr = dcr[dcr[:,2].argsort()]
ncr = dcr[0:302,]
dcr = dcr[303:len(dcr),]

extra = np.concatenate((ncr,nur,ngr),0)
extra = extra[extra[:,3].argsort()[::-1]]
efo = extra[0:7424,:]
efo[:,9] = 11
dfo = lol[lol[:,9] == 11,:]

eot = extra[7425:7435,:]
eot[:,9] = 14
dot = lol[lol[:,9] == 14,:]

ex_nc = extra[7436:len(extra),:]
ex_nc[:,9] = 0
d_nc = lol[lol[:,9] == 0,:]


#dnc = lol[lol[:,9] == -9999,:]

final = np.concatenate((dur,dgr,dcr,dfo,dot,efo,eot,ex_nc),0)

savetxt('npy_test.csv', final, fmt='%f6',delimiter=',')

savetxt('grid_test.csv', limga, fmt='%f6',delimiter=',')

np.save('grid_test',limga)

#diffs1 = np.array([icr-1368,ifo-40132,igr-5124,iot-10])
#nn = np.where(diffs == np.amax(diffs))[0][0]




#ttt = lol[lol[:,nn+2].argsort()[::-1]]



#if nn > 0:
    #lol = lol[lol[:,nn+2].argsort()[::-1]]
    


for i in range(0,len(diffs),1):
    if diff[i] != 1368:
        
    

#lol = lol[lol[:,2].argsort()[::-1]]


# convert the lat, lon and array into an image
# def toImage(lats,lons,data):
 
#     # get the unique coordinates
#     uniqueLats = np.unique(lats)
#     uniqueLons = np.unique(lons)
#     #print(type(data))
#     # get number of columns and rows from coordinates
#     ncols = len(uniqueLons)
#     nrows = len(uniqueLats)
 
#     # determine pixelsizes
#     ys = uniqueLats[1] - uniqueLats[0]
#     xs = uniqueLons[1] - uniqueLons[0]
 
#     # create an array with dimensions of image
#     arr = np.zeros([nrows, ncols], np.float32) #-9999
 
#     # fill the array with values
#     counter = 0
#     #nnn = 0
    
#     # for i in range(0,ncols,1):
#     #     for j in range(0,nrows,1):
#     #         while uniqueLats[j]  != lats []
    
#     for y in range(0,len(arr),1):
#         for x in range(0,len(arr[0]),1):
#             if lats[counter] == uniqueLats[y] and lons[counter] == uniqueLons[x] and counter < len(lats)-1:
#                 arr[len(uniqueLats)-1-y,x] = data[counter] # we start from lower left corner
#                 counter+=1
#                 #print(counter)
#     return arr
 
#lat, lon, pc, pf, pg, pu, po = LatLonImg(probs,sa_test)
 


#np.save('sla_601021010',limg)

# ee.batch.Export.table.toDrive(
#             collection= ee.FeatureCollection(iFF),
#             description= 'ttt_lat',
#             fileFormat= 'CSV').start()
# 1d to 2d array
#image  = toImage(lat,lon,data)
#rgb_img = np.vstack([lon, lat,  pc, pf, pg, pu, po]).transpose()

# Expand the dimensions of the images so they can be concatenated into 3-D.
# np_arr_b4 = np.expand_dims(lat, 2)
# np_arr_b5 = np.expand_dims(lon, 2)
# np_arr_b6 = np.expand_dims(data, 2)
# print(np_arr_b4.shape)
# print(np_arr_b5.shape)
# print(np_arr_b6.shape)

# Stack the individual bands to make a 3-D array.
#rgb_img = np.concatenate((np_arr_b6, np_arr_b5, np_arr_b4), 2)


#rgb_img = np.dstack(lon, lat, data)

#print(rgb_img.shape())
#savetxt('image_test.csv', image, delimiter=',')

#plt.imshow(rgb_img)
#plt.show()