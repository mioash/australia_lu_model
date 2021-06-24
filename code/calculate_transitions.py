# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 04:39:24 2020

@author: -
"""
import ee 
ee.Initialize()
vic_hex = ee.FeatureCollection('users/mioash/vic_hex_clipped')
wa_hex = ee.FeatureCollection('users/mioash/wa_hex_clipped')
sa_hex = ee.FeatureCollection('users/mioash/sa_hex_clipped')
qld_hex = ee.FeatureCollection('users/mioash/qld_hex_clipped')
nt_hex = ee.FeatureCollection('users/mioash/nt_hex_clipped')
nsw_hex = ee.FeatureCollection('users/mioash/nsw_hex_clipped')
act_hex = ee.FeatureCollection('users/mioash/act_hex_clipped')
tas_hex = ee.FeatureCollection('users/mioash/tas_hex_clipped2')

sa2 = ee.FeatureCollection('users/mioash/drivers_lcaus/boundaries/SA2')
lc_all= ee.Image("users/mioash/Calderon_etal_Australian_land-cover/Lc_Aus_1985_2015_v1")
lc_imc = ee.FeatureCollection([lc_all.select('b1985'),lc_all.select('b1990'),lc_all.select('b1995'),lc_all.select('b2000'),lc_all.select('b2005'),lc_all.select('b2010'),lc_all.select('b2015')])
#Map.addLayer(vic_hex,'','vic_hex')

def myproperties (feature):
    feature = ee.Feature(feature).setGeometry(None)
    return feature

def transitions_area (fc_analysis,id_fc,im_ts,im_i,im_f,lc,mode,name_out,fold):
    imi = ee.Image(im_ts.toList(7).get(im_i))
    imf = ee.Image(im_ts.toList(7).get(im_f))
    imi = imi.add(ee.Image(10))
    imi = imi.updateMask(imi.eq(lc))
    for i in range(0,6):
        im_t1 = imf.updateMask(imi)
        im_t1 = im_t1.updateMask(im_t1.eq(i)).clip(fc_analysis)
        im_t2 = ee.Image(1).updateMask(im_t1.add(10))
        if mode=='area':
            area_cc = im_t2.multiply(ee.Image.pixelArea()).reduceRegions(
                collection= fc_analysis.select([id_fc]),
                reducer= ee.Reducer.sum(), 
                scale= 30,
                tileScale=8)
        elif mode == 'pixels':
            area_cc = im_t2.reduceRegions(
                collection= fc_analysis.select([id_fc]),
                reducer= ee.Reducer.count(), 
                scale= 30,
                tileScale=8)
        area_cc = area_cc.map(myproperties)
        ee.batch.Export.table.toDrive(
            collection= ee.FeatureCollection(area_cc),
            folder = fold,
            description= name_out+'_to_'+str(i),
            fileNamePrefix= name_out+'_to_'+str(i),
            fileFormat= 'CSV').start()
        #p=p+1

#act_1985_1990_area_crop_to_0
#transitions_area(act_hex,'Id_hex_new',lc_imc,0,1,10,'test_act_crop')
immi = 5 #0-5, 1985==0
immf = 6 #1-6
fc_ans = sa2.filterMetadata('STE_CODE11','equals','6')
#fc_ans = fc_ans.filterMetadata('SA2_MAIN11','equals','601021005')
nmm ='tas_2010_2015'
stt='tas'
per = '10-15'
##nsw=1, vic=2, tas=6, act=8

transitions_area(fc_ans,'SA2_MAIN11',lc_imc,immi,immf,10,'pixels',str(nmm)+'_pixel_crop_sa2','transitions_sa2/'+str(per)+'-pixels/'+str(stt))
transitions_area(fc_ans,'SA2_MAIN11',lc_imc,immi,immf,11,'pixels',str(nmm)+'_pixel_forest_sa2','transitions_sa2/'+str(per)+'-pixels/'+str(stt))
transitions_area(fc_ans,'SA2_MAIN11',lc_imc,immi,immf,12,'pixels',str(nmm)+'_pixel_grass_sa2','transitions_sa2/'+str(per)+'-pixels/'+str(stt))
transitions_area(fc_ans,'SA2_MAIN11',lc_imc,immi,immf,13,'pixels',str(nmm)+'_pixel_urban_sa2','transitions_sa2/'+str(per)+'-pixels/'+str(stt))
transitions_area(fc_ans,'SA2_MAIN11',lc_imc,immi,immf,14,'pixels',str(nmm)+'_pixel_water_sa2','transitions_sa2/'+str(per)+'-pixels/'+str(stt))
transitions_area(fc_ans,'SA2_MAIN11',lc_imc,immi,immf,15,'pixels',str(nmm)+'_pixel_other_sa2','transitions_sa2/'+str(per)+'-pixels/'+str(stt))

    