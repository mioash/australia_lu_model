# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 11:17:35 2021

@author: mcalderonloor
"""
import ee

def allocate(lc_ts,FC,lcclasses_vector,prob_list,lcclass_prob,lcclass_name,lc_number,method,prob_threshold,scenario,geom,id_look,sa2_id,mask_prev,im_tomask):
    
    lcpt = ee.Image(lc_ts).select('lc_t1')
    lcpt = lcpt.add(1).clip(geom)
    
    FC_filt = FC.filterMetadata(id_look,'equals',sa2_id)
    #[1,2,3,4]
    
    if lcclass_name == 'crop':
        if mask_prev == 'mask_prev':
            lcc= [2,3]
        else:
            lcc= [2,3,4]
    elif lcclass_name == 'forest':
        if mask_prev == 'mask_prev':
            lcc= [1,3]
        else:
            lcc= [1,3,4]
    elif lcclass_name == 'grass':
        if mask_prev == 'mask_prev':
            lcc= [1,2]
        else:
            lcc= [1,2,4]
    elif lcclass_name == 'urban':
        lcc= [1,2,3]
    
    lc_nn = lc_number+1
    
    lc_cl = lcclass_prob.select('classification')#.updateMask(lcpt)
    lc_cl = lc_cl.updateMask(lcpt.eq(int(lc_nn)))
    
    if mask_prev == 'mask_prev':
        lc_cl = lc_cl.where(im_tomask,-99999)
        lc_cl = lc_cl.updateMask(lc_cl.neq(-99999))
    
    #lc_cl = lcpt.updateMask(lcpt.eq(int(lc_nn))).rename('classification')
    lcclasses_vector1 = lcclasses_vector.remove(str(lcclass_name))
    #print(lc_cl.bandNames().getInfo())
    
    if scenario == 'unrestricted':       
        lc_count = lc_cl.reduceRegion(reducer=ee.Reducer.count(),
                                      geometry=geom,
                                      scale=30,
                                      maxPixels=1e13)
        print('prev_lcover',lc_count.getInfo())
        change = ee.Number(FC_filt.first().get(str(lcclass_name)+'_to_'+str(lcclass_name)))#.divide(900)
        print('desired_change',change.getInfo())
        
        cutoff = ee.Number(1).subtract(change.divide(ee.Number(lc_count.get('classification')).float()))
        print('cutoff',cutoff.getInfo())
        if cutoff.getInfo() < 0 :
            cutoff = ee.Number(0)
            print('cutoff changed to 0')
        
        lc_cut = lc_cl.reduceRegion(
            reducer = ee.Reducer.percentile([cutoff.multiply(100).float()]),
            geometry = geom,    
            scale = 30,
            maxPixels=1e13)
        #print('lccut',lc_cut.getInfo())
        lcv1 = ee.Image(lc_cl.updateMask(lc_cl.gte(ee.Number(lc_cut.get('classification')).float())))
        lvc = lcv1.reduceRegion(
                    reducer=ee.Reducer.count(),
                    geometry=geom,
                    scale=30,
                    maxPixels=1e13)
        print('lc_stab',lvc.getInfo())               
        new_prop = ee.Number(1)
    elif scenario == 'expansion':
        #lc_count = lc_cl.reduceRegion(reducer=ee.Reducer.count(),
        #                              geometry=geom,
        #                              scale=30)
        #change = ee.Number(lc_count.get('classification')).getInfo()
        #cutoff = ee.Number(1).subtract(change.divide(ee.Number(lc_count.get('classification'))))
        #cutoff = ee.Number(0)
        #lc_cut = lc_cl.reduceRegion(
        #    reducer = ee.Reducer.percentile([cutoff.multiply(100)]),
        #    geometry = geom,    
        #    scale = 30)
        #print('lccut',lc_cut.getInfo())
        #lcv1 = ee.Image(lc_cl.updateMask(lc_cl.gte(ee.Number(lc_cut.get('classification')))))
        #lcv1 = lc_cl
        lcv1 = lcpt
        lcv1 = lcv1.updateMask(lcv1.eq(lc_nn)).rename('classification').toByte()
        area_diff= FC_filt.first().get(str(lcclass_name)+'_diff')
        
        print('_area_diff',area_diff.getInfo())
        
        if area_diff.getInfo() == 0:
            area_diff = ee.Number(0)
        #area_next = ee.Number(FC_filt.first().get(str(lcclass_name)+'_to_'+str(lcclass_name)))#.divide(900)
        
        #lc_cb = lc_count.get('classification').getInfo()
        changb = 0
        for i in range(0,lcclasses_vector1.size().getInfo()):
            # print(lc_cb)
            changb1 = ee.Number(FC_filt.first().get(lcclasses_vector1.get(i).getInfo()+str('_to_')+str(lcclass_name)))
            changb = changb+changb1.getInfo()
        print('rest_of_change',changb)
        new_prop = ee.Number(area_diff.getInfo()/changb).abs()
        print('new_prop',new_prop.getInfo())
        #lcv1 = ee.Image(lc_cl.updateMask(lc_cl.gte(ee.Number(0))))             
    
    #print(lcclasses_vector1.getInfo())
    
    #print('area_urb_urb',lc_counta.getInfo())
    
    for i in range(0,lcclasses_vector1.size().getInfo()):
        lc_cl = lcclass_prob.select('classification').clip(geom)
        lc_cl = lc_cl.updateMask(lcpt.eq(lcc[i]))
        #lcvtemp = ee.ImageCollection(ee.List([lcv1]).flatten()).mosaic()
        #lc_cl = lc_cl.updateMask(lcvtemp)
        
        if mask_prev == 'mask_prev':
            lc_cl = lc_cl.where(im_tomask,-99999)
            lc_cl = lc_cl.updateMask(lc_cl.neq(-99999))
        #lc_cl = lc_cl.unmask()
        
        print('from',lcclasses_vector1.get(i).getInfo())
        print('to',lcclass_name)
        #print(lcc[i])
        prob1 = prob_threshold
        change = ee.Number(FC_filt.first().get(lcclasses_vector1.get(i).getInfo()+str('_to_')+str(lcclass_name))).multiply(new_prop)
        change = change.round()
        print('desired_change',change.getInfo())
        #print('change',change.getInfo())
        if method == 'high_prob':
            
            lc_cl = lc_cl.updateMask(lc_cl.gt(ee.Image(prob_list.get(i)).select('classification')))
            lc_count = lc_cl.reduceRegion(
                reducer=ee.Reducer.count(),
                geometry=geom,
                scale=30,
                maxPixels=1e13)
            cutoff = ee.Number(1).subtract(change.divide(ee.Number(lc_count.get('classification'))))#.round()
            #print('lc_cl',lc_cl.get('classification').getInfo())
            print('cutoff',cutoff.getInfo())
            #lvc = ee.Number(-9999999)
                        #to_mask= ee.Number(lc_cut.get('classification'))
            
            #lcv = lc_cl.updateMask(lc_cl.gte(to_mask)).toByte()                
            
        elif method == 'lowest_prob':
            cutoff1 = -1.1
            while cutoff1 < float(0):
                #print('begin prob',prob1)
                #print('i',i)
                lc_cl = lcclass_prob.select('classification').clip(geom)
                lc_cl = lc_cl.updateMask(lcpt.eq(lcc[i]))
                if mask_prev == 'mask_prev':
                    lc_cl = lc_cl.where(im_tomask,-99999)
                    lc_cl = lc_cl.updateMask(lc_cl.neq(-99999))
                lc_cl = lc_cl.updateMask(ee.Image(prob_list.get(i)).select('classification').lt(prob1))
                lc_count = lc_cl.reduceRegion( 
                    reducer = ee.Reducer.count(),
                    geometry = geom,
                    scale = 30,
                    maxPixels=1e13)
                #change = ee.Number(FC.filterMetadata(id_look,'equals',sa2_id).first().get(lcclasses_vector1.get(i).getInfo()+str('_to_')+str(lcclass_name))).divide(900)
                cutoff = ee.Number(1).subtract(change.divide(ee.Number(lc_count.get('classification'))))
                cutoff1 = ee.Number(cutoff).getInfo()
                #print('lc_count',lc_count.getInfo())
                #print('lc_change',change.getInfo())
                #print('cutoff1',cutoff1)
                if cutoff1 < 0:
                    prob1 = prob1 + 1000
        elif method == 'low_high_prob':
            cutoff1 = -1.1
            while cutoff1 < float(0):
                #print('begin prob',prob1)
                #print('i',i)
                lc_cl = lcclass_prob.select('classification').clip(geom)
                lc_cl = lc_cl.updateMask(lcpt.eq(lcc[i]))
                
                if mask_prev == 'mask_prev':
                    lc_cl = lc_cl.where(im_tomask,-99999)
                    lc_cl = lc_cl.updateMask(lc_cl.neq(-99999))
        
                lc_cl = lc_cl.updateMask(ee.Image(prob_list.get(i)).select('classification').lt(prob1))
                lc_cl = lc_cl.updateMask(lc_cl.gt(ee.Image(prob_list.get(i)).select('classification')))
                
                lc_count = lc_cl.reduceRegion( 
                    reducer = ee.Reducer.count(),
                    geometry = geom,
                    scale = 30,
                    maxPixels=1e13)
                #change = ee.Number(FC.filterMetadata(id_look,'equals',sa2_id).first().get(lcclasses_vector1.get(i).getInfo()+str('_to_')+str(lcclass_name))).divide(900)
                cutoff = ee.Number(1).subtract(change.divide(ee.Number(lc_count.get('classification'))))
                cutoff1 = ee.Number(cutoff).getInfo()
                if cutoff1 < 0:
                    prob1 = prob1 + 1000
        else:
            lc_count = lc_cl.reduceRegion(
                reducer=ee.Reducer.count(),
                geometry=geom,
                scale=30,
                maxPixels=1e13)
            cutoff = ee.Number(1).subtract(change.divide(ee.Number(lc_count.get('classification'))))
                            
           
        # lc_cut = lc_cl.reduceRegion(
        # reducer = ee.Reducer.percentile([cutoff.multiply(100)]),
        # geometry = geom,
        # scale = 30)
        # #print('lc_cut',lc_cut.getInfo())
        if cutoff.getInfo() < 0 :
                cutoff = ee.Number(0)
        lc_cut = lc_cl.reduceRegion(
                    reducer = ee.Reducer.percentile([cutoff.multiply(100)]),
                    geometry = geom,
                    scale = 30,
                    maxPixels=1e13)
        print(lc_cut.get('classification').getInfo())
        lcv = lc_cl.updateMask(lc_cl.gt(ee.Number(lc_cut.get('classification')))).toByte()
        
        lvc = lcv.reduceRegion(
                    reducer=ee.Reducer.count(),
                    geometry=geom,
                    scale=30,
                    maxPixels=1e13)
        print('lc_final',lvc.getInfo())        
        
        lcv1 = ee.List([lcv1,ee.Image(lcv)])
    
    
    new_lc = ee.ImageCollection(lcv1.flatten()).mosaic()
    
    return ee.Image(1).updateMask(new_lc.add(100)).toByte()
