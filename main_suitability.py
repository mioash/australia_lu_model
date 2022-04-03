# -*- coding: utf-8 -*-
"""
Created on Sat Jun 26 09:56:12 2021

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

bound= ee.Image('users/mioash/drivers_lcaus/boundaries/boundaries_ready')
sa2 = ee.FeatureCollection('users/mioash/drivers_lcaus/boundaries/SA2')

def id_parse(FCC):
    return ee.Feature(FCC).set('SA2_MAIN11', ee.Number.parse(FCC.get('SA2_MAIN11')))

def myproperties(feature):
    feature=ee.Feature(feature).setGeometry(None)
    return feature    

def add_dist_calc(img_dist):
    immm = img_dist.select('lc_t1').rename('lc_temp')
    #immm = lc.select('b2005')
    imf = immm.add(1)
    immm1 = imf
    
    for i in range(1,6):
        #idist = add_distance(idist,lcclasses[i],'lc_temp',geom,cpixels[i])
        imf1 = imf.updateMask(imf.eq(i))
        patchsize = imf1.connectedPixelCount(100, False)
        imf1 =  imf1.updateMask(patchsize.gt(15))
        dist = imf1.fastDistanceTransform(1000).sqrt().multiply(ee.Image.pixelArea().sqrt())
        dist = dist.select('distance').rename('distto_'+str(i-1))
        dist= dist.unitScale(0,100000).multiply(255).round().toByte()
        dist = dist.updateMask(immm1)
        immm = immm.addBands(dist)
    return immm.select(['distto_0','distto_1','distto_2','distto_3','distto_4'])
    
def suitability (train_year,pred_year,lc_class,lc_t1,lc_t2,lc_t3,lc_t4,lc_t5,neigh_suit,neigh_img,exp_ptos):
    #val_year = 2015
#    train_year = train_year
    #i_year = 1983
    #lc_img = define_image(50,0,'tmov',train_year,5) #temporal mov threshold >= 40, 
    #lc_img_classify = define_image(0,0,'none',train_year,5)
    lim_temp_f = ee.FeatureCollection('users/mioash/aust_cd66states')
    geom = lim_temp_f
    
    llx = 108.76 
    lly = -44 
    urx = 155 
    ury = -10  #australia
    geometri = [[llx,lly], [llx,ury], [urx,ury], [urx,lly]]
    
    ##for masking out points with high uncertainty
    lc_img_for_masking = define_image(50,0,'tmov',train_year,5)
    ##image to be used for training the model (always the same 1985-2005)
    lc_img_train = define_image(0,0,'none',train_year,5)
    
    #Images to be used for classification
    lc_t1 = lc_t1.rename('lc_t1') ## t - 1 (pred_year - 5)
    lc_t2 = lc_t2.rename('lc_t2') ## t - 2
    lc_t3 = lc_t3.rename('lc_t3')
    lc_t4 = lc_t4.rename('lc_t4')
    lc_t5 = lc_t5.rename('lc_t5') ## t - 5
    
    #Create the time-series image for doing the classification in pred_year
    lc_img_classify = lc_t1.addBands(lc_t2).addBands(lc_t3).addBands(lc_t4).addBands(lc_t5)
    img_pred = lc_t1
    
    #Call and create the stack with images that doesn't change with time
    #Output connectivity.select(['cindex']).addBands(physical).addBands(soil).addBands(anu_clim)
    fst_img = stacking(train_year-10,train_year-5,'mean',False,False) # initial year, end year, reducer for climate, get climate, get precipitation
       
    proj = img_pred.projection()
    
    pop = ee.ImageCollection("projects/sat-io/open-datasets/hrsl/hrslpop").median().clip(geom)
    pop = pop.unmask(0).reproject(proj.atScale(30))#.clip(geom)
    pop = pop.select('b1').rename('pop')

    ##Call image collection with info about distances
    ##This will be the same image for training the model always
    ##The following will be added to each future time-step 'distto_npa','dist_roads','dist_wways' 
    dist_2005 = ee.Image('users/mioash/drivers_lcaus/Distances_nigh_national_2005_byte')
    distto = dist_2005.select(['distto_npa','dist_roads','dist_wways'])
    #bfst = fst_img.bandNames().getInfo()   
    
    #################CHANGE ACCESSIBILITY
        
    lcclasses = [0,1,2,3,4]
    cpixels = [15,15,15,15,15]
    ##If neighbourhood is not available run this first
    if neigh_suit =='neigh':
        # idist = img_pred.select('lc_t1').rename('lc_temp')
        
        # for i in range(0,len(lcclasses)):
        #     idist = add_distance(idist,lcclasses[i],'lc_temp',geom,cpixels[i])
        
        # idist = idist.select(idist.bandNames().remove(str('lc_t1')).getInfo())
        # idist = idist.select(idist.bandNames().remove(str('lc_temp')).getInfo())
        
        # nat_npa = ee.Image("users/mioash/drivers_lcaus/natural_areas/nat_npa_ready")
        # nat_npa = nat_npa.select('npa')
        
        ineigh = ee.Image(1).addBands(add_neigh(img_pred.select('lc_t1'),3,1,False,'_nb')) #-> r=1
        ineigh = ineigh.addBands(add_neigh(img_pred.select('lc_t1'),9,3,True,'_3b')) #-> r=4
        
        ineigh = ineigh.addBands(add_neigh(img_pred.select('lc_t1'),27,9,True,'_9b')) #-> r=13
        
        ineigh = ineigh.addBands(add_neigh(img_pred.select('lc_t1'),81,27,True,'_27b')) #-> r=33
        
        #im_to_exp = idist.addBands(ineigh)
        #idist= idist.unitScale(0,1372959).multiply(255).round().toByte()

        #neigh = distn.select(nneigh).rename(nneigh1)
        ineigh = ineigh.unitScale(0,100).multiply(255).round().toByte()
        im_to_exp = ineigh#idist.addBands(ineigh)
        
        im_to_exp  = im_to_exp.toByte()
        #im_to_exp  = idist.toByte()
        # ee.batch.Export.image.toAsset(image=im_to_exp,
        #                               description='Only_distances_national_'+str(pred_year-5),
        #                               assetId='users/mioash/drivers_lcaus/Only_distances_national_'+str(pred_year-5),scale=30,
        #                              region = geometri,maxPixels=1e13,crs='EPSG:4326').start()
        
        ee.batch.Export.image.toAsset(image=im_to_exp,
                                      description='Only_neigh_national_gt_'+str(pred_year-5)+'for_pred_'+str(pred_year),
                                      assetId='users/mioash/drivers_lcaus/Only_neigh_national_gt_'+str(pred_year-5)+'for_pred_'+str(pred_year),scale=30,
                                      region = geometri,maxPixels=1e13,crs='EPSG:4326').start()
    elif neigh_suit =='suitability':
        
        #lcclasses = [0,1,2,3,4]
        #cpixels = [15,15,15,15,15]
        lc_alt= ee.Image("users/mioash/Calderon_etal_Australian_land-cover/Lc_Aus_1985_2015_v1")
        lc_2005 = lc_alt.select('b2005').rename('lc_t1')
        
        idist_training = add_dist_calc(lc_2005)
        idist_pred = add_dist_calc(img_pred)
        print(idist_training.bandNames().getInfo())
        print(idist_pred.bandNames().getInfo())
        # immm = img_pred.select('lc_t1').rename('lc_temp')
        # #immm = lc.select('b2005')
        # imf = immm.add(1)
        # immm1 = imf
        
        # #idist = img_pred.select('lc_t1').rename('lc_temp')
        
        # for i in range(1,6):
        #     #idist = add_distance(idist,lcclasses[i],'lc_temp',geom,cpixels[i])
        #     imf1 = imf.updateMask(imf.eq(i))
        #     patchsize = imf1.connectedPixelCount(100, False)
        #     imf1 =  imf1.updateMask(patchsize.gt(15))
        #     dist = imf1.fastDistanceTransform(1000).sqrt().multiply(ee.Image.pixelArea().sqrt())
        #     dist = dist.select('distance').rename('distto_'+str(i-1))
        #     dist= dist.unitScale(0,100000).multiply(255).round().toByte()
        #     dist = dist.updateMask(immm1)
        #     immm = immm.addBands(dist)
        # #idist = idist.select(idist.bandNames().remove(str('lc_t1')).getInfo())
        
        
        # # for (var i=1;i<6;i++){
          
        # #   var imf1 = imf.updateMask(imf.eq(i))
        # #   //Map.addLayer(imf1,'','mask'+(i-1))
        # #   var patchsize = imf1.connectedPixelCount(100, false)
        # #   imf1 =  imf1.updateMask(patchsize.gt(15))
        # #   var dist = imf1.fastDistanceTransform(1000).sqrt().multiply(ee.Image.pixelArea().sqrt())
        # #   dist = dist.select('distance').rename('distto_'+(i-1))
        # #   dist= dist.unitScale(0,100000).multiply(255).round().toByte()
        # #   dist = dist.updateMask(immm1)
        # #   immm = immm.addBands(dist)
        # # }
        
        # # //imf = imf.addBands(dist)
        # immm = immm.select(['distto_0','distto_1','distto_2','distto_3','distto_4'])


        sa21 = sa2.map(id_parse)
        sa2_img = ee.Image(-9999).paint(sa21,'SA2_MAIN11').rename('sla_id').updateMask(lc_t1.add(1).select('lc_t1'))
        boundaries = bound.select(['ste','ecoregion'])
        ##building image for training
        dist_neigh_2005 = ee.Image('users/mioash/drivers_lcaus/Distances_nigh_national_2005_byte')
        
        bands_dist = dist_neigh_2005.bandNames().getInfo()        
        item_dist = [e for e in bands_dist if e not in ('distto_0','distto_1','distto_2','distto_3','distto_4')]
        dist_neigh_2005 = dist_neigh_2005.select(item_dist).addBands(idist_training) 
        
        bands_dist_pred = neigh_img.bandNames().getInfo()        
        item_dist_pred = [e for e in bands_dist_pred if e not in ('distto_0','distto_1','distto_2','distto_3','distto_4')]
        neigh_img = neigh_img.select(item_dist_pred).addBands(idist_pred) 
        
        
        img_ready = lc_img_train.addBands(fst_img).addBands(dist_neigh_2005).addBands(pop).addBands(sa2_img).addBands(boundaries)
        
        img_readya = img_ready.updateMask(lc_img_for_masking.add(1).select('lc_dep'))
        img_readya = img_readya.unmask(-9999,False).updateMask(lc_img_train.add(ee.Image(1)).select('lc_t1'))
        
        bbands = img_ready.bandNames().getInfo()
        print('bands',bbands)
        
        #Build individual suitability layers
        lc_cnc_c = suit_layer(lc_img_train,'binary',True,0,True)#c
        lc_cnc_f = suit_layer(lc_img_train,'binary',True,1,True)#f
        lc_cnc_g = suit_layer(lc_img_train,'binary',True,2,True)#g
        lc_cnc_u = suit_layer(lc_img_train,'binary',True,3,True)#u
        
        abands = ['lc_dep', 'lc_t1', 'lc_t2', 'lc_t3', 'lc_t4', 'lc_t5', 
       'cindex', 'elevation', 'slope', 'aspect', 'landforms', 'strata',
       'DES_000_200_EV', 'SOC', 'BDW', 'CLY', 'SLT', 'SND', 'PH', 'AWC', 'NTO', 'PTO', 
       'p01', 'p05', 'p06', 'p08', 'p09', 'p10', 'p11', 'p12', 'p13', 'p14', 'p16', 'p17', 'p18', 'p19', 
       'distto_0', 'distto_1', 'distto_2', 'distto_3', 'distto_4', 'distto_npa', 'dist_roads', 'dist_wways', 
       'c_3_nb', 'f_3_nb', 'g_3_nb', 'b_3_nb', 'w_3_nb', 'o_3_nb', 
       'c_9_3b', 'f_9_3b', 'g_9_3b', 'b_9_3b', 'w_9_3b', 'o_9_3b', 
       'c_27_9b', 'f_27_9b', 'g_27_9b', 'b_27_9b', 'w_27_9b', 'o_27_9b', 
       'c_81_27b', 'f_81_27b', 'g_81_27b', 'b_81_27b', 'w_81_27b', 'o_81_27b', 
       'pop', 'sla_id', 'ste', 'ecoregion']
        
        lcclass = lc_class#'cropland'
        if lc_class == 'cropland':
            lc_imgb = lc_cnc_c
            dont_use_bands = ['lc_dep','strata',
                ##low_importance
                'p14','p16', 'p17', 'p18', 'p19', 'BDW', 'SND', 'AWC', 'NTO', 'PTO','DES_000_200_EV','aspect','landforms',
                 #correlated
                'p05', 'p11', 'p13']

        elif lc_class == 'forest':
            lc_imgb = lc_cnc_f
            dont_use_bands = ['lc_dep','strata',
                ##low_importance
                    'p16', 'p17', 'p18', 'p19','SOC','CLY','PH', 'BDW', 'SND', 'AWC', 'NTO', 'PTO','DES_000_200_EV',
                   'aspect','landforms']

        elif lc_class == 'grassland':
            lc_imgb = lc_cnc_g
            dont_use_bands = ['lc_dep','strata',
                ##low_importance
                'SOC','CLY', 'BDW', 'NTO', 'SLT', 'PTO','DES_000_200_EV','elevation','slope','cindex',
                    'aspect','landforms','AWC','distto_npa','dist_wways','pop',
                  'p08','p12','p13','p09','p14','p16', 'p18','SND','dist_roads']

        elif lc_class == 'urban':
            lc_imgb = lc_cnc_u
            dont_use_bands = ['lc_dep','strata',
                ##low_importance
                'SOC','CLY','PH', 'BDW', 'NTO', 'PTO','DES_000_200_EV',
                   'aspect','landforms','AWC','distto_npa','dist_wways',
                  'p09','p14','p13','p16', 'p17', 'p18', 'p19','SLT']

        
        band_multi = img_ready.bandNames().getInfo()        
        item_list = [e for e in band_multi if e not in ('lc_dep', 'lc_t1', 'lc_t2', 'lc_t3', 'lc_t4', 'lc_t5')]
        lc_imgb = lc_imgb.addBands(img_ready.select(item_list)) 
        
        lc_imgba = lc_imgb.updateMask(lc_img_for_masking.add(1).select('lc_dep'))
        lc_imgba = lc_imgb.unmask(-9999,False).updateMask(lc_img_classify.add(ee.Image(1)).select('lc_t1'))
        #print(lc_imgb.bandNames().getInfo())
                
        if exp_ptos:
            strat = make_strata(lc_imgba,2,True)
            #strat = make_strata(img,6,False)
            ssamples = prop_allocation(lc_imgba.addBands(ee.Image(strat).toInt()),geom,25000)
            ee.batch.Export.table.toAsset(collection=ssamples,
                              description='train_AUS_25k_'+lcclass,
                              assetId='users/mioash/ALUM/train_AUS_25k_'+str(lcclass)).start()
        else:                       
            ##For classification    
            dist_neigh_classify = neigh_img.addBands(distto)        
            img_readyc = lc_img_classify.addBands(fst_img) \
                .addBands(dist_neigh_classify) \
                    .addBands(pop) \
                        .addBands(sa2_img) \
                            .addBands(boundaries)
        
            img_readyac = img_readyc#.updateMask(lc_img_for_masking.add(1).select('lc_dep'))
            img_readyac = img_readyac.unmask(-9999,False).updateMask(lc_img_train.add(ee.Image(1)).select('lc_t1'))
            
            lcclass = lc_class#'cropland'
            if lc_class == 'cropland':
                
                lc_imgc = suit_layer(lc_img_classify,'binary',True,0,False)#c
                #ssamples= ee.FeatureCollection('users/mioash/Calderon_etal_Australian_land-cover/C_NC_20k_cropland')
                ssamples= ee.FeatureCollection('users/mioash/ALUM/train_AUS_25k_cropland')
            elif lc_class == 'forest':
                lc_imgc = suit_layer(lc_img_classify,'binary',True,1,False)#f
                ssamples = ee.FeatureCollection('users/mioash/ALUM/train_AUS_25k_forest')

            elif lc_class == 'grassland':
                lc_imgc = suit_layer(lc_img_classify,'binary',True,2,False)#g
                ssamples = ee.FeatureCollection('users/mioash/ALUM/train_AUS_25k_grassland')

            elif lc_class == 'urban':
                lc_imgc = suit_layer(lc_img_classify,'binary',True,3,False)#u
                ssamples = ee.FeatureCollection('users/mioash/ALUM/train_AUS_25k_urban')
                
            for bnd in dont_use_bands:
                abands.remove(bnd)
            
            lc_imgc = lc_imgc.addBands(img_readyac.select(item_list)) 
            
        
            lc_imgc = lc_imgc#.updateMask(lc_img_for_masking.add(1).select('lc_dep'))
            lc_imgc = lc_imgc.unmask(-9999,False).updateMask(lc_img_classify.add(ee.Image(1)).select('lc_t1'))
        
            print(lc_imgc.bandNames().getInfo())
            #print('bin',abands)
            #ssamples= ee.FeatureCollection('users/mioash/ALUM/train_AUS_20k_'+str(lcclass))
            mtry = round(len(abands)*(2/3),0)
            
            # classifier1 = ee.Classifier.smileRandomForest(numberOfTrees=100,
            #                                                variablesPerSplit =mtry ,
            #                                               minLeafPopulation=4,
            #                                              maxNodes = 40, seed = 55).setOutputMode('PROBABILITY')
            classifier1 = ee.Classifier.smileRandomForest(numberOfTrees=150,
                                                           variablesPerSplit =mtry ,
                                                          minLeafPopulation=4,
                                                         maxNodes = 30, seed = 155).setOutputMode('PROBABILITY')
            #Train the classifier.
            trainer1 = classifier1.train(features=ssamples,classProperty ='lc_dep',
                                        inputProperties= abands)
            #trainer2 = classifier2.train(trainingvals,'lc_dep')           
            classified1 = lc_imgc.classify(trainer1).toDouble()
            
            # Export image            
            ee.batch.Export.image.toAsset(image=classified1,
                                          description=str(pred_year)+'_SUIT_AUS_v3_'+lcclass,
                                          assetId='users/mioash/Calderon_etal_Australian_land-cover/'+str(pred_year)+'_SUIT_AUS_v3_25k_150trees_'+lcclass,scale=30,
                                         region = geometri,maxPixels=1e13,crs='EPSG:4326').start()