#!/usr/bin/env python
# coding: utf-8
import ee 
from ee import batch
ee.Initialize()
# In[26]:

def make_climate(y_ini,y_end,redcr,varname,redd):

    tmin = ee.List([ee.Image('users/mioash/drivers_lcaus/climate/tmin/tmin1982')])
    tmax = ee.List([ee.Image('users/mioash/drivers_lcaus/climate/tmax/tmax1982')])
    rain = ee.List([ee.Image('users/mioash/drivers_lcaus/climate/rainfall/rain1982')])

    for l in range(1983,2020):
        tmin = tmin.cat(ee.List([ee.Image('users/mioash/drivers_lcaus/climate/tmin/tmin'+str(l))]))
        tmax = tmax.cat(ee.List([ee.Image('users/mioash/drivers_lcaus/climate/tmax/tmax'+str(l))]))
        rain = rain.cat(ee.List([ee.Image('users/mioash/drivers_lcaus/climate/rainfall/rain'+str(l))]))
    
    tmean = ee.List([ee.Image(tmin.get(0)).add(ee.Image(tmax.get(0))).divide(2)])
    for r in range (1,38):
        tmean = tmean.cat(ee.List([ee.Image(tmin.get(r)).add(ee.Image(tmax.get(r))).divide(2)]))
    #print(tmean.length().getInfo())    
    if varname == 'tmin':
        image = tmin
    elif varname == 'tmax':
        image = tmax
    elif varname == 'tmean':
        image = tmean
    elif varname == 'rain':
        image = rain
    
    def long_term (image,lm,renam,ini,end,redc):
        if renam == 'DJF':
            M1 = ee.List([ee.Image(image.get(ini-1983)).select(ee.String(lm.get(0))).rename(renam)])
            M2 = ee.List([ee.Image(image.get(ini-1982)).select(ee.String(lm.get(1))).rename(renam)])
            M3 = ee.List([ee.Image(image.get(ini-1982)).select(ee.String(lm.get(2))).rename(renam)])
        else:
            M1 = ee.List([ee.Image(image.get(ini-1982)).select(ee.String(lm.get(0))).rename(renam)])
            M2 = ee.List([ee.Image(image.get(ini-1982)).select(ee.String(lm.get(1))).rename(renam)])
            M3 = ee.List([ee.Image(image.get(ini-1982)).select(ee.String(lm.get(2))).rename(renam)])
        imc = ee.ImageCollection(M1).merge(ee.ImageCollection(M2)).merge(ee.ImageCollection(M3))
        if varname == 'rain':
            seas = ee.List([imc.sum()])
        else:
            seas = ee.List([imc.mean()])
        if renam=='DJF':
            for i in (ini-1981,end-1982): #1982
                MM1 = ee.List([ee.Image(image.get(i-1)).select(ee.String(lm.get(0))).rename(renam)])
                MM2 = ee.List([ee.Image(image.get(i)).select(ee.String(lm.get(1))).rename(renam)])
                MM3 = ee.List([ee.Image(image.get(i)).select(ee.String(lm.get(2))).rename(renam)])
                imc = ee.ImageCollection(MM1).merge(ee.ImageCollection(MM2)).merge(ee.ImageCollection(MM3))
                if varname=='rain':
                    seas1 = ee.List([imc.sum()])
                else:
                    seas1 = ee.List([imc.mean()])
            seas = seas.cat(seas1)
        else:
            for j in (ini-1982,end-1982): #1982
                MM1 = ee.List([ee.Image(image.get(j)).select(ee.String(lm.get(0))).rename(renam)])
                MM2 = ee.List([ee.Image(image.get(j)).select(ee.String(lm.get(1))).rename(renam)])
                MM3 = ee.List([ee.Image(image.get(j)).select(ee.String(lm.get(2))).rename(renam)])
                imc = ee.ImageCollection(MM1).merge(ee.ImageCollection(MM2)).merge(ee.ImageCollection(MM3))
                if varname=='rain':
                    seas1 = ee.List([imc.sum()])
                else:
                    seas1 = ee.List([imc.mean()])
                seas = seas.cat(seas1)
        return ee.ImageCollection(seas).reduce(redc)

    def make_clim (imagea,y_inia,y_enda,redcra,varnamea):
        v_DJF = long_term(imagea,ee.List(['b12','b1','b2']),'DJF',y_inia,y_enda,redcra).rename(varnamea+'_djf_'+str(y_end+1)+'_'+str(redd))
        v_MAM = long_term(imagea,ee.List(['b3','b4','b5']),'MAM',y_inia,y_enda,redcra).rename(varnamea+'_mam_'+str(y_end+1)+'_'+str(redd))
        v_JJA = long_term(imagea,ee.List(['b6','b7','b8']),'JJA',y_inia,y_enda,redcra).rename(varnamea+'_jja_'+str(y_end+1)+'_'+str(redd))
        v_SON = long_term(imagea,ee.List(['b9','b10','b11']),'SON',y_inia,y_enda,redcra).rename(varnamea+'_son_'+str(y_end+1)+'_'+str(redd))
        if varname=='rain':
            v_year = v_DJF.add(v_MAM).add(v_JJA).add(v_SON).rename(varnamea+'_year_'+str(y_enda+1))
        else:
            v_year = v_DJF.add(v_MAM).add(v_JJA).add(v_SON).divide(4).rename(varnamea+'_year_'+str(y_enda+1))
        im_out = ee.ImageCollection([v_DJF,v_MAM,v_JJA,v_SON,v_year])
        return im_out
    return make_clim (image,y_ini,y_end,redcr,varname)


