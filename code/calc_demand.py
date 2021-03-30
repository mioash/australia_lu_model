# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 10:15:38 2020

@author: -
"""

import ee

tas_csv = ee.FeatureCollection('users/mioash/transitions/tas_summary_hex')
tas_hex = ee.FeatureCollection('users/mioash/tas_hex_clipped2')

def add_prop (FCh,FCc,period,size):
    tas_csv1 = FCc.filterMetadata('period','equals',period)
    def nf (feat):
        idd = ee.Feature(feat).get('hex_new')
        return ee.Feature(feat).set("Id_hex_new", idd)
    tas_csv1 = ee.FeatureCollection(tas_csv1.map(nf))
    tas_csv1 = tas_csv1.sort('Id_hex_new')
    l = tas_csv1.toList(size)
    FCh = FCh.sort('Id_hex_new')
    ll = FCh.toList(size)
    f = ee.Feature(l.get(0)).setGeometry(ee.Feature(ll.get(0)).geometry())
    f = ee.FeatureCollection([f,ee.Feature(l.get(1)).setGeometry(ee.Feature(ll.get(1)).geometry())]) 
    for i in range(2,size):
        g = ee.Feature(l.get(i)).setGeometry(ee.Feature(ll.get(i)).geometry())
        f = f.merge(ee.Feature(g))
    return f

#lcclass = 'urban'

def to_img (FCa,lclass):
    iml = ee.Image().clip(tas_hex).paint(FCa,str(lclass)+'_loss')
    imw = ee.Image().clip(tas_hex).paint(FCa,str(lclass)+'_win')
    #var imc = ee.Image().clip(tas_hex).paint(FCa,lclass+'_stable')
    imw = imw.subtract(iml)#//.add(imc)
    return imw.divide(1000000)
##calc_demand(FCg=FC with geometry, FCc=FC csv, period = period for calculating the demand, ltrend = Fit a linear regression (T/F), ssize = number of features)
def calc_demand_period(FCg,FCc,period,year,ssize,lcclass):
    p8590 = add_prop(FCg,FCc,period,ssize)
    #p9095 = add_prop(tas_hex,tas_csv,'90-95',33)
    #p9500 = add_prop(tas_hex,tas_csv,'95-00',33)
    #p0005 = add_prop(tas_hex,tas_csv,'00-05',33)
    #p0510 = add_prop(tas_hex,tas_csv,'05-10',33)
    #p1015 = add_prop(tas_hex,tas_csv,'10-15',33)
    return to_img(p8590,lcclass).addBands(ee.Image(year).toInt().rename('year')).set('system:time_start',ee.Date.fromYMD(year,1,1).millis())
#c09 = to_img(p9095,lcclass).addBands(ee.Image(1995).toInt().rename('year')).set('system:time_start',ee.Date.fromYMD(1995,1,1).millis())
#c50 = to_img(p9500,lcclass).addBands(ee.Image(2000).toInt().rename('year')).set('system:time_start',ee.Date.fromYMD(2000,1,1).millis())
#c00 = to_img(p0005,lcclass).addBands(ee.Image(2005).toInt().rename('year')).set('system:time_start',ee.Date.fromYMD(2005,1,1).millis())
#c51 = to_img(p0510,lcclass).addBands(ee.Image(2010).toInt().rename('year')).set('system:time_start',ee.Date.fromYMD(2010,1,1).millis())
#c01 = to_img(p1015,lcclass).addBands(ee.Image(2015).toInt().rename('year')).set('system:time_start',ee.Date.fromYMD(2015,1,1).millis())
def calc_demand_regression(imc,iyear,eyear,pyear):
#imc = ee.ImageCollection([c59,c09,c50,c00,c51,c01])
    wp =  imc.select(['year', 'constant']).reduce(ee.Reducer.linearFit())
    years = ee.List.sequence(iyear,eyear,5)
    def add_time(y):
        t = ee.Date.fromYMD(y,1,1).millis()
        img = wp.select("offset").add(wp.select("scale").multiply(ee.Number(y)))
        return img.set("system:time_start",t)
    collection = ee.ImageCollection(years.map(add_time))
    return collection.filterDate(ee.Date.fromYMD(pyear,1,1),ee.Date.fromYMD(pyear,12,31))