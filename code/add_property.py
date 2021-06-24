# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 07:31:43 2021

@author: mcalderonloor
"""
#var sa2_tas = sa2.filterMetadata('STE_CODE11','equals','6')

import ee

def add_prop (FCh,FCc,period,size):
    tas_csv1 = FCc.filterMetadata('period','equals',period)
    tas_csv1 = tas_csv1.sort('SA2_MAIN11')
    l = tas_csv1.toList(size)
    FCh = FCh.sort('SA2_MAIN11')
    ll = FCh.toList(size)
    f = ee.Feature(l.get(0)).setGeometry(ee.Feature(ll.get(0)).geometry().simplify(10))
    f = ee.FeatureCollection([f,ee.Feature(l.get(1)).setGeometry(ee.Feature(ll.get(1)).geometry().simplify(10))])
    for i in range(2,size):
        g = ee.Feature(l.get(i)).setGeometry(ee.Feature(ll.get(i)).geometry().simplify(10))
        f = f.merge(ee.Feature(g))
    
    idList = ee.List.sequence(0,f.size().subtract(1))
    
    llist = f.toList(f.size())
    
    def nf (newSysIndex):
        feat = ee.Feature(llist.get(newSysIndex))
        indexString = ee.Number(newSysIndex).format('%03d')
        return feat.set('system:index', indexString, 'ID', indexString)
       
    return ee.FeatureCollection(idList.map(nf))
   