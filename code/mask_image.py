# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 06:15:49 2020

@author: mcalderonloor
"""

import ee 
#from ee_plugin import Map 

lc_all= ee.Image("users/mioash/Calderon_etal_Australian_land-cover/Lc_Aus_1985_2015_v1")
ent_all = ee.Image('users/mioash/Calderon_etal_Australian_land-cover/Australia_ent_v1')
mov_all = ee.Image('users/mioash/Calderon_etal_Australian_land-cover/margin_of_victory')

# define_image (mov threshold, entropy threshold, type of filter ('ent','mov','both','tmov'),prediction's year, number of time-steps)
def define_image(mov_th,entr_th,filt,year,ntspts):
    blc = 'b'+str(year)
    elc = 'e'+str(year)
    mlc = 'mov'+str(year)
    all_years = range(year-(5*ntspts),year-4,5)
    blci = ["{}{}".format(str('b'),i) for i in all_years]
    beni = ["{}{}".format(str('e'),i) for i in all_years]
    bmvi = ["{}{}".format(str('mov'),i) for i in all_years]
    
    lc_dep = lc_all.select(blc).rename('lc_dep')
    l_stps = range(1,len(all_years)+1)
    
    renam = ["{}{}".format(str('lc_t'),i) for i in l_stps]
    
    ld_indep = lc_all.select(blci).rename(renam)
    
    if filt=='ent':
        elcf = ent_all.select(elc)
        elcf= elcf.updateMask(elcf.gte(entr_th))
        maskk = elcf
        return lc_dep.addBands(ld_indep).updateMask(maskk)
    elif filt== 'mov':
        mlcf = mov_all.select(mlc)
        mlcf = mlcf.updateMask(mlcf.gte(mov_th))
        maskk= mlcf
        return lc_dep.addBands(ld_indep).updateMask(maskk)
    elif filt=='both':
        elcf = ent_all.select(elc)
        elcf = elcf.updateMask(elcf.gte(entr_th))
        mlcf = mov_all.select(mlc)
        mlcf = mlcf.updateMask(mlcf.gte(mov_th))
        maskk = elcf.upadteMask(mlcf)
        return lc_dep.addBands(ld_indep).updateMask(maskk)
    elif filt=='tmov':
        #mlcf = mov_all.select(bmvi)
        #mlcf = mlcf.reduce(ee.Reducer.median())
        #mlcf = mlcf.updateMask(mlcf.gte(mov_th))
        #maskk = mlcf
        mlcf = mov_all.select(mlc)
        mlcf = mlcf.updateMask(mlcf.gte(mov_th))
        lc_dep = lc_dep.updateMask(mlcf.gte(mov_th))
        for i in (0,len(bmvi)-1):
            my = mov_all.select(bmvi[i])
            my = my.updateMask(my.gte(mov_th))
            lc_dep = lc_dep.updateMask(my)
        return lc_dep.addBands(ld_indep).updateMask(lc_dep.add(1))
    elif filt=='none':
        return lc_dep.addBands(ld_indep)#.updateMask(maskk)
        