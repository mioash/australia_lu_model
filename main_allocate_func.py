# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 10:03:12 2021

@author: mcalderonloor
"""
import ee
ee.Initialize()
import numpy as np

from dwn_main_func import build_df
#from dwn_main_func import LatLonImg
from dwn_main_func import dwn_save
year = 2010

STT = 'Northern Territory'

if STT == 'New South Wales':        
    stt = 'nsw'
elif STT == 'Victoria':
    stt = 'vic'
elif STT == 'Queensland':
    stt = 'qld'
elif STT == 'South Australia':
    stt = 'sa'
elif STT == 'Western Australia':
    stt = 'wa'
elif STT == 'Tasmania':
    stt = 'tas'
elif STT == 'Northern Territory':
    stt = 'nt'
elif STT == 'Australian Capital Territory':
    stt = 'act'

probs, sgrid, sgridsize = build_df(year,STT,702031057)

#llist = list(range(0,sgridsize))
llist = list(range(1,sgridsize+1))

#lend = (sgridsize//100)+1
#for i in range(0,sgridsize-100,100):
for i in range(1700,1800,100):
    print(i)
    #dwn_save(i,i+100)
    dwn_save (stt,i,i+100,sgrid,llist,probs,year)


dwn_save (stt,i+100,sgridsize,sgrid,llist,probs,year)
dwn_save (stt,0,sgridsize,sgrid,llist,probs,year,702031057)
#dwn_save(i+100,sgridsize)


#bad nt = [702031057,702051066,702041063,702031061]



    