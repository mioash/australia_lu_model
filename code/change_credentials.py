# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 06:20:08 2021

@author: mcalderonloor
"""
import shutil
import os

def change_credentials (usser):
    src = r'C:/Users/mcalderonloor/.config/earthengine/'+usser
    dest = r'C:/Users/mcalderonloor/.config/earthengine'
    files = os.listdir(src)
    os.chdir(src)
    for file in files:
        if os.path.isfile(file):
            shutil.copy(file, dest)


# change_credentials('kumi')
# import ee
# ee.Initialize()
# import geemap


# print(geemap.ee_user_id())
        
