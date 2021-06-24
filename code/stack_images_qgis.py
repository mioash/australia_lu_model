import ee 
#from ee_plugin import Map 
from make_clim import make_climate

connectivity = ee.Image("users/mioash/drivers_lcaus/aria/connectivity_ready")
fires = ee.Image("users/mioash/drivers_lcaus/fires/ffrequency_ready")
nat_npa = ee.Image("users/mioash/drivers_lcaus/natural_areas/nat_npa_ready")
physical = ee.Image("users/mioash/drivers_lcaus/physical/physical_ready")
pdensity = ee.Image("users/mioash/drivers_lcaus/population/pdensity_ready")
soil = ee.Image("users/mioash/drivers_lcaus/soil/soil_ready")
tenure = ee.Image("users/mioash/drivers_lcaus/tenure/tenure_ready")
soil = ee.Image("users/mioash/drivers_lcaus/soil/soil_ready")

lclim = ['p05','p06','p08','p09','p10','p11','p12','p13','p14','p16','p17','p18','p19']
anu_clim = ee.Image("users/mioash/drivers_lcaus/climate/anuclim/anu_p01").rename('p01')

for i in range(0,len(lclim),1):
    px = ee.Image("users/mioash/drivers_lcaus/climate/anuclim/anu_"+str(lclim[i])).rename(lclim[i])
    anu_clim = anu_clim.addBands(px)
    
#p6 = ee.Image("users/mioash/drivers_lcaus/climate/anuclim/anu_p06").rename('p6')
#p12 = ee.Image("users/mioash/drivers_lcaus/climate/anuclim/anu_p12").rename('p12')
#p14 = ee.Image("users/mioash/drivers_lcaus/climate/anuclim/anu_p14").rename('p14')

#anu_clim = p1.addBands(p5).addBands(p6).addBands(p12).addBands(p14)

proj = connectivity.select('accessibility').projection();
anu_clim = anu_clim.reproject(proj.atScale(30))

def fill_holes (img,n_iter,min_scale,max_scale,increment):
  vals1 = ee.List.sequence(min_scale,max_scale,increment)
  for i in range(0, n_iter):
    val = vals1.get(ee.Number(i))
    imm1 = img.reproject(proj.atScale(val))
    img = img.unmask(imm1,False)
  return img.updateMask(fires.select('ffreq').add(100))

con1 = fill_holes(connectivity.select('accessibility'),10,1000,10000,1000)

def stacking(yearl,yeara,red_clim,just_clim,rrain):
    
    # if yeara==1985:
    #     yearl = 1983
    # else:
    #     yearl = yeara-4
    
    if red_clim == 'mean':
        redc = ee.Reducer.mean()
    elif red_clim == 'min':
        redc = ee.Reducer.min()
    elif red_clim == 'max':
        redc = ee.Reducer.max()
    elif red_clim == 'p10':
        redc = ee.Reducer.percentile([10])
    elif red_clim == 'p50':
        redc = ee.Reducer.percentile([50])
    elif red_clim == 'p90':
        redc = ee.Reducer.percentile([90])
    
    if just_clim is True:
        tmin = make_climate(yearl,yeara,redc,'tmin',red_clim).toBands()
        tmax = make_climate(yearl,yeara,redc,'tmax',red_clim).toBands()
        tmean = make_climate(yearl,yeara,redc,'tmean',red_clim).toBands()
        rain = make_climate(yearl,yeara,redc,'rain',red_clim).toBands()
        if rrain is True:
            return tmin.addBands(tmax).addBands(tmean).addBands(rain)
        else:
            return tmin.addBands(tmax).addBands(tmean)
    else:
        #return connectivity.select(['friction','cindex','aria_1km']).addBands(con1).addBands(fires).addBands(physical).addBands(soil).addBands(anu_clim).addBands(tenure)#.addBands(clim)
        return connectivity.select(['cindex']).addBands(physical).addBands(soil).addBands(anu_clim).addBands(tenure)#.addBands(clim)

#rain_mean = make_climate(1983,2010,ee.Reducer.mean(),'tmin')