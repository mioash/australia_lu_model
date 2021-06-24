import ee 
from ee_plugin import Map 
from make_clim import make_climate

connectivity = ee.Image("users/mioash/drivers_lcaus/aria/connectivity_ready"),
fires = ee.Image("users/mioash/drivers_lcaus/fires/ffrequency_ready"),
nat_npa = ee.Image("users/mioash/drivers_lcaus/natural_areas/nat_npa_ready"),
physical = ee.Image("users/mioash/drivers_lcaus/physical/physical_ready"),
pdensity = ee.Image("users/mioash/drivers_lcaus/population/pdensity_ready"),
soil = ee.Image("users/mioash/drivers_lcaus/soil/soil_ready")

def stacking(yeara,red_clim,just_clim):
    if yeara==1985:
        yearl = 1983
    else:
        yearl = yeara-5
    
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
        tmin = make_climate(yearl,yeara,redc,'tmin').toBands()
        tmax = make_climate(yearl,yeara,redc,'tmax').toBands()
        tmean = make_climate(yearl,yeara,redc,'tmean').toBands()
        rain = make_climate(yearl,yeara,redc,'rain').toBands()
        return tmin.addBands(tmax).addBands(tmean).addBands(rain)
    else:
        return connectivity.addBands(fires).addBands(nat_npa).addBands(physical).addBands(soil)#.addBands(clim)


#rain_mean = make_climate(1983,2010,ee.Reducer.mean(),'tmin')