"""
nonlinear CMAP : another function (works for both positive and negative levels)

an alternate plotting

"""
import xarray as xr
import numpy as np
import pandas as pd

from matplotlib.colors import LinearSegmentedColormap

class nlcmap(LinearSegmentedColormap):
    """A nonlinear colormap"""

    name = 'nlcmap'

    def __init__(self, cmap, levels):
        self.cmap = cmap
        self.monochrome = self.cmap.monochrome
        self.levels = np.asarray(levels, dtype='float64')
        self.levmax = self.levels.max()
        self.levmin = self.levels.min()
        self._x = (self.levels - self.levmin) / (self.levmax - self.levmin)
        self._y = np.linspace(0, 1, len(self.levels))

    def __call__(self, xi, alpha=1.0, **kw):
        yi = np.interp(xi, self._x, self._y)
        return self.cmap(yi, alpha)

def extract_lat_lon(cmip,model):

    variable = 'land_area_and_fraction'

    data_dir = '/Users/zo0/Documents/my_data/'+cmip+'/'+model+'/'+variable+'/area*.nc'

    nc = xr.open_mfdataset(data_dir)

    lats = np.array(nc.variables['lat'])
    lons = np.array(nc.variables['lon'])
    lons-=180 # using standard 0-178.75 & -180 - 0

    _ = np.meshgrid(lons,lats)
    lons_1d = _[0].flatten()
    lats_1d = _[1].flatten()

    return lons,lats,lons_1d,lats_1d

def rearrange_data_lon_lat(lats,lons,data):
    data_ = np.zeros([len(lats),len(lons)]);data_[:]=np.nan
    data_[:,len(lons)//2:len(lons)]= data[:,0:len(lons)//2]
    data_[:,0:len(lons)//2]= data[:,len(lons)//2:len(lons)]
    return data_

def land_mask_1degree(cmip,model):

    variable = 'land_area_and_fraction'

    data_dir = '/Users/zo0/Documents/my_data/remapping_cmip6_historical/MeanCMIP5conv/sftlf'

    nc = xr.open_mfdataset(data_dir)

    lats = np.array(nc.variables['lat'])
    lons = np.array(nc.variables['lon'])
    lons[len(lons)//2:] = lons[len(lons)//2:] - 360 # using standard 0-178.75 & -180 - 0

    _ = np.meshgrid(lons,lats)
    lons_1d = _[0].flatten()
    lats_1d = _[1].flatten()

    return lons,lats,lons_1d,lats_1d

#def nlmap_plt(i,fig,lats,lons,data_decadal_1996_2005,cmip,model):
