#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 10:43:42 2020

!! using machine learning algorithm to predict SOC change between 1996-2005 and 2091-2100

Note1: d_npp may be proxied by contempary fertilaztion effects for the sack of emergent
constraint (will compare this with d_npp)

Note2: tried log10(tau) to take care of the outliers; may consider directly removing them

Note 3: MIP scenario ssp585 missing one file...

Note 4: adding more variables such as rh, tau_rh, soil water, soil temperature

Note 5: tsl first two layers (35cm). (3 layers in total down to 8 some meters)


@author: zo0
"""
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import netCDF4
from scipy.stats import linregress

import sys
sys.path.insert(0, './cmip_ml/')
import cmip_ml_soc_preprocessing as preproc
import numpy.ma as ma

import lib_plotting

from tabulate import tabulate

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
#%% functions
def land_area(cmip,model):
    file_id='areacella*.nc'
    data_dir = '/global/cfs/cdirs/m2467/zhengshi/my_data/'+cmip+'/'+model+'/land_area_and_fraction/'+file_id
    nc_area = xr.open_mfdataset(data_dir)   
    land_area = nc_area['areacella'] # unit: m2
    return land_area
   
def land_fraction(cmip,model):
    file_id='sftlf*.nc'
    data_dir = '/global/cfs/cdirs/m2467/zhengshi/my_data/'+cmip+'/'+model+'/land_area_and_fraction/'+file_id
    nc_fraction = xr.open_mfdataset(data_dir)
    land_fraction = nc_fraction['sftlf'] # unit: percentage * 100
    land_fraction/=100
    return land_fraction

def indicies_of_outliers(x):
    a = 4
    q1,q3 = np.percentile(x,[25,75])
    iqr = q3 - q1
    lower_bound = q1 - (iqr * a)
    upper_bound = q3 + (iqr * a)
    return np.where((x>upper_bound)|(x<lower_bound))

def extract_lat_lon_map(cmip,model,file):
    
    variable = 'land_area_and_fraction'
    
    data_dir = '/global/cfs/cdirs/m2467/zhengshi/my_data/'+cmip+'/'+model+'/'+variable+'/'+file
        
    nc = xr.open_mfdataset(data_dir)
    
    lats = np.array(nc.variables['lat'])
    lons = np.array(nc.variables['lon'])
    # lons[len(lons)//2:] = lons[len(lons)//2:] - 360 # using standard 0-178.75 & -180 - 0    
    return lons, lats

def extract_lat_lon(cmip,model,file):
    
    variable = 'land_area_and_fraction'
    
    data_dir = '/global/cfs/cdirs/m2467/zhengshi/my_data/'+cmip+'/'+model+'/'+variable+'/'+file
        
    nc = xr.open_mfdataset(data_dir)
    
    lats = np.array(nc.variables['lat'])
    lons = np.array(nc.variables['lon'])
    lons[len(lons)//2:] = lons[len(lons)//2:] - 360 # using standard 0-178.75 & -180 - 0

    _ = np.meshgrid(lons,lats)
    lons_1d = _[0].flatten()
    lats_1d = _[1].flatten()  
    
    return lons,lats,lons_1d, lats_1d
    
def change_lon_lat(lats,lons,data):
    data_ = np.zeros([len(lats),len(lons)]);data_[:]=np.nan
    data_[:,len(lons)//2:len(lons)]= data[:,0:len(lons)//2]
    data_[:,0:len(lons)//2]= data[:,len(lons)//2:len(lons)]
    return data_
        
        
def extract_feat_hist(cmip,model,variable,file):   
    
    data_dir = '/global/cfs/cdirs/m2467/zhengshi/my_data/'+cmip+'/'+model+'/'+variable+'/'+file
        
    nc = xr.open_mfdataset(data_dir,combine = 'by_coords', concat_dim="time") # reading the nc files altogether and creating Dataset 
        
    data = np.array(nc.variables[variable]) 
    
    if variable == 'npp' or variable == 'gpp' or variable == 'rh':
        scalar_month = 1000 * 30 * 24 *3600
        data_ = data*scalar_month # change kg/m2/s to g/m2/month
        data_ave_1996_2005 = np.nansum(data_[-(2014-1996+1)*12:-(2014-2005)*12,:,:],axis=0)/10
        data_ave_1996_2005[(data_ave_1996_2005>1e6)|(data_ave_1996_2005<=0)] = np.nan
    elif variable == 'pr':  
        scalar_month = 30 * 24 *3600
        data_ = data*scalar_month/1000*1000 # change kg/m2/s to mm/m2/month (1000kg/m3)
        data_ave_1996_2005 = np.nansum(data_[-(2014-1996+1)*12:-(2014-2005)*12,:,:],axis=0)/10
        data_ave_1996_2005[(data_ave_1996_2005>1e6)] = np.nan
    elif variable == 'tsl':  
        if model=='ipsl' and cmip=='cmip6':
            depth = np.array(nc.variables['solth'])
        else:
            depth = np.array(nc.variables['depth'])           
        
        id_depth30cm = np.where(depth>0.3)[0][0]-1
        data_ave_1996_2005 = np.nanmean(data[-(2014-1996+1)*12:-(2014-2005)*12,id_depth30cm,:,:],axis=0)
        data_ave_1996_2005[(data_ave_1996_2005>1e6)] = np.nan
    else:
        data_ave_1996_2005 = np.nanmean(data[-(2014-1996+1)*12:-(2014-2005)*12,:,:],axis=0)
        data_ave_1996_2005[(data_ave_1996_2005>1e6)] = np.nan
        
    # if model=='mohc_ukesm' and variable == 'tas' and cmip=='cmip5': # variable = 'ta' under cimp5
    #     data_ave_1996_2005 = np.delete(data_ave_1996_2005, 0, 1)
        
    data_ave_1996_2005_1d = data_ave_1996_2005.flatten()
    
    return data_ave_1996_2005,data_ave_1996_2005_1d

def extract_feat_scenario(cmip,model,scenario,variable,file): 
    
    data_dir = '/global/cfs/cdirs/m2467/zhengshi/my_data/'+cmip+'/'+model+'/'+scenario+'/'+variable+'/'+file
    
    nc = xr.open_mfdataset(data_dir,combine = 'by_coords', concat_dim="time") # reading the nc files altogether and creating Dataset   
        
    data = np.array(nc.variables[variable])   
    
    if variable == 'npp' or variable == 'gpp':
        scalar_month = 1000 * 30 * 24 *3600
        data_ = data*scalar_month # change kg/m2/s to g/m2/month
        data_ave_2091_2100 = np.nansum(data_[-(2100-2091+1)*12:,:,:],axis=0)/10
        data_ave_2091_2100[(data_ave_2091_2100>1e6)|(data_ave_2091_2100<=0)] = np.nan
    elif variable == 'pr':  
        scalar_month = 30 * 24 *3600
        data_ = data*scalar_month/1000*1000 # change kg/m2/s to mm/m2/month (1000kg/m3)
        data_ave_2091_2100 = np.nansum(data_[-(2100-2091+1)*12:,:,:],axis=0)/10
        data_ave_2091_2100[(data_ave_2091_2100>1e6)] = np.nan
    elif variable == 'tsl':  
        if model=='ipsl' and 'cmip5':
            depth = np.array(nc.variables['solth'])
        else:
            depth = np.array(nc.variables['depth'])
        
        id_depth30cm = np.where(depth>0.3)[0][0]-1
        data_ave_2091_2100 = np.nanmean(data[(2100-2091+1)*12:,id_depth30cm,:,:],axis=0)
        data_ave_2091_2100[(data_ave_2091_2100>1e6)] = np.nan
    else:
        data_ave_2091_2100 = np.nanmean(data[-(2100-2091+1)*12:,:,:],axis=0)
        data_ave_2091_2100[(data_ave_2091_2100>1e6)] = np.nan
        
    if model=='ncc' and variable=='tas' and cmip=='cmip5': # the reason for tease out ncc, see notes
        data_file1 ='/global/cfs/cdirs/m2467/zhengshi/my_data/cmip5/ncc/ssp585/tas/tas_Amon_NorESM1-ME_rcp85_r1i1p1_200601-204412.nc'
        data_file2 ='/global/cfs/cdirs/m2467/zhengshi/my_data/cmip5/ncc/ssp585/tas/tas_Amon_NorESM1-ME_rcp85_r1i1p1_204501-210012.nc'
        nc1 = xr.open_dataset(data_file1)
        nc2 = xr.open_dataset(data_file2)
        data1 = nc1.variables['tas']
        data2 = nc2.variables['tas']
        data_together = np.append(data1,data2,axis=0)
        data_ave_2091_2100 = np.nanmean(data_together[-(2100-2091+1)*12:,:,:],axis=0)
        
    if model=='ncc' and variable=='pr' and cmip=='cmip5':
        data_file1 ='/global/cfs/cdirs/m2467/zhengshi/my_data/cmip5/ncc/ssp585/pr/pr_Amon_NorESM1-ME_rcp85_r1i1p1_200601-204412.nc'
        data_file2 ='/global/cfs/cdirs/m2467/zhengshi/my_data/cmip5/ncc/ssp585/pr/pr_Amon_NorESM1-ME_rcp85_r1i1p1_204501-210012.nc'
        nc1 = xr.open_dataset(data_file1)
        nc2 = xr.open_dataset(data_file2)
        data1 = nc1.variables['pr']
        data2 = nc2.variables['pr']
        data = np.append(data1,data2,axis=0)
        scalar_month = 30 * 24 *3600
        data_ = data*scalar_month/1000*1000 # change kg/m2/s to mm/m2/month (1000kg/m3)
        data_ave_2091_2100 = np.nansum(data_[-(2100-2091+1)*12:,:,:],axis=0)/10
        data_ave_2091_2100[(data_ave_2091_2100>1e6)] = np.nan
    
    data_ave_2091_2100_1d = data_ave_2091_2100.flatten()
       
    return data_ave_2091_2100,data_ave_2091_2100_1d

# def seasonality_of_features(cmip,model,variable,file): 
def fertilization_gpp(cmip,model,scenario,variable,file='*.nc'):
    
    data_dir = '/global/cfs/cdirs/m2467/zhengshi/my_data/'+cmip+'/'+model+'/'+scenario+'/'+variable+'/'+file
    
    nc = xr.open_mfdataset(data_dir) # reading the nc files altogether and creating Dataset
    
    data = np.array(nc.variables[variable])
    data_=[np.nansum(data[(i-1)*12:i*12,:,:],axis=0) for i in np.arange(1,2100-2015+2)]
    
    scalar_month = 1000 * 30 * 24 *3600   
    
    data_year = [np.nansum(data_[i]*scalar_month*land_area(cmip,model)*land_fraction(cmip,model))\
                 for i in np.arange(86)]
    data_year_=np.asarray(data_year)
    data_year_/=1e15
    
    return data_year_

def historical_gpp_npp_trend_total(cmip,model,variable,file):
    
    data_dir = '/global/cfs/cdirs/m2467/zhengshi/my_data/'+cmip+'/'+model+'/'+variable+'/'+file
        
    nc = xr.open_mfdataset(data_dir) # reading the nc files altogether and creating Dataset
    
    data = np.array(nc.variables[variable])
    
    data_=[np.nansum(data[(i-1)*12:i*12,:,:],axis=0) for i in np.arange(1,2014-1850+2)]
    
    scalar_month = 1000 * 30 * 24 *3600   
    
    data_year = [np.nansum(data_[i]*scalar_month*land_area(cmip,model)*land_fraction(cmip,model))\
                 for i in np.arange(165)]
    data_year_=np.asarray(data_year)
    
    data_year_/=1e15
    
    data_year_trend = linregress(np.arange(10),data_year_[-(2014-1996+1):-(2014-2005)])
    
    return data_year_trend[0],data_year_

def historical_gpp_npp_trend_grid(cmip,model,variable,file):
    
    data_dir = '/global/cfs/cdirs/m2467/zhengshi/my_data/'+cmip+'/'+model+'/'+variable+'/'+file
        
    nc = xr.open_mfdataset(data_dir) # reading the nc files altogether and creating Dataset
    
    data = np.array(nc.variables[variable])
    
    scalar_month = 1000 * 30 * 24 *3600 
    
    data*=scalar_month
    
    data_=[np.nansum(data[(i-1)*12:i*12,:,:],axis=0) for i in np.arange(1,2014-1850+2)]
    
    data_array = np.asarray(data_)
    data_array[data_array<=0]=np.nan
    
    data_year_trend = np.zeros([data_array.shape[1],data_array.shape[2]])
    data_year_trend[:] = np.nan
    for lat in np.arange(data_array.shape[1]):
        for lon in np.arange(data_array.shape[2]):
            data_year_trend_ = linregress(np.arange(10),data_array[-(2014-1996+1):-(2014-2005),lat,lon])
            data_year_trend[lat,lon] = data_year_trend_[0]
    
    # data_year_trend[data_year_trend==0]=np.nan
    return data_year_trend
#%%
""" Kathy's data for biogoescience 2013 
    the files are the same: csv, pft
"""  
# the pdf nc file data
data_dir = '/global/cfs/cdirs/m2467/zhengshi/my_data/land_cover_Kathy/'
file_name = 'pft.nc'

nc = netCDF4.Dataset(data_dir+file_name) # reading the nc file and creating Dataset

data_ = np.array(nc['var2d'])
data_lc = np.transpose(data_)
data_lc [(data_lc==1)|(data_lc==10)] = np.nan # 1: water & 10 is ice

lons_lc = np.array(nc['lon'])
lats_lc = np.array(nc['lat'])
#%% data processing

# cmips=['cmip5','cmip6']

cmips=['cmip5']

if cmips==['cmip5']:
    models = ['bcc','can_esm','cesm','ipsl','miroc','mpi','ncc','mohc_ukesm','bnu','gfdl','giss','inm']  
# models_cmip5 = ['bnu']
else:
    models = ['bcc','can_esm','cesm','ipsl','miroc','mpi','ncc','mohc_ukesm','access','cmcc','cnrm','tai']
# 

# cmip,model,file = 'cmip5','bcc','*.nc'
file = '*.nc'

# record_stats = pd.DataFrame(columns=['cmip','model']+['r2_{}'.format(i) for i in range(5)]+['MSE_{}'.format(i) for i in range(5)])
fig1,axes1 = plt.subplots(4,2,figsize=(12,12))
m=0


lc_id = 9
for i in np.arange(2,10):
    print(i)
    
    ii=0
    
    feature_names = ['cSoil', 'npp', 'tas', 'pr', 'tau_rh', 'd_npp', 'd_ta', 'd_pr','d_tau_rh']
    df_fi_biome = pd.DataFrame({'features':feature_names})
    df_fi_biome.set_index('features')
    
    for model in models:
        print(model)

        for cmip in cmips:

            if cmip=='cmip6' and (model=='ncc' or model=='cesm'):
                variables = ['npp','cSoilAbove1m','tas','pr','rh']
            else:
                variables = ['npp','cSoil','tas','pr','rh']

            for variable in variables:
                data_temp = preproc.full_temporal(cmip,model,variable,file) 
                start_year,end_year = preproc.start_end_years_simulation(cmip,model,variable,file)        

                if variable == 'npp' or variable == 'gpp' or variable == 'rh':

                    scalar_month = 1000 * 30 * 24 *3600              

                    data_temp_scaled = data_temp * scalar_month 

                    data_1996_2005 = data_temp_scaled[-(end_year-1996+1)*12:-(end_year-2005)*12,:,:].copy()

                    if end_year==2100:
                        data_2091_2100 = data_temp_scaled[-(end_year-2091+1)*12:,:,:].copy()
                    else:
                        data_2091_2100 = data_temp_scaled[-(end_year-2091+1)*12:-(end_year-2100)*12,:,:].copy()

                    vars()[variable] = np.nansum(data_1996_2005,axis=0)/10
                    vars()[variable+'_ssp585'] = np.nansum(data_2091_2100,axis=0)/10

                elif variable == 'pr':  

                    scalar_month = 30 * 24 *3600
                    data_temp_scaled = data_temp*scalar_month/1000*1000 # change kg/m2/s to mm/m2/month (1000kg/m3)

                    data_1996_2005 = data_temp_scaled[-(end_year-1996+1)*12:-(end_year-2005)*12,:,:].copy()

                    if end_year==2100:
                        data_2091_2100 = data_temp_scaled[-(end_year-2091+1)*12:,:,:].copy()
                    else:
                        data_2091_2100 = data_temp_scaled[-(end_year-2091+1)*12:-(end_year-2100)*12,:,:].copy()               

                    vars()[variable] = np.nansum(data_1996_2005,axis=0)/10 
                    vars()[variable+'_ssp585'] = np.nansum(data_2091_2100,axis=0)/10
                else:
                    data_temp_scaled = data_temp.copy()

                    data_1996_2005 = data_temp_scaled[-(end_year-1996+1)*12:-(end_year-2005)*12,:,:].copy()

                    if end_year==2100:
                        data_2091_2100 = data_temp_scaled[-(end_year-2091+1)*12:,:,:].copy()
                    else:
                        data_2091_2100 = data_temp_scaled[-(end_year-2091+1)*12:-(end_year-2100)*12,:,:].copy()

                    if cmip=='cmip6' and (model == 'cesm' or model == 'ncc') and variable =='cSoilAbove1m':
                        variable = 'cSoil'   
                    vars()[variable] = np.nanmean(data_1996_2005,axis=0)
                    vars()[variable+'_ssp585'] = np.nanmean(data_2091_2100,axis=0)

            lons,lats,_,_ = preproc.extract_lat_lon(cmip,model,'area*.nc')
            cSoil = lib_plotting.rearrange_data_lon_lat(lats,lons,cSoil)
            cSoil_ssp585 = lib_plotting.rearrange_data_lon_lat(lats,lons,cSoil_ssp585)

            npp = lib_plotting.rearrange_data_lon_lat(lats,lons,npp)
            npp_ssp585 = lib_plotting.rearrange_data_lon_lat(lats,lons,npp_ssp585)

            pr = lib_plotting.rearrange_data_lon_lat(lats,lons,pr)
            pr_ssp585 = lib_plotting.rearrange_data_lon_lat(lats,lons,pr_ssp585)

            tas = lib_plotting.rearrange_data_lon_lat(lats,lons,tas)
            tas_ssp585 = lib_plotting.rearrange_data_lon_lat(lats,lons,tas_ssp585)

            npp[npp<=0] = np.nan
            npp_ssp585[npp_ssp585<=0] = np.nan
            tau_npp = cSoil*1000/npp 
            tau_ssp585_npp = cSoil_ssp585*1000/npp_ssp585

            rh[rh<=2] = np.nan
            rh_ssp585[rh_ssp585<=2] = np.nan
            tau_rh = cSoil*1000/rh  
            tau_ssp585_rh = cSoil_ssp585*1000/rh_ssp585      
            d_tau_rh = tau_ssp585_rh - tau_rh

            # model_soc_tt = np.nansum(model_soc*land_area_lc*land_fraction_lc)/1e12
            #% generating lc for model soil carbon
            model_lc = np.zeros([len(lats),len(lons)]);model_lc[:]=np.nan
            for ik in np.arange(len(lats)):
                rows = np.argmin((lats[ik] - lats_lc)**2)
                for jk in np.arange(len(lons)):
                    cols = np.argmin((lons[jk]- lons_lc)**2)        
                    """script below to guarantee lc assignment of all soil carbon"""
                    if cSoil[ik,jk]>0: 
                        k=1
                        while True:               
                            lc_square = data_lc[rows-k:rows+k+1,cols-k:cols+k+1].copy()
                            k+=1
                            if np.nansum(lc_square)>0:
                                break
                        unique, counts = np.unique(lc_square[~np.isnan(lc_square)], return_counts=True)                        
                        model_lc[ik,jk] = unique[np.argmax(counts)]                       
            # ============================================================================================================  
            lc_types = ['Tundra','Boreal forest','Tropical forest','Temperate forest','Desert and shrublands','Grassland and savannas',
                'Cropland','Permanent wetlands']    # 2-9
            # plt.imshow(np.flipud(model_lc),vmin=1.5,vmax= 9.5,cmap=plt.cm.get_cmap('Accent_r',8),interpolation=None)  
            # plt.colorbar() 
            

            # lc_id = 9
            # for i in np.arange(2,10):
            cSoil_1d = cSoil[model_lc==i].flatten()
            d_soc = cSoil_ssp585 - cSoil
            d_soc_1d = d_soc[model_lc==i].flatten()

            npp_1d = npp[model_lc==i].flatten()
            d_npp = npp_ssp585 - npp
            d_npp_1d = d_npp[model_lc==i].flatten()

            pr_1d = pr[model_lc==i].flatten()
            d_pr = pr_ssp585 - pr
            d_pr_1d = d_pr[model_lc==i].flatten()

            tas_1d = tas[model_lc==i].flatten()
            d_ta = np.array(tas_ssp585 - tas)
            d_ta_1d = np.array(d_ta[model_lc==i].flatten()) 

            tau_rh_1d = tau_rh[model_lc==i].flatten() 
            d_tau_rh_1d = d_tau_rh[model_lc==i].flatten()

            # mrso_1d = mrso.flatten() 
            # d_mrso = mrso_ssp585 - mrso
            # d_mrso_1d = d_mrso.flatten()                

            d = {'cSoil':cSoil_1d,
                  'npp':npp_1d,
                  'tas':tas_1d,
                  'pr':pr_1d,#'mrso':mrso_1d,
                  'tau_rh':tau_rh_1d,
                   'd_npp':d_npp_1d,
                   'd_ta':d_ta_1d,
                   'd_pr':d_pr_1d,#'d_mrso':d_mrso_1d,
                   'd_tau_rh':d_tau_rh_1d,
                  'd_soc':d_soc_1d
                  }

            data_mlx = pd.DataFrame(data=d)
            # data_mlx[data_mlx['d_soc']<-15] = np.nan

            notNaNs = ~np.any(np.isnan(data_mlx),axis=1)
            data_ml = data_mlx[notNaNs]

            """ machine learning algorithms """
             #%% randome forest
            # manually 5 fold cross validation 
            from sklearn.model_selection import KFold
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.metrics import r2_score
            from sklearn.metrics import mean_squared_error
            import seaborn as sns
            sns.set()

            # data_ml_shuffled = data_ml.values
            # np.random.shuffle(data_ml_shuffled)
            X = data_ml.values[:,:-1]
            y = data_ml.values[:,-1]
            kf = KFold(n_splits=5,shuffle=True)

            rf_reg2 = RandomForestRegressor(n_estimators=300,max_depth=20,min_samples_split=4,random_state=20)

            # fig,axes = plt.subplots(2,3,figsize=(12,8))

            feature_importances_mean = 0
            n = 0
            r2_record = []
            MSE_record = []
            for train, test in kf.split(X):
                Xtrain,Xtest = X[train],X[test]
                ytrain,ytest = y[train],y[test]

                rf_reg2.fit(Xtrain,ytrain)
                yhat = rf_reg2.predict(Xtest)
                print(len(test),r2_score(ytest,yhat))
                feature_importances = rf_reg2.feature_importances_/kf.n_splits
                feature_importances_mean+=feature_importances

                # axes[n//3,n%3].scatter(ytest,yhat,s=1) 
                # axes[n//3,n%3].set_xlim(min(min(yhat),min(ytest)),max(max(yhat),max(ytest)))
                # axes[n//3,n%3].set_ylim(min(min(yhat),min(ytest)),max(max(yhat),max(ytest)))
                # axes[n//3,n%3].plot([min(min(yhat),min(ytest)),max(max(yhat),max(ytest))],[min(min(yhat),min(ytest)),max(max(yhat),max(ytest))])

                # axes[n//3,n%3].text(min(min(yhat),min(ytest))+.5,max(max(yhat),max(ytest))-1.5,r'$R^2$ = {:.2f}'.format(r2_score(ytest,yhat)))
                # if n//3 == 1:
                #         axes[n//3,n%3].set_xlabel('$\Delta$SOC (gC $m^{-2}$)',fontsize=18)
                # if n%3 == 0:
                #         axes[n//3,n%3].set_ylabel('$\Delta$SOC (gC $m^{-2}$)',fontsize=18)

                n+=1

                r2_record.append(r2_score(ytest,yhat))
                MSE_record.append(mean_squared_error(ytest,yhat))

            # record_stats.loc[len(record_stats)] = [cmip,model]+r2_record+MSE_record
            # np.save('record_r2_mse_cv5_for_obs.npy',record_stats.values)

            # feature_names = data_ml.columns[:-1]
            # f_imp = pd.DataFrame({'features':feature_names,'scores':feature_importances_mean})
            # f_imp = f_imp.sort_values(by='scores',ascending=False)

            # sns.set(font_scale=1)       
            # sns.barplot(y='features',x='scores', color='blue',data=f_imp,ax=axes[n//3,n%3])
            # axes[1,2].text(max(f_imp['scores'])/4,5,cmip+'_'+model,fontsize=22)

            # plt.tight_layout()

            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            model_label_cmip5 = ['bcc-csm1-1','CanESM2','CESM1-BGC','IPSL-CM5A-LR','MIROC-ESM','MPI-ESM-LR','NorESM1-ME','HadGEM2-ES',
                                 'BNU-ESM','GFDL-ESM2G','GISS-E2-R','inmcm4']
            model_label_cmip6 = ['BCC-CSM2-MR','CanESM5','CESM2','IPSL-CM6A-LR','MIROC-ES2L','MPI-ESM1-2-LR','NorESM2-LM','UKESM1-0-LL',
                                 'ACCESS-ESM1-5','CMCC-ESM2','CNRM-ESM2-1','TaiESM1']

            if cmip=='cmip5':
                model_id = model_label_cmip5
            else:
                model_id = model_label_cmip6

            # model_id = ['bcc-csm1-1','BCC-CSM2-MR','CanESM2','CanESM5','CESM1-BGC','CESM2',
            #     'IPSL-CM5A-LR','IPSL-CM6A-LR','MIROC-ESM','MIROC-ES2L','MPI-ESM-LR','MPI-ESM1-2-LR',
            #     'NorESM1-ME','NorESM2-LM','HadGEM2-ES','UKESM1-0-LL']
            feature_names = data_ml.columns[:-1]

            df_fi_biome[model_id[ii]] = feature_importances_mean
            ii+=1
            f_imp = pd.DataFrame({'features':feature_names,'scores':feature_importances_mean})
            f_imp = f_imp.sort_values(by='scores',ascending=False)

        
        
        
        
    import seaborn as sns
    sns.set()
    fig = df_fi_biome.set_index('features').T.plot(
        kind='bar', stacked=True,legend=False,grid=False,ax=axes1[m//2,m%2])
    axes1[m//2,m%2].set_title(lc_types[m],fontsize=12)
    axes1[m//2,m%2].set(xticklabels=[])
    if m//2>=3:
        axes1[m//2,m%2].set_xticklabels(model_id,rotation = 45, ha="right")

    if m%2==0:
        axes1[m//2,m%2].set_ylabel('Relative importance',fontsize=12)
    m+=1
        # axes1[m//3,m%3].grid(False)
    # if cmip=='cmip5':
    #     cmip='cmip6'
    # else:
    #     cmip='cmip5'

    
    
    
    
    
# labels= ['cSoil', 'npp', 'tas', 'pr', 'tau_rh', 'd_npp', 'd_ta', 'd_pr','d_tau_rh'] 
labels = ['cSoil','NPP','Ta','P','\u03C4','\u0394NPP','\u0394Ta','\u0394P','\u0394\u03C4']
# fig1.legend(labels, ncol=9, 
#             bbox_to_anchor=(0.0675, 0.86, 0.75, .10), loc='lower left',mode="expand") #
# 0.1, 0.9, 0.75, .10
      
"""            
            sns.set(font_scale=1)       
            sns.barplot(y='features',x='scores', color='blue',data=f_imp,ax=axes1[m//4,m%4])
            axins = inset_axes(axes1[m//4,m%4], width=1.3, height=0.9,loc=4)
            axins.scatter(data_ml[f_imp['features'].iloc[0]],data_ml['d_soc'],s=1)
            axins.axes.xaxis.set_visible(False)
            axins.axes.yaxis.set_visible(False)
            # axes1[m//4,m%4].text(max(f_imp['scores'])/4,5,model_id[models.index(model)]+'\n'+lc_types[i-2],fontsize=12)
            axes1[m//4,m%4].set_title(model_id[7]+'\n'+lc_types[i-2],fontsize=12)
            # axes1[m//4,m%4].set_title(model_id[models.index(model)]+'\n'+lc_types[i-2],fontsize=12)
            
            m+=1
"""            
#plt.tight_layout()
            
# fig1.savefig('/Users/zo0/Documents/my_projects/cmip/figures/fig_relative_importance_biomes_{}.png'.format(model),bbox_inches='tight',dpi=300)
fig1.savefig('./figures/20221216_fig_relative_importance_biomes_CMIP5_alternative.png',bbox_inches='tight',dpi=300)  

"""
another figure for publication in terms of putting all models together
bar plot for feature importance for each biome
"""
# import seaborn as sns
# sns.set()
# fig = df_fi_biome.set_index('features').T.plot(
#     kind='bar', stacked=True,legend=False)

