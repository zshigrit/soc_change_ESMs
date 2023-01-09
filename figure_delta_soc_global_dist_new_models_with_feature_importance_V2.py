import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import netCDF4
from mpl_toolkits.basemap import Basemap
from scipy.stats import linregress
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy.ma as ma

import sys
sys.path.insert(0, './cmip_ml/')
import cmip_ml_soc_preprocessing as preproc

import lib_plotting
from matplotlib import gridspec
#%%
def land_area(cmip,model,file_id):
    data_dir = '/global/cfs/cdirs/m2467/zhengshi/my_data/'+cmip+'/'+model+'/land_area_and_fraction/'+file_id
    nc_area = netCDF4.Dataset(data_dir)   
    land_area = np.array(nc_area['areacella']) # unit: m2
    if model=='bnu':
        land_area/=1.
    return land_area
   
def land_fraction(cmip,model,file_id):
    data_dir = '/global/cfs/cdirs/m2467/zhengshi/my_data/'+cmip+'/'+model+'/land_area_and_fraction/'+file_id
    nc_fraction = netCDF4.Dataset(data_dir)
    land_fraction = nc_fraction['sftlf'] # unit: percentage * 100
    return land_fraction
#%% data files
files_land_area = {'cesm':{'cmip6':'areacella_fx_CESM2_historical_r11i1p1f1_gn.nc',
                           'cmip5':'areacella_fx_CESM1-BGC_historical_r0i0p0.nc'},
                   'can_esm':{'cmip6':'areacella_fx_CanESM5_historical_r1i1p1f1_gn.nc',
                              'cmip5':'areacella_fx_CanESM2_historical_r0i0p0.nc'},
                   'gfdl':{'cmip6':'areacella_fx_GFDL-ESM4_historical_r1i1p1f1_gr1.nc',
                           'cmip5':'areacella_fx_GFDL-ESM2G_historical_r0i0p0.nc'},
                   'mohc_ukesm':{'cmip6':'areacella_fx_UKESM1-0-LL_piControl_r1i1p1f2_gn.nc',
                                 'cmip5':'areacella_fx_HadGEM2-ES_historical_r1i1p1.nc'},
                   'ncc':{'cmip6':'areacella_fx_NorESM2-LM_historical_r1i1p1f1_gn.nc',
                          'cmip5':'areacella_fx_NorESM1-ME_historical_r0i0p0.nc'},
                   'bcc':{'cmip6':'areacella_fx_BCC-CSM2-MR_hist-resIPO_r1i1p1f1_gn.nc',
                          'cmip5':'areacella_fx_bcc-csm1-1_piControl_r0i0p0.nc'},
                   'ipsl':{'cmip6':'areacella_fx_IPSL-CM6A-LR_historical_r1i1p1f1_gr.nc',
                           'cmip5':'areacella_fx_IPSL-CM5A-LR_historical_r0i0p0.nc'},
                   'miroc':{'cmip6':'areacella_fx_MIROC-ES2L_historical_r1i1p1f2_gn.nc',
                            'cmip5':'areacella_fx_MIROC-ESM_historical_r0i0p0.nc'},
                   'mpi':{'cmip6':'areacella_fx_MPI-ESM1-2-LR_historical_r1i1p1f1_gn.nc',
                          'cmip5':'areacella_fx_MPI-ESM-LR_decadal2000_r0i0p0.nc'},
                   'bnu':{'cmip6':'*.nc',
                          'cmip5':'areacella_fx_BNU-ESM_historical_r0i0p0.nc'},
                   'giss':{'cmip6':'*.nc',
                          'cmip5':'areacella_fx_GISS-E2-R_historical_r0i0p0.nc'},
                   'inm':{'cmip6':'*.nc',
                          'cmip5':'areacella_fx_inmcm4_rcp85_r0i0p0.nc'},
                   'access':{'cmip6':'areacella_fx_ACCESS-ESM1-5_historical_r1i1p1f1_gn.nc',
                             'cmip5':'*.nc'},
                   'cmcc':{'cmip6':'areacella_fx_CMCC-ESM2_ssp585_r1i1p1f1_gn.nc',
                             'cmip5':'*.nc'},
                   'cnrm':{'cmip6':'areacella_fx_CNRM-ESM2-1_historical_r11i1p1f2_gr.nc',
                             'cmip5':'*.nc'},
                   'tai':{'cmip6':'areacella_fx_TaiESM1_ssp585_r1i1p1f1_gn.nc',
                             'cmip5':'*.nc'}                  
                   }

files_land_fraction = {'cesm':{'cmip6':'sftlf_fx_CESM2_historical_r11i1p1f1_gn.nc',
                           'cmip5':'sftlf_fx_CESM1-BGC_historical_r0i0p0.nc'},
                   'can_esm':{'cmip6':'sftlf_fx_CanESM5_historical_r1i1p1f1_gn.nc',
                              'cmip5':'sftlf_fx_CanESM2_historical_r0i0p0.nc'},
                   'gfdl':{'cmip6':'sftlf_fx_GFDL-ESM4_historical_r1i1p1f1_gr1.nc',
                           'cmip5':'sftlf_fx_GFDL-ESM2G_historical_r0i0p0.nc'},
                   'mohc_ukesm':{'cmip6':'sftlf_fx_UKESM1-0-LL_piControl_r1i1p1f2_gn.nc',
                                 'cmip5':'sftlf_fx_HadGEM2-ES_historical_r1i1p1.nc'},
                   'ncc':{'cmip6':'sftlf_fx_NorESM2-LM_historical_r1i1p1f1_gn.nc',
                          'cmip5':'sftlf_fx_NorESM1-ME_historical_r0i0p0.nc'},
                   'bcc':{'cmip6':'sftlf_fx_BCC-CSM2-MR_hist-resIPO_r1i1p1f1_gn.nc',
                          'cmip5':'sftlf_fx_bcc-csm1-1_piControl_r0i0p0.nc'},
                   'ipsl':{'cmip6':'sftlf_fx_IPSL-CM6A-LR_historical_r1i1p1f1_gr.nc',
                           'cmip5':'sftlf_fx_IPSL-CM5A-LR_historical_r0i0p0.nc'},
                   'miroc':{'cmip6':'sftlf_fx_MIROC-ES2L_historical_r1i1p1f2_gn.nc',
                            'cmip5':'sftlf_fx_MIROC-ESM_historical_r0i0p0.nc'},
                   'mpi':{'cmip6':'sftlf_fx_MPI-ESM1-2-LR_historical_r1i1p1f1_gn.nc',
                          'cmip5':'sftlf_fx_MPI-ESM-LR_historical_r0i0p0.nc'},
                   'bnu':{'cmip6':'*.nc',
                          'cmip5':'sftlf_fx_BNU-ESM_historical_r0i0p0.nc'},
                   'giss':{'cmip6':'*.nc',
                          'cmip5':'sftlf_fx_GISS-E2-R_historical_r0i0p0.nc'},
                   'inm':{'cmip6':'*.nc',
                          'cmip5':'sftlf_fx_inmcm4_rcp85_r0i0p0.nc'},
                   'access':{'cmip6':'sftlf_fx_ACCESS-ESM1-5_historical_r1i1p1f1_gn.nc',
                             'cmip5':'*.nc'},
                   'cmcc':{'cmip6':'sftlf_fx_CMCC-ESM2_ssp585_r1i1p1f1_gn.nc',
                             'cmip5':'*.nc'},
                   'cnrm':{'cmip6':'sftlf_fx_CNRM-ESM2-1_historical_r11i1p1f2_gr.nc',
                             'cmip5':'*.nc'},
                   'tai':{'cmip6':'sftlf_fx_TaiESM1_ssp585_r1i1p1f1_gn.nc',
                             'cmip5':'*.nc'}
                   }   

#%% 
model_labels = {'cmip5':
          ['bcc-csm1-1','CanESM2','CESM1-BGC','IPSL-CM5A-LR','MIROC-ESM','MPI-ESM-LR','NorESM1-ME','HadGEM2-ES',
           'BNU-ESM','GFDL-ESM2G','GISS-E2-R','inmcm4'],
          'cmip6':
          ['BCC-CSM2-MR','CanESM5','CESM2','IPSL-CM6A-LR','MIROC-ES2L','MPI-ESM1-2-LR','NorESM2-LM','UKESM1-0-LL',
           'ACCESS-ESM1-5','CMCC-ESM2','CNRM-ESM2-1','TaiESM1']}
    
main_models_cmip5 = ['bcc','can_esm','cesm','ipsl','miroc','mpi','ncc','mohc_ukesm','bnu','gfdl','giss','inm']    
main_models_cmip6 = ['bcc','can_esm','cesm','ipsl','miroc','mpi','ncc','mohc_ukesm','access','cmcc','cnrm','tai'] 
   
cmips=['cmip5','cmip6']
cmips=['cmip5']
# variables = ['npp','cSoil','tas','pr','mrso']

file='*.nc'
# cmip='cmip5'
# main_model = 'mpi'
# model=['bcc']
dSOC_models = pd.DataFrame()
for cmip in cmips:

    # fig,ax = plt.subplots(4,6,figsize=(12,12),gridspec_kw={'width_ratios': [2, 1]*3},sharey=False)
    fig,ax = plt.subplots(4,4,figsize=(12,8),sharey=False)
    fig.tight_layout()
    # fig = plt.figure(figsize=(12,12))
    # i = 0
    
    # spec = gridspec.GridSpec(ncols=6, nrows=4,
    #                      width_ratios=[3, 1]*3)
    #                      # height_ratios=[1, 1]*2)
    
    if cmip=='cmip5':
        main_models = main_models_cmip5.copy()
    else:
        main_models = main_models_cmip6.copy()
        
    main_models = ['bcc','bcc','can_esm','can_esm']    
    fig_id = 0
    # main_models = ['bcc','bcc','can_esm','can_esm','cesm','cesm','ipsl','ipsl',
    #                 'miroc','miroc','mpi','mpi','ncc','ncc','mohc_ukesm','mohc_ukesm']
    for main_model in main_models: 
        lons,lats,_,_ = preproc.extract_lat_lon(cmip,main_model,'area*.nc')
        
        if cmip=='cmip6' and (main_model == 'cesm' or main_model == 'ncc'):
            variables = ['npp','cSoilAbove1m','tas','pr','rh'] 
        else:
            variables = ['npp','cSoil','tas','pr','rh']
            
        for variable in variables:
            
            data_temp = preproc.full_temporal(cmip,main_model,variable,file) 
            start_year,end_year = preproc.start_end_years_simulation(cmip,main_model,variable,file)        
            
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
                if cmip=='cmip6' and (main_model == 'cesm' or main_model == 'ncc') and variable =='cSoilAbove1m':
                    variable = 'cSoil'    
                vars()[variable] = np.nanmean(data_1996_2005,axis=0)
                vars()[variable+'_ssp585'] = np.nanmean(data_2091_2100,axis=0)
        
        cSoil_1d = cSoil.flatten()
        d_soc = cSoil_ssp585 - cSoil
        d_soc_1d = d_soc.flatten()
        
        # # plt.imshow(d_soc);plt.colorbar()
        
        npp_1d = npp.flatten()
        d_npp = npp_ssp585 - npp
        d_npp_1d = d_npp.flatten()
        
        rh_1d = rh.flatten()
        d_rh = rh_ssp585 - rh
        d_rh_1d = d_rh.flatten()
        
        pr_1d = pr.flatten()
        d_pr = pr_ssp585 - pr
        d_pr_1d = d_pr.flatten()
        
        tas_1d = tas.flatten()
        d_ta = np.array(tas_ssp585 - tas)
        d_ta_1d = np.array(d_ta.flatten()) 
        
        rh[rh<=2] = np.nan
        rh_ssp585[rh_ssp585<=2] = np.nan
        tau_rh = cSoil*1000/rh  
        tau_rh_1d = tau_rh.flatten() 
        tau_ssp585_rh = cSoil_ssp585*1000/rh_ssp585      
        d_tau_rh = tau_ssp585_rh - tau_rh
        d_tau_rh_1d = d_tau_rh.flatten()
        
        d = {'cSoil':cSoil_1d,
              'npp':npp_1d,
              'tas':tas_1d,
              'pr':pr_1d,
              'tau_rh':tau_rh_1d,
              'd_npp':d_npp_1d,
              'd_ta':d_ta_1d,
              'd_pr':d_pr_1d,
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
        # n = 0
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
                
            # n+=1
            
            r2_record.append(r2_score(ytest,yhat))
            
            MSE_record.append(mean_squared_error(ytest,yhat))
            
        
        # record_stats.loc[len(record_stats)] = [cmip,model]+r2_record+MSE_record
        # np.save('record_r2_mse_cv5_for_obs.npy',record_stats.values)

        r2_record_mean = np.mean(r2_record)
        MSE_record_mean = np.mean(MSE_record)
        # feature_names = data_ml.columns[:-1]
        feature_names = ['cSoil','NPP','Ta','P','\u03C4','\u0394NPP','\u0394Ta','\u0394P','\u0394\u03C4']
        f_imp = pd.DataFrame({'features':feature_names,'scores':feature_importances_mean})
        f_imp = f_imp.sort_values(by='scores',ascending=False)
            
        
    #%% plotting
        
        # lons,lats = np.linspace(-180,180,360),np.linspace(-90,90,180)
        
        datax = d_soc.copy()
        datay_ = lib_plotting.rearrange_data_lon_lat(lats,lons,datax)
        
        land_fraction_ = preproc.land_fraction(cmip,main_model)
        land_fraction = lib_plotting.rearrange_data_lon_lat(lats,lons,land_fraction_)
        land_mask = land_fraction==0
        datay = ma.masked_array(datay_,mask=land_mask)
        
        # plotting
        lonmin=min(lons);lonmax=max(lons);latmin=min(lats);latmax=max(lats)
        lon_0 = (lonmin + lonmax)/2
        m = Basemap(projection='cyl', lon_0=lon_0, llcrnrlon=lonmin, \
                    llcrnrlat=latmin, urcrnrlon=lonmax, urcrnrlat=latmax,ax = ax[fig_id//4,fig_id%4])
        m.drawcoastlines(linewidth=0.4)
        # m.drawparallels(np.arange(-90.,120.,30.),labels=[1,0,0,0],fontsize=8)
        # m.drawmeridians(np.arange(0.,420.,60.),labels=[0,0,0,1],fontsize=8)
        # show data
        # Lon, Lat = np.meshgrid(lons,lats)
        
        # X,Y = m(Lon,Lat)
        cm = m.pcolormesh(lons,lats,datay,cmap=plt.get_cmap('RdYlGn'),shading='auto',vmin=-10,vmax=10)
        # cbar = m.colorbar(cm, extend="max")
        if np.nanmax(datay)>10 and np.nanmin(datay)<-10:
            dummy='both'
        elif np.nanmax(datay)>10 and np.nanmin(datay)>=-10:
            dummy = 'max'
        elif np.nanmax(datay)<=10 and np.nanmin(datay)<-10:
            dummy = 'min'
        else:
            dummy='neither'
        
        ######## colorbar    
        cbar = m.colorbar(cm, extend=dummy)
        if fig_id%4==3:
            labelsize1 = 6
        else:
            labelsize1 = 0
            # cbar.ax.tick_params(size=0)
            cbar.set_ticks([])
        
        cbar.ax.tick_params(labelsize=labelsize1)
        # # if model == 'HadGEM2-ES':
        # #     model = 'UK-HadGEM2-ES'
        
        ax[fig_id//4,fig_id%4].set_title(model_labels[cmip][main_models.index(main_model)//2],fontsize=14)
        
        axins = inset_axes(ax[fig_id//4,fig_id%4], width=.5, height=0.6,loc=3)
        # sns.set(font_scale = .5)
        p = sns.barplot(y='features',x='scores', color='blue',data=f_imp,ax=axins)
        p.tick_params(axis='both',labelsize=6) # labelsize=6
        
        # _, ylabels = plt.yticks()
        # p.set_yticklabels(ylabels,size=8)
        p.set(xlabel=None)
        p.set(ylabel=None)

        axins.text(0.1,8.0,r'MSE={:.2f}'.format(MSE_record_mean),size=6) # size = 6
        axins.text(0.1,7,r'$R^2$={:.2f}'.format(r2_record_mean),size=6)
        # _, xlabels = plt.xticks()
        # p.set_xticklabels(xlabels,size=8)
        # axins.barh(data_ml[f_imp['features'].iloc[0]],data_ml['d_soc'],s=1)
        # ax = fig.add_subplot(6, 4, i+2)
        # ax = fig.add_subplot(spec[i+1])
        # fig_id+=1
        # d_soc_lat_mean = np.nanmean(np.flipud(datay),axis=1)
        # ax[fig_id//6,fig_id%6].plot(d_soc_lat_mean,np.flipud(lats))
        # ax[fig_id//6,fig_id%6].set_yticklabels([])
        # ax[fig_id//6,fig_id%6].tick_params(axis='both', which='major', labelsize=15)
        
        # plt.plot(d_soc_lat_mean,np.flipud(lats))
        fig_id+=1

        if cmip=='cmip5':
            cmip='cmip6'
        else:
            cmip='cmip5'
        
        # plt.gca().set_aspect('equal', adjustable='datalim')
fig.subplots_adjust(wspace=0)
plt.tight_layout()

fig.savefig('./figures/20221216_FIG_global_map_delta_soc_cmip5&6commonmodels.png',bbox_inches='tight',dpi=300)                                                       






