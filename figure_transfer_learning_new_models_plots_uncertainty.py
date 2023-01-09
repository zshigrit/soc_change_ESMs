import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import netCDF4
from scipy.stats import linregress
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import sys
sys.path.insert(0, './cmip_ml/')
import cmip_ml_soc_preprocessing as preproc

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
main_models_cmip5 = ['bcc','can_esm','cesm','ipsl','miroc','mpi','ncc','mohc_ukesm','bnu','gfdl','giss','inm']    
main_models_cmip6 = ['bcc','can_esm','cesm','ipsl','miroc','mpi','ncc','mohc_ukesm','access','cmcc','cnrm','tai']      
cmips=['cmip5','cmip6']
# cmips=['cmip5']
# variables = ['npp','cSoil','tas','pr','mrso']

file='*.nc'
# # cmip='cmip5'
# # main_model = 'mpi'
# # model=['bcc']
# dSOC_models = pd.DataFrame()
# for cmip in cmips:
#     if cmip=='cmip5':
#         main_models = main_models_cmip5.copy()
#     else:
#         main_models = main_models_cmip6.copy()
#     # main_models = ['bcc']    
#     for main_model in main_models: 
#         if cmip=='cmip6' and (main_model == 'cesm' or main_model == 'ncc'):
#             variables = ['npp','cSoilAbove1m','tas','pr'] 
#         else:
#             variables = ['npp','cSoil','tas','pr']
            
#         for variable in variables:
            
#             data_temp = preproc.full_temporal(cmip,main_model,variable,file) 
#             start_year,end_year = preproc.start_end_years_simulation(cmip,main_model,variable,file)        
            
#             if variable == 'npp' or variable == 'gpp' or variable == 'rh':
                
#                 scalar_month = 1000 * 30 * 24 *3600              
                
#                 data_temp_scaled = data_temp * scalar_month 
                
#                 data_1996_2005 = data_temp_scaled[-(end_year-1996+1)*12:-(end_year-2005)*12,:,:].copy()
                
#                 if end_year==2100:
#                     data_2091_2100 = data_temp_scaled[-(end_year-2091+1)*12:,:,:].copy()
#                 else:
#                     data_2091_2100 = data_temp_scaled[-(end_year-2091+1)*12:-(end_year-2100)*12,:,:].copy()
                
#                 vars()[variable] = np.nansum(data_1996_2005,axis=0)/10
#                 vars()[variable+'_ssp585'] = np.nansum(data_2091_2100,axis=0)/10

#             elif variable == 'pr':  
                
#                 scalar_month = 30 * 24 *3600
#                 data_temp_scaled = data_temp*scalar_month/1000*1000 # change kg/m2/s to mm/m2/month (1000kg/m3)
                
#                 data_1996_2005 = data_temp_scaled[-(end_year-1996+1)*12:-(end_year-2005)*12,:,:].copy()
                
#                 if end_year==2100:
#                     data_2091_2100 = data_temp_scaled[-(end_year-2091+1)*12:,:,:].copy()
#                 else:
#                     data_2091_2100 = data_temp_scaled[-(end_year-2091+1)*12:-(end_year-2100)*12,:,:].copy()               
                
#                 vars()[variable] = np.nansum(data_1996_2005,axis=0)/10 
#                 vars()[variable+'_ssp585'] = np.nansum(data_2091_2100,axis=0)/10
#             else:
#                 data_temp_scaled = data_temp.copy()
                
#                 data_1996_2005 = data_temp_scaled[-(end_year-1996+1)*12:-(end_year-2005)*12,:,:].copy()
                
#                 if end_year==2100:
#                     data_2091_2100 = data_temp_scaled[-(end_year-2091+1)*12:,:,:].copy()
#                 else:
#                     data_2091_2100 = data_temp_scaled[-(end_year-2091+1)*12:-(end_year-2100)*12,:,:].copy()
#                 if cmip=='cmip6' and (main_model == 'cesm' or main_model == 'ncc') and variable =='cSoilAbove1m':
#                     variable = 'cSoil'    
#                 vars()[variable] = np.nanmean(data_1996_2005,axis=0)
#                 vars()[variable+'_ssp585'] = np.nanmean(data_2091_2100,axis=0)
        
#         cSoil_1d = cSoil.flatten()
#         d_soc = cSoil_ssp585 - cSoil
#         d_soc_1d = d_soc.flatten()
        
#         # plt.imshow(d_soc);plt.colorbar()
        
#         npp_1d = npp.flatten()
#         d_npp = npp_ssp585 - npp
#         d_npp_1d = d_npp.flatten()
        
#         pr_1d = pr.flatten()
#         d_pr = pr_ssp585 - pr
#         d_pr_1d = d_pr.flatten()
        
#         tas_1d = tas.flatten()
#         d_ta = np.array(tas_ssp585 - tas)
#         d_ta_1d = np.array(d_ta.flatten()) 
        
#         npp[npp<=0] = np.nan
#         npp_ssp585[npp_ssp585<=0] = np.nan
#         tau_npp = cSoil*1000/npp  
#         tau_npp_1d = tau_npp.flatten() 
#         tau_ssp585_npp = cSoil_ssp585*1000/npp_ssp585      
#         d_tau_npp = tau_ssp585_npp - tau_npp
#         d_tau_npp_1d = d_tau_npp.flatten()
        
#         # mrso_1d = mrso.flatten() 
#         # d_mrso = mrso_ssp585 - mrso
#         # d_mrso_1d = d_mrso.flatten()                

#         d = {'cSoil':cSoil_1d,
#               'npp':npp_1d,
#               'tas':tas_1d,
#               'pr':pr_1d,
#               'tau_npp':tau_npp_1d,
#               'd_npp':d_npp_1d,
#               'd_ta':d_ta_1d,
#               'd_pr':d_pr_1d,
#               'd_tau_npp':d_tau_npp_1d,
#               'd_soc':d_soc_1d
#               }

#         data_mlx = pd.DataFrame(data=d)
#         # data_mlx[data_mlx['d_soc']<-15] = np.nan
        
#         notNaNs = ~np.any(np.isnan(data_mlx),axis=1)
#         data_ml = data_mlx[notNaNs]
        
#         """ machine learning algorithms """
#         #%% transfer learning
        
#         from sklearn.model_selection import KFold
#         from sklearn.ensemble import RandomForestRegressor
#         from sklearn.metrics import r2_score
#         import seaborn as sns
#         sns.set()
        
#         X = data_ml.values[:,:-1].copy()
#         y = data_ml.values[:,-1].copy()
        
#         rf_reg = RandomForestRegressor(n_estimators=100,max_depth=20,min_samples_split=4,random_state=20)
#         rf_reg.fit(X,y)
        
#         # r2_record = pd.DataFrame(r2_score(y,rf_reg.predict(X)))
               
#         # cmips=['cmip5','cmip6']
#         # cmip='cmip5'
#         file='*.nc'
#         # target_models = ['bcc','can_esm','cesm','ipsl','miroc','mpi','ncc','mohc_ukesm']
#         # target_models = ['can_esm']
        
#         d_soc_target_total_record=[]
        
#         for cmip_ in cmips:
#             if cmip_ == 'cmip5':
#                 target_models = main_models_cmip5.copy()
#             else:
#                 target_models = main_models_cmip6.copy()
                
#             # target_models = ['bnu']
#             for target_model in target_models: 
                
#                 if cmip_=='cmip6' and (target_model == 'cesm' or target_model == 'ncc'):
#                     variables = ['npp','cSoilAbove1m','tas','pr']
#                 else:
#                     variables = ['npp','cSoil','tas','pr']
                    
#                 for variable in variables:
                
#                     data_temp = preproc.full_temporal(cmip_,target_model,variable,file)
#                     start_year,end_year = preproc.start_end_years_simulation(cmip_,target_model,variable,file)
                
#                     if variable == 'npp' or variable == 'gpp' or variable == 'rh':
                
#                         scalar_month = 1000 * 30 * 24 *3600              
                        
#                         data_temp_scaled = data_temp * scalar_month 
                        
#                         data_1996_2005 = data_temp_scaled[-(end_year-1996+1)*12:-(end_year-2005)*12,:,:].copy()
                        
#                         if end_year==2100:
#                             data_2091_2100 = data_temp_scaled[-(end_year-2091+1)*12:,:,:].copy()
#                         else:
#                             data_2091_2100 = data_temp_scaled[-(end_year-2091+1)*12:-(end_year-2100)*12,:,:].copy()
                        
#                         vars()[variable] = np.nansum(data_1996_2005,axis=0)/10
#                         vars()[variable+'_ssp585'] = np.nansum(data_2091_2100,axis=0)/10

#                     elif variable == 'pr':  
                
#                         scalar_month = 30 * 24 *3600
#                         data_temp_scaled = data_temp*scalar_month/1000*1000 # change kg/m2/s to mm/m2/month (1000kg/m3)
                        
#                         data_1996_2005 = data_temp_scaled[-(end_year-1996+1)*12:-(end_year-2005)*12,:,:].copy()
                        
#                         if end_year==2100:
#                             data_2091_2100 = data_temp_scaled[-(end_year-2091+1)*12:,:,:].copy()
#                         else:
#                             data_2091_2100 = data_temp_scaled[-(end_year-2091+1)*12:-(end_year-2100)*12,:,:].copy()               
                        
#                         vars()[variable] = np.nansum(data_1996_2005,axis=0)/10 
#                         vars()[variable+'_ssp585'] = np.nansum(data_2091_2100,axis=0)/10
#                     else:
#                         data_temp_scaled = data_temp.copy()
                        
#                         data_1996_2005 = data_temp_scaled[-(end_year-1996+1)*12:-(end_year-2005)*12,:,:].copy()
                        
#                         if end_year==2100:
#                             data_2091_2100 = data_temp_scaled[-(end_year-2091+1)*12:,:,:].copy()
#                         else:
#                             data_2091_2100 = data_temp_scaled[-(end_year-2091+1)*12:-(end_year-2100)*12,:,:].copy()
#                         if cmip_=='cmip6' and (target_model == 'cesm' or target_model == 'ncc') and variable =='cSoilAbove1m':
#                              variable = 'cSoil'     
#                         vars()[variable] = np.nanmean(data_1996_2005,axis=0)
#                         vars()[variable+'_ssp585'] = np.nanmean(data_2091_2100,axis=0)
            
#                 cSoil_1d = cSoil.flatten()
#                 d_soc = cSoil_ssp585 - cSoil
#                 d_soc_1d = d_soc.flatten()
                
#                 npp_1d = npp.flatten()
#                 d_npp = npp_ssp585 - npp
#                 d_npp_1d = d_npp.flatten()
                
#                 pr_1d = pr.flatten()
#                 d_pr = pr_ssp585 - pr
#                 d_pr_1d = d_pr.flatten()
                
#                 tas_1d = tas.flatten()
#                 d_ta = np.array(tas_ssp585 - tas)
#                 d_ta_1d = np.array(d_ta.flatten()) 
                
#                 npp[npp<=0] = np.nan
#                 npp_ssp585[npp_ssp585<=0] = np.nan
#                 tau_npp = cSoil*1000/npp  
#                 tau_npp_1d = tau_npp.flatten() 
#                 tau_ssp585_npp = cSoil_ssp585*1000/npp_ssp585      
#                 d_tau_npp = tau_ssp585_npp - tau_npp
#                 d_tau_npp_1d = d_tau_npp.flatten()
                
#                 # mrso_1d = mrso.flatten() 
#                 # d_mrso = mrso_ssp585 - mrso
#                 # d_mrso_1d = d_mrso.flatten()                
        
#                 d = {'cSoil':cSoil_1d,
#                       'npp':npp_1d,
#                       'tas':tas_1d,
#                       'pr':pr_1d,
#                       'tau_npp':tau_npp_1d,
#                       'd_npp':d_npp_1d,
#                       'd_ta':d_ta_1d,
#                       'd_pr':d_pr_1d,
#                       'd_tau_npp':d_tau_npp_1d,
#                       'd_soc':d_soc_1d
#                       }
                   
#                 data_mlx = pd.DataFrame(data=d)
#                 # data_mlx[data_mlx['d_soc']<-15] = np.nan
                
#                 notNaNs = ~np.any(np.isnan(data_mlx),axis=1)
#                 data_ml_target = data_mlx[notNaNs]
            
            
#                 X_target = data_ml_target.values[:,:-1].copy()
#                 y_target = data_ml_target.values[:,-1].copy()
            
#                 ypred = rf_reg.predict(X_target)
#                 yhat = np.empty((data_mlx.shape[0])); yhat[:] = np.nan
#                 yhat[notNaNs] = ypred
#                 out = np.reshape(yhat,(cSoil.shape[0],cSoil.shape[1]))
#                 # lons,lats,_,_ = preproc.extract_lat_lon(cmip_, target_model, file)
#                 # datax = preproc.change_lon_lat(lats,lons,out)# rearrange the out
                
#                 land_area_ = land_area(cmip_,target_model,
#                                            file_id = files_land_area[target_model][cmip_])
#                 land_fraction_ = land_fraction(cmip_,target_model,
#                                            file_id = files_land_fraction[target_model][cmip_])
                
#                 d_soc_target_grid = out * land_area_*land_fraction_ *1000/100
                
#                 d_soc_target_total = np.nansum(d_soc_target_grid)/(10**15) 
                  
#                 d_soc_target_total_record.append(d_soc_target_total)
                
#         dSOC_models[cmip+main_model] = d_soc_target_total_record
#         data=dSOC_models.values.copy()
#         np.save('dSOC_model.npy',data)
#         # model_labels = {'cmip5':
#         #   ['bcc-csm1-1','CanESM2','CESM1-BGC','IPSL-CM5A-LR','MIROC-ESM','MPI-ESM-LR','NorESM1-ME','HadGEM2-ES'],
#         #   'cmip6':
#         #   ['BCC-CSM2-MR','CanESM5','CESM2','IPSL-CM6A-LR','MIROC-ES2L','MPI-ESM1-2-LR','NorESM2-LM','UKESM1-0-LL']}
#         # dSOC_models_ = np.load('dSOC_model.npy')
#         # dSOC_models = pd.DataFrame(dSOC_models_,columns=model_labels['cmip5'] + model_labels['cmip6'])
"""
plotting
plotting
plotting
"""
#%%        

# plt.imshow(np.flipud(out),vmin=-3,vmax=3,cmap='bwr');plt.colorbar()
# plt.scatter(y_target,ypred)

# m.imshow(d_soc,vmin=-3,vmax=3,cmap='bwr');m.colorbar()
# plt.imshow(np.flipud(d_soc),vmin=-4,vmax=4,cmap='bwr_r');plt.colorbar()
model_labels = {'cmip5':
      ['bcc-csm1-1','CanESM2','CESM1-BGC','IPSL-CM5A-LR','MIROC-ESM','MPI-ESM-LR','NorESM1-ME','HadGEM2-ES',
       'BNU-ESM','GFDL-ESM2G','GISS-E2-R','inmcm4'],
      'cmip6':
      ['BCC-CSM2-MR','CanESM5','CESM2','IPSL-CM6A-LR','MIROC-ES2L','MPI-ESM1-2-LR','NorESM2-LM','UKESM1-0-LL',
           'ACCESS-ESM1-5','CMCC-ESM2','CNRM-ESM2-1','TaiESM1']}

# dSOC_models.to_pickle('dSOC_models')
data = np.load('dSOC_model.npy')
columns = model_labels['cmip5'] + model_labels['cmip6']
dSOC_models = pd.DataFrame(data,columns)

fig,(ax1,ax2) = plt.subplots(2,1,figsize=(20,16))
dSOC_models.boxplot(fontsize=20,ax=ax1)
ax1.set_ylabel('$\Delta$SOC (Pg)',fontsize=25)
# ax1.set_xlabel('RF-Models',fontsize=25)
ax1.set_xticklabels([])
ax1.xaxis.grid(False)
ax1.set_ylim(-400,400)

ax1.set_title('A',loc='right',fontsize=20,fontweight="bold")
# [ax1.get_xticklabels()[i].set_color("blue") for i in np.arange(0,12)]
# [ax1.get_xticklabels()[i].set_color("r") for i in np.arange(12,24)]    
# fig.savefig('./figures/fig_uncertainty_among_models_delta_soc.png',bbox_inches='tight',dpi=300)

dSOC_input = pd.DataFrame(dSOC_models.values.T,columns=model_labels['cmip5'] + model_labels['cmip6'])
# fig,ax = plt.subplots(figsize=(20,8))
dSOC_input.boxplot(fontsize=20,ax=ax2)
ax2.set_ylabel('$\Delta$SOC (Pg)',fontsize=25)
ax2.set_ylim(-400,400)
# ax2.xlabel('RF-Inputs',fontsize=25)
xlabels = columns.copy()
ax2.set_xticklabels(xlabels,rotation='90',fontsize=18)
[ax2.get_xticklabels()[i].set_color("blue") for i in np.arange(0,12)]
[ax2.get_xticklabels()[i].set_color("r") for i in np.arange(12,24)]
ax2.set_title('B',loc='right',fontsize=20,fontweight="bold")
ax2.xaxis.grid(False)
fig.savefig('./figures/fig_uncertainty_delta_soc.png',bbox_inches='tight',dpi=300)

    # dSOC_models.values()
    # plt.xticks(np.arange(1,17), dSOC_models.columns,rotation=90,fontsize=18)
    # plt.yticks(fontsize=18)
    
    # plt.boxplot(dSOC_models.values[:,0:8].flatten())
    # plt.boxplot(dSOC_models.values[:,8:16].flatten())


# # plt.hist(dSOC_models.values.flatten())
# import scipy

# fig,(ax1,ax2) = plt.subplots(2,1,figsize=(12,12))

# mu, sigma = scipy.stats.norm.fit(dSOC_models.values.flatten())
# ax1.hist(dSOC_models.values.flatten(),bins=10, density =True, alpha=0.25,label = 'RF-$\Delta$SOC',edgecolor='black')
# xmin, xmax = plt.xlim()
# x = np.linspace(-300, 400, 100)
# best_fit_line = scipy.stats.norm.pdf(x, mu, sigma)
# ax1.plot(x, best_fit_line,'blue',linewidth=2)

# ax1.set_title('A',loc='right',fontsize=20,fontweight="bold")

# # _, bins, _ = ax1.hist(dSOC_models.values.flatten(), 30, density =1, alpha=0.25,label = 'RF-$\Delta$SOC')
# # mu, sigma = scipy.stats.norm.fit(dSOC_models.values.flatten())
# # # sigma/=np.sqrt(len(dSOC_models.values.flatten()))
# # # x = np.linspace(-300, 400, 100)
# # # best_fit_line = scipy.stats.norm.pdf(x, mu, sigma)
# # best_fit_line = scipy.stats.norm.pdf(bins, mu, sigma)
# # ax1.plot(bins, best_fit_line,'blue')
# # ax1.set_xlabel('$\Delta$SOC (Pg)',fontsize=25)
# ax1.set_ylabel('Probability',fontsize = 18)
# ax1.tick_params(axis="x", labelsize=18)
# ax1.tick_params(axis="y", labelsize=18)
# ax1.set_xlim(-400,400)
# ax1.set_ylim(0,0.006)
# # ax1.set_yticks(np.arange(0,0.01,0.002))

# # delta_soc needed from running "figure_total_soc_comparisions.py'
# # plt.scatter(delta_soc['d_soc'][delta_soc['cmip']=='cmip5'].values,[0.001]*12,c='k',s=35,alpha=0.65,label='CMIP5')
# # plt.scatter(delta_soc['d_soc'][delta_soc['cmip']=='cmip6'].values,[0.002]*8,c='r',s=35,alpha=0.65,label='CMIP6')
# # plt.legend(fontsize=20)
# # fig.savefig('/Users/zo0/Documents/my_projects/cmip/figures/fig_histogram_rfmodels_delta_soc.png',bbox_inches='tight',dpi=300)

# # dSOC_original = [dSOC_models.values[i,i] for i in np.arange(16)]
# # _, bins, _ = plt.hist(dSOC_original, 15, density =1, alpha=0.5,label = 'only model data')
# # mu, sigma = scipy.stats.norm.fit(dSOC_original)
# # best_fit_line = scipy.stats.norm.pdf(bins, mu, sigma)
# # plt.plot(bins, best_fit_line,'orange')

# # plt.legend(loc='upper right')

# # plt.xlabel('$\Delta$SOC (Pg)',fontsize=18)
# # plt.ylabel('Probability',fontsize = 18)


# # # cmip5 and cmip6
# # data_cmip5 = dSOC_models.values[:8,:8]
# # data_cmip6 = dSOC_models.values[8:16,8:16]
# # _, bins, _ = plt.hist(data_cmip5.flatten(), 15, density =1, alpha=0.1,label = 'CMIP5')
# # mu, sigma = scipy.stats.norm.fit(data_cmip5.flatten())
# # # sigma/=np.sqrt(len(dSOC_models.values.flatten()))
# # best_fit_line = scipy.stats.norm.pdf(bins, mu, sigma)
# # plt.plot(bins, best_fit_line,'blue')
# # plt.xlabel('$\Delta$SOC (Pg)',fontsize=18)
# # plt.ylabel('Probability',fontsize = 18)

# # _, bins, _ = plt.hist(data_cmip6.flatten(), 15, density =1, alpha=0.1,label = 'CMIP6')
# # mu, sigma = scipy.stats.norm.fit(data_cmip6.flatten())
# # # sigma/=np.sqrt(len(dSOC_models.values.flatten()))
# # best_fit_line = scipy.stats.norm.pdf(bins, mu, sigma)
# # plt.plot(bins, best_fit_line,'orange')

# # plt.legend(loc='upper right')

# # plt.xlabel('$\Delta$SOC (Pg)',fontsize=18)
# # plt.ylabel('Probability',fontsize = 18)


# #%% calculate sum squares of differences
# #  sum square of subjects 

# data = dSOC_models.values.copy()
# # data = data_.copy()
# # data = data_[8:16,8:16]
# n=24 #n=16

# ss_subject = sum(n*((data.mean(axis=0)-data.mean())**2)) # ss of subjects (i.e. MODLES)

# ss_input = sum(n*((data.mean(axis=1)-data.mean())**2)) # SS OF INPUT

# ss_interaction = 0.
# for i in range(n):
#     for j in range(n):
#         ss_interaction_ = (data[i,j]-data.mean(axis=0)[j]-data.mean(axis=1)[i]+data.mean())**2
#         ss_interaction+=ss_interaction_

# ss_total = ss_subject + ss_input + ss_interaction
# print(ss_subject/ss_total,ss_input/ss_total,ss_interaction/ss_total)

# labels = 'RF model', 'RF input', 'Interactions'
# sizes = [ss_subject/ss_total, ss_input/ss_total, ss_interaction/ss_total]
# explode = (0, 0., 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

# # fig1, ax1 = plt.subplots()
# axins = inset_axes(ax1, width=2.5, height=2.5,loc='upper left',bbox_to_anchor=(0.08,-0.1,2,1.075), bbox_transform=ax1.transAxes)
# axins.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',textprops={'fontsize': 15},
#         shadow=False, startangle=90)
# # axins.axes.xaxis.set_visible(False)
# # axins.axes.yaxis.set_visible(False)
            
# # ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
# #         shadow=True, startangle=90)
# # ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

# # fig.savefig('/Users/zo0/Documents/my_projects/cmip/figures/histogram_cmip56_delta_soc.png',bbox_inches='tight',dpi=300)
# # plt.show()

# #%% 
# """plot probability distribution of cmip5, cmip6 and obs-constrained"""

# import matplotlib.pyplot as plt

# import scipy

# import seaborn as sns

# import numpy as np


# # fig = plt.figure(figsize=(10,8))
# # 
# mu, sigma = scipy.stats.norm.fit(data[0:12,0:12].flatten())
# ax2.hist(data[0:12,0:12].flatten(),bins=10, density =True, alpha=0.,color='b')
# xmin, xmax = plt.xlim()
# x = np.linspace(-300, 400, 100)
# best_fit_line = scipy.stats.norm.pdf(x, mu, sigma)
# ax2.plot(x, best_fit_line,'b',linewidth=2,label='CMIP5')
# # plt.legend()

# mu, sigma = scipy.stats.norm.fit(data[12:24,0:12].flatten())
# ax2.hist(data[12:24,0:12].flatten(),bins=10, density =True, alpha=0.,color='b')
# xmin, xmax = plt.xlim()
# x = np.linspace(-200, 300, 100)
# best_fit_line = scipy.stats.norm.pdf(x, mu, sigma)
# ax2.plot(x, best_fit_line,'b--',linewidth=2,label='CMIP5 with CMIP6 Inputs')
# # plt.legend()

# mu, sigma = scipy.stats.norm.fit(data[12:24,12:24].flatten())
# ax2.hist(data[12:24,12:24].flatten(),bins=10, density =True, alpha=0.,color='r')
# xmin, xmax = plt.xlim()
# x = np.linspace(-200, 300, 100)
# best_fit_line = scipy.stats.norm.pdf(x, mu, sigma)
# ax2.plot(x, best_fit_line,'r',linewidth=2,label='CMIP6')
# # plt.legend()

# mu, sigma = scipy.stats.norm.fit(data[0:12,12:24].flatten())
# ax2.hist(data[0:12,12:24].flatten(),bins=10, density =True, alpha=0.,color='r')
# xmin, xmax = plt.xlim()
# x = np.linspace(-200, 300, 100)
# best_fit_line = scipy.stats.norm.pdf(x, mu, sigma)
# ax2.plot(x, best_fit_line,'r--',linewidth=2,label='CMIP6 with CMIP5 Inputs')
# # plt.legend()

# # _, bins, _ = ax.hist(data[8:16,8:16].flatten(), 10, density = 1, alpha=0.)
# # mu, sigma = scipy.stats.norm.fit(data[8:16,8:16].flatten())
# # best_fit_line = scipy.stats.norm.pdf(bins, mu, sigma)
# # ax.plot(bins, best_fit_line,'brown',label='CMIP 6')

# # _, bins, _ = ax.hist(data[0:8,8:16].flatten(), 10, density = 1, alpha=0.)
# # mu, sigma = scipy.stats.norm.fit(data[0:8,8:16].flatten())
# # best_fit_line = scipy.stats.norm.pdf(bins, mu, sigma)
# # ax.plot(bins, best_fit_line,'brown',linestyle='dashed',label='CMIP 6 with CMIP5 Inputs')

# # y = [0.004]*8; y[-1],y[-3]=0.0035,0.0045
# # ax.scatter(data_000[8:16],y,c='brown',s=50,label = 'Obs-constrained (CMIP 6)')
# # box = ax.boxplot(data_000[8:16],vert=False,positions=[0.004],widths=0.0015,
# #                  whiskerprops=dict(linestyle='--'),boxprops=dict(linestyle='--'))

# # for _, line_list in box.items():
# #     for line in line_list:
# #         line.set_color('brown')
# ax2.set_xlim(-400,400)
# ax2.set_ylim([0,0.008])
# ax2.set_yticks(np.arange(0,0.01,0.002))
# ax2.tick_params(axis="x", labelsize=18)
# ax2.tick_params(axis="y", labelsize=18)

# # plt.set_xticks(np.arange(-200,350,100))
# # plt.set_xticklabels(np.arange(-200,350,100),fontsize=18)
# # plt.set_yticklabels(np.arange(0,0.012,0.002),fontsize=18)
# ax2.set_ylabel('Probablity',fontsize=18)
# ax2.set_xlabel('$\Delta$SOC (Pg)',fontsize=18)

# ax2.set_title('B',loc='right',fontsize=20,fontweight="bold")

# ax2.legend(loc='upper left')

# """ add the pie plot for relative contribution """
# data1 = dSOC_models.values[0:12,0:12].copy()
# # data = data_.copy()
# # data = data_[8:16,8:16]
# n=12 #n=16

# ss_subject = sum(n*((data1.mean(axis=0)-data1.mean())**2)) # ss of subjects (i.e. MODLES)

# ss_input = sum(n*((data1.mean(axis=1)-data1.mean())**2)) # SS OF INPUT

# ss_interaction = 0.
# for i in range(n):
#     for j in range(n):
#         ss_interaction_ = (data1[i,j]-data1.mean(axis=0)[j]-data1.mean(axis=1)[i]+data1.mean())**2
#         ss_interaction+=ss_interaction_

# ss_total = ss_subject + ss_input + ss_interaction
# print(ss_subject/ss_total,ss_input/ss_total,ss_interaction/ss_total)

# labels = 'RF model', 'RF input', 'Interactions'
# sizes = [ss_subject/ss_total, ss_input/ss_total, ss_interaction/ss_total]
# explode = (0, 0., 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

# # fig1, ax1 = plt.subplots()
# axins = inset_axes(ax2, width=2.5, height=2.5,loc='upper left',bbox_to_anchor=(0.08,-1.5,2,1.075), bbox_transform=ax1.transAxes)
# axins.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',textprops={'fontsize': 15},
#         shadow=False, startangle=90)
# axins.text(-.1,-1.2,'CMIP5',c='b',size=16)

# data2 = dSOC_models.values[12:24,12:24].copy()
# # data = data_.copy()
# # data = data_[8:16,8:16]
# n=12 #n=16

# ss_subject = sum(n*((data2.mean(axis=0)-data2.mean())**2)) # ss of subjects (i.e. MODLES)

# ss_input = sum(n*((data2.mean(axis=1)-data2.mean())**2)) # SS OF INPUT

# ss_interaction = 0.
# for i in range(n):
#     for j in range(n):
#         ss_interaction_ = (data2[i,j]-data2.mean(axis=0)[j]-data2.mean(axis=1)[i]+data2.mean())**2
#         ss_interaction+=ss_interaction_

# ss_total = ss_subject + ss_input + ss_interaction
# print(ss_subject/ss_total,ss_input/ss_total,ss_interaction/ss_total)

# labels = 'RF model', 'RF input', 'Interactions'
# sizes = [ss_subject/ss_total, ss_input/ss_total, ss_interaction/ss_total]
# explode = (0, 0., 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

# # fig1, ax1 = plt.subplots()
# axins = inset_axes(ax2, width=2.5, height=2.5,loc='upper left',bbox_to_anchor=(.65,-0.05,2,1.075), bbox_transform=ax2.transAxes)
# axins.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',textprops={'fontsize': 15},
#         shadow=False, startangle=90)
# axins.text(1.,.05,'CMIP6',c='r',size=16)

# """ end """

# fig.savefig('./figures/FIG_histogram_cmip56_delta_soc_test.png',bbox_inches='tight',dpi=300)



