#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 17:03:45 2021

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

import seaborn as sns
sns.set()
#%% 
model_labels = {'cmip5':
          ['bcc-csm1-1','CanESM2','CESM1-BGC','IPSL-CM5A-LR','MIROC-ESM','MPI-ESM-LR','NorESM1-ME','HadGEM2-ES',
           'BNU-ESM','GFDL-ESM2G','GISS-E2-R','inmcm4'],
          'cmip6':
          ['BCC-CSM2-MR','CanESM5','CESM2','IPSL-CM6A-LR','MIROC-ES2L','MPI-ESM1-2-LR','NorESM2-LM','UKESM1-0-LL',
           'ACCESS-ESM1-5','CMCC-ESM2','CNRM-ESM2-1','TaiESM1']}

color_cmip5 =  ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf','#FFFF00','#FF0000']

color_cmip6 =  ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#458B74', '#000000','#0000FF','#EE00EE'] #''#6495ED
    
models_cmip5 = ['bcc','can_esm','cesm','ipsl','miroc','mpi','ncc','mohc_ukesm','bnu','gfdl','giss','inm']  
# models_cmip5 = ['bnu']
models_cmip6 = ['bcc','can_esm','cesm','ipsl','miroc','mpi','ncc','mohc_ukesm','access','cmcc','cnrm','tai']
# models = ['mohc_ukesm']        
cmips=['cmip5','cmip6']
# cmips=['cmip5']
file='*.nc'

delta_soc = pd.DataFrame(columns=(['cmip','model','d_soc']))


# figure #2: scenario ssp585
# plt.rcParams.update({'font.size': 16})
fig,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=(12,6),gridspec_kw={'width_ratios': [3, 1,1.5]},sharey=False)


for cmip in cmips:
    if cmip == 'cmip5':
        models = models_cmip5.copy()
    else:
        models = models_cmip6.copy()
    for model in models:
        print(model)
        variables=['cSoil']
        # if model=='mpi':
        #     continue
        if cmip=='cmip6' and (model == 'cesm' or model == 'ncc'):
            variables = ['cSoilAbove1m']
        
# """        
#         variables = ['npp','cSoil','tas','pr','mrsos']
#         if model == 'mpi':
#             variables = ['npp','cSoil','tas','pr','mrso']
#         if cmip=='cmip6' and (model == 'cesm' or model == 'ncc'):
#             variables = ['npp','cSoilAbove1m','tas','pr','mrsos']
# """        
        for variable in variables:
            
            data_temp = preproc.full_temporal(cmip,model,variable,file)
            start_year,end_year = preproc.start_end_years_simulation(cmip,model,variable,file)
            # if end_year<2100: record_error = record_error.append(pd.DataFrame({'cmip':[cmip],'model':[model],'end_yr':[end_year]}))
            
            land_area = preproc.land_area(cmip,model)
            land_fraction = preproc.land_fraction(cmip,model)
            land_area_ratio = (land_area*land_fraction)/np.nansum(land_area*land_fraction)
            # lf_mask = land_fraction.reshape ((1,land_fraction.shape[0],land_fraction.shape[1]))
            lf_mask = np.zeros(data_temp.shape,dtype=bool)
            lf_mask[:,:,:] = land_fraction[np.newaxis,:,:]==0
            data_temp_masked = ma.masked_array(data_temp,mask=lf_mask)
           
            if variable == 'npp' or variable == 'gpp' or variable == 'rh':
                scalar_month = 1000 * 30 * 24 *3600              
                
                data_temp_scaled = data_temp_masked * scalar_month * \
                                land_area.reshape((1,land_area.shape[0],land_area.shape[1])) \
                                * land_fraction.reshape ((1,land_fraction.shape[0],land_fraction.shape[1]))
                           
                data_temporal_yearly = [np.nansum(data_temp_scaled[i*12:(i+1)*12,:,:])/1e15 \
                                        for i in np.arange(round(data_temp_scaled.shape[0]/12))]
                
                axs.plot(np.arange(start_year,2100+1),data_temporal_yearly[:2101-start_year],label=model)             
                axs.title.set_text('npp')
                axs.legend(bbox_to_anchor=(1.05, 1))
            elif variable == 'pr':  
                scalar_month = 30 * 24 *3600
                data_temp_scaled = data_temp_masked*scalar_month/1000*1000 * \
                            land_area_ratio.reshape ((1,land_area_ratio.shape[0],land_area_ratio.shape[1])) # change kg/m2/s to mm/m2/month (1000kg/m3)
                
                data_temporal_yearly = [np.nansum(data_temp_scaled[i*12:(i+1)*12,:,:]) \
                                        for i in np.arange(round(data_temp_scaled.shape[0]/12))]
                
                # axs[0,1].plot(np.arange(start_year,2100+1),data_temporal_yearly[:2101-start_year],label=variable)
                # axs[0,1].title.set_text('pr')
                
                axs.plot(np.arange(start_year,2100+1),data_temporal_yearly[:2101-start_year],label=model)
                axs.title.set_text('pr') 
                axs.legend(bbox_to_anchor=(1.05, 1))
                
            elif variable=='cSoil' or variable=='cSoilAbove1m':
                data_temp_scaled = data_temp_masked * \
                                land_area.reshape((1,land_area.shape[0],land_area.shape[1])) \
                                * land_fraction.reshape ((1,land_fraction.shape[0],land_fraction.shape[1]))
                                
                data_temporal_yearly = [np.nansum(np.nanmean(data_temp_scaled[i*12:(i+1)*12,:,:],axis=0))/1e12 \
                                        for i in np.arange(round(data_temp_scaled.shape[0]/12))]
                if model == 'e3sm':
                    data_temporal_2005_2100 = data_temporal_yearly[2007-start_year:2101-start_year]
                else:
                    data_temporal_2005_2100 = data_temporal_yearly[1996-start_year:2101-start_year].copy()
                    
                data_temporal_2005_2100_rel = data_temporal_2005_2100-data_temporal_2005_2100[0]
                delta_soc.loc[len(delta_soc)]=[cmip,model_labels[cmip][models.index(model)],data_temporal_2005_2100_rel[-1]]
                
                if cmip=='cmip5':
                   ax1.plot(np.arange(1996,2100+1),data_temporal_2005_2100_rel,linestyle='--',
                         label=model_labels[cmip][models.index(model)],c=color_cmip5[models.index(model)])
                else:
                    ax1.plot(np.arange(1996,2100+1),data_temporal_2005_2100_rel,
                    label=model_labels[cmip][models.index(model)],c=color_cmip6[models.index(model)])
                    
                
                
                # # rectangular box plot
                # bplot1 = ax1.boxplot(all_data,
                #      vert=True,  # vertical box alignment
                #      patch_artist=True,  # fill with color
                #      labels=labels)  # will be used to label x-ticks


                # plt.savefig('/Users/zo0/Documents/my_projects/cmip/figures/SSP585.png',bbox_inches='tight',dpi=600)
                # axs.title.set_text('cSoil') 
                # axs.legend(bbox_to_anchor=(1.05, 1))
            elif variable=='tas':
                data_temp_scaled = (data_temp_masked-273.15)*\
                            land_area_ratio.reshape ((1,land_area_ratio.shape[0],land_area_ratio.shape[1]))
                data_temp_global_ave = np.nansum(data_temp_scaled,axis=(1,2))
                data_temporal_yearly = [np.nanmean(data_temp_global_ave[i*12:(i+1)*12]) \
                                        for i in np.arange(round(data_temp_scaled.shape[0]/12))]
                
                # axs[1,1].plot(np.arange(start_year,2100+1),data_temporal_yearly[:2101-start_year],label=variable)
                # axs[1,1].title.set_text('tas')
                
                axs.plot(np.arange(start_year,2100+1),data_temporal_yearly[:2101-start_year],label=model)
                axs.title.set_text('tas') 
                axs.legend(bbox_to_anchor=(1.05, 1))
            else:
                data_temp_scaled = data_temp_masked*\
                            land_area_ratio.reshape ((1,land_area_ratio.shape[0],land_area_ratio.shape[1]))
                data_temp_global_ave = np.nansum(data_temp_scaled,axis=(1,2))
                data_temporal_yearly = [np.nanmean(data_temp_global_ave[i*12:(i+1)*12]) \
                                        for i in np.arange(round(data_temp_scaled.shape[0]/12))]
                
                # axs[2,0].plot(np.arange(start_year,2100+1),data_temporal_yearly[:2101-start_year],label=model)
                # axs[2,0].title.set_text('mrsos')
                # axs[2,0].legend(bbox_to_anchor=(1.05, 1))
                
                axs.plot(np.arange(start_year,2100+1),data_temporal_yearly[:2101-start_year],label=model)
                # axs.title.set_text('pr') 
                axs.legend(bbox_to_anchor=(1.05, 1))
#% some empty figures to bring the legend labels equal
# ax1.plot(np.arange(1996,2100+1), np.zeros(2101-1996), color='w', alpha=0, label=' ')
# ax1.plot(np.arange(1996,2100+1), np.zeros(2101-1996), color='w', alpha=0, label=' ')
# ax1.plot(np.arange(1996,2100+1), np.zeros(2101-1996), color='w', alpha=0, label=' ')
# ax1.plot(np.arange(1996,2100+1), np.zeros(2101-1996), color='w', alpha=0, label=' ')

leg = ax1.legend(loc='upper left',prop={'size': 7.5},ncol=2,framealpha=0)

ax1.text(2090,310,'A',fontsize=16,fontweight="bold")

my_color = ['blue']*12+['r']*12
i=0    
# leg = ax.legend(bbox_to_anchor=(1.05, 1.05))
for text in leg.get_texts():
    text.set_color(my_color[i])
    i=i+1
ax1.set_ylim(-200,300)
ax1.set_ylabel('Cumulative carbon change (Pg)',fontsize=15)
ax1.set_xlabel('Year',fontsize=15)
ax1.grid(True) 
ax1.tick_params(axis='both', which='major', labelsize=15)
# ax1.set_yticks(np.arange(-200,400,100))
# ax1.set_yticklabels(np.arange(-200,400,100))
# ax1.set_title('SSP585: CMIP6 vs CMIP5')
# data_boxplot = pd.DataFrame({'CMIP5':delta_soc['d_soc'][delta_soc['cmip']=='cmip5'].values,
#                              'CMIP6':delta_soc['d_soc'][delta_soc['cmip']=='cmip6'].values})

import scipy



mu, sigma = scipy.stats.norm.fit(delta_soc['d_soc'].values)
ax2.hist(delta_soc['d_soc'].values,bins=7, density =True, alpha=1,edgecolor='k',facecolor='none',orientation='horizontal')
xmin, xmax = plt.xlim()
y = np.linspace(-200, 300, 100)
best_fit_line = scipy.stats.norm.pdf(y, mu, sigma)
ax2.plot(best_fit_line,y,'k-',linewidth=2,label='',alpha=1)
# ax2.axis('off')
ax2.xaxis.tick_top()
ax2.set_xlabel('Probability',fontsize=15)    
ax2.xaxis.set_label_position('top') 
ax2.set_yticklabels([])
ax2.set_ylim(-200,300)
ax2.grid(False)
ax2.text(0.007,310,'B',fontsize=16,fontweight="bold")
# ax2.set_yticklabels([])

                            
boxplot = delta_soc.boxplot(by='cmip',widths = .5,ax=ax3)

boxplot.get_figure().gca().set_title("")
boxplot.get_figure().suptitle('')
boxplot.get_figure().gca().set_xlabel("")

ax3.scatter([1]*12,delta_soc['d_soc'][delta_soc['cmip']=='cmip5'].values)
ax3.scatter([2]*12,delta_soc['d_soc'][delta_soc['cmip']=='cmip6'].values)
# delta_soc.boxplot(by='cmip',widths=0.5)
ax3.set_xticklabels(['CMIP5','CMIP6'])
ax3.set_yticklabels([])
ax3.set_ylim(-200,300)
ax3.tick_params(axis='both', which='major', labelsize=15)
ax3.set_title('C',loc='right',fontsize=16,fontweight="bold")
# ax3.axis('off')
# ax3.set_xticklabels([])
# ax3.set_yticklabels([])

fig.subplots_adjust(wspace=0)
# axs.boxplot(delta_soc['d_soc'][delta_soc['cmip']=='cmip5'],showmeans=True,widths = 15,
#               linewidth=0,positions=[2110])   
# axs.boxplot(delta_soc['d_soc'][delta_soc['cmip']=='cmip6'],showmeans=True,widths = 15,positions=[2120])
# # plt.legend(bbox_to_anchor=(1.05, 1))
# plt.ylabel('Cumulative carbon (Pg)')
# plt.xlabel('Year')
# plt.grid(True) 
# plt.title('SSP585 run: CMIP6 vs CMIP5')

# plt.tight_layout()                    
                # plt.plot(np.arange(start_year,2100+1),data_temporal_yearly[:2101-start_year],label=variable)
                
fig.savefig('./figures/FIG_temporal_delta_soc_new_model.png',bbox_inches='tight', dpi=300)


# mu, sigma = scipy.stats.norm.fit(delta_soc['d_soc'].values)
# plt.hist(delta_soc['d_soc'].values,bins=7, density =True, alpha=.5,color='k',orientation='horizontal')
# xmin, xmax = plt.xlim()
# y = np.linspace(-200, 300, 100)
# best_fit_line = scipy.stats.norm.pdf(y, mu, sigma)
# plt.plot(best_fit_line,y,'k--',linewidth=2,label='',alpha=.5)
# plt.axis('off')

mean, sigma = np.mean(delta_soc['d_soc']), np.std(delta_soc['d_soc'])

conf_int = scipy.stats.norm.interval(0.95, loc=mean, 
    scale=sigma/np.sqrt(len(delta_soc['d_soc'])))