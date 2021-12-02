import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import jit, njit
import numba
import glob
from glob import glob
import os
import ftplib
import pathlib
import matplotlib.gridspec as gridspec
from datetime import datetime
import pwlf
from matplotlib.widgets import Slider, Button

#funções para plotar dados de observatórios, ainda em desenvovilmento
#adicionar funções para conversão H,D e Z para X, Y e Z, conversão dados sec para min.


def resample_obs_data(dataframe, sample):
    df = pd.DataFrame()
    df = dataframe
    samples = ['H','D','M','Y']
    
    if sample not in samples:
        print('sample must be H, D, M or Y!')
    else:
        if sample == 'H':
            df = (df.resample('H').mean()).shift(30, freq = 'Min')
        if sample == 'D':
            df = df.resample('D').mean().shift(12, freq = 'H')
        if sample == 'M':
            df = df.resample('M').mean().shift(-15, freq = 'D')
        if sample == 'Y':
            df = df.resample('Y').mean().shift(-182.5, freq = 'D')
            
        return df


def load_qd_data(station, path):
    '''
    Function to read and concat observatory data
    
    Sample must be H, D, M, Y   
    
    '''
    #print('Reading files from '+ station.upper() +'...')
    #year  = []
    #for i in range(int(starttime[0:4]),int(endtime[0:4])+ 1):
    #    Y = i
    #    year.append(Y)
    #
    #Years = []
    #Years.extend([str(i) for i in year])
    #Years
    #
    #files_station = []
    #
    L_27 = ['CZT','DRV','PAF']
    L_26 = ['NGK','DRV','MAW','CNB','HAD','TSU','HON','KAK','BOU','KOU','HBK','BMT']
    #
    skiprows = 17
    if station.upper() in L_27:
        skiprows = 27
    if station.upper() in L_26:
        skiprows = 26
    #
    #for Year in Years:

    files_station = glob(path + '/*')
    files_station.sort()

    #d_parser = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S%.f')
    
    df_station = pd.concat( (pd.read_csv(file, sep='\s+',usecols = [0,1,3,4,5,6], 
                   header = None,skiprows = skiprows, 
                   parse_dates = {'Date': ['date', 'Time']},
                   names = ['date','Time','X','Y','Z','F']) for file in files_station), 
                   ignore_index = True)
    df_station['Date'] = pd.to_datetime(df_station['Date'], format = '%Y-%m-%dd %H:%M:%S.%f')     
    #df_station['Hour'] = pd.to_datetime(df_station['Hour'], format = '%H:%M:%S.%f').dt.time               
    df_station.set_index('Date', inplace = True)

    
    df_station.loc[df_station['X'] == 99999.0, 'X'] = np.nan
    df_station.loc[df_station['Y'] == 99999.0, 'Y'] = np.nan
    df_station.loc[df_station['Z'] == 99999.0, 'Z'] = np.nan 
    df_station.loc[df_station['F'] == 88888.0, 'F'] = np.nan 
    
    return df_station

def obs_data_processing(station,path, starttime, endtime, sample = None, plots: bool = False):
    
    df = load_qd_data(station = station, path = path)
    df = df.loc[starttime:endtime]
    if sample == None:
        df = df
    else:
        df = resample_obs_data(df, sample = sample)
    if plots == False:
        pass
    if plots == True:
        #components = ['X','Y','Z','F']
        colors = ['blue','green','black','red']
        fig, axes = plt.subplots(4,1, figsize = (14,10))
        for col, ax, color in zip(df.columns, axes.flatten(), colors):
            ax.plot(df[col], color  = color)
            ax.set_ylabel(col.upper() + '/nT', fontsize = 12)
            
                 
     
    
    return df




def p_diff_obs(obs, pillar,  starttime = None, endtime = None):
    
    if obs == 'ttb':
        path = 'planilhas/Pillar diff ttb0 since 2019.xlsx'
    if obs == 'vss':
        path = 'planilhas/Pillar diff ttb0 since 2019.xlsx'
        
    df_obs =  pd.read_excel(path, usecols = [0,2,3,4,5,6], names = ['Date','Pillar 1','Pillar 2','Pillar Diff', 'JD','DOY'])
    df_obs['Date'] = pd.to_datetime(df_obs['Date'], format = '%Y%m%d')
    df_obs.set_index('Date', inplace = True)
    
    if starttime == None and endtime == None:
    
        fig, ax = plt.subplots(figsize=(18,5)) 
    #gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1]) 
    #Days = ['01/01','03/01','05/01','07/01','09/01','11/01']
    #ax0 = plt.subplot(gs[0])
        ax.plot(df_obs['Pillar ' + str(pillar)], 'or',markersize = 5, label = 'Pillar ' + str(pillar))
#ax.plot(df1['jd'],pl(df1['jd']),'k--',label = 'linear trend = 0.002429x - 50.53', markersize = 4)
#ax[0].plot(df2['doy'], df2['over'] , 'o--',label = 'GSM Over',markersize = 5)
#ax[0].plot(DOY,pl(DOY),'r-',label = 'trend')
#ax[0].set_xticklabels(Days)
#ax.set_ylim(-35,-28)
#ax[0].set_xticks(np.arange(0,365,61))
#ax[0].axhline(y=np.mean(Diff), c='k', linewidth=2, alpha=0.5, linestyle='--',label = 'mean =-32,21 nT')
#ax.set_xlim(7252,7669)
        ax.set_ylabel('Differences(nT)', fontsize = 14)
        ax.set_xlabel('Time', fontsize = 14)
        ax.set_ylim(1.05*df_obs['Pillar ' + str(pillar)].min(),0.95*df_obs['Pillar ' + str(pillar)].max())
        ax.set_title('Pillar differences ' + obs.upper(), fontsize = 16)
        ax.grid()
        ax.legend(loc='best', fontsize = 12)
        plt.show()
        
        print('The maximum difference is:', round(df_obs['Pillar ' + str(pillar)].max(),3))
        print('The minimum difference is:', round(df_obs['Pillar ' + str(pillar)].min(),3))
        print('The median:', round(df_obs['Pillar ' + str(pillar)].median(),3))
        print('The mean is:', round(df_obs['Pillar ' + str(pillar)].mean(),3))
        print('The STD is:', round(df_obs['Pillar ' + str(pillar)].std(),3))
        
    else:
        fig, ax = plt.subplots(figsize=(18,5)) 
    #gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1]) 
    #Days = ['01/01','03/01','05/01','07/01','09/01','11/01']
    #ax0 = plt.subplot(gs[0])
        ax.plot(df_obs['Pillar ' + str(pillar)][starttime:endtime], 'or',markersize = 5, label = 'Pillar ' + str(pillar))
#ax.plot(df1['jd'],pl(df1['jd']),'k--',label = 'linear trend = 0.002429x - 50.53', markersize = 4)
#ax[0].plot(df2['doy'], df2['over'] , 'o--',label = 'GSM Over',markersize = 5)
#ax[0].plot(DOY,pl(DOY),'r-',label = 'trend')
#ax[0].set_xticklabels(Days)
#ax.set_ylim(-35,-28)
#ax[0].set_xticks(np.arange(0,365,61))
#ax[0].axhline(y=np.mean(Diff), c='k', linewidth=2, alpha=0.5, linestyle='--',label = 'mean =-32,21 nT')
#ax.set_xlim(7252,7669)
        ax.set_ylabel('Differences(nT)', fontsize = 14)
        ax.set_ylim(1.05*df_obs['Pillar ' + str(pillar)][starttime:endtime].min(),0.95*df_obs['Pillar ' + str(pillar)][starttime:endtime].max())
        ax.set_xlabel('Time', fontsize = 14)
        ax.set_title('Pillar differences ' + obs.upper(), fontsize = 16)
        ax.grid()
        ax.legend(loc='best', fontsize = 12)
        plt.show()
        
        print('The maximum difference is:', round(df_obs['Pillar ' + str(pillar)][starttime:endtime].max(),3))
        print('The minimum difference is:', round(df_obs['Pillar ' + str(pillar)][starttime:endtime].min(),3))
        print('The median:', round(df_obs['Pillar ' + str(pillar)][starttime:endtime].median(),3))
        print('The mean is:', round(df_obs['Pillar ' + str(pillar)][starttime:endtime].mean(),3))
        print('The STD is:', round(df_obs['Pillar ' + str(pillar)][starttime:endtime].std(),3))
        

def plot_components_sep(x,y,z,label1,label2,label3,title, ndays, days):
    fig, ax = plt.subplots(3,1,figsize=(16, 10)) 
    #gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1]) 
    Days = days
    #ax0 = plt.subplot(gs[0])
    ax[0].plot(x, 'b',label = label1)
    #ax[0].plot(dXSPT, 'r',label = 'dXSPT')
    ax[0].set_xticklabels(Days)
    #ax[0].set_ylim(np.min(x)*1.1,np.max(x)*1.1)
    ax[0].set_xticks(np.arange(0,x.size,x.size/ndays))
    ax[0].set_xlim(0,x.size)
    ax[0].set_title(title, fontsize = 18)
    ax[0].set_ylabel('X/nT',fontsize = 14)
    ax[0].grid()
    ax[0].legend(loc='best', fontsize = 12)

    #ax1 = plt.subplot(gs[0])
    ax[1].plot(y, 'g', label = label2)
    #ax[1].plot(dYSPT, 'r', label = 'dYSPT')
    ax[1].set_xticklabels(Days)
    #ax[1].set_ylim((np.min(y)*1.1,np.max(y)*1.1))
    ax[1].set_xticks(np.arange(0,y.size,y.size/ndays))
    ax[1].set_xlim(0,y.size)
    ax[1].set_ylabel('Y/nT',fontsize = 14)
    ax[1].legend(loc='best', fontsize = 12)
    ax[1].grid()

    #ax1 = plt.subplot(gs[0])
    ax[2].plot(z, 'r', label = label3)
    #ax[1].plot(dYSPT, 'r', label = 'dYSPT')
    ax[2].set_xticklabels(Days)
    #ax[2].set_ylim(np.min(z)*1.1,np.max(z)*1.1)
    ax[2].set_xticks(np.arange(0,z.size,z.size/ndays))
    ax[2].set_xlim(0,z.size)
    ax[2].legend(loc='best', fontsize = 12)
    ax[2].set_ylabel('Z/nT',fontsize = 14)
    ax[2].set_xlabel('Tempo(Dias)',fontsize = 14)
    ax[2].grid()

    plt.show()


def plot_components_sep_compa(x,x2,y,y2,z,z2,label1,label12,label2,label22,label3,label32,title,ndays,days):
    fig, ax = plt.subplots(3,1,figsize=(16, 10)) 
    #gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1]) 
    Days = days
    #ax0 = plt.subplot(gs[0])
    ax[0].plot(x - x[1], 'b',label = label1)
    ax[0].plot(x2 - x2[1], 'k',label = label12)
    #ax[0].plot(dXSPT, 'r',label = 'dXSPT')
    ax[0].set_xticklabels(Days)
    #ax[0].set_ylim(np.min(x)*1.1,np.max(x)*1.1)
    ax[0].set_xticks(np.arange(0,x.size,x.size/ndays))
    ax[0].set_xlim(0,x.size)
    ax[0].set_title(title, fontsize = 18)
    ax[0].set_ylabel('X/nT',fontsize = 14)
    ax[0].grid()
    ax[0].legend(loc='best', fontsize = 12)

    #ax1 = plt.subplot(gs[0])
    ax[1].plot(y - y[1], 'g', label = label2)
    ax[1].plot(y2 -y2[1], 'k', label = label22)
    #ax[1].plot(dYSPT, 'r', label = 'dYSPT')
    ax[1].set_xticklabels(Days)
    #ax[1].set_ylim((np.min(y)*1.1,np.max(y)*1.1))
    ax[1].set_xticks(np.arange(0,y.size,y.size/ndays))
    ax[1].set_xlim(0,y.size)
    ax[1].set_ylabel('Y/nT',fontsize = 14)
    ax[1].legend(loc='best', fontsize = 12)
    ax[1].grid()

    #ax1 = plt.subplot(gs[0])
    ax[2].plot(z - z[1], 'r', label = label3)
    ax[2].plot(z2 - z2[1], 'k', label = label32)
    #ax[1].plot(dYSPT, 'r', label = 'dYSPT')
    ax[2].set_xticklabels(Days)
    #ax[2].set_ylim(np.min(z)*1.1,np.max(z)*1.1)
    ax[2].set_xticks(np.arange(0,z.size,z.size/ndays))
    ax[2].set_xlim(0,z.size)
    ax[2].legend(loc='best', fontsize = 12)
    ax[2].set_ylabel('Z/nT',fontsize = 14)
    ax[2].set_xlabel('Tempo(Dias)',fontsize = 14)
    ax[2].grid()

   
    #plt.savefig('Plotagem componentes H, D e Z de COI - 21-05 a 23-05-2021_out.jpeg', bbox_inches='tight')

    plt.show()
    
    
def plot_components_tog(x,title,ndays,label1, days,x2 = None,x3 = None,label2 = None,label3 = None):
    '''
    add comments about de function
    '''
    
    fig, ax = plt.subplots(figsize=(18,5)) 
#gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1]) 
    Days = days
    
    if x2 is not None and label2 is not None:
    
        ax.plot(x2 - x2[1], 'r',label = label2)
        
    if x3 is not None and label3 is not None:
        ax.plot(x3 - x3[1], 'b',label = label3)
        
    ax.plot(x - x[1], 'k',label = label1)
#ax[0].plot(DOY,pl(DOY),'r-',label = 'trend')
    ax.set_xticklabels(Days)
    #ax.set_ylim((np.min(z)*1.1),(np.max(z)*1.1))
    ax.set_xticks(np.arange(0,x.size,x.size/ndays))
#ax[0].axhline(y=np.mean(Diff), c='k', linewidth=2, alpha=0.5, linestyle='--',label = 'mean =-32,21 nT')
    ax.set_xlim(0,x.size)
    ax.set_ylabel('Variação/nT', fontsize = 14)
    ax.set_xlabel('Tempo(dias)', fontsize = 14)
    ax.legend(loc='best', bbox_to_anchor=(1.0, 0.5), fontsize = 12)
    ax.set_title(title, fontsize = 16)
    ax.grid()
    plt.show()


def plot_diff_tog(x,y,z,label1,label2,label3,title,ndays, days, x2 = None,y2 = None,z2 = None, label4 = None):
    fig, ax = plt.subplots(figsize=(18,5)) 
#gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1]) 
    Days = days
    
    if x2 is not None and label4 is not None:
    
        ax.plot(x2.diff(), 'k',label = label4)
        
    if y2 is not None and z2 is not None:
        
        ax.plot(y2.diff() + 2, 'k')
        
        ax.plot(z2.diff() - 2, 'k')
        
    #ax0 = plt.subplot(gs[0])
    ax.plot(x.diff() , 'b',label = label1)
    ax.plot(y.diff() + 2, 'g',label = label2)
    ax.plot(z.diff() -2, 'r', label = label3)
#ax[0].plot(DOY,pl(DOY),'r-',label = 'trend')
    ax.set_xticklabels(Days)
    ax.set_ylim(-3.5,3.5)
    ax.set_xticks(np.arange(0,x.size,x.size/ndays))
#ax[0].axhline(y=np.mean(Diff), c='k', linewidth=2, alpha=0.5, linestyle='--',label = 'mean =-32,21 nT')
    ax.set_xlim(0,x.size)
    ax.set_ylabel('Variação/nT', fontsize = 14)
    ax.set_xlabel('Tempo(dias)', fontsize = 14)
    ax.set_title(title, fontsize = 16)
    ax.grid()
    ax.legend(loc='best', bbox_to_anchor=(1.0, 0.5), fontsize = 14)
    plt.show()
    
    
def plot_temp(t1, label1, title,ndays, days, t2 = None,label2 = None):    
    
    fig, ax = plt.subplots(figsize=(18,5)) 
    Days = days
    
    
    if t2 is not None and label2 is not None:
        ax.plot(t2, 'g', label = label2)
        #if label2 is not None:
            #ax.plot(t2, 'g', label = label2)
    #gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1]) 
    #ax0 = plt.subplot(gs[0])
    ax.plot(t1, 'b',label = label1)
    #ax[0].plot(DOY,pl(DOY),'r-',label = 'trend')
    ax.set_xticklabels(Days)
    #ax.set_ylim(-93,-91)
    ax.set_xticks(np.arange(0,t1.size,t1.size/ndays))
    #ax[0].axhline(y=np.mean(Diff), c='k', linewidth=2, alpha=0.5, linestyle='--',label = 'mean     =-32,21 nT')
    ax.set_xlim(0,t1.size)
    ax.set_ylabel('Temp/°C', fontsize = 14)
    ax.set_xlabel('Tempo(dias)', fontsize = 14)
    ax.set_title(title, fontsize = 16)
    ax.grid()
    ax.legend(loc='best', bbox_to_anchor=(1.0, 0.5), fontsize = 14)

    plt.show()
    
def data_filter_basic(component, vmin, vmax):
    
    Filtered_comp = np.asarray(component)
    #comp_new = np.asarray(component)
    
    
    Filtered_comp = [number for number in component if number < vmin or number > vmax]
    #print(Filtered_comp)
    component = component.replace(Filtered_comp,np.nan)
    
    return component

#def sec2min_converter_notinuse(comp1, comp2 = None, comp3 = None):
#    ndata = comp1.size # Número de Elementos 
#    N = np.arange(0, comp1.size, 60)
#    filtered_vector1 = []
#    if comp2 is not None:
#        filtered_vector2 = []
#        N2 = np.arange(0, comp2.size, 60)
#        for i in N2:
#            filtered_vector2 = np.append(filtered_vector2, np.nanmean(comp2[i:i+60]))
#    else:
#        for i in N:               
#            filtered_vector1 = np.append(filtered_vector1, np.nanmean(comp1[i:i+60]))
#    if comp2 is not None and comp3 is not None:
#        filtered_vector2 = []
#        filtered_vector3 = []
#        N2 = np.arange(0, comp2.size, 60)
#        N3 = np.arange(0, comp3.size, 60)
#        for i in N:
#            filtered_vector1 = np.append(filtered_vector1, np.nanmean(comp1[i:i+60]))
#        for i in N2:
#            filtered_vector2 = np.append(filtered_vector2, np.nanmean(comp2[i:i+60]))
#        for i in N3:
#            filtered_vector3 = np.append(filtered_vector3, np.nanmean(comp3[i:i+60]))
#
#    else:
#        for i in N:
#            filtered_vector1 = np.append(filtered_vector1, np.nanmean(comp1[i:i+60]))
#                      
#    
#    return filtered_vector1, filtered_vector2, filtered_vector3
    
def sec2min_converter(data, winsize):
    ndata = data.size # Número de Elementos 
    N = np.arange(0, data.size, winsize)
    filtered_data = []
    for i in N:
        filtered_data = np.append(filtered_data, np.nanmean(data[i:i+winsize]))
    return filtered_data

@jit
def hampel_filter_forloop_numba(input_series, window_size, n_sigmas=3):

    n = len(input_series)
    new_series = input_series.copy()
    k = 1.4826 # scale factor for Gaussian distribution
    
    indices = []
    
    # possibly use np.nanmedian 
    for i in range((window_size),(n - window_size)):
        x0 = np.median(input_series[(i - window_size):(i + window_size)])
        S0 = k * np.median(np.abs(input_series[(i - window_size):(i + window_size)] - x0))
        if (np.abs(input_series[i] - x0) > n_sigmas * S0):
            new_series[i] = x0
    
    fig, ax = plt.subplots(figsize = (16,5))
    ax.plot(input_series, 'k', label = 'Removed Outliers')
    ax.plot(new_series, 'r', label = 'New Series')
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize = 12)
    plt.grid()
    plt.show()

    return new_series
    
    
    
    
    
    
 