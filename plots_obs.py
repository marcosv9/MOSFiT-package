import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import jit, njit
import numba

#funções para plotar dados de observatórios, ainda em desenvovilmento
#adicionar funções para conversão H,D e Z para X, Y e Z, conversão dados sec para min.


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
    ax.set_xlim(0,new_series.size)
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize = 12)
    plt.grid()
    plt.show()

    return new_series
    
    
    
    
    
    
 