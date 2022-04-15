import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from pandas.tseries.frequencies import to_offset
import glob
import os
import ftplib
import pathlib
import matplotlib.gridspec as gridspec
from datetime import datetime
import pwlf
import chaosmagpy as cp
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from Thesis_Marcos import thesis_functions as mvs
from Thesis_Marcos import utilities_tools as utt
from Thesis_Marcos import support_functions as spf
from Thesis_Marcos import data_processing_tools as dpt

def polynomial_jerk_detection(station, window_start, 
                              window_end, 
                              starttime, 
                              endtime,
                              df_station = None,
                              plot_detection: bool = True,
                              CHAOS_correction: bool = True):
    
    '''
    '''
    
    for i in [starttime,endtime,window_start,window_end]:
        spf.validate(i)
        
    assert len(station) == 3, 'station must be a three letters IAGA Code'
    
    assert isinstance(df_station,pd.DataFrame) or df_station == None, 'df_station must be a pandas dataframe or None'
     
    station = station
    window_start = window_start
    window_end = window_end
    starttime = starttime
    endtime = endtime
    
        
    if df_station is not None:
        
        df_station = df_station
        
    else:
        df_station = mvs.load_INTERMAGNET_files(station = station, starttime = starttime, endtime = endtime)
        
    if CHAOS_correction == True:
        
        df_station, df_chaos = dpt.external_field_correction_chaos_model(station = station,
                                                  starttime = starttime,
                                                  endtime = endtime,
                                                  df_station = df_station,
                                                  df_chaos = None)
    else:
        pass
    
    #calculating SV
    df_SV = dpt.calculate_SV(dataframe = df_station, starttime = starttime, endtime = endtime, info = 'ADMM', columns = None)
    
    #starting with polynomial jerk detection
    
    t = np.arange(0,df_SV.loc[window_start:window_end].size/3).reshape(-1,1)
    jerk_prediction = pd.DataFrame()
    jerk_prediction.index = df_SV.loc[window_start:window_end].index
    
    for column in df_SV:

        polynomial_features= PolynomialFeatures(degree=3)
        x_poly = polynomial_features.fit_transform(t)
        
        model = LinearRegression()
        model.fit(x_poly, df_SV[column].loc[window_start:window_end])
        jerk_prediction[column] = model.predict(x_poly)
        
    if plot_detection == True:
        colors = ['blue','green','black']
        fig, axes = plt.subplots(3,1,figsize = (10,8))
        plt.suptitle(station.upper() +' secular variation', fontsize = 14, y = 0.92)
        plt.xlabel('Date (Years)', fontsize = 12)
        
        for col, ax, color in zip(df_SV.columns, axes.flatten(), colors):
            ax.plot(df_SV[col],'o-',color = color)
            ax.plot(df_SV[col].loc[window_start:window_end].index,
                    jerk_prediction[col],color = 'red', linewidth = 3, label = '3rd order polynomial')
            ax.set_ylabel('d' + col.upper() +'/dt (nT)', fontsize = 12)
            ax.legend()
        
        
        fig, axes = plt.subplots(1,3,figsize = (16,4))
        plt.suptitle(station.upper() +' secular variation', fontsize = 14, y = 0.93)
        fig.text(0.5, 0.04, 'Date (Years)', ha='center')
     
        upper_limit = int(str(datetime.strptime(window_end ,'%Y-%m-%d'))[0:4]) +1
        lower_limit = int(str(datetime.strptime(window_start ,'%Y-%m-%d'))[0:4]) -1

        for col, ax, color in zip(df_SV.columns, axes.flatten(), colors):
            ax.plot(df_SV[col].loc[str(lower_limit):str(upper_limit)],'o-',color = color)
            ax.plot(jerk_prediction[col],color = 'red', linewidth = 3)
            ax.set_ylabel('d' + col.upper() +'/dt (nT)', fontsize = 12)
            
        
        
    else:
        pass
        
    return df_SV, jerk_prediction 

def jerk_detection(station, dataframe,linear_segments, starttime, endtime):
    
    '''
    adopt piecewise linear segments on the secular variation for a informed number o linear segments.
    
    Built using PWLF python library
    
    @Manual{pwlf,
           author = {Jekel, Charles F. and Venter, Gerhard},
           title = {{pwlf:} A Python Library for Fitting 1D Continuous Piecewise Linear Functions},
           year = {2019},
           url = {https://github.com/cjekel/piecewise_linear_fit_py}}
    
    The breakpoints of the linear segments are adopted based on global optmization, 
    which minimize the sum of the square errors.  
    
    --------------------------------------------------------------------------------
    Inputs:
    
    station - 3 letters IAGA code for a INTERMAGNET observatory.
    
    starttime - first day of the data (format = 'yyyy-mm-dd)
    
    endtime - last day of the data (format = 'yyyy-mm-dd)
    
    linear_segments - number of linear segments per component, must be a list.
    
    dataframe - pandas dataframe, must be a SV data.
    
    --------------------------------------------------------------------------------
    
    usage example:
    jerk_detection(station = 'VSS',
                   dataframe = df_VSS,
                   linear_segments = [3,4,3],
                   starttime = '2005-01-01',
                   endtime = '2021-09-30')
    
    '''
    
    assert len(station) == 3, 'station must be a three letters IAGA Code'
    
    assert isinstance(dataframe,pd.DataFrame), 'dataframe must be a pandas dataframe'
    
    assert isinstance(linear_segments,(list,int)), 'x must be a list of integers'
    
    assert len(linear_segments) == 3, 'x must be a list with 3 integers'
    
    df_IMOS = pd.read_csv('Thesis_Marcos/Data/Imos informations/Imos_INTERMAGNET.txt',
                          skiprows = 1,
                          sep = '\s+',
                          usecols=[0,1,2,3],
                          names = ['Imos','Latitude','Longitude','Elevation'],
                          index_col= ['Imos'])
    
    assert station in df_IMOS.index, 'station must be an INTERMAGNET observatory IAGA code'
    
    for i in [starttime,endtime]:
        spf.validate(i)
    
    
    #linear_segments = []
    df_station = dataframe
    df_SV = pd.DataFrame()
    #fit your data (x and y)
    components = ['X','Y','Z']
    
    
    df_SV = calculate_SV(df_station, starttime = starttime, endtime = endtime)
    
    X = np.arange(0,len(df_SV.index))
    
    for (component,i) in zip(components,linear_segments):
        
        myPWLF = pwlf.PiecewiseLinFit(X,df_SV[component])
        
        #fit the data for n line segments
        z = myPWLF.fit(i)
        #calculate slopes
        #slopes = myPWLF.calc_slopes()
        
        # predict for the determined points
        xHat = X 
        yHat = myPWLF.predict(xHat)
        
        #calculate statistics
        #p = myPWLF.p_values(method='non-linear', step_size=1e-4) #p-values
        #se = myPWLF.se  # standard errors
        
        df_SV[str(component) + 'p'] = yHat
        
        
    directory = 'Filtered_data/'+ station +'_data'
    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)      
        
    fig, ax = plt.subplots(3,1,figsize = (14,10))


    ax[0].plot(df_SV['X'],'o', color = 'blue')
    ax[0].plot(df_SV['Xp'],'-', color = 'red')
    #ax01].plot(y_poly_pred,'-')
    ax[0].set_xlim(df_SV['X'].index[0],df_SV['X'].index[-1])
    ax[0].set_ylabel('dX/dt (nT)', fontsize = 14)
    ax[0].set_ylim(df_SV['X'].min() - 3,df_SV['X'].max() + 3)
    ax[0].set_title('Automatic detection - ' + station.upper(), fontsize = 16)
    ax[0].grid()
    
    
    ax[1].plot(df_SV['Y'],'o',color = 'green')
    ax[1].plot(df_SV['Yp'],'-', color = 'red')
    #ax11].plot(y_poly_pred,'-')
    ax[1].set_xlim(df_SV['Y'].index[0],df_SV['Y'].index[-1])
    ax[1].set_ylabel('dY/dt (nT)', fontsize = 14)
    ax[1].set_ylim(df_SV['Y'].min() - 3,df_SV['Y'].max() + 3)
    ax[1].grid()
    
    ax[2].plot(df_SV['Z'],'o',color = 'black')
    ax[2].plot(df_SV['Zp'],'-', color = 'red')
    #ax21].plot(y_poly_pred,'-')
    ax[2].set_xlim(df_SV['Z'].index[0],df_SV['Z'].index[-1])
    ax[2].set_ylabel('dZ/dt (nT)', fontsize = 14)
    ax[2].set_xlabel('Years', fontsize = 14 )
    ax[2].set_ylim(df_SV['Z'].min() - 3,df_SV['Z'].max() + 3)
    ax[2].grid()
    
    plt.savefig(directory + '/' + station + '_SV_LFit.jpeg', bbox_inches='tight')
    plt.show()