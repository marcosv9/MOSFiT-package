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

def HDZ_to_XYZ_conversion(station,
                          dataframe,
                          starttime,
                          endtime,
                          files_path = None):
    '''
    Automatically indentify the existence H, D and Z components in the 
    geomagnetic data, and convert to X, Y and Z.
    
    --------------------------------------------------------    
    
    Inputs:
    
    station - 3 letters IAGA code for a INTERMAGNET observatory.
    
    dataframe - a pandas dataframe with geomagnetic data.
    
    starttime - first day of the data (format = 'yyyy-mm-dd)
    
    endtime - last day of the data (format = 'yyyy-mm-dd)
    
    ----------------------------------------------------------------
    
    Usage example:
    
    HDZ_to_XYZ_conversion(station = 'VSS',dataframe = 'name_of_datafrme', starttime = '2000-01-01', endtime = '2021-10-20')
    
    ------------------------------------------------------------------
    
    Return a dataframe with only X, Y and Z components
    
    '''
    
    assert len(station) == 3, 'station must be a three letters IAGA Code'
    
    assert isinstance(dataframe,pd.DataFrame), 'dataframe must be a pandas DataFrame'
    
    df_IMOS = pd.read_csv('Thesis_Marcos/Data/Imos informations/Imos_INTERMAGNET.txt',
                          skiprows = 1,
                          sep = '\s+',
                          usecols=[0,1,2,3],
                          names = ['Imos','Latitude','Longitude','Elevation'],
                          index_col= ['Imos'])
    
    assert station.upper() in df_IMOS.index, 'station must be an INTERMAGNET observatory IAGA code'
    
    for i in [starttime,endtime]:
        spf.validate(i)
        
    if files_path != None:
        if files_path[-1] == '/':
            pass
        else:
            files_path = files_path + '/'    
    
    df_station = dataframe
    
    year  = []
    Reported = []

    years_interval = np.arange(int(starttime[0:4]),int(endtime[0:4])+ 1)
    files_station = []

    
    if files_path == None:
        for year in years_interval:
            files_station.extend(glob.glob('Dados OBS\\' + str(year) + '/*/' + station + '*'))
            files_station.sort()
    else:
        files_station.extend(glob.glob(files_path + station + '*'))
        files_station.sort()
            
            
    df_reported = pd.DataFrame()
    df_reported = pd.concat( (pd.read_csv(file,sep = '\s+',
                                         header = None,                                         
                                         skiprows = 7,
                                         nrows = 25,
                                         usecols = [0,1],
                                         names = ['Date','Reported'])
                             for file in files_station))
    
    Reported = df_reported.loc[[0],['Reported']]

    date = df_reported.loc[[21],['Date']]
    #date.set_index('Date', inplace = True)

    Reported.set_index(date['Date'], inplace = True)
    
    Reported.sort_index()
    #print(df_data_type)
    
    #Reported_HDZ = Reported.loc[Reported['Reported'] == 'HDZF']
    
    Date_HDZ = pd.to_datetime(Reported.loc[Reported['Reported'] == 'HDZF'].index)
    
    for date in Date_HDZ.year.drop_duplicates():
    #print(date)
    #Data_HDZ = pd.concat([Data_HDZ,df_HER.loc[str(date)]])
        D = np.deg2rad(df_station['Y'].loc[str(date)]/60)
        X = df_station['X'].loc[str(date)]*np.cos(D)
        Y = df_station['X'].loc[str(date)]*np.sin(D)
        
        df_station['X'].loc[str(date)] = X
        df_station['Y'].loc[str(date)] = Y
    
                 #for Reported in df_data_type['Reported']: 
    #    
    #    Reported = df_data_type.loc[df_data_type['Reported'] == 'HDZF']
    #calculating declination
    #D = np.deg2rad(df_station['Y'].loc[(df_station.index >= Reported.index[0] + ' 00:00:00') &
    #                               (df_station.index <= Reported.index[-1] + ' 23:59:59')]/60)
    #
    ##converting H and D to X and Y.
    #
    #x = df_station['X'].loc[(df_station.index >= Reported.index[0] + ' 00:00:00') &
    #                    (df_station.index <= Reported.index[-1] + ' 23:59:59')]*np.cos(D)
    #
    #y = df_station['X'].loc[(df_station.index >= Reported.index[0] + ' 00:00:00') &
    #                    (df_station.index <= Reported.index[-1] + ' 23:59:59')]*np.sin(D)
    #
    #df_station['X'].loc[(df_station.index >= Reported.index[0] + ' 00:00:00') & 
    #                (df_station.index <= Reported.index[-1] + ' 23:59:59')] = x
    #df_station['Y'].loc[(df_station.index >= Reported.index[0] + ' 00:00:00') &
    #                (df_station.index <= Reported.index[-1] + ' 23:59:59')] = y 
#
    
    
    return df_station

def update_qd_and_dd(data):
    '''
    
    '''
    
    if data not in ['DD','QD']:
        
        print('Data must be QD or DD!')
        
    path = f'Thesis_Marcos/Data/Disturbed and Quiet Days/' 
    files = glob.glob(f'{path}qd*')
    files.sort()
    
    if data == 'DD':

        df = pd.concat((pd.read_csv(file,skiprows = 4,sep = '\s+',
                        header = None,
                        usecols = [0,1,12,13,14,15,16],
                        names = ['Month','Year','D1','D2','D3','D4','D5'])
                        for file in files),
                        ignore_index=True)
        
         
        columns = ['D1','D2',
                   'D3','D4',
                   'D5'
                  ]
        
        df['Month'] = pd.to_datetime(df.Month, format='%b').dt.month
        
        for col in columns:
            df['Test' +  col] = df['Year'].astype(str)+'-'+df['Month'].astype(str)+'-'+df[col].astype(str)
        for col in columns:
            df['Test' + col] = df['Test' + col].str.replace('*','')
        
        df_DD = pd.DataFrame()
        
        df_DD['DD'] = pd.concat([df['TestD1'] ,df['TestD2'],
                                 df['TestD3'], df['TestD4'],
                                df['TestD5']]
                               )
        
        df_DD['DD'] = pd.to_datetime(df_DD['DD'], infer_datetime_format=True)
        
        
        df_DD.set_index('DD', inplace = True)
        
        df_DD = df_DD.sort_index()
        
        df_DD.to_csv(path + 'Disturbed_Days_list.txt',index = True)
        
    if data == 'QD':
        df = pd.concat((pd.read_csv(file,
                                    skiprows = 4,
                                    sep = '\s+',
                                    header = None,
                                    usecols = [0, 1, 2,
                                               3, 4, 5,
                                               6, 7, 8,
                                               9, 10, 11
                                               ],
                                    names = ['Month', 'Year', 'Q1',
                                             'Q2', 'Q3', 'Q4', 'Q5',
                                             'Q6', 'Q7', 'Q8', 'Q9',
                                             'Q10'
                                             ]) for file in files),
                                                ignore_index = True
                       )
        
        columns = [f'Q{i}' for i in range(1, 11)]
        
        df['Month'] = pd.to_datetime(df.Month, format='%b').dt.month
        
        for col in columns:
            df['Test' +  col] = df['Year'].astype(str) + '-' + df['Month'].astype(str) + '-' + df[col].astype(str)
        for col in columns:
            df['Test' + col] = df['Test' + col].str.replace('A','')
        for col in columns:
            df['Test' + col] = df['Test' + col].str.replace('K','')
        
        df_QD = pd.DataFrame()
        df_QD['QD'] = pd.concat([df['TestQ1'],df['TestQ2'],df['TestQ3'],df['TestQ4'],df['TestQ5'],df['TestQ6'],df['TestQ7'],df['TestQ8'],df['TestQ9'],df['TestQ10']])
        
        df_QD['QD'] = pd.to_datetime(df_QD['QD'],infer_datetime_format=True)
        
        
        df_QD.set_index('QD', inplace = True)
        
        df_QD = df_QD.sort_index()
        
        df_QD.to_csv(path + 'Quiet_Days_list.txt',index = True)    