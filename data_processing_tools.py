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


def remove_Disturbed_Days(dataframe, starttime, endtime):
    ''' 
    Function created to remove geomagnetic disturbed 
    days from observatory geomagnetic data.

    Ther list of geomagnetic disturbed days used is available
    on https://www.gfz-potsdam.de/en/kp-index/. It is update every month.
    
    ------------------------------------------------------
    
    Distubed days available until December-2021.
    
    --------------------------------------------------------
    
    Inputs:
    
    dataframe - a pandas dataframe with geomagnetic data.
    
    starttime - first day of the data (format = 'yyyy-mm-dd)
    
    endtime - last day of the data (format = 'yyyy-mm-dd)
    
    ---------------------------------------------------------------------
    Example of use:
    
    remove_Disturbed_Days(df_NGK, starttime = '2010-01-01' , endtime = '2019-12-31')
    
    ----------------------------------------------------------------------
    
    Return a dataframe without the geomagnetic disturbed days.
    
    '''
    
    assert isinstance(dataframe,pd.DataFrame), 'dataframe must be a pandas dataframe'
    
    for i in [starttime,endtime]:
        mvs.validate(i)
    
    #df = pd.DataFrame()
    #df = dataframe.resample('D').mean()
    
    df = dataframe.loc[starttime:endtime]
    
    disturbed_index = pd.DataFrame()
    
    df_d = pd.read_csv('Dados OBS/Data/Disturbed and Quiet Days/Disturbed_Days_list.txt',
                       skiprows = 1, 
                     usecols = [0],
                     names = ['dd'],
                     parse_dates = {'D-Days': ['dd']},
                    )
    
    df_d['D-Days'] = pd.to_datetime(df_d['D-Days'], format = '%YYYY-%mm-%dd')

    df_d.set_index('D-Days', inplace = True)
    
    df_d = df_d.sort_values('D-Days')
    
    
    first_day = str(df.index[0].year) + '-' + str(df.index[0].month) + '-' + str(df.index[0].day)
    last_day = str(df.index[-1].year) + '-' + str(df.index[-1].month) + '-' + str(df.index[-1].day)
    
    df_d = df_d.loc[first_day:last_day]
    
    for date in df_d.index.date:
        disturbed_index = disturbed_index.append(df.loc[str(date)])
        
    
    df = df.drop(disturbed_index.index)
    
    
    return df

def keep_Q_Days(dataframe, starttime, endtime):
    ''' 
    Function created to keep only geomagnetic quiet 
    days from observatory geomagnetic data.

    Ther list of geomagnetic quiet days used is available
    on https://www.gfz-potsdam.de/en/kp-index/. It is update every month.
    
    --------------------------------------------------------------------------
    
    Quiet days are available until December-2021.
    
    --------------------------------------------------------------------------
    
    Inputs:
    
    dataframe - a pandas dataframe with geomagnetic data.
    
    starttime - first day of the data (format = 'yyyy-mm-dd)
    
    endtime - last day of the data (format = 'yyyy-mm-dd)
    
    ---------------------------------------------------------------------------
    Example of use:
    
    keep_Q_Days(dataframe = my_dataframe, starttime = '2010-01-01' , endtime = '2019-12-31')
    
    ---------------------------------------------------------------------------
    Return a dataframe containing only geomagnetic quiet days.
    
    '''
    
    assert isinstance(dataframe,pd.DataFrame), 'dataframe must be a pandas dataframe'
    
    for i in [starttime,endtime]:
        mvs.validate(i)
    
    df = dataframe.loc[starttime:endtime]
    
    quiet_index = pd.DataFrame()

    df_q = pd.read_csv('Dados OBS/Data/Disturbed and Quiet Days/Disturbed_Days_list.txt',
                       header = None,
                       skiprows = 1, 
                     usecols = [0],
                     names = ['qd'],
                     parse_dates = {'Q-Days': ['qd']},
                    )
    
    df_q['Q-Days'] = pd.to_datetime(df_q['Q-Days'], format = '%YYYY-%mm-%dd')
    
    df_q = df_q.sort_values('Q-Days')
    
    df_q.set_index('Q-Days', inplace = True)
    
    df_q = df_q.loc[starttime:endtime]

    for date in df_q.index.date:
        quiet_index = quiet_index.append(df.loc[str(date)])
    df = df.reindex(quiet_index.index).dropna()

    return df

def calculate_SV(dataframe, starttime, endtime, method = 'ADMM', columns = None):
    '''
    Calculate the secular variation of geomagnetic observatory data.
    
    Two different methods are available, Annual differences of monthly means (ADMM) or Yearly differences (YD). 
    
    -------------------------------------------------------------------------------------
    
    inputs:

    dataframe - a pandas dataframe with geomagnetic data.
    
    starttime - first day of the data (format = 'yyyy-mm-dd)
    
    endtime - last day of the data (format = 'yyyy-mm-dd)
    
    method- 'ADMM' or 'YD'
    
    columns - name of the geomagnetic components columns of your dataframe.
              None if the columns are X, Y and Z or must be passed as a list.
    
    --------------------------------------------------------------------------------------------
    Example of use:
    
    calculate_SV(dataframe = name_of_your_dataframe, starttime = 'yyyy-mm-dd', endtime = 'yyyy-mm-dd',
                 method = 'ADMM', columns = ['X','Y','Z']) 
                
    
    --------------------------------------------------------------------------------------------
    Return a dataframe with the secular variation
    
    '''
    
    assert isinstance(dataframe,pd.DataFrame), 'dataframe must be a pandas dataframe'

    df = dataframe
    df = df.loc[starttime:endtime]
    
    Method = ['ADMM','YD']
    df_SV = pd.DataFrame()
    
    #if info is not in Info:
    #    print('info must be ADMM or YD')
#   
    df_ADMM = resample_obs_data(dataframe = df, sample = 'M')
    df_YD = resample_obs_data(dataframe = df, sample = 'Y')

    if columns == None:
        columns = ['X','Y','Z']
    else:
        columns = columns
        
    if method not in Method:
        print('Info musdf_SV = pd.DataFrame()t be ADMM or YD')
    
    if method == 'ADMM':
        for col in columns:
            SV = (df_ADMM[col].diff(6) - df_ADMM[col].diff(-6)).round(3).dropna()
            df_SV[col] = SV  
    if method == 'YD':
        for col in columns:
            SV = df[col].diff().round(3).dropna()
            df_SV[col] = SV 
            
    return df_SV

def Kp_index_correction(dataframe, starttime, endtime, kp):
    '''
    Function o filter geomagnetic data based on Kp index
    
    ---------------------------------------------------------
    inputs:
    
    dataframe - a pandas dataframe with geomagnetic data.
    
    starttime - first day of the data (format = 'yyyy-mm-dd)
    
    endtime - last day of the data (format = 'yyyy-mm-dd)
    
    kp = limit kp index value (float or int), from 0 to 9.
    --------------------------------------------------------
    Example of use:
    
    Kp_index_correction(dataframe = name_of_dataframe, starttime = 'yyyy-mm-dd', endtime = 'yyyy-mm-dd',
                        kp = 3.3)    
    
    ----------------------------------------------------------
    return a dataframe filtered by the selected Kp-index
    
    '''
    
    assert isinstance(dataframe,pd.DataFrame), 'dataframe must be a pandas DataFrame'
    
    assert isinstance(kp, int) or isinstance(kp, float), 'kp must be a number from 0 to 9'
    
    for i in [starttime,endtime]:
        mvs.validate(i)
            
    if (kp <=0 ) or (kp>= 9): 
        print('kp must be a number from 0 to 9, try again!')
    df = pd.DataFrame()
    df = dataframe
    
    KP_ = pd.read_csv('Dados OBS/Data/Kp index/Kp_ap_since_1932.txt', skiprows = 30,
                  header = None,
                  sep = '\s+', 
                  usecols = [7,8],
                  names = ['Kp','Ap'],
                 )
    Date = pd.date_range('1932-01-01 00:00:00','2022-01-29 21:00:00', freq = '3H')    
    KP_.index = Date
    
    x=pd.DataFrame()
    x['Date'] = KP_[starttime:endtime].loc[KP_['Kp'] > kp].index.date
    x['Date'] = x['Date'].drop_duplicates()
    x.index = x['Date']
    x =x.dropna()
    
    dataframe = dataframe.resample('D').mean().drop(x.index)
    
    return dataframe

def chaos_model_prediction(station, starttime, endtime):
    '''
    Compute the CHAOS-7.9 model geomagnetic field prediction for a INTERMAGNET observatory.
    
    Hourly values are computed based on starttime and endtime
    
    Find the model in the website - http://www.spacecenter.dk/files/magnetic-models/CHAOS-7/
    
    References
    Finlay, C.C., Kloss, C., Olsen, N., Hammer, M. Toeffner-Clausen, L., Grayver, A and Kuvshinov, A. (2020),
    The CHAOS-7 geomagnetic field model and observed changes in the South Atlantic Anomaly,
    Earth Planets and Space 72, doi:10.1186/s40623-020-01252-9
    --------------------------------------------------------------------------------
    Inputs:
    
    station - 3 letters IAGA code for a INTERMAGNET observatory.
    
    starttime - first day of the data (format = 'yyyy-mm-dd)
    
    endtime - last day of the data (format = 'yyyy-mm-dd)
    
    ----------------------------------------------------------------------------------
    
    Example of use:
    chaos_model_prediction(station = 'VSS', starttime = 'yyyy-mm-dd', endtime = 'yyyy-mm-dd')
    
    ----------------------------------------------------------------------------------
    
    return a dataframe with the X, Y and Z geomagnetic components CHAOS prediction for total field,
    internal field and external field.
    
    '''
    
    assert len(station) == 3, 'station must be a three letters IAGA Code'
    
    for i in [starttime,endtime]:
        mvs.validate(i)
        
        
    model = cp.load_CHAOS_matfile('chaosmagpy_package_0.8/data/CHAOS-7.9.mat')
    
    station = station.upper()
    df_IMOS = pd.read_csv('Dados OBS/Data/Imos informations/Imos_INTERMAGNET.txt',
                          skiprows = 1,
                          sep = '\s+',
                          usecols=[0,1,2,3],
                          names = ['Imos','Latitude','Longitude','Elevation'],
                          index_col= ['Imos'])
    
    
    R_REF = 6371.2
    
    if station not in df_IMOS.index:
        print('Station must be an observatory IAGA CODE!')

    Longitude = df_IMOS.loc[station]['Longitude']
    

    Latitude = 90 - df_IMOS.loc[station]['Latitude']
 

    Elevation = (df_IMOS.loc[station]['Elevation']/1000) + R_REF

    Date = pd.date_range(starttime,endtime + ' 23:00:00', freq = 'H')
    Time =cp.data_utils.mjd2000(Date)
    
    # Internal field
    print('Initiating geomagnetic field computation for ' + station.upper() +'.')
    print('Computing core field.')
    B_core = model.synth_values_tdep(time = Time,
                                     radius = Elevation,
                                     theta = Latitude ,
                                     phi = Longitude)

    print('Computing crustal field up to degree 70.')
    B_crust = model.synth_values_static(radius = Elevation,
                                        theta = Latitude,
                                        phi = Longitude,
                                        nmax=70)
    
    # complete internal contribution
    B_radius_int = B_core[0] + B_crust[0]
    B_theta_int = B_core[1] + B_crust[1]
    B_phi_int = B_core[2] + B_crust[2]
    
    print('Computing field due to external sources, incl. induced field: GSM.')
    B_gsm = model.synth_values_gsm(time = Time,
                                   radius = Elevation, 
                                   theta = Latitude,
                                   phi = Longitude, 
                                   source='all')

    if endtime <= '2021-06-30':
        
        print('Computing field due to external sources, incl. induced field: SM.')
        B_sm = model.synth_values_sm(time = Time,
                                 radius = Elevation,
                                 theta = Latitude,
                                 phi = Longitude,
                                 source='all')
    else:
        Date_RC = pd.date_range(starttime,'2021-06-30' + ' 23:00:00', freq = 'H')
        Date_NO_RC = pd.date_range('2021-07-01',endtime + ' 23:00:00', freq = 'H')
        Time =cp.data_utils.mjd2000(Date_RC)
        Time_sm =cp.data_utils.mjd2000(Date_NO_RC)
        
        print('Computing field due to external sources, incl. induced field: SM.')
        B_sm_with_rc = model.synth_values_sm(time = Time,
                                 radius = Elevation,
                                 theta = Latitude,
                                 phi = Longitude,
                                 source='all')
        
        B_sm_without_rc = model.synth_values_sm(time = Time_sm,
                                 radius = Elevation,
                                 theta = Latitude,
                                 phi = Longitude,
                                 source='all',rc_e = False, rc_i = False)
        
        B_sm = []
        B_sm_x = np.append(B_sm_with_rc[0],B_sm_without_rc[0])
        B_sm_y = np.append(B_sm_with_rc[1],B_sm_without_rc[1])
        B_sm_z = np.append(B_sm_with_rc[2],B_sm_without_rc[2])
        B_sm = [B_sm_x,B_sm_y,B_sm_z]

    # complete external field contribution
    B_radius_ext = B_gsm[0] + B_sm[0]
    B_theta_ext = B_gsm[1] + B_sm[1]
    B_phi_ext = B_gsm[2] + B_sm[2]

    # complete forward computation
    B_radius = B_radius_int + B_radius_ext
    B_theta = B_theta_int + B_theta_ext
    B_phi = B_phi_int + B_phi_ext
    

    df_station = pd.DataFrame()
    df_station.index = Date
    
    df_station['X_tot'] = B_theta.round(3)*-1
    df_station['Y_tot'] = B_phi.round(3)
    df_station['Z_tot'] = B_radius.round(3)*-1    
    
    df_station['X_int'] = B_theta_int.round(3)*-1
    df_station['Y_int'] = B_phi_int.round(3)
    df_station['Z_int'] = B_radius_int.round(3)*-1

    df_station['X_ext_gsm'] = B_gsm[1].round(3)*-1
    df_station['Y_ext_gsm'] = B_gsm[2].round(3)
    df_station['Z_ext_gsm'] = B_gsm[0].round(3)*-1
    
    df_station['X_ext_sm'] = B_sm[1].round(3)*-1
    df_station['Y_ext_sm'] = B_sm[2].round(3)
    df_station['Z_ext_sm'] = B_sm[0].round(3)*-1
    
    return df_station 

def INTERMAGNET_AND_CHAOS_COMPARISION(station, dataframe_Chaos, starttime, endtime, dataframe_intermagnet = None):
    
    directory = 'Chaos_comparison/'+ station +'_data'
    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
    
    Dataframe_intermagnet = pd.DataFrame()
    Dataframe_intermagnet = dataframe_intermagnet
    
    Dataframe_Chaos = pd.DataFrame()
    Dataframe_Chaos = dataframe_Chaos
    
    Dataframe_Chaos['X_ext'] = Dataframe_Chaos['X_ext_gsm'] + Dataframe_Chaos['X_ext_sm']
    Dataframe_Chaos['Y_ext'] = Dataframe_Chaos['Y_ext_gsm'] + Dataframe_Chaos['Y_ext_sm']
    Dataframe_Chaos['Z_ext'] = Dataframe_Chaos['Z_ext_gsm'] + Dataframe_Chaos['Z_ext_sm']
    
    cols_tot = ['X_tot','Y_tot','Z_tot']
    colors = ['blue','green','black']
    
    #plotting total field for chaos data
    
    fig, axes = plt.subplots(3,1, figsize = (14,10))
    axes[0].set_title('Predicted Chaos Data - Total Field', fontsize = 16)
    for col, ax, color in zip(cols_tot, axes.flatten(), colors):
        ax.plot(Dataframe_Chaos[col].loc[starttime:endtime], color  = color)
        ax.set_ylabel(col.upper() + '/nT', fontsize = 12)
    plt.savefig(directory + '/' + station + '_Chaos_Total_field.jpeg', bbox_inches='tight')
    plt.show()
    #plotting internal field for Chaos data
    
    cols_int = ['X_int','Y_int','Z_int']
    fig, axes = plt.subplots(3,1, figsize = (14,10))
    axes[0].set_title('Predicted Chaos Data - Internal Field', fontsize = 16)
    for col, ax, color in zip(cols_int, axes.flatten(), colors):
        ax.plot(Dataframe_Chaos[col].loc[starttime:endtime], color  = color)
        ax.set_ylabel(col.upper() + '/nT', fontsize = 12)
    plt.savefig(directory + '/' + station + '_Chaos_Internal_field.jpeg', bbox_inches='tight')
    plt.show()
    # plotting external field
    
    cols_ext = ['X_ext','Y_ext','Z_ext']
    fig, axes = plt.subplots(3,1, figsize = (14,10))
    axes[0].set_title('Predicted Chaos Data External Field', fontsize = 16)
    for col, ax, color in zip(cols_ext, axes.flatten(), colors):
        
        ax.plot(Dataframe_Chaos[col][starttime:endtime], color  = color)
        ax.set_ylabel(col.upper() + '/nT', fontsize = 12)
    plt.savefig(directory + '/' + station + '_Chaos_External_field.jpeg', bbox_inches='tight')
    plt.show()
        
    if dataframe_intermagnet is not None:
        
        cols_intermagnet = ['X','Y','Z']
        
        #comparing SV - Total field
        fig, axes = plt.subplots(3,1, figsize = (14,10))
        axes[0].set_title(station.upper() + ' - INTERMAGNET and CHAOS SV comparison (Total Field)', fontsize = 16)
        for col, ax, color,column in zip(cols_tot, axes.flatten(), colors, cols_intermagnet):
            ax.plot((Dataframe_Chaos[col].loc[starttime:endtime].resample('M').mean().diff(6) - 
                    Dataframe_Chaos[col].loc[starttime:endtime].resample('M').mean().diff(-6)) ,'o-', color  = 'red')
            ax.plot((Dataframe_intermagnet[column].loc[starttime:endtime].resample('M').mean().diff(6) - 
                    Dataframe_intermagnet[column].loc[starttime:endtime].resample('M').mean().diff(-6)) ,'o-', color  = color)
        plt.savefig(directory + '/' + station + '_Chaos_Total_field_SV_comp.jpeg', bbox_inches='tight') 
        plt.show()
        
        #comparing SV - Internal field
        fig, axes = plt.subplots(3,1, figsize = (14,10))
        axes[0].set_title(station.upper() + ' - INTERMAGNET and CHAOS SV comparison (Internal Field)', fontsize = 16)
        for col, ax, color,column in zip(cols_int, axes.flatten(), colors, cols_intermagnet):
            ax.plot((Dataframe_Chaos[col].loc[starttime:endtime].resample('M').mean().diff(6) - 
                    Dataframe_Chaos[col].loc[starttime:endtime].resample('M').mean().diff(-6)) ,'o-', color  = 'red')
            ax.plot((Dataframe_intermagnet[column].loc[starttime:endtime].resample('M').mean().diff(6) - 
                    Dataframe_intermagnet[column].loc[starttime:endtime].resample('M').mean().diff(-6)) ,'o-', color  = color)                          
            
            ax.set_ylabel(col.upper() + '/nT', fontsize = 12)
        plt.savefig(directory + '/' + station + '_Chaos_internal_field_SV_comp.jpeg', bbox_inches='tight')
        plt.show()
        
        #comparing intermagnet SV corrected for external field 
        
        fig, axes = plt.subplots(3,1, figsize = (14,10))
        axes[0].set_title(station.upper() + ' - INTERMAGNET (corrected for external field) and CHAOS SV comparison', fontsize = 16)
        for col, cols, ax, color,column in zip(cols_int,cols_ext, axes.flatten(), colors, cols_intermagnet):
            ax.plot((Dataframe_Chaos[col].loc[starttime:endtime].resample('M').mean().diff(6) - 
                    Dataframe_Chaos[col].loc[starttime:endtime].resample('M').mean().diff(-6)) ,'o-', color  = 'red')
            ax.plot(((Dataframe_intermagnet[column].loc[starttime:endtime].resample('H').mean() - Dataframe_Chaos[cols]).resample('M').mean().diff(6) - 
                    (Dataframe_intermagnet[column].loc[starttime:endtime].resample('H').mean() - Dataframe_Chaos[cols]).resample('M').mean().diff(-6)) ,'o-', color  = color)                          
            
            ax.set_ylabel(col.upper() + '/nT', fontsize = 12)
        plt.savefig(directory + '/' + station + '_SV_comp_external_corrected.jpeg', bbox_inches='tight')
        plt.show()
        
def external_field_correction_chaos_model(station, starttime, endtime,df_station = None, df_chaos = None):    
    '''
    Correct the INTERMAGNET observatory data with the CHAOS-7.9 model external geomagnetic field prediction.
    
    Find the model in the website - http://www.spacecenter.dk/files/magnetic-models/CHAOS-7/
    
    References
    Finlay, C.C., Kloss, C., Olsen, N., Hammer, M. Toeffner-Clausen, L., Grayver, A and Kuvshinov, A. (2020),
    The CHAOS-7 geomagnetic field model and observed changes in the South Atlantic Anomaly,
    Earth Planets and Space 72, doi:10.1186/s40623-020-01252-9
    --------------------------------------------------------------------------------
    Inputs:
    
    station - 3 letters IAGA code for a INTERMAGNET observatory.
    
    starttime - first day of the data (format = 'yyyy-mm-dd)
    
    endtime - last day of the data (format = 'yyyy-mm-dd)
    
    df_station - dataframe with INTERMAGNET data or None (compute the INTERMAGNET data)
    
    df_chaos - dataframe with CHAOS predicted data or None (compute the CHAOS model data)
    
    ----------------------------------------------------------------------------------
    
    Return a hourly mean dataframe corrected from CHAOS-7 model external field
    
    
    '''
    
    assert len(station) == 3, 'station must be a three letters IAGA Code'
    
    assert isinstance(df_station,pd.DataFrame) or df_station == None, 'df_station must be a pandas dataframe or None'
    
    assert isinstance(df_chaos,pd.DataFrame) or df_chaos == None, 'df_station must be a pandas dataframe or None'
    
    
    station = station.upper()
    
    df_IMOS = pd.read_csv('Dados OBS/Data/Imos informations/Imos_INTERMAGNET.txt',
                          skiprows = 1,
                          sep = '\s+',
                          usecols=[0,1,2,3],
                          names = ['Imos','Latitude','Longitude','Elevation'],
                          index_col= ['Imos'])
    if station not in df_IMOS.index:
        print('Station must be an observatory IAGA CODE!')
        
    
    
    if df_chaos is not None:
        df_chaos = df_chaos
        df_chaos.loc[starttime:endtime] = df_chaos
    
    else:
        
        df_chaos = chaos_model_prediction(station = station,
                                         starttime = starttime,
                                         endtime = endtime)
        
        
    df_chaos['X_ext'] = df_chaos['X_ext_gsm'] + df_chaos['X_ext_sm']
    df_chaos['Y_ext'] = df_chaos['Y_ext_gsm'] + df_chaos['Y_ext_sm']
    df_chaos['Z_ext'] = df_chaos['Z_ext_gsm'] + df_chaos['Z_ext_sm']
    
    if df_station is not None:
        
        df_station = df_station
        df_station.loc[starttime:endtime] = df_station
    else:
        df_station = mvs.load_INTERMAGNET_files(station = station,
                                                  starttime = starttime,
                                                  endtime = endtime)
        
        df_station = df_station.loc[starttime:endtime]
    df_station = df_station.resample('H').mean()
        
    df_station['X'] = df_station['X'] - df_chaos['X_ext']
    df_station['Y'] = df_station['Y'] - df_chaos['Y_ext']
    df_station['Z'] = df_station['Z'] - df_chaos['Z_ext'] 
    
    df_station = resample_obs_data(df_station, 'H')
        
    return df_station, df_chaos    

def rms(predictions, real_data):
    columns = ['X_int','Y_int','Z_int']
    x = []
    for col,cols in zip(columns,real_data.columns):
        y = (real_data[cols].resample('M').mean().diff(6) - real_data[cols].resample('M').mean().diff(-6)).dropna()
        ypred = (predictions[col].resample('M').mean().diff(6) - predictions[col].resample('M').mean().diff(-6)).dropna()
        ypred = ypred.reindex(y.index)
        rms = np.sqrt(((ypred - y) ** 2).mean()).round(3)
        x.append(rms)
        #print('the rmse for ' + str(cols) + ' component is ' + str(rms) + '.')
    return x

def night_time_selection(station, dataframe, starttime, endtime):
    
    '''
    Function to select the night time period (from 23 PM to 5 AM) from the geomagnetic data.
     
    ---------------------------------------------------------------------
    Inputs:
    
    station - 3 letters IAGA code for a INTERMAGNET observatory.
    
    dataframe - a pandas dataframe with geomagnetic data.
    
    dataframe - 3 letters IAGA code for a INTERMAGNET observatory.
    
    starttime - first day of the data (format = 'yyyy-mm-dd)
    
    endtime - last day of the data (format = 'yyyy-mm-dd)
    
    ---------------------------------------------------------------------------
        
    Example of use:
    night_time_selection(station = 'VSS',dataframe = 'name_of_dataframe',
                         starttime = 'yyyy-mm-dd', endtime = 'yyyy-mm-dd')
    
    ------------------------------------------------------------------------------
    
    return a dataframe with only the night time period.
    
    '''
    assert len(station) == 3, 'station must be a three letters IAGA Code'
    
    assert isinstance(dataframe,pd.DataFrame), 'dataframe must be a pandas dataframe'
    
    for i in [starttime,endtime]:
        mvs.validate(i)
    
    f = []
    f.extend(glob.glob('Dados OBS/*/*/' + station + '*'))
    f.sort()
    
    df_IMOS = pd.read_csv('Dados OBS/Data/Imos informations/Imos_INTERMAGNET.txt',
                          skiprows = 1,
                          sep = '\s+',
                          usecols=[0,1,2,3],
                          names = ['Imos','Latitude','Longitude','Elevation'],
                          index_col= ['Imos'])
    
    Longitude = df_IMOS.loc[station]['Longitude']
    
    dif =  Longitude/15

    
    df = dataframe
    df = df.loc[starttime:endtime]
    
    df_lt = df.shift(round(dif, 3), freq = 'H')
    
    
    df_NT_lt = df_lt.drop(df_lt.loc[(df_lt.index.hour > 5) & (df_lt.index.hour < 23)].index).dropna()
    
    df_NT = pd.DataFrame()
    df_NT = df_NT_lt.shift(round(-dif, 3), freq = 'H')
    
    return df_NT

def jerk_detection(station, dataframe,linear_segments, starttime, endtime):
    
    '''
    adopt piecewise linear segments on the secular variation
    
    ---------------------------------------------------------------------------------
    
    Inputs:
    
    station = IAGA code
    dataframe = Pandas dataframe with the secular variation
    from the observatory.
    linear_segments = Number of linear segments, must be a list containing 3 numbers (one for each geomagnetic component)
    starttime = initial period
    endtime = final period
    
    --------------------------------------------------------------------------------
    
    usage example:
    jerk_detection(station = 'VSS', dataframe = df_VSS, linear_segments = [3,4,3],
                   starttime = '2005-01-01', endtime = '2021-09-30')
    
    '''
    
    assert len(station) == 3, 'station must be a three letters IAGA Code'
    
    assert isinstance(dataframe,pd.DataFrame), 'dataframe must be a pandas dataframe'
    
    assert isinstance(linear_segments,(list,int)), 'x must be a list of integers'
    assert len(linear_segments) == 3, 'x must be a list with 3 integers'
    
    for i in [starttime,endtime]:
        mvs.validate(i)
    
    
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
    
def hampel_filter_denoising(dataframe, window_size, n_sigmas=3):
    '''
    
    
    
    
    
    '''
    
    assert isinstance(dataframe,pd.DataFrame), 'dataframe must be a pandas dataframe.'
    
    assert isinstance(window_size, int), 'window_size must be an integer.'
    
    dataframe = resample_obs_data(dataframe,'H')
    denoised_dataframe = dataframe.copy()
    for column in dataframe:
        
        n = len(dataframe[column])
        #denoised_dataframe = dataframe.copy()
        k = 1.4826 # scale factor for Gaussian distribution
        
        indices = []
        
        # possibly use np.nanmedian 
        for i in range((window_size),(n - window_size)):
            x0 = np.median(dataframe[column][(i - window_size):(i + window_size)])
            S0 = k * np.median(np.abs(dataframe[column][(i - window_size):(i + window_size)] - x0))
            if (np.abs(dataframe[column][i] - x0) > n_sigmas * S0):
                denoised_dataframe[column][i] = x0
        
        fig, ax = plt.subplots(figsize = (16,4))
        ax.plot(dataframe[column], 'k', label = 'Removed Outliers')
        ax.plot(denoised_dataframe[column], 'r', label = 'Denoised ' + column)
        ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize = 12)
        plt.grid()
        plt.show()
        
    return denoised_dataframe

def polynomial_jerk_detection(station, window_start, 
                              window_end, 
                              starttime, 
                              endtime,
                              df_station = None,
                              plot_detection: bool = True,
                              CHAOS_correction: bool = True):
    
    for i in [starttime,endtime,window_start,window_end]:
        validate(i)
        
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
        
        df_station, df_chaos = external_field_correction_chaos_model(station = station,
                                                  starttime = starttime,
                                                  endtime = endtime,
                                                  df_station = df_station,
                                                  df_chaos = None)
    else:
        pass
    
    #calculating SV
    df_SV = calculate_SV(dataframe = df_station, starttime = starttime, endtime = endtime, info = 'ADMM', columns = None)
    
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

def resample_obs_data(dataframe, sample):
    '''
    Resample a pd.DataFrame to hourly, daily, monthly or annual means
    
    The new sample is set in the middle of the sample range. Example daily mean is set in the middle of the day, 12h.
    
    
    '''
    
    assert isinstance(dataframe,pd.DataFrame), 'dataframe must be a pandas dataframe.'
    
    df_station = dataframe
    
    samples = ['min','H','D','M','Y']
    
    
    if sample not in samples:
        print('sample must be min, H, D, M or Y!')
    else:
        
        if sample == 'min':
            
            df_station = df_station
            
        if sample == 'H':
            
            df_station = df_station.resample('H').mean()
            df_station.index = df_station.index + to_offset('30min')
            
        if sample == 'D':
            
            df_station = df_station.resample('D').mean()
            df_station.index = df_station.index + to_offset('12H')
            
        if sample == 'M':
            
            df_station = df_station.resample('M').mean()
            df_station.index = df_station.index + to_offset('-1M') + to_offset('15D')
            
            
        if sample == 'Y':
            
            df_station = df_station.resample('Y').mean()
            df_station.index = df_station.index + to_offset('-6M') + to_offset('-15D')
            
    return df_station


