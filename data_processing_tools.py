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
        spf.validate(i)    
    
    df = dataframe.loc[starttime:endtime]
    
    disturbed_index = pd.DataFrame()
    
    df_d = pd.read_csv('Thesis_Marcos/Data/Disturbed and Quiet Days/Disturbed_Days_list.txt',
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
        disturbed_index = pd.concat([disturbed_index,df.loc[str(date)]])
        
    
    df = df.drop(disturbed_index.index)
    
    print('Top 5 disturbed days for each month were removed from the data.')
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
        spf.validate(i)
    
    df = dataframe.loc[starttime:endtime]
    
    quiet_index = pd.DataFrame()

    df_q = pd.read_csv('Thesis_Marcos/Data/Disturbed and Quiet Days/Quiet_Days_list.txt',
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
        try:
            quiet_index = pd.concat([quiet_index,df.loc[str(date)]])
        except:
            pass
    df = quiet_index
    
    print('Only quiet top 10 quiet days for each month were kept in the data.')
    return df

def calculate_SV(dataframe, starttime, endtime, method = 'ADMM', columns = None, apply_percentage:bool = False):
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

    if columns == None:
        columns = ['X','Y','Z']
    else:
        columns = columns
        
    if method not in Method:
        print('Info musdf_SV = pd.DataFrame()t be ADMM or YD')
    
    if method == 'ADMM':
        df_ADMM = resample_obs_data(dataframe = df, sample = 'M',apply_percentage = apply_percentage)
        
        for col in columns:
            SV = (df_ADMM[col].diff(6) - df_ADMM[col].diff(-6)).round(3).dropna()
            df_SV[col] = SV  
    if method == 'YD':
        df_YD = resample_obs_data(dataframe = df, sample = 'Y',apply_percentage = apply_percentage)
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
        spf.validate(i)
            
    if (kp <=0 ) or (kp>= 9): 
        print('kp must be a number from 0 to 9, try again!')
    df = pd.DataFrame()
    df = dataframe
    
    KP_ = pd.read_csv('Data/Kp index/Kp_ap_since_1932.txt', skiprows = 30,
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
    
    return a hourly dataframe with the X, Y and Z geomagnetic components CHAOS prediction for total field,
    internal field and external field.
    
    '''
    
    assert len(station) == 3, 'station must be a three letters IAGA Code'
    
    for i in [starttime,endtime]:
        spf.validate(i)
        
        
    model = cp.load_CHAOS_matfile('chaosmagpy_package_0.8/data/CHAOS-7.9.mat')
    
    station = station.upper()
    df_IMOS = pd.read_csv('Thesis_Marcos/Data/Imos informations/Imos_INTERMAGNET.txt',
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
        
def external_field_correction_chaos_model(station, starttime, endtime,df_station = None, df_chaos = None, apply_percentage:bool = False):    
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
    
    df_IMOS = pd.read_csv('Thesis_Marcos/Data/Imos informations/Imos_INTERMAGNET.txt',
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
        
    #df_chaos.index = df_chaos.index + to_offset('30min')
        
        
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
    
    df_station = resample_obs_data(df_station, 'H', apply_percentage = False)   
    
    print('The external field predicted using CHAOS-model was removed from the data.')
    return df_station, df_chaos    

def rms(predictions, real_data):
    '''
    Function to calculate the Root mean square error.
    
    Built to work with geomagnetic data (dataframe with X, Y and Z components).
    
    Will calculate the RMSE for each component
    
    ----------------------------------------------------------
    Inputs:
    
    predictions -> must be a dataframe
    
    real_data ->  must be a dataframe
    
    --------------------------------------------------------------------
    Return a list with the RMSE for each X, Y and Z
    '''
    
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
        spf.validate(i)
    
    f = []
    f.extend(glob.glob('Dados OBS/*/*/' + station + '*'))
    f.sort()
    
    df_IMOS = pd.read_csv('Thesis_Marcos/Data/Imos informations/Imos_INTERMAGNET.txt',
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
    
    print('The night time period was selected.')
    return df_NT

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
    jerk_detection(station = 'VSS', dataframe = df_VSS, linear_segments = [3,4,3],
                   starttime = '2005-01-01', endtime = '2021-09-30')
    
    '''
    
    assert len(station) == 3, 'station must be a three letters IAGA Code'
    
    assert isinstance(dataframe,pd.DataFrame), 'dataframe must be a pandas dataframe'
    
    assert isinstance(linear_segments,(list,int)), 'x must be a list of integers'
    assert len(linear_segments) == 3, 'x must be a list with 3 integers'
    
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
    
       
def hampel_filter_denoising(dataframe, window_size, n_sigmas=3, plot_figure:bool = False):
    '''
    
    
    
    
    ------------------------------------------------------------------------------------
    
    Inputs:
    
    dataframe - pandas dataframe 
    
    window_size - integer, size of the moving window to calculate the absolute median
    
    n_sigmas - Number of standard deviations to be consider as a outlier
    
    plot_figure - boolean, option to plot a comparison between real and denoised data.
    
    ------------------------------------------------------------------------------------
    
    Return a hourly dataframe denoised 
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
        if plot_figure == True:
            for i in range((window_size),(n - window_size)):
                x0 = np.median(dataframe[column][(i - window_size):(i + window_size)])
                S0 = k * np.median(np.abs(dataframe[column][(i - window_size):(i + window_size)] - x0))
                if (np.abs(dataframe[column][i] - x0) > n_sigmas * S0):
                    denoised_dataframe[column][i] = x0
            
            fig, ax = plt.subplots(figsize = (16,4))
            ax.plot(dataframe[column], 'k', label = 'Removed Outliers')
            ax.plot(denoised_dataframe[column], 'r', label = 'Denoised ' + column)
            ax.set_xlim(dataframe[column].index[0],dataframe[column].index[-1])
            ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize = 12)
            plt.grid()
            plt.show()
        else:
            pass
        
    return denoised_dataframe

def polynomial_jerk_detection(station, window_start, 
                              window_end, 
                              starttime, 
                              endtime,
                              df_station = None,
                              plot_detection: bool = True,
                              CHAOS_correction: bool = True):
    
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

def resample_obs_data(dataframe, sample, apply_percentage:bool = False):
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
        
        if sample == 'min' and apply_percentage == False:
            
            df_station = df_station
        if sample == 'min' and apply_percentage == True:
            df_station = df_station
            
        if sample == 'H' and apply_percentage == False:
            
            df_station = df_station.resample('H').mean()
            df_station.index = df_station.index + to_offset('30min')
            
        if sample == 'H' and apply_percentage == True:
                        
            
            tmp = df_station.groupby(pd.Grouper(freq='H')).agg(['mean','count']).swaplevel(0,1,axis=1)
            
            if tmp['count'].median().any() <= 1:
                
                df_station = tmp['mean'].where(tmp['count']>=1*0.85)
            else:
                df_station = tmp['mean'].where(tmp['count']>=60*0.85)
            
            #df_station = df_station.resample('H').mean()
            df_station.index = df_station.index + to_offset('30min')
            
        if sample == 'D' and apply_percentage == False:
            
            df_station = df_station.resample('D').mean()
            df_station.index = df_station.index + to_offset('12H')        
        
        if sample == 'D' and apply_percentage == True:
            
            tmp = df_station.groupby(pd.Grouper(freq='D')).agg(['mean','count']).swaplevel(0,1,axis=1)
            
            if tmp['count'].median().any() <= 30:
                
                df_station = tmp['mean'].where(tmp['count']>=24*0.85)
                
            else:
            
                df_station = tmp['mean'].where(tmp['count']>=1440*0.85)
            
            #df_station = df_station.resample('D').mean()
            df_station.index = df_station.index + to_offset('12H')
            
        if sample == 'M' and apply_percentage == False:
            
            
            df_station = df_station.resample('M').mean()
            df_station.index = df_station.index + to_offset('-1M') + to_offset('15D')
            
        if sample == 'M' and apply_percentage == True:
            
            tmp = df_station.groupby(pd.Grouper(freq='M')).agg(['mean','count']).swaplevel(0,1,axis=1)
            
            if tmp['count'].median().any() <= 800:
                tmp['full day'] = df_station.resample('M').mean().index.days_in_month*24

            else:
            #tmp = df_station.groupby(pd.Grouper(freq='M')).agg(['mean','count']).swaplevel(0,1,axis=1)
                tmp['full day'] = df_station.resample('M').mean().index.days_in_month*24*60
                
                
            X = tmp['mean','X'].loc[tmp['count','X'] >= tmp['full day']*0.85]
            Y = tmp['mean','Y'].loc[tmp['count','Y'] >= tmp['full day']*0.85]
            Z = tmp['mean','Z'].loc[tmp['count','Z'] >= tmp['full day']*0.85]

            
            df_station = pd.DataFrame()
            df_station['X'] = X
            df_station['Y'] = Y
            df_station['Z'] = Z
            
            #df_station = df_station.resample('M').mean()
            df_station.index = df_station.index + to_offset('-1M') + to_offset('15D')
            
            
        if sample == 'Y' and apply_percentage == False:
            
            df_station = df_station.resample('Y').mean()
            df_station.index = df_station.index + to_offset('-6M') + to_offset('-15D')
            
        if sample == 'Y' and apply_percentage == True:
            
            Days = df_station.groupby(pd.Grouper(freq='M')).agg(['count']).swaplevel(0,1,axis=1)
            Days['Days'] = df_station.resample('M').mean().index.days_in_month

            tmp = df_station.groupby(pd.Grouper(freq='Y')).agg(['mean','count']).swaplevel(0,1,axis=1)
            tmp['Days'] = Days['Days'].resample('Y').sum()
            
            if tmp['count'].median().any() <= 8784:
                X = tmp['mean','X'].loc[tmp['count','X'] >= tmp['Days']*0.85*24]
                Y = tmp['mean','Y'].loc[tmp['count','Y'] >= tmp['Days']*0.85*24]
                Z = tmp['mean','Z'].loc[tmp['count','Z'] >= tmp['Days']*0.85*24]
            else:

                X = tmp['mean','X'].loc[tmp['count','X'] >= tmp['Days']*0.85*24*60]
                Y = tmp['mean','Y'].loc[tmp['count','Y'] >= tmp['Days']*0.85*24*60]
                Z = tmp['mean','Z'].loc[tmp['count','Z'] >= tmp['Days']*0.85*24*60]
            
            df_station = pd.DataFrame()
            df_station['X'] = X
            df_station['Y'] = Y
            df_station['Z'] = Z
            df_station.index = df_station.index + to_offset('-6M') + to_offset('-15D')
            
    return df_station

def jerk_detection_window(station, window_start, 
                              window_end, 
                              starttime, 
                              endtime,
                              df_station = None,
                              df_CHAOS = None,
                              plot_detection: bool = True,
                              CHAOS_correction: bool = True,
                              plot_CHAOS_prediction:bool = False):
    
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
    
    #computing dataframe from observatory files
    else:
        df_station = mvs.load_INTERMAGNET_files(station = station,
                                                starttime = starttime,
                                                endtime = endtime)
    if df_CHAOS is not None:
        
        df_chaos = df_CHAOS
    
    if CHAOS_correction == True and df_CHAOS is not None:
        
        df_station, df_chaos = external_field_correction_chaos_model(station = station,
                                                  starttime = starttime,
                                                  endtime = endtime,
                                                  df_station = df_station,
                                                  df_chaos= df_chaos)
    elif CHAOS_correction == True and df_CHAOS == None:
        df_station, df_chaos = external_field_correction_chaos_model(station = station,
                                                  starttime = starttime,
                                                  endtime = endtime,
                                                  df_station = df_station,
                                                  df_chaos = None)
    
    else:
        pass
    
    #calculating SV from intermagnet files
    df_SV = calculate_SV(dataframe = df_station,
                             starttime = starttime,
                             endtime = endtime,
                             method = 'ADMM',
                             columns = None)
    
    if CHAOS_correction and plot_CHAOS_prediction == True:
        
        df_CHAOS_SV = calculate_SV(dataframe = df_chaos,
                                       starttime = starttime,
                                       endtime = endtime,
                                       method = 'ADMM',
                                       columns = ['X_int','Y_int','Z_int'])
    else:
        
        pass
    
    df_jerk_window = pd.DataFrame()
    
    df_jerk_window.index = df_SV.loc[window_start:window_end].index
    
    z = pd.DataFrame()
    #starting with window jerk detection
    
    #t = np.arange(0,df_SV.loc[window_start:window_end].size/3).reshape(-1,1)
    #jerk_prediction = pd.DataFrame()
    #jerk_prediction.index = df_SV.loc[window_start:window_end].index
    
    X = np.arange(0,len(df_jerk_window.index))
    
    df_slopes = pd.DataFrame()
    #rsq = pd.DataFrame()
    df_rsq = pd.DataFrame()

    
    #eqn_list = []
    
    for column in df_SV.columns:

        myPWLF = pwlf.PiecewiseLinFit(X,df_SV.loc[window_start:window_end][column])
        
        z[column] = myPWLF.fit(2)
        
        #calculate slopes
        slopes = myPWLF.calc_slopes()
        df_slopes[column] = slopes
        
        #calculate r_squared
        
        print('\nR^2 for the ' + column + ' component: ' + str((myPWLF.r_squared()).round(2)))

        #se = myPWLF.se
        #print(se)
        xHat = X 
        yHat = myPWLF.predict(xHat)
        
        df_jerk_window[str(column)] = yHat
        
        #for i in range(myPWLF.n_segments):
        #    eqn_list.append(get_symbolic_eqn(myPWLF, i + 1))
        #    #print('Equation number: ',(i + 1) + 'for ' + str(column) + 'component')
        #    print(eqn_list[-1])
        #    #f_list.append(lambdify(x, eqn_list[-1]))
        
    if plot_detection == True and plot_CHAOS_prediction == False or CHAOS_correction == False:
        colors = ['blue','green','black']
        fig, axes = plt.subplots(3,1,figsize = (10,8))
        plt.suptitle(station.upper() +' secular variation', fontsize = 14, y = 0.92)
        plt.xlabel('Date (Years)', fontsize = 12)
        
        for col, ax, color in zip(df_SV.columns, axes.flatten(), colors):
            ax.plot(df_SV[col],'o-',color = color)
            ax.plot(df_jerk_window[col].index,
                    df_jerk_window[col],color = 'red', linewidth = 3, label = 't0 ' + 
                    str(round((df_jerk_window.index[int(z[col][1].round())].year+
                               (df_jerk_window.index[int(z[col][1].round())].dayofyear -1)/365),2)))
            ax.set_xlim(df_SV[col].index[0],df_SV[col].index[-1])
            ax.set_ylabel('d' + col.upper() +'/dt (nT)', fontsize = 12)
            ax.legend()
        
        
        fig, axes = plt.subplots(1,3,figsize = (16,4))
        plt.suptitle(station.upper() +' secular variation', fontsize = 14, y = 0.93)
        fig.text(0.5, 0.04, 'Date (Years)', ha='center')
     
        upper_limit = int(str(datetime.strptime(window_end ,'%Y-%m-%d'))[0:4]) +1
        lower_limit = int(str(datetime.strptime(window_start ,'%Y-%m-%d'))[0:4]) -1

        for col, ax, color in zip(df_SV.columns, axes.flatten(), colors):
            
            ax.plot(df_SV[col].loc[str(lower_limit):str(upper_limit)],'o-',color = color)
            ax.plot(df_jerk_window[col],color = 'red', linewidth = 3)
            ax.set_ylabel('d' + col.upper() +'/dt (nT)', fontsize = 12)
    
    elif plot_detection and plot_CHAOS_prediction and CHAOS_correction == True:
        
        colors = ['blue','green','black']
        fig, axes = plt.subplots(3,1,figsize = (12,10))
        plt.suptitle(station.upper() +' secular variation', fontsize = 14, y = 0.92)
        plt.xlabel('Date (Years)', fontsize = 12)
        
        for col,chaos_col, ax, color in zip(df_SV.columns,df_CHAOS_SV.columns, axes.flatten(), colors):
            ax.plot(df_SV[col],'o-',color = color)
            
            ax.plot(df_CHAOS_SV[chaos_col],
                    '-',
                    linewidth = 2,
                    label = 'CHAOS prediction')
            
            ax.plot(df_jerk_window[col].index,
                    df_jerk_window[col],color = 'red', linewidth = 3, label = 't0 ' + 
                    str(round((df_jerk_window.index[int(z[col][1].round())].year+
                               (df_jerk_window.index[int(z[col][1].round())].dayofyear -1)/365),2)))
            ax.set_xlim(df_SV[col].index[0],df_SV[col].index[-1])
            ax.set_ylabel('d' + col.upper() +'/dt (nT)', fontsize = 12)
            ax.legend()
        plt.show()
        
        
        fig, axes = plt.subplots(1,3,figsize = (16,4))
        plt.suptitle(station.upper() +' secular variation', fontsize = 14, y = 0.93)
        fig.text(0.5, 0.04, 'Date (Years)', ha='center')
     
        upper_limit = int(str(datetime.strptime(window_end ,'%Y-%m-%d'))[0:4]) +1
        lower_limit = int(str(datetime.strptime(window_start ,'%Y-%m-%d'))[0:4]) -1

        for col,chaos_col, ax, color in zip(df_SV.columns,df_CHAOS_SV, axes.flatten(), colors):
            
            ax.plot(df_SV[col].loc[str(lower_limit):str(upper_limit)],
                    'o-',
                    color = color)
            
            ax.plot(df_CHAOS_SV[chaos_col].loc[str(lower_limit):str(upper_limit)],
                    '-',
                    linewidth = 2,
                    label = 'CHAOS prediction')
            
            ax.plot(df_jerk_window[col],color = 'red', linewidth = 3)
            ax.set_ylabel('d' + col.upper() +'/dt (nT)', fontsize = 12)
            ax.legend()
        plt.show()
        
        
    
    return df_jerk_window, df_slopes


