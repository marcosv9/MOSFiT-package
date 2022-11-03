import sys
from time import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from pandas.tseries.frequencies import to_offset
import h5py
import glob
import os
import ftplib
import pathlib
import matplotlib.gridspec as gridspec
import matplotlib.dates as md
from datetime import datetime, timedelta
import pwlf
import chaosmagpy as cp
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import main_functions as mvs
import utilities_tools as utt
import support_functions as spf
from chaosmagpy.data_utils import save_RC_h5file


def project_directory():
    '''
    Get the project directory 
    '''
    return os.getcwd()

def remove_disturbed_days(dataframe: pd.DataFrame()):
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
    
    ---------------------------------------------------------------------
    Example of use:
    
    remove_disturbed_days(df_NGK, starttime = '2010-01-01' , endtime = '2019-12-31')
    
    ----------------------------------------------------------------------
    
    Return a dataframe without the geomagnetic disturbed days.
    
    '''
    
    assert isinstance(dataframe, pd.DataFrame), 'dataframe must be a pandas dataframe'
      
    df = dataframe
    
    disturbed_index = pd.DataFrame()
    
    working_directory = project_directory()
    
    dd_list_directory = pathlib.Path(os.path.join(working_directory,
                                                  'Data',
                                                  'Disturbed and Quiet Days',
                                                  'Disturbed_Days_list.txt'
                                                  )
                                     )
    
    
    #updating disturbed days list

    df_d = pd.read_csv(dd_list_directory,
                       skiprows = 1, 
                       usecols = [0],
                       names = ['dd'],
                       parse_dates = {'D-Days': ['dd']},
                       index_col = ['D-Days']
                       )
    if df_d.index[-1].date().strftime('%Y-%m') != (datetime.today().date() - timedelta(days=30)).strftime('%Y-%m'):
        spf.update_qd_and_dd(data = 'DD')
      
    #df_d['D-Days'] = pd.to_datetime(df_d['D-Days'], format = '%YYYY-%mm-%dd')

    #df_d.set_index('D-Days', inplace = True)
    
    #df_d = df_d.sort_values('D-Days')
    
    
    #first_day = str(df.index[0].year) + '-' + str(df.index[0].month) + '-' + str(df.index[0].day)
    #last_day = str(df.index[-1].year) + '-' + str(df.index[-1].month) + '-' + str(df.index[-1].day)
    
    df_d = df_d.loc[str(df.index[0].date()):str(df.index[-1].date())]
    
    for date in df_d.index.date:
        try:
            disturbed_index = pd.concat([disturbed_index, df.loc[str(date)]])
        except:
            pass

    
    df = df.drop(disturbed_index.index)
    
    print('Top 5 disturbed days for each month were removed from the data.')
    return df

def keep_quiet_days(dataframe: pd.DataFrame()):
    
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
    
    ---------------------------------------------------------------------------
    Example of use:
    
    keep_quiet_days(dataframe = my_dataframe, starttime = '2010-01-01' , endtime = '2019-12-31')
    
    ---------------------------------------------------------------------------
    Return a dataframe containing only geomagnetic quiet days.
    
    '''
    
    assert isinstance(dataframe,pd.DataFrame), 'dataframe must be a pandas dataframe'
    
    df = dataframe
    
    quiet_index = pd.DataFrame()

    #updating quiet days list
    
    #spf.update_qd_and_dd(data = 'QD')
    
    working_directory = project_directory()
    
    qd_list_directory = pathlib.Path(os.path.join(working_directory,
                                                  'Data',
                                                  'Disturbed and Quiet Days',
                                                  'Disturbed_Days_list.txt'
                                                  )
                                     )

    df_q = pd.read_csv(qd_list_directory,
                       header = None,
                       skiprows = 1, 
                       usecols = [0],
                       names = ['qd'],
                       parse_dates = {'Q-Days': ['qd']},
                       index_col= ['Q-Days']
                       )
    
    if df_q.index[-1].date().strftime('%Y-%m') != (datetime.today().date() - timedelta(days=30)).strftime('%Y-%m'):
        spf.update_qd_and_dd(data = 'QD')
    
    #df_q['Q-Days'] = pd.to_datetime(df_q['Q-Days'], format = '%YYYY-%mm-%dd')
    #
    #df_q = df_q.sort_values('Q-Days')
    #
    #df_q.set_index('Q-Days', inplace = True)
    #
    df_q = df_q.loc[str(df.index[0].date()):str(df.index[-1].date())]

    for date in df_q.index.date:
        try:
            quiet_index = pd.concat([quiet_index, df.loc[str(date)]])
        except:
            pass
    df = quiet_index
    
    print('Only top 10 quiet days for each month were kept in the data.')
    return df

def calculate_sv(dataframe: pd.DataFrame(),
                 method: str = 'ADMM',
                 source: str = None,
                 apply_percentage:bool = False
                ):
    '''
    Calculate the secular variation of geomagnetic observatory data.
    
    Two different methods are available, Annual differences of monthly means (ADMM) or Yearly differences (YD). 
    
    -------------------------------------------------------------------------------------
    
    inputs:

    dataframe - a pandas dataframe with geomagnetic data.
    
    method (str) - 'ADMM' or 'YD'
    
    source (str) - geomagnetic source to calculate the SV. used for chaos model prediction
                    must be 'tot' (total field) or 'int' (core field). Default is None, used for
                    intermagnet observatories 
    
    --------------------------------------------------------------------------------------------
    Example of use:
    
    calculate_sv(dataframe = name_of_your_dataframe,
                 method = 'ADMM',
                 source = None) 
                
    
    --------------------------------------------------------------------------------------------
    Return a dataframe with the secular variation
    
    '''
    #validating inputs
    
    assert isinstance(dataframe, pd.DataFrame), 'dataframe must be a pandas dataframe'
 
    assert method.upper() in ['ADMM', 'YD'], 'method must be ADMM or YD'
    
    assert source in ['tot', 'int', None], 'method must be ADMM or YD'
    
    assert isinstance(apply_percentage, bool), 'apply_percentage must be True or False'
        
    df = dataframe
    
    df_sv = pd.DataFrame()
    
    if source is None:
            columns = ['X', 'Y', 'Z']
    else:
        if source == 'tot':
            columns = ['X_tot', 'Y_tot', 'Z_tot']
        if source == 'int':
            columns = ['X_int', 'Y_int', 'Z_int']
    #computing SV from ADMM
    if method == 'ADMM':
        df_admm = resample_obs_data(dataframe = df,
                                    sample = 'M',
                                    apply_percentage = apply_percentage
                                   )
        for col in columns:
            SV = (df_admm[col].diff(6) - df_admm[col].diff(-6)).round(3).dropna()
            df_sv[col] = SV 
        
    #computing SV from YD          
    if method == 'YD':
        df_yd = resample_obs_data(dataframe = df,
                                  sample = 'Y',
                                  apply_percentage = apply_percentage
                                 )
        for col in columns:
            sv = df_yd[col].diff().round(3).dropna()
            df_sv[col] = sv
            
    return df_sv

def kp_index_correction(dataframe: pd.DataFrame(),
                        kp: float = 2,
                        ):
    '''
    Function o filter geomagnetic data based on Kp index
    
    ---------------------------------------------------------
    inputs:
    
    dataframe - a pandas dataframe with geomagnetic data.
    
    kp = limit kp index value (float or int), from 0 to 9.
    --------------------------------------------------------
    Example of use:
    
    Kp_index_correction(dataframe = name_of_dataframe,
                        kp = 3.3)    
    
    ----------------------------------------------------------
    return a dataframe filtered by the selected Kp-index
    
    '''
    #validating the inputs
    
    assert isinstance(dataframe,pd.DataFrame), 'dataframe must be a pandas DataFrame'
    
    assert isinstance(kp, int) or isinstance(kp, float), 'kp must be a number from 0 to 9'
    
    assert kp >= 0 and kp <= 9, 'kp must be between 0 and 9'

    df_station = dataframe.copy()
    
    working_directory = project_directory()
    
    kp_directory = pathlib.Path(os.path.join(working_directory,
                                             'Data',
                                             'Kp index',
                                             'kp_index_since_1932.txt'
                                             )
                                )
    
    KP_ = pd.read_csv(kp_directory,
                      sep = '\t',
                      index_col = ['Date'])
    
    KP_.index = pd.to_datetime(KP_.index, format = '%Y-%m-%d %H:%M:%S')
    
    if datetime.today() > KP_.index[-1]:
        print('Updating the index')
    
    #updating the Kp_index for the most recent data
        KP_ = pd.read_csv('https://www-app3.gfz-potsdam.de/kp_index/Kp_ap_since_1932.txt',
                          skiprows = 30,
                          header = None,
                          sep = '\s+', 
                          usecols = [0,1,2,3,7,8],
                          parse_dates = {'Date': ['Y', 'M','D','H']},
                          names = ['Y','M','D','H','Kp','Ap']
                          )
    
        KP_.index = pd.to_datetime(KP_['Date'], format = '%Y %m %d %H.%f')
    
    df_kp = pd.DataFrame()
    df_kp['Date'] = KP_[str(df_station.index[0].date()):str(df_station.index[-1].date())].loc[KP_['Kp'] > kp].index.date
    df_kp['Date'] = df_kp['Date'].drop_duplicates()
    df_kp.index = df_kp['Date']
    df_kp = df_kp.dropna()
    
    df_disturbed = pd.DataFrame()
    
    for i in df_kp.index:
        df_disturbed = pd.concat([df_disturbed, df_station.loc[str(i)]])
        #df_station.drop(df_station.loc[str(i)].index, inplace = True)
    df_station.drop(df_disturbed.index, inplace = True)
    
    return df_station

def chaos_model_prediction(station: str,
                           starttime: str,
                           endtime: str
                           ):
    '''
    Compute the CHAOS-7.10 model geomagnetic field prediction for a INTERMAGNET observatory.
    
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
    chaos_model_prediction(station = 'VSS',
                           starttime = 'yyyy-mm-dd',
                           endtime = 'yyyy-mm-dd')
    
    ----------------------------------------------------------------------------------
    
    return a hourly dataframe with the X, Y and Z geomagnetic components CHAOS prediction for total field,
    internal field and external field.
    
    '''
    
    assert len(station) == 3, 'station must be a three letters IAGA Code'
    
    for i in [starttime, endtime]:
        spf.validate(i)
        
    if utt.IMO.check_existence(station) is False:
        raise ValueError(f'Station must be an observatory IAGA CODE!')
    
            
    working_directory = project_directory()
    
    #loading CHAOS model    
    chaos_path = glob.glob(os.path.join(working_directory,
                                        'chaosmagpy_package_*.*',
                                        'data',
                                        'CHAOS*'
                                        )
                           ) 

    model = cp.load_CHAOS_matfile(chaos_path[0])
    
    station = station.upper()
    
    rc_directory = pathlib.Path(os.path.join(working_directory,
                                             'Data',
                                             'chaos rc',
                                             'newest_RC_file.h5'
                                             )
                                )                        
    rc_data = h5py.File(rc_directory)
    
    if int(cp.data_utils.mjd2000(datetime.today())) != int(rc_data['time'][-1]):
        
        rc_data.close()
        save_RC_h5file(rc_directory)
        cp.basicConfig['file.RC_index'] = rc_directory
    else:
        rc_data.close()
        cp.basicConfig['file.RC_index'] = rc_directory    
    #setting the Earth radius reference
    R_REF = 6371.2

    #getting coordenates for the stations
    Longitude = utt.IMO.longitude(station)

    Latitude = 90 - utt.IMO.latitude(station)

    Elevation = (utt.IMO.elevation(station)/1000) +R_REF
    #Longitude = df_IMOS.loc[station]['Longitude']
    

    #Latitude = 90 - df_IMOS.loc[station]['Latitude']
 

    #Elevation = (df_IMOS.loc[station]['Elevation']/1000) + R_REF

    Date = pd.date_range(starttime, endtime + ' 23:00:00', freq = 'H')
    Time =cp.data_utils.mjd2000(Date)
    
    # Internal field
    print(f'Initiating geomagnetic field computation for {station.upper()}.')
    print(f'Computing core field.')
    B_core = model.synth_values_tdep(time = Time,
                                     radius = Elevation,
                                     theta = Latitude ,
                                     phi = Longitude,
                                     nmax = 20
                                    )

    print(f'Computing crustal field up to degree 110.')

    B_crust = model.synth_values_static(radius = Elevation,
                                        theta = Latitude,
                                        phi = Longitude,
                                        nmax = 110
                                       )
    
    # complete internal contribution
    B_radius_int = B_core[0] + B_crust[0]
    B_theta_int = B_core[1] + B_crust[1]
    B_phi_int = B_core[2] + B_crust[2]
    
    print('Computing field due to external sources, incl. induced field: GSM.')
    B_gsm = model.synth_values_gsm(time = Time,
                                   radius = Elevation, 
                                   theta = Latitude,
                                   phi = Longitude, 
                                   source='all',
                                   nmax = 2
                                   )
    
    B_sm = model.synth_values_sm(time = Time,
                                 radius = Elevation,
                                 theta = Latitude,
                                 phi = Longitude,
                                 source='all',
                                 nmax = 2,
                                 )

    #if endtime <= '2022-02-28':
    #    
    #    print('Computing field due to external sources, incl. induced field: SM.')
    #    B_sm = model.synth_values_sm(time = Time,
    #                             radius = Elevation,
    #                             theta = Latitude,
    #                             phi = Longitude,
    #                             source='all',
    #                             nmax = 2,
    #                             )
    #else:
    #    Date_RC = pd.date_range(starttime, '2022-02-28' + ' 23:00:00', freq = 'H')
    #    Date_NO_RC = pd.date_range('2022-03-01', endtime + ' 23:00:00', freq = 'H')
    #    Time =cp.data_utils.mjd2000(Date_RC)
    #    Time_sm =cp.data_utils.mjd2000(Date_NO_RC)
    #    
    #    print('Computing field due to external sources, incl. induced field: SM.')
    #    B_sm_with_rc = model.synth_values_sm(time = Time,
    #                             radius = Elevation,
    #                             theta = Latitude,
    #                             phi = Longitude,
    #                             source='all'
    #                             )
    #    
    #    B_sm_without_rc = model.synth_values_sm(time = Time_sm,
    #                             radius = Elevation,
    #                             theta = Latitude,
    #                             phi = Longitude,
    #                             source='all',
    #                             rc_e = False,
    #                             rc_i = False
    #                             )
    #    
    #    B_sm = []
    #    B_sm_x = np.append(B_sm_with_rc[0], B_sm_without_rc[0])
    #    B_sm_y = np.append(B_sm_with_rc[1], B_sm_without_rc[1])
    #    B_sm_z = np.append(B_sm_with_rc[2], B_sm_without_rc[2])
    #    B_sm = [B_sm_x, B_sm_y, B_sm_z]

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
    
    df_station['X_int'] = B_core[1].round(3)*-1
    df_station['Y_int'] = B_core[2].round(3)
    df_station['Z_int'] = B_core[0].round(3)*-1
    
    df_station['X_crust'] = B_crust[1].round(3)*-1
    df_station['Y_crust'] = B_crust[2].round(3)
    df_station['Z_crust'] = B_crust[0].round(3)*-1
    

    #df_station['X_ext_gsm'] = B_gsm[1].round(3)*-1
    #df_station['Y_ext_gsm'] = B_gsm[2].round(3)
    #df_station['Z_ext_gsm'] = B_gsm[0].round(3)*-1
    #
    #df_station['X_ext_sm'] = B_sm[1].round(3)*-1
    #df_station['Y_ext_sm'] = B_sm[2].round(3)
    #df_station['Z_ext_sm'] = B_sm[0].round(3)*-1
    
    df_station['X_ext'] = B_theta_ext.round(3)*-1
    df_station['Y_ext'] = B_phi_ext.round(3)
    df_station['Z_ext'] = B_radius_ext.round(3)*-1
    
    return df_station 
        
def external_field_correction_chaos_model(station: str,
                                          starttime: str = None,
                                          endtime: str = None,
                                          df_station = None,
                                          df_chaos = None,
                                          files_path = None,
                                          apply_percentage: bool = False
                                          ):    
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
    
    #validant inputs
    assert len(station) == 3, 'station must be a three letters IAGA Code'
    
    assert isinstance(df_station, pd.DataFrame) or df_station == None, 'df_station must be a pandas dataframe or None'
    
    assert isinstance(df_chaos, pd.DataFrame) or df_chaos == None, 'df_station must be a pandas dataframe or None'
    
    assert isinstance(files_path, str) or files_path == None, 'files_path must be a string or None'
    
    assert isinstance(apply_percentage, bool), 'apply_percentage must be True or False'
    
    if not [i for i in (starttime, endtime) if i is None]:
        for i in [starttime, endtime]:
            spf.validate(i)
    else:
        if files_path is None:
            raise ValueError('if starttime and endtime are None, you must inform files_path.')   
    
    station = station.upper()
    
    if utt.IMO.check_existence(station) is False:
        raise ValueError(f'Station must be an observatory IAGA CODE!')  
    
    if df_station is not None:
            
        df_station = df_station.loc[starttime:endtime].copy()
        
    else:
        df_station = mvs.load_intermagnet_files(station = station,
                                                starttime = starttime,
                                                endtime = endtime,
                                                files_path = files_path
                                                )
        
        df_station = df_station.loc[starttime:endtime]
    
    if starttime is None and endtime is None:
        starttime = str(df_station.index[0].date())
        endtime = str(df_station.index[-1].date())
        
    
    if df_chaos is not None:

        df_chaos = df_chaos.loc[starttime:endtime].copy()
    
    else:
        
        df_chaos = chaos_model_prediction(station = station,
                                          starttime = starttime,
                                          endtime = endtime
                                         )
        
    #df_chaos.index = df_chaos.index + to_offset('30min')
        
        
    #df_chaos['X_ext'] = df_chaos['X_ext_gsm'] + df_chaos['X_ext_sm']
    #df_chaos['Y_ext'] = df_chaos['Y_ext_gsm'] + df_chaos['Y_ext_sm']
    #df_chaos['Z_ext'] = df_chaos['Z_ext_gsm'] + df_chaos['Z_ext_sm']
    
    df_station = df_station.resample('H').mean()
        
    df_station['X'] = df_station['X'] - df_chaos['X_ext']
    df_station['Y'] = df_station['Y'] - df_chaos['Y_ext']
    df_station['Z'] = df_station['Z'] - df_chaos['Z_ext'] 
    
    df_station = resample_obs_data(df_station,
                                   'H',
                                   apply_percentage = apply_percentage
                                   )   
    
    print('The external field predicted using CHAOS-model was removed from the data.')
    
    return df_station, df_chaos    

def rms(predictions: pd.DataFrame(),
        observed_data: pd.DataFrame()
        ):
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
    
    #columns = ['X_int','Y_int','Z_int']
    
    assert isinstance(predictions, pd.DataFrame), 'predictions must be a pandas dataframe'
    
    assert isinstance(observed_data, pd.DataFrame) or observed_data is None, 'observed_data must be a pandas dataframe'
    x = []
    
    for col,cols in zip(['X_int', 'Y_int', 'Z_int'], observed_data.columns):
        
        y = pd.DataFrame()
        ypred = pd.DataFrame()
        
        y = calculate_sv(observed_data, apply_percentage=True)
        ypred = calculate_sv(predictions, source = 'int')
        
        #y = (observed_data[cols].resample('M').mean().diff(6) - observed_data[cols].resample('M').mean().diff(-6)).dropna()
        #print(y)
        #ypred = (predictions[col].resample('M').mean().diff(6) - predictions[col].resample('M').mean().diff(-6)).dropna()
        ypred = ypred.reindex(y.index)
        
        rms = np.sqrt(((ypred[col] - y[cols]) ** 2).mean()).round(3)
        #rms2 = mean_squared_error(y[cols].values, ypred[col].values, squared = False)
        
        x.append(rms)
        #x2.append(rms2)
        #print('the rmse for ' + str(cols) + ' component is ' + str(rms) + '.')
    return x

def night_time_selection(station: str,
                         dataframe: pd.DataFrame(),
                         h_min:int = 5,
                         h_max:int = 23
                         ):
    
    '''
    Function to select the night time period (from 23 PM to 5 AM) from the geomagnetic data.
     
    ---------------------------------------------------------------------
    Inputs:
    
    station (str) - 3 letters IAGA code for a INTERMAGNET observatory.
    
    dataframe (pd.DataFrame()) - a pandas dataframe with geomagnetic data.
    
    h_min (int) - minimun hour for the nighttime interval (Default is 5)
    
    h_max (int) - maximun hour for the nighttime interval (Default is 23)
    ---------------------------------------------------------------------------
        
    Example of use:
    night_time_selection(station = 'VSS',
                         dataframe = name_of_dataframe)
    
    ------------------------------------------------------------------------------
    
    return a dataframe with only the night time period.
    
    '''
    assert len(station) == 3, 'station must be a three letters IAGA Code'
    
    assert isinstance(dataframe, pd.DataFrame), 'dataframe must be a pandas dataframe'
    
    if utt.IMO.check_existence(station) is False:
        raise ValueError(f'Station must be an observatory IAGA CODE!') 
        
    station = station.upper()    

    Longitude = utt.IMO.longitude(station)
    
    time_to_shift =  Longitude/15

    df_obs = dataframe
    
    df_lt = df_obs.shift(round(time_to_shift, 2), freq = 'H')
    
    df_NT_lt = pd.DataFrame()
    df_NT_lt = df_lt.drop(df_lt.loc[(df_lt.index.hour > h_min) & (df_lt.index.hour < h_max)].index).dropna()
    
    df_NT = pd.DataFrame()
    df_NT = df_NT_lt.shift(round(-time_to_shift, 2), freq = 'H')
    
    print('The night time period was selected.')
    return df_NT

def hampel_filter_denoising(dataframe: pd.DataFrame(),
                            window_size: int,
                            n_sigmas=3,
                            plot_figure:bool = False,
                            apply_percentage = False
                            ):
    '''

    
    ------------------------------------------------------------------------------------
    
    Inputs:
    
    dataframe - a pandas dataframe with geomagnetic data. 
    
    window_size - integer, size of the moving window to calculate the absolute median
    
    n_sigmas - Number of standard deviations to be consider as a outlier
    
    plot_figure - boolean, option to plot a comparison between real and denoised data.
    
    ------------------------------------------------------------------------------------
    
    Return a hourly dataframe denoised 
    '''
    
    #validating the inputs
    
    assert isinstance(dataframe, pd.DataFrame), 'dataframe must be a pandas dataframe.'
    
    assert isinstance(window_size, int), 'window_size must be an integer.'
    
    dataframe = resample_obs_data(dataframe,'H',
                                  apply_percentage = apply_percentage
                                  )
    
    assert isinstance(plot_figure, bool), 'plot_figure must be True or False'
    
    assert isinstance(apply_percentage, bool), 'apply_percentage must be True or False'
    
    df_denoised = dataframe.copy()

    print('Denoising the data')

    for column in dataframe:
        
        n = len(dataframe[column])
        #denoised_dataframe = dataframe.copy()
        k = 1.4826 # scale factor for Gaussian distribution
        
        for i in range((window_size),(n - window_size)):
            x0 = np.median(dataframe[column][(i - window_size):(i + window_size)])
            S0 = k * np.median(np.abs(dataframe[column][(i - window_size):(i + window_size)] - x0))
            if (np.abs(dataframe[column][i] - x0) > n_sigmas * S0):
                df_denoised[column][i] = x0
                
    if plot_figure is True:
        
        fig, axis = plt.subplots(3 ,1, figsize = (16,10))
        for col, ax in zip(dataframe.columns, axis.flatten()):
            
            ax.plot(dataframe[col], 'k', label = 'Removed Outliers')
            ax.plot(df_denoised[col], 'r', label = 'Denoised ' + col)
            ax.set_xlim(dataframe[col].index[0], dataframe[col].index[-1])
            ax.legend(loc='best', fontsize = 12)
            ax.grid()
        plt.show()
               
    else:
        pass
        
    return df_denoised

def resample_obs_data(dataframe: pd.DataFrame(),
                      sample: str,
                      apply_percentage:bool = False
                      ):
    '''
    Resample a pd.DataFrame to hourly, daily, monthly or annual means
    
    The new sample is set in the middle of the sample range. 
    
    Example daily mean is set in the middle of the day, 12h.
    
    ------------------------------------------------------------------------
    Inputs:
    
    dataframe - a pandas dataframe with geomagnetic data. 
    
    sample - string, must be 'min','H','D','M' or 'Y'
             *min - Minute mean data
             *H - Hourly mean data
             *D - Daily mean data
             *M - Monthly mean data
             *Y - Annual mean data
             
    apply_percentage - a condition to resample the data. If True, must have at least 90%
    of data availability in the interval to be resampled. If False, all the intervals are resampled.
    
    ---------------------------------------------------------------------
    Use example:
    resample_obs_data(dataframe = my_data,
                      sample = 'M',
                      apply_percentage = False)
    
    Return a pandas dataframe converted for the selected sample
    '''
    
    assert isinstance(dataframe, pd.DataFrame), 'dataframe must be a pandas dataframe.'
    
    assert isinstance(apply_percentage, bool), 'apply_percentage must be True or False'
    
    samples = ['min', 'H', 'D',
               'M', 'Y'
              ]
    
    assert sample in samples, 'sample must be one of %s' % samples
    
    df_station = dataframe
        
    if sample == 'min' and apply_percentage == False:
        
        df_station = df_station

    if sample == 'min' and apply_percentage == True:
        
        df_station = df_station
        
    if sample == 'H' and apply_percentage == False:
            
        df_station = df_station.resample('H').mean()
        df_station.index = df_station.index + to_offset('30min')
            
    if sample == 'H' and apply_percentage == True:
                          
        tmp = df_station.groupby(pd.Grouper(freq='H')).agg(['mean','count']).swaplevel(0,1,axis=1)
            
        if any(tmp['count'].median()) <= 1 == True:

            df_station = df_station.resample('H').mean()
                
            df_station = tmp['mean'].where(tmp['count']>=1*0.9)
        else:
            df_station = tmp['mean'].where(tmp['count']>=60*0.9)
            
            df_station = df_station.resample('H').mean()

        df_station.index = df_station.index + to_offset('30min')
            
    if sample == 'D' and apply_percentage == False:
            
        df_station = df_station.resample('D').mean()
        df_station.index = df_station.index + to_offset('12H')       
        
    elif sample == 'D' and apply_percentage == True:
        
        tmp = df_station.groupby(pd.Grouper(freq='D')).agg(['mean','count']).swaplevel(0,1,axis=1)
        
        if any(tmp['count'].median() <= 30) == True:
            
            df_station = df_station.resample('H').mean()
            
            df_station = tmp['mean'].where(tmp['count']>=24*0.9)
            
        else:
        
            df_station = tmp['mean'].where(tmp['count']>=1440*0.9)
        
        df_station = df_station.resample('D').mean()
        df_station.index = df_station.index  + to_offset('12H')
            
    if sample == 'M' and apply_percentage == False:
        
        
        df_station = df_station.resample('M').mean()
        idx1 = df_station.loc[df_station.index.month == 2].index + to_offset('-1M') + to_offset('14D')
        idx2 = df_station.loc[df_station.index.month != 2].index + to_offset('-1M') + to_offset('15D')
        df_station.index = idx1.union(idx2)
        
    if sample == 'M' and apply_percentage == True:
        
        tmp = df_station.groupby(pd.Grouper(freq='M')).agg(['mean','count']).swaplevel(0,1,axis=1)
        
        if any(tmp['count'].median() <= 800) == True:
            
            df_station = df_station.resample('H').mean()

            tmp['full day'] = df_station.resample('M').mean().index.days_in_month*24
            
        else:
        
            tmp['full day'] = df_station.resample('M').mean().index.days_in_month*24*60
        #print(tmp['count'].median())    
        #print(tmp['full day'])    
        X = tmp['mean','X'].loc[tmp['count','X'] >= tmp['full day']*0.9]
        Y = tmp['mean','Y'].loc[tmp['count','Y'] >= tmp['full day']*0.9]
        Z = tmp['mean','Z'].loc[tmp['count','Z'] >= tmp['full day']*0.9]
        
            
            
        df_station = df_station.resample('M').mean()
        #print(df_station_resampled)
        df_station['X'] = X
        df_station['Y'] = Y
        df_station['Z'] = Z
        
        #df_station = df_station_resampled
        df_station = df_station.resample('M').mean()
        idx1 = df_station.loc[df_station.index.month == 2].index + to_offset('-1M') + to_offset('14D')
        idx2 = df_station.loc[df_station.index.month != 2].index + to_offset('-1M') + to_offset('15D')
        df_station.index = idx1.union(idx2)
            
            
    if sample == 'Y' and apply_percentage == False:
        
        df_station = df_station.resample('Y').mean()
        leap_year = []
        
        for years in df_station.index.year:
            
            if spf.year.check_leap_year(years) == True:
                
                leap_year.append(years)
                
        idx_leap = df_station.loc[df_station.index.year.isin(leap_year)].index + to_offset('-6M') + to_offset('+1D') + to_offset('23H') + to_offset('59min') + to_offset('30s')
        idx_normal = df_station.loc[~df_station.index.year.isin(leap_year)].index + to_offset('-6M') + to_offset('+2D') + to_offset('11H') + to_offset('59min') + to_offset('30s')        
        df_station.index = idx_normal.union(idx_leap)
        
    if sample == 'Y' and apply_percentage == True:
        
        Days = df_station.groupby(pd.Grouper(freq='M')).agg(['count']).swaplevel(0,1,axis=1)
        Days['Days'] = df_station.resample('M').mean().index.days_in_month
        tmp = df_station.groupby(pd.Grouper(freq='Y')).agg(['mean','count']).swaplevel(0,1,axis=1)
        tmp['Days'] = Days['Days'].resample('Y').sum()
        
        if tmp['count'].median().any() <= 8784:
            
            df_station = df_station.resample('H').mean()
            
            X = tmp['mean','X'].loc[tmp['count','X'] >= tmp['Days']*0.60*24]
            Y = tmp['mean','Y'].loc[tmp['count','Y'] >= tmp['Days']*0.60*24]
            Z = tmp['mean','Z'].loc[tmp['count','Z'] >= tmp['Days']*0.60*24]
        else:
            X = tmp['mean','X'].loc[tmp['count','X'] >= tmp['Days']*0.60*24*60]
            Y = tmp['mean','Y'].loc[tmp['count','Y'] >= tmp['Days']*0.60*24*60]
            Z = tmp['mean','Z'].loc[tmp['count','Z'] >= tmp['Days']*0.60*24*60]
        
        
        df_station['X'] = X
        df_station['Y'] = Y
        df_station['Z'] = Z
        df_station = df_station.resample('Y').mean()
        leap_year = []
        
        for years in df_station.index.year:
            
            if spf.year.check_leap_year(years) == True:
                leap_year.append(years)
                
        idx_leap = df_station.loc[df_station.index.year.isin(leap_year)].index + to_offset('-6M') + to_offset('+1D') + to_offset('23H') + to_offset('59min') + to_offset('30s')
        idx_normal = df_station.loc[~df_station.index.year.isin(leap_year)].index + to_offset('-6M') + to_offset('+2D') + to_offset('11H') + to_offset('59min') + to_offset('30s')        
        df_station.index = idx_normal.union(idx_leap)
        
    return df_station

def jerk_detection_window(station: str,
                          window_start: str, 
                          window_end: str, 
                          starttime: str = None, 
                          endtime: str = None,
                          df_station = None,
                          df_chaos = None,
                          files_path = None,
                          plot_detection: bool = True,
                          chaos_correction: bool = True,
                          plot_chaos_prediction:bool = False,
                          convert_hdz_to_xyz:bool = False,
                          save_plots:bool = False
                          ):
    '''
    Geomagnetic jerk detection based on two linear segments adoption in a
    chosen time window.
    
    ----------------------------------------------------------
    Inputs:
    
    station - String, 3 letters IAGA code.
    
    window_start - first day of the geomagnetic jerk window (format = 'yyyy-mm-dd)
    
    window_end - last day of the geomagnetic jerk window (format = 'yyyy-mm-dd)
    
    starttime - first day of the data (format = 'yyyy-mm-dd)
    
    endtime - last day of the data (format = 'yyyy-mm-dd)
    
    df_station - Must be a pandas dataframe with the geomagnetic data or None.
                 If None, the data will be computed using load_INTERMAGNET_files.
    
    df_chaos - Must be a pandas dataframe with the predicted CHAOS geomagnetic data or None.
               If None, the data will be computed using chaos_model_prediction.
    
    plot_detection - boolean (True or False). If True, the jerk detection will be shown.
    
    chaos_correction - boolean (True or False). If True, the geomagnetic data from 'df_station'
                       will be corrected using the CHAOS-model prediciton
                       
    plot_chaos_prediction - boolean (True or False). If True, the SV from CHAOS will be shown.
    
    -----------------------------------------------------------------------
    Use example:
        jerk_detection_window(station = 'ngk',
                              window_start = '2012-01-15', 
                              window_end = '2018-01-15', 
                              starttime = '2005-01-01', 
                              endtime = '2021-06-30',
                              df_station = None,
                              df_chaos = None,
                              plot_detection = True,
                              chaos_correction = True,
                              plot_chaos_prediction=True)
                              
    -----------------------------------------------------------------------
    Return - plots of the SV and the jerk detection for X, Y and Z.
             Jerk occurence time (t0)
             Jerk amplitude
             R² between the data and the linear segments.
    '''
    
    #validating the inputs
    
    if not [i for i in (starttime, endtime) if i is None]:
        for i in [starttime, endtime]:
            spf.validate(i)
    else:
        if files_path is None:
            raise ValueError('if starttime and endtime are None, you must inform files_path.') 

    for i in [window_start, window_end]:
        spf.validate_ym(i)
        
    assert len(station) == 3, 'station must be a three letters IAGA Code'
    
    assert isinstance(df_station, pd.DataFrame) or df_station == None, 'df_station must be a pandas dataframe or None'
    
    assert isinstance(df_chaos, pd.DataFrame) or df_chaos == None, 'df_station must be a pandas dataframe or None'
     
    for check_bool in [plot_detection, plot_chaos_prediction,
                       convert_hdz_to_xyz, save_plots]:
        
        assert isinstance(check_bool, bool) , f'{check_bool} must be True or False'
         
    station = station
    window_start = window_start + '-15'
    window_end = window_end + '-15'
    
    working_directory = project_directory()
    
    directory = pathlib.Path(os.path.join(working_directory,
                                          'Filtered_data',
                                          f'{station}_data'
                                          )
                             )
           
    #directory = f'Filtered_data/{station}_data'
    
    df_chaos = df_chaos
    
    # creating directory if it doesn't exist
    
    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
    
    if df_station is not None:
        
        df_station = df_station
    
    #computing dataframe from observatory files
    else:
        df_station = mvs.load_intermagnet_files(station = station,
                                                starttime = starttime,
                                                endtime = endtime,
                                                files_path = files_path
                                                )

    if starttime is None and endtime is None:
        starttime = str(df_station.index[0].date())
        endtime = str(df_station.index[-1].date())
        
    #cheking existence of HDZ components
    if convert_hdz_to_xyz == True:    
        df_station = utt.hdz_to_xyz_conversion(station = station,
                                               dataframe = df_station,
                                               files_path = files_path
                                               )
    else: 
        pass
    
    # conditions to load CHAOS dataframe
    if df_chaos is not None:
        
        df_chaos = df_chaos
    
    if chaos_correction is True and df_chaos is not None:
        
        df_station, df_chaos = external_field_correction_chaos_model(station = station,
                                                                     starttime = starttime,
                                                                     endtime = endtime,
                                                                     df_station = df_station,
                                                                     df_chaos= df_chaos,
                                                                     files_path = None
                                                                     )
        
    elif chaos_correction is True and df_chaos is None:
        df_station, df_chaos = external_field_correction_chaos_model(station = station,
                                                                     starttime = starttime,
                                                                     endtime = endtime,
                                                                     df_station = df_station,
                                                                     df_chaos = None,
                                                                     files_path = None
                                                                     )
    
    #calculating SV from intermagnet files
    df_sv = calculate_sv(dataframe = df_station,
                         method = 'ADMM',
                         source = None
                         )
    
    #if plot_chaos_prediction == False:
    #    
    #    pass
    
    if chaos_correction is True and plot_chaos_prediction is True:
        
        df_chaos_sv = calculate_sv(dataframe = df_chaos,
                                   method = 'ADMM',
                                   source = 'int'
                                   )
    else:
        
        pass
    
    
    df_jerk_window = pd.DataFrame()
    
    df_jerk_window.index = df_sv.loc[window_start:window_end].index
    
    date_jerk= [spf.date_to_decinal_year_converter(date) for date in df_jerk_window.index]
    
    breakpoints = pd.DataFrame()
    #starting with window jerk detection
    
    df_slopes = pd.DataFrame()
    #rsq = pd.DataFrame()
    df_rsq = pd.DataFrame()

    #eqn_list = []
    r2 = []
    for column in df_sv.columns:

        myPWLF = pwlf.PiecewiseLinFit(date_jerk, df_sv.loc[window_start:window_end][column])
        
        breakpoints[column] = myPWLF.fit(2)
        
        #calculate slopes
        slopes = myPWLF.calc_slopes()
        df_slopes[column] = slopes
        
        #calculate r_squared
        
        r2.append(myPWLF.r_squared().round(2))
        
        #se = myPWLF.se
        #print(se)
        xHat = date_jerk
        yHat = myPWLF.predict(xHat)
        
        df_jerk_window[str(column)] = yHat
        
        print('\n' + station.upper() + ' Jerk statistics for the ' + column + ' component.')
        print('\nJerk occurence time -t0-: ' + str(breakpoints[column][1].round(2)))
        print('Jerk amplitute: ' + str(df_slopes[column].diff()[1].round(2)))
        print('R^2: ' + str((myPWLF.r_squared()).round(2)))
        print('\n****************************************************************')
        #for i in range(myPWLF.n_segments):
        #    eqn_list.append(get_symbolic_eqn(myPWLF, i + 1))
        #    #print('Equation number: ',(i + 1) + 'for ' + str(column) + 'component')
        #    print(eqn_list[-1])
        #    #f_list.append(lambdify(x, eqn_list[-1]))
    if plot_detection == False:
        pass
    else:

        #plotting single figure

        if plot_detection is True and plot_chaos_prediction is False or chaos_correction is False:
            colors = ['blue', 'green', 'black']
            fig, axes = plt.subplots(3,1,figsize = (12,8), sharex = True)
            plt.subplots_adjust(hspace=0.05)
            plt.suptitle(f'{station.upper()} secular variation', fontsize = 12, y = 0.94)
            plt.xlabel('Date (Years)', fontsize = 12)
            
            for col, ax, color in zip(df_sv.columns, axes.flatten(), colors):
                ax.plot(df_sv[col],
                        'o',
                        color = color
                        )
                ax.plot(df_jerk_window[col].index,
                        df_jerk_window[col],
                        color = 'red',
                        linewidth = 3,
                        label = 'jerk detection'
                        ) 
                        #label = 'JOT ' + 
                        #str(round((df_jerk_window.index[int(z[col][1].round())].year+
                        #           (df_jerk_window.index[int(z[col][1].round())].dayofyear -1)/365),2)))
                ax.set_ylabel(f'd{col.upper()}/dt (nT)', fontsize = 12)
                ax.set_xlim(df_sv[col].index[0], df_sv[col].index[-1])
                ax.xaxis.set_major_locator(md.MonthLocator(interval=12)) 
                ax.xaxis.set_major_formatter(md.DateFormatter('%Y-%m'))
                ax.xaxis.get_ticklocs(minor=True)
                ax.yaxis.set_tick_params(which='minor', bottom=False)
                ax.minorticks_on() 
                ax.grid(alpha = 0.5)
                ax.legend()
            if save_plots == True:
                
                plt.savefig(os.path.join(directory,
                                         f'{station}_jerk_detection.jpeg',
                                         bbox_inches='tight'
                                         )
                            )
                plt.show()
 
            #plotting multiple figure// small xaxis

            fig, axes = plt.subplots(1,3,figsize = (17,6))
            plt.suptitle(f'{station.upper()} secular variation',
                         fontsize = 12,
                         y = 0.94
                         )
            fig.text(0.5,
                     0.04,
                     'Years',
                     ha='center',
                     fontsize = 12
                     )
         
            upper_limit = int(str(datetime.strptime(window_end ,'%Y-%m-%d'))[0:4]) +1
            lower_limit = int(str(datetime.strptime(window_start ,'%Y-%m-%d'))[0:4]) -1
    
            for col, ax, color in zip(df_sv.columns, axes.flatten(), colors):
                
                ax.plot(df_sv[col].loc[str(lower_limit):str(upper_limit)],
                        'o'
                        )
                ax.plot(df_jerk_window[col],
                        linewidth = 3,
                        label = 'jerk detection'
                        )
                #ax.set_ylabel(f'd{col.upper()}/dt (nT)', fontsize = 12)
                ax.xaxis.set_major_locator(md.MonthLocator(interval=24)) 
                ax.xaxis.set_major_formatter(md.DateFormatter('%Y'))
                ax.xaxis.get_ticklocs(minor=True)
                ax.yaxis.set_tick_params(which='minor', bottom=False)
                ax.set_ylabel(f'd{col.upper()}/dt (nT)', fontsize = 12)
                ax.minorticks_on()
                ax.grid(alpha = 0.5)
            #fig.text(0.08, 0.5, f'dY/dt (nT/yr)', ha='center', va='center', rotation='vertical',fontsize = 10) 
            if save_plots == True:
                plt.savefig(os.path.join(directory,
                                         '{station}_jerk_detection_2.jpeg'
                                         ), 
                            dpi = 300,
                            bbox_inches='tight'
                            )
                plt.show()
                
            mvs.plot_tdep_map(time = str(spf.decimal_year_to_date(breakpoints['Y'][1].round(2))),
                              deriv = 2,
                              plot_changes = True,
                              station = [station.upper()]) 
        
        elif plot_detection is True and plot_chaos_prediction is True and chaos_correction is True:
            
            #plotting single figure

            colors = ['blue', 'green', 'black']
            
            fig, axes = plt.subplots(3,1,figsize = (12,8), sharex=True)
            plt.suptitle(f'{station.upper()} secular variation',
                         fontsize = 14,
                         y = 0.92)
            plt.subplots_adjust(hspace=0.05)
            plt.xlabel('Date (Years)', fontsize = 12)
            
            for col, chaos_col, ax, color in zip(df_sv.columns, df_chaos_sv.columns, axes.flatten(), colors):
                
                ax.plot(df_sv[col],
                        'o-',
                        color = color
                        )
                                
                ax.plot(df_chaos_sv[chaos_col],
                        linewidth = 2,
                        label = 'CHAOS prediction'
                        )
                
                ax.plot(df_jerk_window[col].index,
                        df_jerk_window[col],
                        color = 'red',
                        linewidth = 3,
                        label = 'jerk detection'
                        ) 
                        #label = 't0 ' + 
                        #str(round((df_jerk_window.index[int(z[col][1].round())].year+
                        #           (df_jerk_window.index[int(z[col][1].round())].dayofyear -1)/365),2)))
                ax.set_ylabel(f'd{col.upper()}/dt (nT)', fontsize = 12)
                ax.set_xlim(df_sv[col].index[0], df_sv[col].index[-1])
                ax.xaxis.set_major_locator(md.MonthLocator(interval=12)) 
                ax.xaxis.set_major_formatter(md.DateFormatter('%Y-%m'))
                ax.xaxis.get_ticklocs(minor=True)
                ax.yaxis.set_tick_params(which='minor', bottom=False)
                ax.minorticks_on()
                ax.grid(alpha = 0.5) 
                ax.legend()
                
            if save_plots == True:
                
                plt.savefig(os.path.join(directory,
                                         '{station}_jerk_detection.jpeg'
                                         ),
                            dpi= 300,
                            bbox_inches='tight')
                plt.show()
            
            #plotting multiple figure

            fig, axes = plt.subplots(1,3,figsize = (17,6))
            plt.suptitle(f'{station.upper()} secular variation', fontsize = 12, y = 0.93)
            fig.text(0.5, 0.04, 'Years', ha='center', fontsize = 12)
         
            upper_limit = int(str(datetime.strptime(window_end,'%Y-%m-%d'))[0:4]) +1
            lower_limit = int(str(datetime.strptime(window_start ,'%Y-%m-%d'))[0:4]) -1

            for col, chaos_col, ax, color in zip(df_sv.columns, df_chaos_sv, axes.flatten(), colors):
                
                ax.plot(df_sv[col].loc[str(lower_limit):str(upper_limit)],
                        'o'
                        )
                
                ax.plot(df_chaos_sv[chaos_col].loc[str(lower_limit):str(upper_limit)],
                        '-',
                        linewidth = 2,
                        label = 'CHAOS prediction'
                        )
                
                ax.plot(df_jerk_window[col],
                        linewidth = 3,
                        label = 'jerk detection'
                        )
                ax.xaxis.set_major_locator(md.MonthLocator(interval=24)) 
                ax.xaxis.set_major_formatter(md.DateFormatter('%Y')) 
                ax.set_ylabel(f'd{col.upper()}/dt (nT)', fontsize = 12)
                ax.xaxis.get_ticklocs(minor=True)
                ax.yaxis.set_tick_params(which='minor', bottom=False)
                ax.minorticks_on()
                ax.grid(alpha = 0.5) 
                ax.legend()
            #fig.text(0.08,
            #         0.5,
            #         f'dY/dt (nT/yr)',
            #         ha='center',
            #         va='center',
            #         rotation='vertical',
            #         fontsize = 10
            #         )
            
            if save_plots == True:
                plt.savefig(os.path.join(directory,
                                         f'{station}_jerk_detection_2.jpeg'),
                            dpi= 300,
                            bbox_inches='tight'
                            )
                plt.show()
            
            mvs.plot_tdep_map(time = str(spf.decimal_year_to_date(breakpoints['Y'][1].round(2))),
                              deriv = 2,
                              plot_changes = True,
                              station = [station.upper()])
        
    return df_jerk_window, df_slopes, breakpoints, r2
