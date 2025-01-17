import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas.tseries.frequencies import to_offset
import h5py
import glob
import pyIGRF
import os
import pathlib
import matplotlib.dates as md
from datetime import datetime, timedelta
import pwlf
import chaosmagpy as cp
import main_functions as mvs
import utilities_tools as utt
import support_functions as spf
from chaosmagpy.data_utils import save_RC_h5file


def project_directory():
    '''
    Get the project directory 
    '''
    return os.getcwd()

def remove_disturbed_days(dataframe: pd.DataFrame):
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
    
    assert isinstance(dataframe, pd.DataFrame), 'dataframe must be a pd.DataFrame()'
      
    config = spf.get_config()  
      
    df = dataframe
    
    dd_list_directory = pathlib.Path(os.path.join(config.directory.qd_dd,
                                                  config.filenames.disturbed_days
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
    
    df_d = df_d.loc[str(df.index[0].date()):str(df.index[-1].date())]
    
    mask = np.where(np.isin(df.index.date, np.unique(df_d.index.date)), True, False)

    df = df[~mask]
    
    print('Top 5 disturbed days for each month were removed from the data.')
    return df

def keep_quiet_days(dataframe: pd.DataFrame):
    
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
    
    assert isinstance(dataframe,pd.DataFrame), 'dataframe must be a pd.DataFrame()'
    
    config = spf.get_config() 
    
    df = dataframe
    
    qd_list_directory = pathlib.Path(os.path.join(config.directory.qd_dd,
                                                  config.filenames.quiet_days
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

    mask = np.where(np.isin(df.index.date, np.unique(df_q.index.date)), True, False)

    df[mask]

    if df_q.index[-1].date().strftime('%Y-%m') != (datetime.today().date() - timedelta(days=30)).strftime('%Y-%m'):
        spf.update_qd_and_dd(data = 'QD')
    
    df_q = df_q.loc[str(df.index[0].date()):str(df.index[-1].date())]

    mask = np.where(np.isin(df.index.date, np.unique(df_q.index.date)), True, False)

    df = df[mask]

    print('Only top 10 quiet days for each month were kept in the data.')
    return df

def calculate_sv(dataframe: pd.DataFrame,
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

def kp_index_correction(dataframe: pd.DataFrame,
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
    
    assert isinstance(kp, (int, float)), 'kp must be a number from 0 to 9'
    
    assert kp >= 0 and kp <= 9, 'kp must be between 0 and 9'

    df_station = dataframe.copy()
    
    if abs(df_station.index[1] - df_station.index[0]).seconds == 3600:
        sample_rate = 'H'
        df_station = df_station.resample('H').mean()
    else:
        sample_rate = 'min'
    
    config = spf.get_config()
    
    kp_directory = pathlib.Path(os.path.join(config.directory.kp_index,
                                             config.filenames.kp_index
                                             )
                                )
    
    KP_ = pd.read_csv(kp_directory,
                      sep = '\t',
                      index_col = ['Date'])
    
    
    KP_.index = pd.to_datetime(KP_.index, format = '%Y-%m-%d %H:%M:%S')
    
    if df_station.index[-1] > KP_.index[-1]:
        print('Updating the Kp index list.')
    
    #updating the Kp_index for the most recent data
        KP_ = pd.read_csv(config.url.kp_index,
                          skiprows = 30,
                          header = None,
                          sep = '\s+', 
                          usecols = [0,1,2,3,7,8],
                          parse_dates = {'Date': ['Y', 'M','D','H']},
                          names = ['Y','M','D','H','Kp','Ap']
                          )
    
        KP_.index = pd.to_datetime(KP_['Date'], format = '%Y %m %d %H.%f')
        
        KP_.pop('Date')
        
        KP_.to_csv(kp_directory,
                   sep = '\t',
                   index = ['Date'])
        
    kp_new_index = pd.date_range(str(df_station.index[0].date()),
                                 f'{str(df_station.index[-1].date())} 21:00:00',
                                 freq = sample_rate)
    
    KP_ = KP_.reindex(kp_new_index).ffill()
    
    KP_ = KP_.loc[KP_['Kp'] <= kp]
    
    df_station = df_station.reindex(KP_.index)
    
    if sample_rate == 'H':
        df_station = resample_obs_data(df_station, 'H')
    
    return df_station

def chaos_model_prediction(station: str,
                           starttime: str,
                           endtime: str,
                           n_core = 20,
                           n_crust = 110,
                           n_gsm = 2,
                           n_sm = 2
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
        raise ValueError(f'station must be an observatory IAGA CODE!')
            
    
    config = spf.get_config()
    
    spf.check_chaos_local_version()
    
    #loading CHAOS model    
    chaos_path = glob.glob(os.path.join(config.directory.chaos_model,
                                        'data',
                                        'CHAOS*'
                                        )
                           ) 

    model = cp.load_CHAOS_matfile(chaos_path[0])
    
    station = station.upper()
    
    rc_directory = pathlib.Path(os.path.join(config.directory.rc_index,
                                             config.filenames.rc_index
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

    colatitude = 90 - utt.IMO.latitude(station)
    
    elevation, colatitude, sd, cd = spf.gg_to_geo(utt.IMO.elevation(station)/1000, colatitude)

    if (pd.to_datetime(endtime).date() == datetime.today().date()) is True:
        Date = pd.date_range(starttime, datetime.utcnow().strftime(format = "%Y-%m-%d %H:00:00"), freq = 'H')
        Time = cp.data_utils.mjd2000(Date)
    else:
        Date = pd.date_range(starttime, f"{endtime} 23:00:00", freq = 'H')
        Time = cp.data_utils.mjd2000(Date)
    
    # Internal field
    print(f'Initiating geomagnetic field computation for {station.upper()}.')
    print(f'Computing core field.')
    B_core = model.synth_values_tdep(time = Time,
                                     radius = round(elevation, 2),
                                     theta = round(colatitude, 2) ,
                                     phi = Longitude,
                                     nmax = n_core
                                     )

    print(f'Computing crustal field up to degree 110.')

    B_crust = model.synth_values_static(radius = elevation,
                                        theta = colatitude,
                                        phi = Longitude,
                                        nmax = n_crust
                                        )         
    
    # complete internal contribution
    B_radius_int = B_core[0].astype('float32') + B_crust[0].astype('float32')
    B_theta_int = B_core[1].astype('float32') + B_crust[1].astype('float32')
    B_phi_int = B_core[2].astype('float32') + B_crust[2].astype('float32')
    
    print('Computing field due to external sources, incl. induced field: GSM.')
    B_gsm = model.synth_values_gsm(time = Time,
                                   radius = elevation, 
                                   theta = colatitude,
                                   phi = Longitude, 
                                   source='all',
                                   nmax = n_gsm
                                   )
    
    B_sm = model.synth_values_sm(time = Time,
                                 radius = elevation,
                                 theta = colatitude,
                                 phi = Longitude,
                                 source='all',
                                 nmax = n_sm,
                                 )

    # complete external field contribution
    B_radius_ext = B_gsm[0].astype('float32') + B_sm[0].astype('float32')
    B_theta_ext = B_gsm[1].astype('float32') + B_sm[1].astype('float32')
    B_phi_ext = B_gsm[2].astype('float32') + B_sm[2].astype('float32')

    # complete forward computation
    B_radius = B_radius_int + B_radius_ext
    B_theta = B_theta_int + B_theta_ext
    B_phi = B_phi_int + B_phi_ext
    

    df_station = pd.DataFrame()
    df_station.index = Date
    
    df_station['X_tot'] = (B_theta*-1)*cd + (B_radius*-1)*sd
    df_station['Y_tot'] = B_phi
    df_station['Z_tot'] = (B_radius*-1)*cd - (B_theta*-1)*sd    
    
    df_station['X_int'] = (B_core[1]*-1)*cd + (B_core[0]*-1)*sd
    df_station['Y_int'] = B_core[2]
    df_station['Z_int'] = (B_core[0]*-1)*cd - (B_core[1]*-1)*sd
    
    df_station['X_crust'] = (B_crust[1]*-1)*cd + (B_crust[0]*-1)*sd
    df_station['Y_crust'] = B_crust[2]
    df_station['Z_crust'] = (B_crust[0]*-1)*cd - (B_crust[1]*-1)*sd
    
    df_station['X_ext'] = (B_theta_ext*-1)*cd + (B_radius_ext*-1)*sd 
    df_station['Y_ext'] = B_phi_ext
    df_station['Z_ext'] = (B_radius_ext*-1)*cd - (B_theta_ext*-1)*sd
    
    return df_station 

def nighttime_selection_sz(station:str, df_station:pd.DataFrame):
    """_summary_

    Args:
        station (str): _description_
        df_station (pd.DataFrame): _description_
    """
    
    starttime = df_station.index[0]
    endtime = df_station.index[-1]

    df_sz = utt.get_solar_zenith(station, starttime, endtime)
    #df_sz = pd.DataFrame(index=df_station.index, columns={"sz":sz} )
    
    df_station['sun_p'] = df_sz['sz']
    df_station['sz'] = np.nan
    df_station.loc[(df_station['sun_p'] >=100),'sz'] = 1
    df_station.loc[(df_station['sun_p'] <100),'sz'] = 0
    df_station.pop('sun_p')
    df_station = df_station.loc[(df_station['sz'] == 1)]
    df_station.pop("sz")
    return df_station


def chaos_magnetospheric_field_prediction(station: str,
                                          starttime: str,
                                          endtime: str,
                                          n_gsm = 2,
                                          n_sm = 2,
                                          ):    
    '''
    Correct the INTERMAGNET observatory data with the CHAOS-7.11 model external geomagnetic field prediction.
    
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
    
    Return a hourly mean dataframe corrected from CHAOS-7 model external field
    
    
    '''
    
    #validant inputs
    assert len(station) == 3, 'station must be a three letters IAGA Code'
    
    
    for i in [starttime, endtime]:
        spf.validate(i) 
    
    station = station.upper()
    
    if utt.IMO.check_existence(station) is False:
        raise ValueError(f'station must be an observatory IAGA CODE!')  
        
        #df_station = df_station.loc[starttime:endtime]
    
    else:
        config = spf.get_config()
    
        spf.check_chaos_local_version()

        #loading CHAOS model    
        chaos_path = glob.glob(os.path.join(config.directory.chaos_model,
                                            'data',
                                            'CHAOS*'
                                            )
                               ) 

        model = cp.load_CHAOS_matfile(chaos_path[0])

        station = station.upper()

        rc_directory = pathlib.Path(os.path.join(config.directory.rc_index,
                                                 config.filenames.rc_index
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
            
        R_REF = 6371.2

        #getting coordenates for the stations
        Longitude = utt.IMO.longitude(station)

        colatitude = 90 - utt.IMO.latitude(station)

        elevation, colatitude, sd, cd = spf.gg_to_geo(utt.IMO.elevation(station)/1000, colatitude)

    if (pd.to_datetime(endtime).date() == datetime.today().date()) is True:
        Date = pd.date_range(starttime, datetime.utcnow().strftime(format = "%Y-%m-%d %H:00:00"), freq = 'H')
        Time = cp.data_utils.mjd2000(Date)
    else:
        Date = pd.date_range(starttime, f"{endtime} 23:00:00", freq = 'H')
        Time = cp.data_utils.mjd2000(Date)

    print('Computing field due to external sources, incl. induced field: GSM.')
    B_gsm = model.synth_values_gsm(time = Time,
                                   radius = elevation, 
                                   theta = colatitude,
                                   phi = Longitude, 
                                   source='all',
                                   nmax = n_gsm
                                   )
    B_sm = model.synth_values_sm(time = Time,
                                 radius = elevation,
                                 theta = colatitude,
                                 phi = Longitude,
                                 source='all',
                                 nmax = n_sm,
                                 )
        
    # complete external field contribution
    B_radius_gsm = B_gsm[0].astype('float32')
    B_theta_gsm = B_gsm[1].astype('float32')    
    B_phi_gsm = B_gsm[2].astype('float32') 
    
    B_radius_sm = B_sm[0].astype('float32')
    B_theta_sm = B_sm[1].astype('float32')
    B_phi_sm = B_sm[2].astype('float32')

    B_radius_ext = B_gsm[0].astype('float32') + B_sm[0].astype('float32')
    B_theta_ext = B_gsm[1].astype('float32') + B_sm[1].astype('float32')
    B_phi_ext = B_gsm[2].astype('float32') + B_sm[2].astype('float32')

    df_chaos = pd.DataFrame(index = Date)
  
    df_chaos["X_gsm"] = (B_theta_gsm*-1)*cd + (B_radius_gsm*-1)*sd
    df_chaos["X_sm"] = (B_theta_sm*-1)*cd + (B_radius_sm*-1)*sd
    
    df_chaos['Y_gsm'] = B_phi_gsm
    df_chaos['Y_sm'] = B_phi_sm
    
    df_chaos['Z_gsm'] = (B_radius_gsm*-1)*cd - (B_theta_gsm*-1)*sd
    df_chaos['Z_sm'] = (B_radius_sm*-1)*cd - (B_theta_sm*-1)*sd

    df_chaos['X_ext'] = (B_theta_ext*-1)*cd + (B_radius_ext*-1)*sd 
    df_chaos['Y_ext'] = B_phi_ext
    df_chaos['Z_ext'] = (B_radius_ext*-1)*cd - (B_theta_ext*-1)*sd
    
    return df_chaos

def chaos_core_field_prediction(station: str,
                                starttime: str,
                                endtime: str,
                                n_core = 20,
                                ): 

    assert len(station) == 3, 'station must be a three letters IAGA Code'
    
    for i in [starttime, endtime]:
        spf.validate(i)
        
    if utt.IMO.check_existence(station) is False:
        raise ValueError(f'station must be an observatory IAGA CODE!')
            
    
    config = spf.get_config()
    
    spf.check_chaos_local_version()
    
    #loading CHAOS model    
    chaos_path = glob.glob(os.path.join(config.directory.chaos_model,
                                        'data',
                                        'CHAOS*'
                                        )
                           ) 

    model = cp.load_CHAOS_matfile(chaos_path[0])
    
    station = station.upper()
    
    #setting the Earth radius reference
    R_REF = 6371.2

    #getting coordenates for the stations
    Longitude = utt.IMO.longitude(station)

    colatitude = 90 - utt.IMO.latitude(station)
    
    elevation, colatitude, sd, cd = spf.gg_to_geo(utt.IMO.elevation(station)/1000, colatitude)

    if (pd.to_datetime(endtime).date() == datetime.today().date()) is True:
        Date = pd.date_range(starttime, datetime.utcnow().strftime(format = "%Y-%m-%d %H:00:00"), freq = 'H')
        Time = cp.data_utils.mjd2000(Date)
    else:
        Date = pd.date_range(starttime, f"{endtime} 23:00:00", freq = 'H')
        Time = cp.data_utils.mjd2000(Date)
    
    # Internal field
    print(f'Initiating geomagnetic field computation for {station.upper()}.')
    print(f'Computing core field.')
    B_core = model.synth_values_tdep(time = Time,
                                     radius = round(elevation, 2),
                                     theta = round(colatitude, 2) ,
                                     phi = Longitude,
                                     nmax = n_core
                                     )        

    df_chaos = pd.DataFrame()
    df_chaos.index = Date
    
    df_chaos['X_int'] = (B_core[1]*-1)*cd + (B_core[0]*-1)*sd
    df_chaos['Y_int'] = B_core[2]
    df_chaos['Z_int'] = (B_core[0]*-1)*cd - (B_core[1]*-1)*sd
    
    return df_chaos 
        
def external_field_correction_chaos_model(station: str,
                                          starttime: str = None,
                                          endtime: str = None,
                                          df_station = None,
                                          df_chaos = None,
                                          n_gsm = 2,
                                          n_sm = 2,
                                          files_path = None,
                                          apply_percentage: bool = False
                                          ):    
    '''
    Correct the INTERMAGNET observatory data with the CHAOS-7.11 model external geomagnetic field prediction.
    
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
    
    assert isinstance(df_station, pd.DataFrame) or df_station is None, 'df_station must be a pandas dataframe or None'
    
    assert isinstance(df_chaos, pd.DataFrame) or df_chaos is None, 'df_station must be a pandas dataframe or None'
    
    assert isinstance(files_path, str) or files_path is None, 'files_path must be a string or None'
    
    assert isinstance(apply_percentage, bool), 'apply_percentage must be True or False'
    
    if not [i for i in (starttime, endtime) if i is None]:
        for i in [starttime, endtime]:
            spf.validate(i)
                
    if df_station is None and files_path is None:   
        raise ValueError('df_station is None, you must inform files_path.')   
    
    if [i for i in (starttime, endtime) if i is None] and df_station is not None:
        starttime = str(df_station.index[0].date())
        endtime = str(df_station.index[-1].date())
    
    station = station.upper()
    
    if utt.IMO.check_existence(station) is False:
        raise ValueError(f'station must be an observatory IAGA CODE!')  
    
    if df_station is not None:
            
        df_station = df_station
        
    else:
        df_station = mvs.load_intermagnet_files(station = station,
                                                starttime = starttime,
                                                endtime = endtime,
                                                files_path = files_path
                                                )
        
        #df_station = df_station.loc[starttime:endtime]
    
    if starttime is None and endtime is None:
        starttime = str(df_station.index[0].date())
        endtime = str(df_station.index[-1].date())
        
    
    if df_chaos is not None:

        df_chaos = df_chaos.loc[starttime:endtime].copy()
    
    else:
        
        df_chaos = chaos_magnetospheric_field_prediction(station = station,
                                                         starttime = starttime,
                                                         endtime = endtime,
                                                         n_gsm = n_gsm,
                                                         n_sm = n_sm  
                                                        )
    
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
        
        ypred = ypred.reindex(y.index)
        
        rms = np.sqrt(((ypred[col] - y[cols]) ** 2).mean()).round(3)
        
        x.append(rms)
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
    
    for i in [h_min, h_max]:
        assert isinstance(i, int), 'h_min and h_max must be integer'
    
    if utt.IMO.check_existence(station) is False:
        raise ValueError(f'station must be an observatory IAGA CODE!') 
        
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

def hampel_filter_denoising(dataframe: pd.DataFrame,
                            window_size: int = 100,
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
    
    if abs(df_station.index[1] - df_station.index[0]).seconds == 3600:
        sample_rate = 'hourly'
    else:
        sample_rate = 'minute'
        
    if sample == 'min' and apply_percentage is False:
        
        df_station = df_station

    if sample == 'min' and apply_percentage is True:
        
        df_station = df_station
        
    if sample == 'H' and apply_percentage is False:
            
        df_station = df_station.resample('H').mean()
        df_station.index = df_station.index + to_offset('30min')
            
    if sample == 'H' and apply_percentage is True:
                          
        tmp = df_station.groupby(pd.Grouper(freq='H')).agg(['mean','count']).swaplevel(0,1,axis=1)
            
        if sample_rate == 'hourly':
            
            df_station = df_station

        else:
            df_station = tmp['mean'].where(tmp['count']>=60*0.9)
            
            df_station = df_station.resample('H').mean()

        df_station.index = df_station.index + to_offset('30min')
            
    if sample == 'D' and apply_percentage is False:
            
        df_station = df_station.resample('D').mean()
        df_station.index = df_station.index + to_offset('12H')       
        
    elif sample == 'D' and apply_percentage is True:
        
        tmp = df_station.groupby(pd.Grouper(freq='D')).agg(['mean','count']).swaplevel(0,1,axis=1)
        
        if sample_rate == 'hourly':
            
            df_station = tmp['mean'].where(tmp['count']>=24*0.9)
            
        else:
        
            df_station = tmp['mean'].where(tmp['count']>=1440*0.9)
        
        df_station = df_station.resample('D').mean()
        df_station.index = df_station.index  + to_offset('12H')
            
    if sample == 'M' and apply_percentage is False:
        
        
        df_station = df_station.resample('M').mean()
        idx1 = df_station.loc[df_station.index.month == 2].index + to_offset('-1M') + to_offset('14D')
        idx2 = df_station.loc[df_station.index.month != 2].index + to_offset('-1M') + to_offset('15D')
        df_station.index = idx1.union(idx2)
        
    if sample == 'M' and apply_percentage is True:
        
        tmp = df_station.groupby(pd.Grouper(freq='M')).agg(['mean','count']).swaplevel(0,1,axis=1)
        
        if sample_rate == 'hourly':

            tmp['full day'] = df_station.resample('M').mean().index.days_in_month*24
            
        else:
        
            tmp['full day'] = df_station.resample('M').mean().index.days_in_month*24*60

        X = tmp['mean','X'].loc[tmp['count','X'] >= tmp['full day']*0.9]
        Y = tmp['mean','Y'].loc[tmp['count','Y'] >= tmp['full day']*0.9]
        Z = tmp['mean','Z'].loc[tmp['count','Z'] >= tmp['full day']*0.9]          
            
        df_station = df_station.resample('M').mean()
        df_station['X'] = X
        df_station['Y'] = Y
        df_station['Z'] = Z
        
        df_station = df_station.resample('M').mean()
        idx1 = df_station.loc[df_station.index.month == 2].index + to_offset('-1M') + to_offset('14D')
        idx2 = df_station.loc[df_station.index.month != 2].index + to_offset('-1M') + to_offset('15D')
        df_station.index = idx1.union(idx2)
            
            
    if sample == 'Y' and apply_percentage is False:
        
        df_station = df_station.resample('Y').mean()
        leap_year = []
        
        for years in df_station.index.year:
            
            if spf.year.check_leap_year(years) is True:
                
                leap_year.append(years)
                
        idx_leap = df_station.loc[df_station.index.year.isin(leap_year)].index + to_offset('-6M') + to_offset('+1D') + to_offset('23H') + to_offset('59min') + to_offset('30s')
        idx_normal = df_station.loc[~df_station.index.year.isin(leap_year)].index + to_offset('-6M') + to_offset('+2D') + to_offset('11H') + to_offset('59min') + to_offset('30s')        
        df_station.index = idx_normal.union(idx_leap)
        
    if sample == 'Y' and apply_percentage is True:
        
        Days = df_station.groupby(pd.Grouper(freq='M')).agg(['count']).swaplevel(0,1,axis=1)
        Days['Days'] = df_station.resample('M').mean().index.days_in_month
        tmp = df_station.groupby(pd.Grouper(freq='Y')).agg(['mean','count']).swaplevel(0,1,axis=1)
        tmp['Days'] = Days['Days'].resample('Y').sum()
        
        if sample_rate == 'hourly':
            
            df_station = df_station.resample('H').mean()
            
            X = tmp['mean','X'].loc[tmp['count','X'] >= tmp['Days']*0.90*24]
            Y = tmp['mean','Y'].loc[tmp['count','Y'] >= tmp['Days']*0.90*24]
            Z = tmp['mean','Z'].loc[tmp['count','Z'] >= tmp['Days']*0.90*24]
        else:
            X = tmp['mean','X'].loc[tmp['count','X'] >= tmp['Days']*0.90*24*60]
            Y = tmp['mean','Y'].loc[tmp['count','Y'] >= tmp['Days']*0.90*24*60]
            Z = tmp['mean','Z'].loc[tmp['count','Z'] >= tmp['Days']*0.90*24*60]
        
        
        df_station['X'] = X
        df_station['Y'] = Y
        df_station['Z'] = Z
        df_station = df_station.resample('Y').mean()
        leap_year = []
        
        for years in df_station.index.year:
            
            if spf.year.check_leap_year(years) is True:
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
                          apply_percentage:bool = False,
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
                
    if df_station is None and files_path is None:   
        raise ValueError('df_station is None, you must inform files_path.')   
    
    if [i for i in (starttime, endtime) if i is None] and df_station is not None:
        starttime = str(df_station.index[0].date())
        endtime = str(df_station.index[-1].date())

    for i in [window_start, window_end]:
        spf.validate_ym(i)
        
    assert len(station) == 3, 'station must be a three letters IAGA Code'
    
    assert isinstance(df_station, pd.DataFrame) or df_station is None, 'df_station must be a pandas dataframe or None'
    
    assert isinstance(df_chaos, pd.DataFrame) or df_chaos is None, 'df_station must be a pandas dataframe or None'
     
    for check_bool in [plot_detection, plot_chaos_prediction,
                       convert_hdz_to_xyz, save_plots]:
        
        assert isinstance(check_bool, bool) , f'{check_bool} must be True or False'
         
    station = station
    window_start = window_start + '-15'
    window_end = window_end + '-15'
    
    working_directory = project_directory()
    
    config = spf.get_config()
    
    directory = pathlib.Path(os.path.join(config.directory.filtered_data,
                                          f'{station}_data'
                                          )
                             )
    
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
    if convert_hdz_to_xyz is True:    
        df_station = utt.hdz_to_xyz_conversion(station = station,
                                               dataframe = df_station,
                                               files_path = files_path
                                               )
    
    # conditions to load CHAOS dataframe
    if df_chaos is not None:
        
        df_chaos = df_chaos
    
    if chaos_correction is True and df_chaos is not None:
        
        df_station, df_chaos = external_field_correction_chaos_model(station = station,
                                                                     starttime = starttime,
                                                                     endtime = endtime,
                                                                     df_station = df_station,
                                                                     df_chaos= df_chaos,
                                                                     files_path = None,
                                                                     apply_percentage = apply_percentage
                                                                     )
        
    elif chaos_correction is True and df_chaos is None:
        df_station, df_chaos = external_field_correction_chaos_model(station = station,
                                                                     starttime = starttime,
                                                                     endtime = endtime,
                                                                     df_station = df_station,
                                                                     df_chaos = None,
                                                                     apply_percentage = apply_percentage,
                                                                     files_path = None
                                                                     )
    
    
    if chaos_correction is True and plot_chaos_prediction is True:

        df_chaos_int = chaos_core_field_prediction(station,
                                                   starttime,
                                                   endtime)
        
        df_chaos_sv = calculate_sv(dataframe = df_chaos_int,
                                   method = 'ADMM',
                                   source = 'int',
                                   apply_percentage = False
                                   )
        
    
    #calculating SV from intermagnet files
    
    df_sv = calculate_sv(dataframe = df_station,
                         method = 'ADMM',
                         source = None,
                         apply_percentage = apply_percentage
                         )    

    df_jerk_window = pd.DataFrame()
    
    df_jerk_window.index = df_sv.loc[window_start:window_end].index
    
    date_jerk= [spf.date_to_decinal_year_converter(date) for date in df_jerk_window.index]
    
    breakpoints = pd.DataFrame()
    #starting with window jerk detection
    
    df_slopes = pd.DataFrame()
    r2 = []
    for column in df_sv.columns:

        myPWLF = pwlf.PiecewiseLinFit(date_jerk, df_sv.loc[window_start:window_end][column])
        
        breakpoints[column] = myPWLF.fit(2)
        
        #calculate slopes
        slopes = myPWLF.calc_slopes()
        df_slopes[column] = slopes
        
        #calculate r_squared
        
        r2.append(myPWLF.r_squared().round(2))
        
        xHat = date_jerk
        yHat = myPWLF.predict(xHat)
        
        df_jerk_window[str(column)] = yHat
        
        print('\n' + station.upper() + ' Jerk statistics for the ' + column + ' component.')
        print('\nJerk occurence time -t0-: ' + str(breakpoints[column][1].round(2)))
        print('Jerk amplitute: ' + str(df_slopes[column].diff()[1].round(2)))
        print('R^2: ' + str((myPWLF.r_squared()).round(2)))
        print('\n****************************************************************')

    if plot_detection is False:
        pass
    else:

        #plotting single figure

        if plot_detection is True and plot_chaos_prediction is False or chaos_correction is False:
            colors = ['blue', 'green', 'black']
            fig, axes = plt.subplots(3,1,figsize = (14,10), sharex = True)
            plt.subplots_adjust(hspace=0.1)
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

                ax.set_ylabel(f'd{col.upper()}/dt (nT)', fontsize = 12)
                ax.set_xlim(df_sv[col].index[0], df_sv[col].index[-1])
                ax.xaxis.set_major_locator(md.MonthLocator(interval=24)) 
                ax.xaxis.set_major_formatter(md.DateFormatter('%Y-%m'))
                ax.xaxis.get_ticklocs(minor=True)
                ax.yaxis.set_tick_params(which='minor', bottom=False)
                ax.minorticks_on() 
                ax.grid(alpha = 0.5)
                ax.legend()
                
            if save_plots is True:
                
                plt.savefig(os.path.join(directory,
                                         f'{station}_jerk_detection.jpeg'
                                         ),
                            bbox_inches='tight',
                            dpi = 300, 
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
                ax.legend()

            if save_plots is True:
                plt.savefig(os.path.join(directory,
                                         f'{station}_jerk_detection_2.jpeg',
                                         dpi = 300
                                         ), 
                            dpi = 300,
                            bbox_inches='tight'
                            )
            plt.show()

            if 'ccrs' in globals():
                
                mvs.plot_tdep_map(time = str(spf.decimal_year_to_date(breakpoints['Y'][1].round(2))),
                                  deriv = 2,
                                  plot_changes = True,
                                  station = [station.upper()]) 
        
        elif plot_detection is True and plot_chaos_prediction is True and chaos_correction is True:
            
            #plotting single figure

            colors = ['blue', 'green', 'black']
            
            fig, axes = plt.subplots(3,1,figsize = (14,10), sharex=True)
            plt.suptitle(f'{station.upper()} secular variation',
                         fontsize = 14,
                         y = 0.92)
            plt.subplots_adjust(hspace=0.1)
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
                ax.set_ylabel(f'd{col.upper()}/dt (nT)', fontsize = 12)
                ax.set_xlim(df_sv[col].index[0], df_sv[col].index[-1])
                ax.xaxis.set_major_locator(md.MonthLocator(interval=24)) 
                ax.xaxis.set_major_formatter(md.DateFormatter('%Y-%m'))
                ax.xaxis.get_ticklocs(minor=True)
                ax.yaxis.set_tick_params(which='minor', bottom=False)
                ax.minorticks_on()
                ax.grid(alpha = 0.5) 
                ax.legend()
                
            if save_plots is True:
                
                plt.savefig(os.path.join(directory,
                                         f'{station}_jerk_detection.jpeg'
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

            if save_plots is True:
                plt.savefig(os.path.join(directory,
                                         f'{station}_jerk_detection_2.jpeg'
                                         ),
                            dpi= 300,
                            bbox_inches='tight'
                            )
            plt.show()
            
            if 'ccrs' in globals():
                mvs.plot_tdep_map(time = str(spf.decimal_year_to_date(breakpoints['Y'][1].round(2))),
                                  deriv = 2,
                                  plot_changes = True,
                                  station = [station.upper()])
        
    return df_jerk_window, df_slopes, breakpoints, r2


if __name__ == '__main__':
    
    chaos_model_prediction("WNG", "2010-01-01", "2011-01-12")
    
    jerk_detection_window(station = 'ngk',
                          window_start = '2012-01', 
                          window_end = '2018-01', 
                          starttime = '2010-01-01', 
                          endtime = '2021-06-30',
                          files_path = "NGK",
                          plot_detection = True,
                          chaos_correction = True,
                          plot_chaos_prediction=True)

