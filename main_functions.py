import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import pathlib
from datetime import timedelta
import matplotlib.gridspec as gridspec
from datetime import datetime
import matplotlib.dates as md
from dateutil.relativedelta import relativedelta
import chaosmagpy as cp
import utilities_tools as utt
import data_processing_tools as dpt
import support_functions as spf
from concurrent.futures import ThreadPoolExecutor
import warnings
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import multiprocessing

try:  # make cartopy optional
    import cartopy.crs as ccrs
except ImportError:
    warnings.warn('Could not import Cartopy package. Jerk detection plots will be incomplete')

 

def project_directory():
    
    return os.getcwd()

def read_csv(filepath):
    
    
    skip_values = spf.skiprows_detection_v2(filepath)    
        
    #reading and concatenating the files

    df_station = pd.DataFrame()
    df_station = pd.read_csv(filepath,
                             sep='\s+',
                             usecols = [0, 1, 3, 4, 5], 
                             header = None,
                             skiprows = skip_values, 
                             parse_dates = {'Date': ['date', 'Time']},
                             names = ['date', 'Time', 'X', 'Y', 'Z']
                             )
    
    try:
        df_station['Date'] = pd.to_datetime(df_station['Date'], format = '%Y-%m-%dd %H:%M:%S.%f')
    except IOError:    
        df_station['Date'] = pd.to_datetime(df_station['Date'], infer_datetime_format=True)
    except:
        print('Wrong date format in the files')
              
    df_station.set_index('Date', inplace = True)

    # replacing 99999.0 values
    
    df_station.loc[df_station['X'] >= 99999.0, 'X'] = np.nan
    df_station.loc[df_station['Y'] >= 99999.0, 'Y'] = np.nan
    df_station.loc[df_station['Z'] >= 99999.0, 'Z'] = np.nan

    #df_station = df_station.sort_index().loc[starttime:endtime]
    
    try:
        df_station.pop('date')
        df_station.pop('Time')
    except:
        pass
    
def load_new_intermagnet(station, starttime, endtime, files_path):
    
    assert len(station) == 3, 'station must be a IAGA code with 3 letters'
    
    if not [i for i in (starttime, endtime) if i is None]:
        for i in [starttime, endtime]:
            spf.validate(i)
    else:
        if files_path is None:
            raise ValueError('if starttime and endtime are None, you must inform files_path.') 
           
    #checking the existence of the station argument
    
    if utt.IMO.check_existence(station) is False:
        raise ValueError(f'station must be an observatory IAGA CODE!')
   
    #creating a list to allocate the file paths
    
    files_station = []
       
    print(f'Reading files from {station.upper()}...')
    
    #default path
    if files_path is None:
        years_interval = np.arange(int(starttime[0:4]), int(endtime[0:4]) + 1)

        for year in years_interval:

            files_station.extend(glob.glob(f'C:\\Users\\marco\\Downloads\\Thesis_notebooks\\Dados OBS\\{str(year)}\\*\\{station}*'))

            files_station.sort()
    
    else:
    # if user set files_path 
        files_station.extend(glob.glob(os.path.join(f'{files_path}',
                                                    f'{station.lower()}*min*'
                                                    )
                                      )
                            )
    
        files_station.sort()
        if starttime is not None and endtime is not None:
            start_index = []
            end_index = []
            for file, i in zip(files_station, np.arange(0,len(files_station))):
                
                if pd.Timestamp(os.path.basename(file)[3:11]).date() == pd.Timestamp(starttime).date():
                    start_index = i
                if pd.Timestamp(os.path.basename(file)[3:11]).date() == pd.Timestamp(endtime).date():
                    end_index = i
            if start_index is []:
                files_station = files_station[:end_index]
            if end_index is []:
                files_station = files_station[start_index:]
            else:
                try:
                    files_station = files_station[start_index:end_index]
                except:
                    files_station = files_station
    
    
    with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        dfs = list(executor.map(read_csv, [os.path.join(files_path, f) for f in files_station]))

# Concatenate all DataFrames into a single DataFrame
    result_df = pd.concat(dfs, ignore_index=True)   
    
    return result_df

def determine_skiprows(file_path, keyword="DATE"):
    with open(file_path, 'r') as file:
        for line_number, line in enumerate(file, start=1):
            if line.startswith(keyword):
                return line_number
    return 0  # Return 0 if the keyword is not found


def load_intermagnet_files(station: str,
                           starttime: str = None,
                           endtime: str = None,
                           files_path: str = None
                           ) -> pd.DataFrame():

    '''
    
    Function to read and concat observatory data.
        Works with every INTERMAGNET Observatory.
    ----------------------------------------------------------
    Data types:
    
    Quasi-definitive and definitive data.
    ----------------------------------------------------------
    Inputs:
    
    station (str) - 3 letters IAGA code
    
    starttime (str, None) - first day of the data (format = 'yyyy-mm-dd', str) or None
    
    endtime (str, None) - last day of the data (format = 'yyyy-mm-dd', str) or None
    
    files_path - path to the IAGA-2002 intermagnet files (str) or None
                 if None it will use the default path for the files
    
    if starttime and endtime are None, all the files will be readed.
    ----------------------------------------------------------
    Example of use:
    
    load_intermagnet_files(station = 'VSS',
                           starttime = '2000-01-25',
                           endtime = '2021-12-31',
                           files_path = 'path//to//files')
    
    ------------------------------------------------------------------------------------------
    
    Return a pandas DataFrame of all readed data with X, Y and Z components.
    
    '''
    #Validating the inputs
    assert len(station) == 3, 'station must be a IAGA code with 3 letters'
    
    if not [i for i in (starttime, endtime) if i is None]:
        for i in [starttime, endtime]:
            spf.validate(i)
    else:
        if files_path is None:
            raise ValueError('if starttime and endtime are None, you must inform files_path.') 
           
    #checking the existence of the station argument
    
    if utt.IMO.check_existence(station) is False:
        raise ValueError(f'station must be an observatory IAGA CODE!')
   
    #creating a list to allocate the file paths
    
    files_station = []
       
    print(f'Reading files from {station.upper()}...')
    
    #default path
    if files_path is None:
        years_interval = np.arange(int(starttime[0:4]), int(endtime[0:4]) + 1)

        for year in years_interval:

            files_station.extend(glob.glob(f'C:\\Users\\marco\\Downloads\\Thesis_notebooks\\Dados OBS\\{str(year)}\\*\\{station}*'))

            files_station.sort()
    
    else:
    # if user set files_path 
        files_station.extend(glob.glob(os.path.join(files_path,
                                                    f'{station.lower()}*min*'
                                                    )
                                      )
                            )
    
        files_station.sort()
        
        if starttime is not None and endtime is not None:
            
            for file in files_station:
                if (pd.Timestamp(os.path.basename(file)[3:11]).date() < pd.Timestamp(starttime).date()) or (pd.Timestamp(os.path.basename(file)[3:11]).date() > pd.Timestamp(endtime).date()):
                    files_station.remove(file)           
    
    skip_values = spf.skiprows_detection(files_station)    

    df_station = pd.DataFrame()
    df_station = pd.concat((pd.read_csv(file,
                                        sep='\s+',
                                        usecols = [0, 1, 3, 4, 5], 
                                        header = None,
                                        skiprows = skiprows, 
                                        parse_dates = {'Date': ['date', 'Time']},
                                        names = ['date', 'Time', 'X', 'Y', 'Z']
                                        ) for skiprows,
                                              file in zip(skip_values[0], skip_values[1])
                                              ), 
                                              ignore_index = True
                           )

    try:
        df_station['Date'] = pd.to_datetime(df_station['Date'], format = '%Y-%m-%dd %H:%M:%S.%f')
    except IOError:    
        df_station['Date'] = pd.to_datetime(df_station['Date'], infer_datetime_format=True)
    except:
        print('Wrong date format in the files')
              
    df_station.set_index('Date', inplace = True)

    # replacing 99999.0 values
    
    df_station.loc[df_station['X'] >= 99999.0, 'X'] = np.nan
    df_station.loc[df_station['Y'] >= 99999.0, 'Y'] = np.nan
    df_station.loc[df_station['Z'] >= 99999.0, 'Z'] = np.nan

    df_station = df_station.sort_index().loc[starttime:endtime]
    
    try:
        df_station.pop('date')
        df_station.pop('Time')
    except:
        pass
    
    return df_station

def load_wdc(filepath, format = "XYZ"):
    
    assert format in ["HDZ", "XYZ"], "format must be HDZ or XYZ"
    
    skiprows = determine_skiprows(filepath)
    df_wdc = pd.read_csv(filepath,sep="\s+", usecols=[0,1,3,4,5,6], skiprows=(skiprows-1), parse_dates = {'Date': ['DATE', 'TIME']})
    
    df_wdc['Date'] = pd.to_datetime(df_wdc['Date'], format = '%Y-%m-%dd %H:%M:%S.%f')
    df_wdc.set_index('Date', inplace = True)
    
    df_wdc = df_wdc.rename(columns={df_wdc.columns[0]:df_wdc.columns[0][3:4],
                                    df_wdc.columns[1]:df_wdc.columns[1][3:4],
                                    df_wdc.columns[2]:df_wdc.columns[2][3:4],
                                    df_wdc.columns[3]:df_wdc.columns[3][3:4]})
    
    df_wdc.loc[df_wdc[df_wdc.columns[0]] >= 99999.0, df_wdc.columns[0]] = np.nan
    df_wdc.loc[df_wdc[df_wdc.columns[1]] >= 99999.0, df_wdc.columns[1]] = np.nan
    df_wdc.loc[df_wdc[df_wdc.columns[2]] >= 99999.0, df_wdc.columns[2]] = np.nan
    df_wdc.loc[df_wdc[df_wdc.columns[3]] >= 99999.0, df_wdc.columns[3]] = np.nan
    
    if (format == 'XYZ') & ("X" not in df_wdc.columns):  

        D = np.deg2rad(df_wdc['D']/60)
        X = df_wdc['H']*np.cos(D)
        Y = df_wdc['H']*np.sin(D)

        df_wdc['X'] = X
        df_wdc['Y'] = Y
    
    if (format == 'HDZ') & ("H" in df_wdc.columns):
        if df_wdc["D"].mean() < 0:
            df_wdc["D"] = ((df_wdc["D"]/60) + 360)*60
    
         
    if (format == 'HDZ') & ("H" not in df_wdc.columns):     
        
        D = np.arctan(df_wdc["Y"]/df_wdc["X"]) #rad
        
        df_wdc["H"] = np.sqrt(df_wdc["X"]**2 + df_wdc["Y"]**2)
        if D.mean() > 0:
            df_wdc["D"] = (360 + np.rad2deg(D))*60 
        else:
            df_wdc["D"] = (360 + np.rad2deg(D))*60
            
    if format == 'XYZ':
        
        return df_wdc[["X","Y","Z"]]
    else:
        return df_wdc[["H","D","Z"]]
    
def load_intermagnet_files_v2(station: str,
                              starttime: str = None,
                              endtime: str = None,
                              files_path: str = None
                              ) -> pd.DataFrame():

    '''
    
    Function to read and concat observatory data.
        Works with every INTERMAGNET Observatory.
    ----------------------------------------------------------
    Data types:
    
    Quasi-definitive and definitive data.
    ----------------------------------------------------------
    Inputs:
    
    station (str) - 3 letters IAGA code
    
    starttime (str, None) - first day of the data (format = 'yyyy-mm-dd', str) or None
    
    endtime (str, None) - last day of the data (format = 'yyyy-mm-dd', str) or None
    
    files_path - path to the IAGA-2002 intermagnet files (str) or None
                 if None it will use the default path for the files
    
    if starttime and endtime are None, all the files will be readed.
    ----------------------------------------------------------
    Example of use:
    
    load_intermagnet_files(station = 'VSS',
                           starttime = '2000-01-25',
                           endtime = '2021-12-31',
                           files_path = 'path//to//files')
    
    ------------------------------------------------------------------------------------------
    
    Return a pandas DataFrame of all readed data with X, Y and Z components.
    
    '''
    #Validating the inputs
    assert len(station) == 3, 'station must be a IAGA code with 3 letters'
    
    if not [i for i in (starttime, endtime) if i is None]:
        for i in [starttime, endtime]:
            spf.validate(i)
    else:
        if files_path is None:
            raise ValueError('if starttime and endtime are None, you must inform files_path.') 
           
    #checking the existence of the station argument
    
    if utt.IMO.check_existence(station) is False:
        raise ValueError(f'station must be an observatory IAGA CODE!')
   
    #creating a list to allocate the file paths
    
       
    print(f'Reading files from {station.upper()}...')
    
    starttime = datetime.strptime(starttime, '%Y-%m-%d')
    endtime = datetime.strptime(endtime, '%Y-%m-%d')
    
    df_station = pd.DataFrame()
    
    while starttime <= endtime:
        
        file_path = glob.glob(os.path.join(files_path,
                                           f"{station.lower()}{starttime.strftime('%Y%m%d')}*.min"))[0] 
    
        if os.path.isfile(file_path) is True:
        
            skiprows = spf.skiprows_detection_v2(file_path)    

            #reading and concatenating the files


            df =   pd.read_csv(file_path,
                                sep='\s+',
                                usecols = [0, 1, 3, 4, 5], 
                                header = None,
                                skiprows = skiprows, 
                                parse_dates = {'Date': ['date', 'Time']},
                                names = ['date', 'Time', 'X', 'Y', 'Z']
                                )
            
            df_station = pd.concat([df_station, df], ignore_index=True)
        else:
            pass
            
        starttime += timedelta(days = 1)

    
    df_station['Date'] = pd.to_datetime(df_station['Date'], format = '%Y-%m-%dd %H:%M:%S.%f')
    #except IOError:    
    #    df_station['Date'] = pd.to_datetime(df_station['Date'], infer_datetime_format=True)
    #except:
    #    print('Wrong date format in the files')
              
    df_station.set_index('Date', inplace = True)

    # replacing 99999.0 values
    
    df_station.loc[df_station['X'] >= 99999.0, 'X'] = np.nan
    df_station.loc[df_station['Y'] >= 99999.0, 'Y'] = np.nan
    df_station.loc[df_station['Z'] >= 99999.0, 'Z'] = np.nan

    #df_station = df_station.sort_index().loc[starttime:endtime]
    
    try:
        df_station.pop('date')
        df_station.pop('Time')
    except:
        pass
    
    return df_station

def sv_obs(station: str,
           starttime: str = None,
           endtime: str = None,
           plot_chaos: bool = False,
           files_path: str = None
           ):
    '''
    Interactive function for INTERMAGNET observatories secular variation data processing
    
    --------------------------------------------
    
    data processing option:
    
    *Denoising filter
    
    *External field reduction - Disturned Days, Quiet days or night time selection
    
    *CHAOS-7 model correction
    
    *Jerk detection based on linear segments
    
    -------------------------------------------
    
    Option to save plot and text files of minute, hourly, daily,
    monthly, annual means and secular variation.
    
    --------------------------------------------
    
    inputs:
        
    station (str) - 3 letters IAGA code
    
    starttime (str) - first day of the data (format = 'yyyy-mm-dd)
    
    endtime (str) - last day of the data (format = 'yyyy-mm-dd)
    
    plot_chaos (boolean) - If the CHAOS model prediction was computed,
                 Will be plotted the comparisons. 
    
    files_path (str) - path to the IAGA-2002 intermagnet files (str) or None
                 if None it will use the default path for the files
    
    -----------------------------------------------
    
    Return a pandas dataframe (processed)
    
    '''
    #reading the files
    
    working_directory = project_directory()
    
    df_station = load_intermagnet_files(station,
                                        starttime,
                                        endtime,
                                        files_path
                                        )
    

    #detecting different data types
    
    if starttime is None and endtime is None:
        
        starttime = str(df_station.index[0].date())
        endtime = str(df_station.index[-1].date())
    
    if pd.to_datetime(endtime) > pd.to_datetime('2018-12-31'):
        
        First_QD_data = spf.data_type(station = station,
                                      starttime = starttime,
                                      endtime = endtime
                                      )
    else:
        First_QD_data = []
    
    
    # HDZ to XYZ conversion
    
    df_station =  utt.hdz_to_xyz_conversion(station = station,
                                            dataframe = df_station,
                                            files_path = files_path
                                            )

    df_station2 = df_station.copy()
    
    # Hampel filter interaction
    
    while True: 
        inp_denoise = input("Do you want to denoise the data based on median absolute deviation (hampel filter)? [y/n]: ")
        if inp_denoise == 'y':
            
            df_station = dpt.hampel_filter_denoising(dataframe = df_station,
                                                     window_size = 100,
                                                     n_sigmas = 3,
                                                     plot_figure = True,
                                                     apply_percentage = True
                                                     )
            break

        if inp_denoise == 'n':

            print('No data changed.')
            break

        else:
            print('You must type y or n, try again!')
    
    # external field reduction interaction
    
    options = ['Q', 'D', 'NT',
               'KP', 'E'
              ]

    while True:

        inp = str(input(f'Press Q to use only Quiet Days, D to remove Disturbed Days, '
                        f'NT to use only the night-time, KP to Kp-Index <=2 or '
                        f'E to Exit without actions [Q/D/NT/KP/E]: '))
        
        if all([inp != option for option in options]):

            print('You must type Q, D, NT, KP or E')

        else:

            break
    
    if inp == 'Q':
        
        df_station = dpt.keep_quiet_days(df_station)
        
    if inp == 'D':
        
        df_station = dpt.remove_disturbed_days(df_station)
        
    if inp == 'NT':
        
        df_station = dpt.night_time_selection(station,
                                              df_station)
    if inp == 'KP':
        
        df_station = dpt.kp_index_correction(dataframe = df_station,
                                             kp = 2)
        
    if inp == 'E':
        print('No action')
        
    
    #condition for data resampling
    
    if inp not in ['Q', 'D',
                   'NT', 'KP'
                   ]:

        resample_condition = True

    else:

        resample_condition = False    
    
    #CHAOS model correction interaction
    
    while True:
    
        input_chaos = input(f"Do You want to correct the external field using the CHAOS model? [y/n]: ")

        if input_chaos == 'y':
            
            df_station_jerk_detection = df_station.copy()
            
            df_station, df_chaos = dpt.external_field_correction_chaos_model(station = station,
                                                                             starttime = starttime,
                                                                             endtime = endtime,
                                                                             df_station = df_station,
                                                                             df_chaos = None,
                                                                             apply_percentage = resample_condition)
            break
        if input_chaos == 'n':

            print('Correction using CHAOS was not applied.')
            
            break

        else:

            print('You must type y or n, try again!')
    
    directory = pathlib.Path(os.path.join(working_directory,
                                          'Filtered_data',
                                          f'{station}_data'
                                          )
                             )
    
    

    pathlib.Path(directory).mkdir(parents = True, exist_ok = True)  
    

    df_sv = dpt.calculate_sv(df_station,
                             apply_percentage = resample_condition
                            )
    
    df_sv_not_corrected = dpt.calculate_sv(df_station2,
                                           apply_percentage = resample_condition
                                           )
    
    if input_chaos == 'y':
        
        df_chaos_sv = dpt.calculate_sv(df_chaos,
                                       source = 'int',
                                       apply_percentage = False
                                      )   
    else:
        
        pass    
    
    #calculating dataframe with minthly mean
    df_monthly_mean = dpt.resample_obs_data(df_station,
                                            sample = 'M',
                                            apply_percentage = resample_condition
                                            )
        
    #option to save txt and plot files
    
    while True: 
        
        inp2 = input(f"Do you want to save files of the different samples for {station}? [y/n]: ")

        if inp2 == 'y':

            print('Saving files...')

            for sample in ['Min', 'H', 'D',
                           'M', 'Y'
                          ]:
                          
                if sample == 'Min':
                
                    file = df_station[starttime:endtime].resample(sample).mean().round(3).replace(np.NaN, 99999.0)
                    
                    file.to_csv(pathlib.Path(os.path.join(f'{directory}',
                                                          f'{station.upper()}_minute_mean_preliminary.zip')),
                                                                     header = [f'{station.upper()}X',
                                                                               f'{station.upper()}Y',
                                                                               f'{station.upper()}Z'
                                                                              ],
                                                                     sep = '\t',
                                                                     index = True
                                                                    )
            
                if sample == 'H':
                    
                    file = dpt.resample_obs_data(df_station,
                                                 'H',
                                                 apply_percentage = resample_condition).round(3).replace(np.NaN, 99999.0)

                    file.to_csv(pathlib.Path(os.path.join(f'{directory}',
                                                          f'{station.upper()}_hourly_mean_preliminary.txt')),
                                                                     header = [f'{station.upper()}X',
                                                                               f'{station.upper()}Y',
                                                                               f'{station.upper()}Z'
                                                                              ],
                                                                     sep = '\t',
                                                                     index = True)
                
                    spf.header_sv_obs_files(station = station,
                                            filename = 'hourly_mean',
                                            data_denoise = inp_denoise,
                                            external_correction = inp,
                                            chaos_model = input_chaos
                                            )               
                if sample == 'D':
                    
                    file = dpt.resample_obs_data(df_station,
                                                 'D',
                                                 apply_percentage = resample_condition).round(3).replace(np.NaN, 99999.0)

                    file.to_csv(pathlib.Path(os.path.join(f'{directory}',
                                                          f'{station.upper()}_daily_mean_preliminary.txt')),
                                                                     header = [f'{station.upper()}X',
                                                                               f'{station.upper()}Y',
                                                                               f'{station.upper()}Z'
                                                                              ],
                                                                     sep = '\t',
                                                                     index = True)
                    
                    spf.header_sv_obs_files(station = station,
                                            filename = 'daily_mean',
                                            data_denoise = inp_denoise,
                                            external_correction = inp,
                                            chaos_model = input_chaos
                                            ) 
                if sample == 'M':
                    
                    file = dpt.resample_obs_data(df_station,
                                                 'M',
                                                 apply_percentage = resample_condition).round(3).replace(np.NaN, 99999.0)
                    
                    file_SV = df_sv.replace(np.NaN, 99999.0)
                    
                    file.to_csv(pathlib.Path(os.path.join(f'{directory}',
                                                          f'{station.upper()}_monthly_mean_preliminary.txt')
                                             ),
                                header = [f'{station.upper()}X',
                                          f'{station.upper()}Y',
                                          f'{station.upper()}Z'
                                         ],
                                sep = '\t',
                                index = True
                                )
                    
                    spf.header_sv_obs_files(station = station,
                                            filename = 'monthly_mean',
                                            data_denoise = inp_denoise,
                                            external_correction = inp,
                                            chaos_model = input_chaos
                                            ) 
                    
                    file_SV.to_csv(pathlib.Path(os.path.join(f'{directory}',
                                                             f'{station.upper()}_secular_variation_preliminary.txt')),
                                                header = [station.upper() + 'SV_X',
                                                          station.upper() + 'SV_Y',
                                                          station.upper() + 'SV_Z'
                                                         ],
                                    sep = '\t',
                                    index = True
                                    )
                    
                    spf.header_sv_obs_files(station = station,
                                            filename = 'secular_variation',
                                            data_denoise = inp_denoise,
                                            external_correction = inp,
                                            chaos_model = input_chaos) 
                if sample == 'Y':
                    
                    file = dpt.resample_obs_data(df_station,
                                                 'Y',
                                                 apply_percentage = resample_condition).round(3).replace(np.NaN, 99999.0)
                    
                    file.to_csv(pathlib.Path(os.path.join(f'{directory}',
                     f'{station.upper()}_annual_mean_preliminary.txt')),
                                header = [f'{station.upper()}X',
                                          f'{station.upper()}Y',
                                          f'{station.upper()}Z'
                                         ],
                                sep = '\t',
                                index = True)
                    
                    spf.header_sv_obs_files(station = station,
                                            filename = 'annual_mean',
                                            data_denoise = inp_denoise,
                                            external_correction = inp,
                                            chaos_model = input_chaos) 
                    
            print(f'Minute, Hourly, Daily, Monthly, Annual means and Secular Variation were saved on directory:')
            print(directory)   
            break

        elif inp2 == 'n':

            print(f'No files saved!')

            break

        else:

            print(f'You must type y or n.')
              
    while True:
        
        inp3 = input(f"Do you want to save plots of the different samples and SV of {station} for X, Y and Z? [y/n]: ")
        if inp3 == 'y':
            directory = pathlib.Path(os.path.join(working_directory,
                                                  'Filtered_data',
                                                  f'{station}_data'
                                                  )
                                     )

            pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
            #plot minute mean
            if input_chaos == 'y' or inp_denoise == 'y':
                
                fig, ax = plt.subplots(3,1, figsize = (13,8), sharex = True)
                plt.subplots_adjust(hspace = 0.05)
                
                ax[0].set_title(f'{station.upper()} minute mean', fontsize = 14)
                ax[0].plot(df_station2['X'],  color  = 'blue')
                ax[0].set_xlim(df_station2['X'].index[0], df_station2['X'].index[-1])
                ax[0].set_ylabel('X(nT)', fontsize = 12)
                ax[0].grid()
                
                ax[1].plot(df_station2['Y'], color  = 'green')
                ax[1].set_xlim(df_station2['Y'].index[0], df_station2['Y'].index[-1])
                ax[1].set_ylabel('Y(nT)', fontsize = 12)
                ax[1].grid()
                
                ax[2].plot(df_station2['Z'], color  =  'black')
                ax[2].set_xlim(df_station2['Z'].index[0], df_station2['Z'].index[-1])
                ax[2].set_ylabel('Z(nT)', fontsize = 12)
                ax[2].grid()

                
                if First_QD_data != []:

                    ax[0].plot(df_station2.loc[df_station2.index > First_QD_data]['X'], color = 'red', label = 'Quasi-definitive data')
                    ax[1].plot(df_station2.loc[df_station2.index > First_QD_data]['Y'], color = 'red', label = 'Quasi-definitive data')
                    ax[2].plot(df_station2.loc[df_station2.index > First_QD_data]['Z'], color = 'red', label = 'Quasi-definitive data')
                    ax[0].legend()
                    ax[1].legend()
                    ax[2].legend()

                
            if input_chaos == 'n' or inp_denoise == 'n':
                
                fig, ax = plt.subplots(3,1, figsize = (13,8), sharex = True)
                plt.subplots_adjust(hspace = 0.05)
                
                ax[0].set_title(f'{station.upper()} minute mean', fontsize = 14)
                ax[0].plot(df_station['X'], color  = 'blue')
                ax[0].set_xlim(df_station['X'].index[0], df_station['X'].index[-1])
                ax[0].set_ylabel('X (nT)', fontsize = 12)
                ax[0].grid()
                
                ax[1].plot(df_station['Y'], color  = 'green')
                ax[1].set_xlim(df_station['Y'].index[0], df_station['Y'].index[-1])
                ax[1].set_ylabel('Y (nT)', fontsize = 12)
                ax[1].grid()
                
                ax[2].plot(df_station['Z'], color  =  'black')
                ax[2].set_xlim(df_station['Z'].index[0], df_station['Z'].index[-1])
                ax[2].set_ylabel('Z (nT)', fontsize = 12)
                ax[2].grid()
                
                if First_QD_data != []:
                    ax[0].plot(df_station.loc[df_station.index > First_QD_data]['X'],
                               color = 'red',
                               label = 'Quasi-definitive data')
                    
                    ax[1].plot(df_station.loc[df_station.index > First_QD_data]['Y'],
                               color = 'red',
                               label = 'Quasi-definitive data')
                    ax[2].plot(df_station.loc[df_station.index > First_QD_data]['Z'],
                               color = 'red',
                               label = 'Quasi-definitive data')
                    ax[0].legend()
                    ax[1].legend()
                    ax[2].legend()
                
            for ax in ax.flatten():
                ax.xaxis.set_major_locator(md.MonthLocator(interval = 12)) 
                ax.xaxis.set_major_formatter(md.DateFormatter('%Y-%m'))
                ax.xaxis.set_tick_params(labelrotation = 30)
                ax.xaxis.get_ticklocs(minor=True)
                ax.yaxis.set_tick_params(which='minor', bottom=False)
                ax.minorticks_on() 
            
            plt.savefig(os.path.join(directory,
                                     f'{station}_minute_mean.jpeg'
                                     ),
                        dpi = 300,
                        bbox_inches='tight'
                        )
            plt.show()
                    
            
            if First_QD_data != []:
            
                plot_samples(station = station,
                             dataframe = df_station.copy(),
                             save_plots = True,
                             plot_data_type = First_QD_data,
                             apply_percentage = resample_condition
                             )
            else:
                plot_samples(station = station,
                             dataframe = df_station.copy(),
                             save_plots = True,
                             plot_data_type = None,
                             apply_percentage = resample_condition
                             )
            
            #plot of secular variation and monthly mean
            
            fig, ax = plt.subplots(3,2, figsize = (18,10))    
            
            ax[0,1].set_title(f'{station.upper()} Secular Variation (ADMM)',
                              fontsize = 14)
            ax[0,1].plot(df_sv['X'],
                         'o',
                         color  = 'blue'
                         )
            ax[0,1].set_xlim(df_sv['X'].index[0], df_sv['X'].index[-1])
            ax[0,1].set_ylabel('dX/dT(nT/yr)', fontsize = 12)
            ax[0,1].grid()
            
            ax[1,1].plot(df_sv['Y'], 'o', color  = 'green')
            ax[1,1].set_xlim(df_sv['Y'].index[0], df_sv['Y'].index[-1])
            ax[1,1].set_ylabel('dY/dT(nT/yr)', fontsize = 12)
            ax[1,1].grid()
            
            ax[2,1].plot(df_sv['Z'], 'o', color  =  'black')
            ax[2,1].set_xlim(df_sv['Z'].index[0], df_sv['Z'].index[-1])
            ax[2,1].set_ylabel('dZ/dT(nT/yr)', fontsize = 12)
            ax[2,1].grid()
            
            
            ax[0,0].set_title(f'{station.upper()} Monthly Mean', fontsize = 14)
            ax[0,0].plot(df_monthly_mean['X'][starttime:endtime], color  = 'blue')
            ax[0,0].set_xlim(df_station['X'].index[0], df_station['X'].index[-1])
            ax[0,0].set_ylabel('X/nT', fontsize = 14)   
            ax[0,0].grid()
            
            ax[1,0].plot(df_monthly_mean['Y'][starttime:endtime], color  = 'green')
            ax[1,0].set_xlim(df_station['Y'].index[0], df_station['Y'].index[-1])
            ax[1,0].set_ylabel('Y/nT', fontsize = 14)           
            ax[1,0].grid()
    
    
            ax[2,0].plot(df_monthly_mean['Z'][starttime:endtime], color  = 'black')
            ax[2,0].set_xlim(df_station['Z'].index[0], df_station['Z'].index[-1])
            ax[2,0].set_ylabel('Z/nT', fontsize = 14)
            ax[2,0].grid()
 
            
            if First_QD_data != []:
                #computing date for SV
                SV_QD_first_data = datetime.strptime(First_QD_data, '%Y-%m-%d') + pd.DateOffset(months=-6)
                
                ax[0,1].plot(df_sv.loc[df_sv.index > SV_QD_first_data]['X'],
                             'o',
                             color  = 'red',
                             label = 'Quasi-definitive')
                
                ax[1,1].plot(df_sv.loc[df_sv.index > SV_QD_first_data]['Y'],
                             'o',
                             color  = 'red',
                             label = 'Quasi-definitive')
                
                ax[2,1].plot(df_sv.loc[df_sv.index > SV_QD_first_data]['Z'],
                             'o',
                             color  = 'red',
                             label = 'Quasi-definitive')
                
                ax[0,1].legend()
                ax[1,1].legend()
                ax[2,1].legend()
                
            for ax in ax.flatten():
                ax.xaxis.set_major_locator(md.MonthLocator(interval = 12)) 
                ax.xaxis.set_major_formatter(md.DateFormatter('%Y-%m'))
                ax.xaxis.get_ticklocs(minor=True)
                ax.yaxis.set_tick_params(which='minor', bottom=False)
                ax.minorticks_on() 

            plt.savefig(os.path.join(directory,
                                     f'{station}_Var_SV.jpeg'
                                     ),
                        dpi = 300,
                        bbox_inches = 'tight'
                        )
            plt.show()      
            
            #plot of SV alone     
                  
            fig, ax = plt.subplots(3, 1, figsize = (13,8), sharex = True)
            plt.subplots_adjust(hspace = 0.05)
            
            ax[0].set_title(f'{station.upper()} Secular Variation (ADMM)', fontsize = 14)
    
            ax[0].plot(df_sv['X'], 'o', color  = 'blue')
            ax[0].set_xlim(df_sv['X'].index[0], df_sv['X'].index[-1])
            ax[0].set_ylim(df_sv['X'].min() - 3, df_sv['X'].max() + 3)
            ax[0].set_ylabel('dX/dT(nT/yr)', fontsize = 12)
            ax[0].grid()
            
            ax[1].plot(df_sv['Y'], 'o', color  = 'green')
            ax[1].set_xlim(df_sv['Y'].index[0], df_sv['Y'].index[-1])
            ax[1].set_ylim(df_sv['Y'].min() - 3, df_sv['Y'].max() + 3)
            ax[1].set_ylabel('dY/dT(nT/yr)', fontsize = 12)
            ax[1].grid()
            
            ax[2].plot(df_sv['Z'], 'o', color  =  'black')
            ax[2].set_xlim(df_sv['Z'].index[0], df_sv['Z'].index[-1])
            ax[2].set_ylim(df_sv['Z'].min() - 3, df_sv['Z'].max() + 3)
            ax[2].set_ylabel('dZ/dT(nT/yr)', fontsize = 12)
            ax[2].grid()           
            
            if First_QD_data != []:
                #computing date for SV
                SV_QD_first_data = datetime.strptime(First_QD_data, '%Y-%m-%d') + pd.DateOffset(months=-6)
                
                ax[0].plot(df_sv.loc[df_sv.index > SV_QD_first_data]['X'],
                             'o',
                             color  = 'red',
                             label = 'Quasi-definitive')
                
                ax[1].plot(df_sv.loc[df_sv.index > SV_QD_first_data]['Y'],
                             'o',
                             color  = 'red',
                             label = 'Quasi-definitive')
                
                ax[2].plot(df_sv.loc[df_sv.index > SV_QD_first_data]['Z'],
                             'o',
                             color  = 'red',
                             label = 'Quasi-definitive')
                
                ax[0].legend()
                ax[1].legend()
                ax[2].legend()
                
            for ax in ax.flatten():
                ax.xaxis.set_major_locator(md.MonthLocator(interval = 12)) 
                ax.xaxis.set_major_formatter(md.DateFormatter('%Y-%m'))
                ax.xaxis.set_tick_params(labelrotation = 30)
                ax.xaxis.get_ticklocs(minor=True)
                ax.yaxis.set_tick_params(which='minor', bottom=False)
                ax.minorticks_on() 

            plt.savefig(os.path.join(directory, f'{station}_SV.jpeg'),
                                     dpi = 300,
                                     bbox_inches='tight')
            plt.show()
            

            if input_chaos == 'y' and plot_chaos is True:
                
                #plotting real SV and corrected SV comparison
                
                fig, ax = plt.subplots(3,1, figsize = (13,8), sharex = True)
                plt.subplots_adjust(hspace = 0.05)
                
                ax[0].set_title(f'{station.upper()} Secular Variation (ADMM) - Observed (raw) SV x corrected SV (CHAOS-correction)', fontsize = 14)
                ax[0].plot(df_sv_not_corrected['X'], 'o', color  = 'red', label = 'Observed (raw) SV')
                ax[0].plot(df_sv['X'], 'o', color  = 'blue', label = 'Observed (corrected) SV')
                ax[0].set_xlim(df_sv['X'].index[0], df_sv['X'].index[-1])
                ax[0].set_ylim(df_sv_not_corrected['X'].min() - 5, df_sv_not_corrected['X'].max() + 5)
                ax[0].set_ylabel('dX/dT(nT/yr)', fontsize = 12)
                ax[0].legend()
                ax[0].grid()
                
                
                ax[1].plot(df_sv_not_corrected['Y'], 'o', color  = 'red', label = 'Observed (raw) SV')
                ax[1].plot(df_sv['Y'], 'o', color  = 'green', label = 'Observed (corrected) SV')
                ax[1].set_xlim(df_sv['Y'].index[0], df_sv['Y'].index[-1])
                ax[1].set_ylim(df_sv_not_corrected['Y'].min() - 5, df_sv_not_corrected['Y'].max() + 5)
                ax[1].set_ylabel('dY/dT(nT/yr)', fontsize = 12)
                ax[1].legend()
                ax[1].grid()
                
                
                ax[2].plot(df_sv_not_corrected['Z'], 'o', color  = 'red', label = 'Observed (raw) SV')
                ax[2].plot(df_sv['Z'], 'o', color  =  'black', label = 'Observed (corrected) SV')
                ax[2].set_xlim(df_sv['Z'].index[0], df_sv['Z'].index[-1])
                ax[2].set_ylim(df_sv_not_corrected['Z'].min() - 5, df_sv_not_corrected['Z'].max() + 5)
                ax[2].set_ylabel('dZ/dT(nT/yr)', fontsize = 12)
                ax[2].legend()
                ax[2].grid()
                
                for ax in ax.flatten():
                    ax.xaxis.set_major_locator(md.MonthLocator(interval = 12)) 
                    ax.xaxis.set_major_formatter(md.DateFormatter('%Y-%m'))
                    ax.xaxis.set_tick_params(labelrotation = 30)
                    ax.xaxis.get_ticklocs(minor=True)
                    ax.yaxis.set_tick_params(which='minor', bottom=False)
                    ax.minorticks_on() 
                
                plt.savefig(os.path.join(directory,
                                         f'{station}_SV_correction_comparison.jpeg'
                                         ),
                            dpi = 300,
                            bbox_inches = 'tight'
                            )
                plt.show()
                
                #plotting chaos predicted and corrected SV
                
                fig, ax = plt.subplots(3,1, figsize = (13,8), sharex = True)
                plt.subplots_adjust(hspace = 0.05)
                
                ax[0].set_title(f'{station.upper()} Secular Variation (ADMM) - Observed (corrected) SV x CHAOS predicted SV (Internal field)', fontsize = 14)
                ax[0].plot(df_chaos_sv['X_int'], 'o', color  = 'red', label = 'CHAOS predicted SV')
                ax[0].plot(df_sv['X'], 'o', color  = 'blue', label = 'Observed (corrected) SV')
                ax[0].set_xlim(df_sv['X'].index[0], df_sv['X'].index[-1])
                ax[0].set_ylim(df_sv['X'].min() - 5, df_sv['X'].max() + 5)
                ax[0].set_ylabel('dX/dT(nT/yr)', fontsize = 12)
                ax[0].legend()
                ax[0].grid()
                
                
                ax[1].plot(df_chaos_sv['Y_int'], 'o', color  = 'red', label = 'CHAOS predicted SV')
                ax[1].plot(df_sv['Y'], 'o', color  = 'green', label = 'Observed (corrected) SV')
                ax[1].set_xlim(df_sv['Y'].index[0], df_sv['Y'].index[-1])
                ax[1].set_ylim(df_sv['Y'].min() - 5, df_sv['Y'].max() + 5)
                ax[1].set_ylabel('dY/dT(nT/yr)', fontsize = 12)
                ax[1].legend()
                ax[1].grid()
                
                
                ax[2].plot(df_chaos_sv['Z_int'], 'o', color  = 'red', label = 'CHAOS predicted SV')
                ax[2].plot(df_sv['Z'], 'o', color  =  'black', label = 'Observed (corrected) SV')
                ax[2].set_xlim(df_sv['Z'].index[0], df_sv['Z'].index[-1])
                ax[2].set_ylim(df_sv['Z'].min() - 5, df_sv['Z'].max() + 5)
                ax[2].set_ylabel('dZ/dT(nT/yr)', fontsize = 12)
                ax[2].legend()
                ax[2].grid()
                
                for ax in ax.flatten():
                    ax.xaxis.set_major_locator(md.MonthLocator(interval = 12)) 
                    ax.xaxis.set_major_formatter(md.DateFormatter('%Y-%m'))
                    ax.xaxis.set_tick_params(labelrotation = 30)
                    ax.xaxis.get_ticklocs(minor=True)
                    ax.yaxis.set_tick_params(which='minor', bottom=False)
                    ax.minorticks_on() 
                
                plt.savefig(os.path.join(directory,
                                         f'{station}_SV_predicted_and_correction_comparison.jpeg'
                                         ),
                            dpi = 300,
                            bbox_inches='tight'
                            )
                plt.show()
                
            
            print('Plots of Minute, Hourly, Daily, Monthly, Yearly means and Secular Variation were saved on directory:')
            print(directory)    
            
            break
        
        elif inp3 == 'n':

            print('No plots saved!')
            
            #plot minute mean

            if input_chaos == 'y' or inp_denoise == 'y':
                
                fig, ax = plt.subplots(3,1, figsize = (13,8), sharex = True)
                plt.subplots_adjust(hspace = 0.05)
                
                ax[0].set_title(f'{station.upper()} minute mean', fontsize = 14)
                ax[0].plot(df_station2['X'],  color  = 'blue')
                ax[0].set_xlim(df_station2['X'].index[0], df_station2['X'].index[-1])
                ax[0].set_ylabel('X(nT)', fontsize = 12)
                ax[0].grid()
                
                ax[1].plot(df_station2['Y'], color  = 'green')
                ax[1].set_xlim(df_station2['Y'].index[0], df_station2['Y'].index[-1])
                ax[1].set_ylabel('Y(nT)', fontsize = 12)
                ax[1].grid()
                
                ax[2].plot(df_station2['Z'], color  =  'black')
                ax[2].set_xlim(df_station2['Z'].index[0], df_station2['Z'].index[-1])
                ax[2].set_ylabel('Z(nT)', fontsize = 12)
                ax[2].grid()

                
                if First_QD_data != []:

                    ax[0].plot(df_station2.loc[df_station2.index > First_QD_data]['X'], color = 'red', label = 'Quasi-definitive data')
                    ax[1].plot(df_station2.loc[df_station2.index > First_QD_data]['Y'], color = 'red', label = 'Quasi-definitive data')
                    ax[2].plot(df_station2.loc[df_station2.index > First_QD_data]['Z'], color = 'red', label = 'Quasi-definitive data')
                    ax[0].legend()
                    ax[1].legend()
                    ax[2].legend()
                plt.show()
         
            if input_chaos == 'n' or inp_denoise == 'n':
                
                fig, ax = plt.subplots(3,1, figsize = (13,8), sharex = True)
                plt.subplots_adjust(hspace = 0.05)
                
                ax[0].set_title(f'{station.upper()} minute mean', fontsize = 14)
                ax[0].plot(df_station['X'], color  = 'blue')
                ax[0].set_xlim(df_station['X'].index[0], df_station['X'].index[-1])
                ax[0].set_ylabel('X (nT)', fontsize = 12)
                ax[0].grid()
                
                ax[1].plot(df_station['Y'], color  = 'green')
                ax[1].set_xlim(df_station['Y'].index[0], df_station['Y'].index[-1])
                ax[1].set_ylabel('Y (nT)', fontsize = 12)
                ax[1].grid()
                
                ax[2].plot(df_station['Z'], color  =  'black')
                ax[2].set_xlim(df_station['Z'].index[0], df_station['Z'].index[-1])
                ax[2].set_ylabel('Z (nT)', fontsize = 12)
                ax[2].grid()
                
                if First_QD_data != []:
                    ax[0].plot(df_station.loc[df_station.index > First_QD_data]['X'],
                               color = 'red',
                               label = 'Quasi-definitive data')
                    
                    ax[1].plot(df_station.loc[df_station.index > First_QD_data]['Y'],
                               color = 'red',
                               label = 'Quasi-definitive data')
                    ax[2].plot(df_station.loc[df_station.index > First_QD_data]['Z'],
                               color = 'red',
                               label = 'Quasi-definitive data')
                    ax[0].legend()
                    ax[1].legend()
                    ax[2].legend()
                
            for ax in ax.flatten():
                ax.xaxis.set_major_locator(md.MonthLocator(interval = 12)) 
                ax.xaxis.set_major_formatter(md.DateFormatter('%Y-%m'))
                ax.xaxis.set_tick_params(labelrotation = 30)
                ax.xaxis.get_ticklocs(minor=True)
                ax.yaxis.set_tick_params(which='minor', bottom=False)
                ax.minorticks_on() 
            plt.show()
                              
            if First_QD_data != []:
            
                plot_samples(station = station,
                             dataframe = df_station.copy(),
                             save_plots = True,
                             plot_data_type = First_QD_data,
                             apply_percentage = resample_condition
                             )
            else:
                plot_samples(station = station,
                             dataframe = df_station.copy(),
                             save_plots = True,
                             plot_data_type = None,
                             apply_percentage = resample_condition
                             )
                
            #plot of secular variation and monthly mean
            
            fig, ax = plt.subplots(3,2, figsize = (18,10))    
        
            ax[0,1].set_title(station.upper() + ' Secular Variation (ADMM)', fontsize = 14)
            ax[0,1].plot(df_sv['X'], 'o', color  = 'blue')
            ax[0,1].set_xlim(df_sv['X'].index[0], df_sv['X'].index[-1])
            ax[0,1].set_ylabel('dX/dT(nT/yr)', fontsize = 12)
            ax[0,1].grid()
            
            ax[1,1].plot(df_sv['Y'], 'o', color  = 'green')
            ax[1,1].set_xlim(df_sv['Y'].index[0], df_sv['Y'].index[-1])
            ax[1,1].set_ylabel('dY/dT(nT/yr)', fontsize = 12)
            ax[1,1].grid()
            
            ax[2,1].plot(df_sv['Z'], 'o', color  =  'black')
            ax[2,1].set_xlim(df_sv['Z'].index[0], df_sv['Z'].index[-1])
            ax[2,1].set_ylabel('dZ/dT(nT/yr)', fontsize = 12)
            ax[2,1].grid()
            
            
            ax[0,0].set_title(station.upper() + ' Monthly Mean', fontsize = 14)
            ax[0,0].plot(df_monthly_mean['X'], color  = 'blue')
            ax[0,0].set_xlim(df_station['X'].index[0], df_station['X'].index[-1])
            ax[0,0].set_ylabel('X/nT', fontsize = 14)   
            ax[0,0].grid()
            
            ax[1,0].plot(df_monthly_mean['Y'], color  = 'green')
            ax[1,0].set_xlim(df_station['Y'].index[0], df_station['Y'].index[-1])
            ax[1,0].set_ylabel('Y/nT', fontsize = 14)           
            ax[1,0].grid()
    
    
            ax[2,0].plot(df_monthly_mean['Z'], color  = 'black')
            ax[2,0].set_xlim(df_station['Z'].index[0], df_station['Z'].index[-1])
            ax[2,0].set_ylabel('Z/nT', fontsize = 14)
            ax[2,0].grid()
            
            for ax in ax.flatten():
                ax.xaxis.set_major_locator(md.MonthLocator(interval = 12)) 
                ax.xaxis.set_major_formatter(md.DateFormatter('%Y-%m'))
                ax.xaxis.get_ticklocs(minor=True)
                ax.yaxis.set_tick_params(which='minor', bottom=False)
                ax.minorticks_on() 
            
            if First_QD_data != []:
                #computing date for SV
                SV_QD_first_data = datetime.strptime(First_QD_data, '%Y-%m-%d') + pd.DateOffset(months=-6)
                
                ax[0,1].plot(df_sv.loc[df_sv.index > SV_QD_first_data]['X'],
                             'o', color  = 'red', label = 'Quasi-definitive')
                
                ax[1,1].plot(df_sv.loc[df_sv.index > SV_QD_first_data]['Y'],
                             'o', color  = 'red', label = 'Quasi-definitive')
                
                ax[2,1].plot(df_sv.loc[df_sv.index > SV_QD_first_data]['Z'],
                             'o', color  = 'red', label = 'Quasi-definitive')
                ax[0,1].legend()
                ax[1,1].legend()
                ax[2,1].legend()
            
            plt.show()         
            
            #plot of SV alone     
                  
            fig, ax = plt.subplots(3,1, figsize = (13,8), sharex = True)
            
            plt.subplots_adjust(hspace = 0.05)
            
            ax[0].set_title(f'{station.upper()} Secular Variation (ADMM)', fontsize = 14)
    
            ax[0].plot(df_sv['X'], 'o', color  = 'blue')
            ax[0].set_xlim(df_sv['X'].index[0], df_sv['X'].index[-1])
            ax[0].set_ylim(df_sv['X'].min() - 3, df_sv['X'].max() + 3)
            ax[0].set_ylabel('dX/dT(nT/yr)', fontsize = 12)
            ax[0].grid()
            
            ax[1].plot(df_sv['Y'], 'o', color  = 'green')
            ax[1].set_xlim(df_sv['Y'].index[0], df_sv['Y'].index[-1])
            ax[1].set_ylim(df_sv['Y'].min() - 3, df_sv['Y'].max() + 3)
            ax[1].set_ylabel('dY/dT(nT/yr)', fontsize = 12)
            ax[1].grid()
            
            ax[2].plot(df_sv['Z'], 'o', color  =  'black')
            ax[2].set_xlim(df_sv['Z'].index[0], df_sv['Z'].index[-1])
            ax[2].set_ylim(df_sv['Z'].min() - 3, df_sv['Z'].max() + 3)
            ax[2].set_ylabel('dZ/dT(nT/yr)', fontsize = 12)
            ax[2].grid()
                       
            if First_QD_data != []:
                #computing date for SV
                SV_QD_first_data = datetime.strptime(First_QD_data, '%Y-%m-%d') + pd.DateOffset(months=-6)
                
                ax[0].plot(df_sv.loc[df_sv.index > SV_QD_first_data]['X'],
                             'o', color  = 'red', label = 'Quasi-definitive')
                
                ax[1].plot(df_sv.loc[df_sv.index > SV_QD_first_data]['Y'],
                             'o', color  = 'red', label = 'Quasi-definitive')
                
                ax[2].plot(df_sv.loc[df_sv.index > SV_QD_first_data]['Z'],
                             'o', color  = 'red', label = 'Quasi-definitive')
                ax[0].legend()
                ax[1].legend()
                ax[2].legend()
                
            for ax in ax.flatten():
                ax.xaxis.set_major_locator(md.MonthLocator(interval=12)) 
                ax.xaxis.set_major_formatter(md.DateFormatter('%Y-%m'))
                ax.xaxis.set_tick_params(labelrotation = 30)
                ax.xaxis.get_ticklocs(minor=True)
                ax.yaxis.set_tick_params(which='minor', bottom=False)
                ax.minorticks_on()  

            plt.show()
            
            
            if input_chaos == 'y' and plot_chaos is True:
                
                #plotting real SV and corrected SV comparison
                
                fig, ax = plt.subplots(3,1, figsize = (13,8), sharex = True)
                plt.subplots_adjust(hspace = 0.05)
                
                ax[0].set_title(f'{station.upper()} Secular Variation (ADMM) - Observed (raw) SV x corrected SV (CHAOS-correction)', fontsize = 14)
                ax[0].plot(df_sv_not_corrected['X'], 'o', color  = 'red', label = 'Observed (raw) SV')
                ax[0].plot(df_sv['X'], 'o', color  = 'blue', label = 'Observed (corrected) SV')
                ax[0].set_xlim(df_sv['X'].index[0],df_sv['X'].index[-1])
                ax[0].set_ylim(df_sv_not_corrected['X'].min() - 5, df_sv_not_corrected['X'].max() + 5)
                ax[0].set_ylabel('dX/dT(nT/yr)', fontsize = 12)
                ax[0].legend()
                ax[0].grid()
                
                
                ax[1].plot(df_sv_not_corrected['Y'], 'o', color  = 'red', label = 'Observed (raw) SV')
                ax[1].plot(df_sv['Y'], 'o', color  = 'green', label = 'Observed (corrected) SV')
                ax[1].set_xlim(df_sv['Y'].index[0],df_sv['Y'].index[-1])
                ax[1].set_ylim(df_sv_not_corrected['Y'].min() - 5, df_sv_not_corrected['Y'].max() + 5)
                ax[1].set_ylabel('dY/dT(nT/yr)', fontsize = 12)
                ax[1].legend()
                ax[1].grid()
                
                
                ax[2].plot(df_sv_not_corrected['Z'], 'o', color  = 'red', label = 'Observed (raw) SV')
                ax[2].plot(df_sv['Z'], 'o', color  =  'black', label = 'Observed (corrected) SV')
                ax[2].set_xlim(df_sv['Z'].index[0],df_sv['Z'].index[-1])
                ax[2].set_ylim(df_sv_not_corrected['Z'].min() - 5, df_sv_not_corrected['Z'].max() + 5)
                ax[2].set_ylabel('dZ/dT(nT/yr)', fontsize = 12)
                ax[2].legend()
                ax[2].grid()
                
                for ax in ax.flatten():
                    ax.xaxis.set_major_locator(md.MonthLocator(interval=12)) 
                    ax.xaxis.set_major_formatter(md.DateFormatter('%Y-%m'))
                    ax.xaxis.set_tick_params(labelrotation = 30)
                    ax.xaxis.get_ticklocs(minor=True)
                    ax.yaxis.set_tick_params(which='minor', bottom=False)
                    ax.minorticks_on()   
                                  
                plt.show()
                
                #plotting CHAOS predicted and corrected SV
                
                fig, ax = plt.subplots(3,1, figsize = (13,8), sharex = True)
                plt.subplots_adjust(hspace = 0.05)
                
                ax[0].set_title(f'{station.upper()} Secular Variation (ADMM) - Corrected SV x CHAOS predicted SV (Internal field)', fontsize = 14)
                ax[0].plot(df_chaos_sv['X_int'], 'o', color  = 'red', label = 'CHAOS predicted SV')
                ax[0].plot(df_sv['X'], 'o', color  = 'blue', label = 'Observed (corrected) SV')
                ax[0].set_xlim(df_sv['X'].index[0],df_sv['X'].index[-1])
                ax[0].set_ylim(df_sv['X'].min() - 5, df_sv['X'].max() + 5)
                ax[0].set_ylabel('dX/dT(nT/yr)', fontsize = 12)
                ax[0].legend()
                ax[0].grid()
                
                
                ax[1].plot(df_chaos_sv['Y_int'], 'o', color  = 'red', label = 'CHAOS predicted SV')
                ax[1].plot(df_sv['Y'], 'o', color  = 'green', label = 'Observed (corrected) SV')
                ax[1].set_xlim(df_sv['Y'].index[0],df_sv['Y'].index[-1])
                ax[1].set_ylim(df_sv['Y'].min() - 5, df_sv['Y'].max() + 5)
                ax[1].set_ylabel('dY/dT(nT/yr)', fontsize = 12)
                ax[1].legend()
                ax[1].grid()
                
                
                ax[2].plot(df_chaos_sv['Z_int'], 'o', color  = 'red', label = 'CHAOS predicted SV')
                ax[2].plot(df_sv['Z'], 'o', color  =  'black', label = 'Observed (corrected) SV')
                ax[2].set_xlim(df_sv['Z'].index[0],df_sv['Z'].index[-1])
                ax[2].set_ylim(df_sv['Z'].min() - 5, df_sv['Z'].max() + 5)
                ax[2].set_ylabel('dZ/dT(nT/yr)', fontsize = 12)
                ax[2].legend()
                ax[2].grid()
                
                for ax in ax.flatten():
                    ax.xaxis.set_major_locator(md.MonthLocator(interval=12)) 
                    ax.xaxis.set_major_formatter(md.DateFormatter('%Y-%m'))
                    ax.xaxis.set_tick_params(labelrotation = 30)
                    ax.xaxis.get_ticklocs(minor=True)
                    ax.yaxis.set_tick_params(which='minor', bottom=False)
                    ax.minorticks_on() 
                    
                plt.show()
         
            break
           
        else:
            print(f'You must type y or n!')
    
    while True:

        condition = input(f"Do you want to detect a geomagnetic jerk? [y/n]: ")
        
        if condition == 'y':
            try:  
                window_start = input(f'type the  start date for the jerk window [yyyy-mm]: ')
                window_end = input(f'type the end date for the jerk window [yyyy-mm]: ')
                
                for i in [str(window_start), str(window_end)]:
                    spf.validate_ym(i)
                if inp3 == 'y':
                    save_plots = True
                else:
                    save_plots = False    
                    
                if input_chaos == 'y':
                    
                    dpt.jerk_detection_window(station = station,
                                              window_start = window_start, 
                                              window_end = window_end, 
                                              starttime = starttime, 
                                              endtime = endtime,
                                              df_station = df_station_jerk_detection,
                                              df_chaos = df_chaos,
                                              plot_detection = True,
                                              chaos_correction = True,
                                              files_path = files_path,
                                              plot_chaos_prediction = True,
                                              convert_hdz_to_xyz = False,
                                              save_plots = save_plots
                                              )
                
                if input_chaos == 'n':
                    
                    dpt.jerk_detection_window(station = station,
                                              window_start = window_start, 
                                              window_end = window_end, 
                                              starttime = starttime, 
                                              endtime = endtime,
                                              files_path = files_path,
                                              df_station = df_station,
                                              df_chaos = None,
                                              plot_detection = True,
                                              chaos_correction = False,
                                              plot_chaos_prediction = False,
                                              convert_hdz_to_xyz = False,
                                              save_plots = save_plots
                                              )
                break
            except:
                print("""This is not the correct format. Please reenter. (correct format: yyyy-mm)""")
                        
           
        if condition == 'n':
            print('No linear segments adopted')
            break
        if condition not in ['y', 'n']:
            print('You must type y or n, try again!')
            
    return df_station[starttime:endtime]        
                

def plot_samples(station: str,
                 dataframe: pd.DataFrame(),
                 save_plots: bool = False,
                 plot_data_type = None,
                 apply_percentage: bool = False
                 ):
                 
    """
    Function to plot Hourly, daily, monthly and annual means from  the dataframe.
    
    ----------------------------------------------------
    Inputs
    
    station: 3 letters IAGA CODE (str)
    
    dataframe: pd.DataFrame containing the data
    
    save_plots: option to save the plots (bool = True or False, default = False)
    
    apply_percentage: option to apply data availability percentage 
                      condition (at least 90%) to resample_the data
                      (bool = True or False, default = False)
    
    -------------------------------------------------------
    Return plots of hourly, daily, monthly and annual means
                      
    """
    
    #validating the inputs
    
    assert len(station) == 3, 'station must be a IAGA code with 3 letters'
    
    if utt.IMO.check_existence(station) is False:
        raise ValueError(f'station must be an observatory IAGA CODE!')    
        
    working_directory = project_directory()
    
    config = spf.get_config()
    
    if save_plots is False and plot_data_type is None:
    
        samples = ['H','D','M','Y']
        colors = ['blue','green','black']
        
        
        for sample, title in zip(samples, ['hourly', 'daily', 'monthly', 'annual']):
            
            df_station = dpt.resample_obs_data(dataframe = dataframe,
                                               sample = sample,
                                               apply_percentage = apply_percentage
                                               )
            
            fig, axes = plt.subplots(3,1,figsize = (13,8), sharex = True)
            
            plt.subplots_adjust(hspace = 0.05)
            
            plt.suptitle(f'{station.upper()} {title} mean',
                         fontsize = 13,
                         y = 0.91
                        )
            
            plt.xlabel('Date (Years)', fontsize = 12)
                          
                          
            for col, ax, color in zip(df_station.columns, axes.flatten(), colors):
            
                ax.plot(df_station[col],'-',color = color)
                ax.set_ylabel(col.upper() +' (nT)', fontsize = 12)
                ax.set_xlim(df_station[col].index[0],df_station[col].index[-1])
                ax.xaxis.set_major_locator(md.MonthLocator(interval=12)) 
                ax.xaxis.set_tick_params(labelrotation = 30)
                ax.xaxis.set_major_formatter(md.DateFormatter('%Y-%m'))
                ax.xaxis.get_ticklocs(minor=True)
                ax.yaxis.set_tick_params(which='minor', bottom=False)
                ax.minorticks_on() 
                ax.grid()

            plt.show()
            
    if save_plots is False and plot_data_type is not None:

        samples = ['H', 'D',
                   'M', 'Y'
                  ]
        
        colors = ['blue', 'green', 'black']
        
        First_QD_data = plot_data_type
    
        for sample, title in zip(samples, ['hourly', 'daily', 'monthly', 'annual']):
            
            df_station = dpt.resample_obs_data(dataframe = dataframe,
                                               sample = sample,
                                               apply_percentage = apply_percentage
                                               )
            
            fig, axes = plt.subplots(3,1,figsize = (13,8), sharex = True)
            
            plt.subplots_adjust(hspace = 0.05)
            plt.suptitle(f'{station.upper()} {title} mean',
                         fontsize = 13,
                         y = 0.91
                        )
            
            plt.xlabel('Date (Years)', fontsize = 12)
                          
            for col, ax, color in zip(df_station.columns, axes.flatten(), colors):
            
                ax.plot(df_station[col], '-', color = color)
                ax.set_ylabel(col.upper() +' (nT)', fontsize = 12)
                ax.set_xlim(df_station[col].index[0], df_station[col].index[-1])
                ax.grid()
                ax.plot(df_station.loc[df_station.index > First_QD_data][col],
                        '-',
                        color = 'red',
                        label = 'Quasi-definitive data')
                
                ax.xaxis.set_major_locator(md.MonthLocator(interval=12)) 
                ax.xaxis.set_major_formatter(md.DateFormatter('%Y-%m'))
                ax.xaxis.get_ticklocs(minor=True)
                ax.yaxis.set_tick_params(which='minor', bottom=False)
                ax.xaxis.set_tick_params(labelrotation = 30)
                ax.minorticks_on() 
                ax.legend()
                
            plt.show()
                       
    if save_plots is True and plot_data_type is None:
        
        directory = pathlib.Path(os.path.join(config.directory.filtered_data,
                                              f'{station}_data'
                                              )
                                 )
        
        pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
        samples = ['H', 'D',
                   'M', 'Y'
                  ]
        colors = ['blue', 'green', 'black']
        
        
        for sample, title in zip(samples, ['hourly', 'daily',
                                           'monthly', 'annual'
                                          ]
                                ):
            
            df_station = dpt.resample_obs_data(dataframe = dataframe,
                                               sample = sample,
                                               apply_percentage = apply_percentage
                                              )
            fig, axes = plt.subplots(3, 1, figsize = (13,8), sharex = True)
            plt.suptitle(station.upper() + ' ' + title + ' mean', fontsize = 14, y = 0.92)
            plt.subplots_adjust(hspace = 0.05)
            plt.xlabel('Date (Years)', fontsize = 12)
                          
            for col, ax, color in zip(df_station.columns, axes.flatten(), colors):
            
                ax.plot(df_station[col], '-', color = color)
                ax.set_ylabel(col.upper() +' (nT)', fontsize = 12)
                ax.set_xlim(df_station[col].index[0], df_station[col].index[-1])
                ax.xaxis.set_major_locator(md.MonthLocator(interval=12)) 
                ax.xaxis.set_major_formatter(md.DateFormatter('%Y-%m'))
                ax.xaxis.get_ticklocs(minor=True)
                ax.yaxis.set_tick_params(which='minor', bottom=False)
                ax.xaxis.set_tick_params(labelrotation = 30)
                ax.minorticks_on() 
                ax.grid()
                
            plt.savefig(os.path.join(config.directory.filtered_data,
                                     f"{station}_data",
                                     f'{station}_{title}_mean.jpeg'
                                     ),
                        dpi = 300,
                        bbox_inches='tight'
                        )
            plt.show()
            
    if save_plots is True and plot_data_type is not None:
        
        First_QD_data = plot_data_type
        directory = pathlib.Path(os.path.join(config.directory.filtered_data,
                                              f'{station}_data'
                                              )
                                 )
        pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
        samples = ['H','D','M','Y']
        colors = ['blue','green','black']
        
        
        for sample, title in zip(samples, ['hourly',
                                           'daily',
                                           'monthly',
                                           'annual'
                                           ]
                                 ):
            
            df_station = dpt.resample_obs_data(dataframe = dataframe,
                                               sample = sample,
                                               apply_percentage = apply_percentage
                                               )
            
            fig, axes = plt.subplots(3,1,figsize = (13,8), sharex = True)
            plt.subplots_adjust(hspace = 0.05)
            plt.suptitle(f'{station.upper()} {title} mean',
                         fontsize = 13,
                         y = 0.91
                         )
            plt.xlabel('Date (Years)', fontsize = 12)
                          
            for col, ax, color in zip(df_station.columns, axes.flatten(), colors):
            
                ax.plot(df_station[col], '-', color = color)
                ax.set_ylabel(col.upper() +' (nT)', fontsize = 12)
                ax.set_xlim(df_station[col].index[0], df_station[col].index[-1])
                ax.plot(df_station.loc[df_station.index > First_QD_data][col],
                        '-',
                        color = 'red',
                        label = 'Quasi-definitive data'
                        )
                ax.legend()
                ax.xaxis.set_major_locator(md.MonthLocator(interval=12)) 
                ax.xaxis.set_major_formatter(md.DateFormatter('%Y-%m'))
                ax.xaxis.get_ticklocs(minor=True)
                ax.yaxis.set_tick_params(which='minor', bottom=False)
                ax.minorticks_on()
                ax.xaxis.set_tick_params(labelrotation = 30) 
                ax.grid()
                
            plt.savefig(os.path.join(config.directory.filtered_data,
                                     f"{station}_data"
                                     f'{station}_{title}_mean.jpeg'
                                     ),
                        dpi = 300,
                        bbox_inches='tight')
            plt.show()
            
def plot_tdep_map(time:str,
                  deriv:int = 1,
                  plot_changes = False,
                  station = None):
    """_summary_

    Args:
        time (str): _description_
        deriv (int, optional): _description_. Defaults to 1.
        plot_changes (bool, optional): _description_. Defaults to False.
        station (_type_, optional): _description_. Defaults to None.
    """
    
    
    #validating inputs
    
    spf.validate(time)
    
    config = spf.get_config()
    
    assert deriv in [1, 2], 'deriv must be 1 or 2'
    
    assert isinstance(plot_changes, bool), 'plot_changes must be True or False'
    
    fig = plt.figure(figsize=(20, 20))
    
    fig.subplots_adjust(
    top=0.98,
    bottom=0.02,
    left=0.013,
    right=0.988,
    hspace=0.0,
    wspace=0.1
    )
       
    if deriv == 1:
        cons = 10
        tdep = 'SV'
    else:
        cons = 3
        tdep = 'SA'
        
    previous_year = str((datetime.strptime(time, "%Y-%m-%d") - relativedelta(months = 6)).date())
    
    next_year = str((datetime.strptime(time, "%Y-%m-%d") + relativedelta(months = 6)).date())
        
    time =cp.data_utils.mjd2000(datetime.strptime(time, '%Y-%m-%d'))
    
    previous_year =cp.data_utils.mjd2000(datetime.strptime(previous_year, '%Y-%m-%d'))
    
    next_year =cp.data_utils.mjd2000(datetime.strptime(next_year, '%Y-%m-%d'))
    
    working_directory = project_directory()
    
    chaos_path = glob.glob(os.path.join(config.directory.chaos_model,
                                        'data',
                                        'CHAOS*'
                                        )
                           )

    model = cp.load_CHAOS_matfile(chaos_path[0])   

    
    radius = 6371.5  # radius of the core surface in km
    theta = np.linspace(1., 179., 181)  # colatitude in degrees
    phi = np.linspace(-180., 180, 361)  # longitude in degrees
    
    # compute radial SV up to degree 16 using CHAOS

    B_radius, B_theta, B_phi = model.synth_values_tdep(time, radius, theta, phi,
                                                       nmax=20, deriv=deriv, grid=True)
    
    B_x = B_theta*-1
    B_y = B_phi
    B_z = B_radius*-1
    
    if plot_changes is True:
        
        #calculating values for previous years
        
        B_radiusp, B_thetap, B_phip = model.synth_values_tdep(previous_year, radius, theta, phi,
                                                              nmax=20, deriv=deriv, grid=True)
    
        B_xp = B_thetap*-1
        B_yp = B_phip
        B_zp = B_radiusp*-1 
        
        #calculating values for next year
        B_radiusn, B_thetan, B_phin = model.synth_values_tdep(next_year, radius, theta, phi,
                                                              nmax=20, deriv=deriv, grid=True)
    
        B_xn = B_thetan*-1
        B_yn = B_phin
        B_zn = B_radiusn*-1 
        
        #calculating SV or SA changes
        B_x = (B_x - B_xn) - (B_x - B_xp)
        B_y = (B_y - B_yn) - (B_y - B_yp)
        B_z = (B_z - B_zn) - (B_z - B_zp)
        cons = 1
    gs = gridspec.GridSpec(1, 3)
    
    axes = [
        plt.subplot(gs[0, 0], projection=ccrs.Robinson()),
        plt.subplot(gs[0, 1], projection=ccrs.Robinson()),
        plt.subplot(gs[0, 2], projection=ccrs.Robinson())
        ]
    
    for ax, comp, name in zip(axes, [B_x, B_y, B_z], ['X','Y','Z']):
        
        if deriv == 1 and plot_changes == False:
            plt.suptitle(f'Secular Variation', fontsize = 13, y = 0.62)
        elif deriv == 1 and plot_changes == True:
            plt.suptitle(f'Secular Variation changes', fontsize = 13, y = 0.62)
        elif deriv == 2 and plot_changes == False:        
            plt.suptitle(f'Secular Acceleration', fontsize = 13, y = 0.62)
        else:
            plt.suptitle(f'Secular Acceleration changes', fontsize = 13, y = 0.62)  
            
              
        pc = ax.pcolormesh(phi, 90. - theta, comp, cmap='PuOr', vmin=- (comp.max() +cons) ,
                           vmax=comp.max() + cons, transform=ccrs.PlateCarree())
        ax.set_title(f'{name} {tdep} (n <= 16)')
        ax.gridlines(linewidth=0.5,
                     linestyle='dashed',
                     ylocs=np.linspace(-90, 90, num=7),  # parallels
                     xlocs=np.linspace(-180, 180, num=13))  # meridians
        ax.coastlines(linewidth=0.5)
        
        # inset axes into global map and move upwards
        cax = inset_axes(ax,
                         width="55%",
                         height="10%",
                         loc='lower center',
                         borderpad=-3.5
                         )
    
        clb = plt.colorbar(pc,
                           cax=cax,
                           extend='both',
                           orientation='horizontal')
        clb.set_label('nT/yr\u00b2',
                      fontsize=14)
    if station != None:
        df_imos = pd.read_csv(os.path.join(config.directory.imos_database,
                                           config.filenames.imos_database
                                           ),
                              sep = '\t',
                              index_col = [0]
                             )
        for imo in station:
            for ax in axes:
                ax.scatter(df_imos.loc[imo]['Longitude'],
                           df_imos.loc[imo]['Latitude'],
                           color = 'black',
                           transform=ccrs.PlateCarree())
                ax.text(df_imos.loc[imo]['Longitude'] + 3,
                        df_imos.loc[imo]['Latitude'] + 4,
                        df_imos.loc[imo].name,
                        horizontalalignment='left',
                        transform=ccrs.Geodetic())
                 
    plt.show()

def plot_sv(station: str,
            starttime: str = None,
            endtime: str = None,
            files_path = None,
            df_station: pd.DataFrame() = None,
            df_chaos: pd.DataFrame() = None,
            apply_percentage: bool = False,
            plot_chaos: bool = False,
            chaos_correction: bool = False,
            save_plot: bool = False,
            convert_hdz_to_xyz: bool = False
            ):
    """
    Function to plot the Secular Variation

    Args:
        station (str): 3 letters IAGA code of the observatory
        
        starttime (str, optional): First day of interest (yyyy-mm-dd). Defaults to None.
        
        endtime (str, optional): Last day of interest (yyyy-mm-dd). Defaults to None.
        
        files_path (_type_, optional): Path to the IAGA-2002 files. Defaults to None.
        
        df_station (pd.DataFrame, optional): pd.Dataframe with the data, must be readed 
                                             with load_intermagnet_files. Defaults to None.
        
        apply_percentage (bool, optional): True or False, 90% of data availability resample 
                                           criteria. Defaults to False.
        
        plot_chaos (bool, optional): True or False, option to plot the time-dependent CHAOS-7 
                                     model prediction. Defaults to False.
        
        chaos_correction (bool, optional): True or False, option to correct external geomagnetic 
                                           field using the CHAOS-7 model prediction. Defaults to False.
        
        save_plot (bool, optional): True or False, option to save the figure. Defaults to False.
        
    """
    #Validating the inputs
    assert len(station) == 3, 'station must be a IAGA code with 3 letters'
    
    if not [i for i in (starttime, endtime) if i is None]:
        for i in [starttime, endtime]:
            spf.validate(i)
            
    if [i for i in (starttime, endtime) if i is None] and df_station is not None:
        starttime = str(df_station.index[0].date())
        endtime = str(df_station.index[-1].date())
    else:    
        if files_path is None:
            raise ValueError('if starttime, endtime and df_station are None, you must inform files_path.')    
    #checking the existence of the station argument
    
    if utt.IMO.check_existence(station) is False:
        raise ValueError(f'station must be an observatory IAGA CODE!')
        
    
    if df_station is None:
        
        df_station = load_intermagnet_files(station,
                                            starttime,
                                            endtime,
                                            files_path)
        
        
    # calculating sv
    
    if convert_hdz_to_xyz is True:
        
        df_station = utt.hdz_to_xyz_conversion(station, df_station, files_path)
        
    if chaos_correction is True:
        df_station_raw = df_station.copy()
        
        df_sv_raw = dpt.calculate_sv(df_station_raw, apply_percentage = apply_percentage)
    
    
    if chaos_correction is True:
        
        df_station, df_chaos= dpt.external_field_correction_chaos_model(station,
                                                                        starttime,
                                                                        endtime,
                                                                        df_station,
                                                                        files_path = files_path,
                                                                        df_chaos = df_chaos
                                                                        )
        
    if chaos_correction is False and plot_chaos is True and df_chaos is None:
        
        df_chaos = dpt.chaos_model_prediction(station, starttime, endtime)
    
    df_sv = dpt.calculate_sv(df_station, apply_percentage = apply_percentage)
        
    if plot_chaos is True:
        
        df_sv_chaos = dpt.calculate_sv(df_chaos, source = 'int')
        
    fig, axes = plt.subplots(3,1 ,figsize = (14,10), sharex = True)
    
    plt.suptitle(f'{station.upper()} Secular Variation', y = 0.91)
    plt.subplots_adjust(hspace=0.05)
    
    if plot_chaos is True:
        for ax, col, cols in zip(axes.flatten(), df_sv.columns, ['X_int','Y_int','Z_int']): 
            ax.plot(df_sv_chaos[cols], color = 'red', label = 'CHAOS prediction')
            ax.plot(df_sv[col], 'o', color = 'black', label = "Corrected SV")
            ax.legend()

        
    else:
        for ax, col in zip(axes.flatten(), df_sv.columns):
            ax.plot(df_sv[col], 'o', color = 'black')
            
    if chaos_correction is True:
        for ax, col in zip(axes.flatten(), df_sv_raw.columns):
            ax.plot(df_sv_raw[col], 'ob', label = "Measured SV")
            ax.legend()

    for ax, col in zip(axes.flatten(), df_sv.columns):
        ax.set_ylabel(f'{df_sv[col].name} SV (nT/Yr)')
        ax.xaxis.set_major_locator(md.MonthLocator(interval=12)) 
        #ax.xaxis.set_major_locator(md.MonthLocator(bymonthday=1, interval=6))
        #ax.xaxis.set_minor_locator(md.MonthLocator(bymonthday=1, interval=1))
        ax.xaxis.set_major_formatter(md.DateFormatter('%Y-%m'))
        ax.xaxis.set_tick_params(labelrotation = 30, width=2)
        ax.xaxis.get_ticklocs(minor=True)
        ax.minorticks_on()
        ax.yaxis.set_tick_params(which='minor', bottom=False)
        ax.grid(alpha = 0.3)
        ax.set_xlim(df_sv[col].index[0], df_sv[col].index[-1])
        ax.set_xticks(list(df_sv[df_sv.index.month.isin([6, 12])].index) + [df_sv.index[-1], df_sv.index[0]])
        #ax.set_xticks([df_sv.index[-1]])
        ax.legend()

    #axes[0].set_title(f"{station.upper()} Secular Variation")
    if save_plot is True:
        plt.savefig(f'{station}_worldmap_sv.jpeg', dpi = 300, bbox_inches = 'tight')
    plt.show()
    
    
    if plot_chaos is True:
        return df_sv, df_sv_chaos
    else:
        return df_sv
    
if __name__ == '__main__':
    #import time
    #start = time.time()
    #df_station = load_intermagnet_files("NGK", "2010-01-01", "2022-12-31", "NGK")
    #
    #print(df_station)
    #
    #end = time.time()
    #print(end - start)
    plot_sv(station = "NGK",
            starttime="2010-01-01",
            endtime = "2021-12-31",
            files_path = "NGK",
            df_station = None,
            df_chaos = None,
            apply_percentage = False,
            plot_chaos = False,
            chaos_correction = False,
            save_plot = False,
            convert_hdz_to_xyz = False
            )
    

