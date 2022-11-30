import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from glob import glob
from pandas.tseries.frequencies import to_offset
import glob
import os
import ftplib
import pathlib
import matplotlib.gridspec as gridspec
from datetime import datetime, timedelta
import pwlf
import chaosmagpy as cp
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import main_functions as mvs
import data_processing_tools as dpt
import utilities_tools as utt
#from SV_project import support_functions as spf


def project_directory():
    '''
    
    Get the project directory 
    
    '''
    return os.getcwd()

def update_qd_and_dd(data: str):
    """
    Update list of quiet and disturbed days.
    
    Used automatically when function to remove disturbed days
    or keep quiet days are used.
    """

    #validating input parameter
    
    assert data in ['DD', 'QD'], 'data must be QD or DD!'
        
    #connecting to the ftp server 
    ftp = ftplib.FTP('ftp.gfz-potsdam.de')
    ftp.login('anonymous', 'email@email.com')
    
    ##path to read the already stored QD and DD
    
    working_directory = project_directory()
    
    path_local = pathlib.Path(os.path.join(working_directory,
                                           'Data/Disturbed and Quiet Days'
                                           )
                              )
    
    ##path inside ftp server
    path_ftp = f'/pub/home/obs/kp-ap/quietdst'
    
    #getting the file
    ftp.cwd(path_ftp)
    
    filenames = ftp.nlst('qdrecen*')
    
    pathlib.Path(path_local).mkdir(parents=True, exist_ok=True)
    #writting file in the local computer
    for filename in filenames:
        local_filename = os.path.join(path_local, filename)
        file = open(local_filename, 'wb')
        ftp.retrbinary('RETR ' + filename, file.write)
        file.close()
    ftp.quit()
        
    
    #files = glob.glob(f'{path}qd*')
    #files.sort()
    
    if data == 'DD':
        
        df = pd.read_csv(os.path.join(f'{path_local}',
                                      'qdrecent.txt'
                                      ),
                         skiprows = 4,
                         sep = '\s+',
                         header = None,
                         usecols = [0,1,12,13,14,15,16],
                         names = ['Month', 'Year', 'DD1',
                                  'DD2', 'DD3', 'DD4', 'DD5'
                                  ]
                         )
                        
         
        columns = ['DD1','DD2',
                   'DD3','DD4',
                   'DD5'
                  ]
        
        df['Month'] = pd.to_datetime(df.Month, format='%b').dt.month
        
        for col in columns:
            df[f'{col}'] = df['Year'].astype(str)+'-'+df['Month'].astype(str)+'-'+df[col].astype(str)
        for col in columns:
            df[f'{col}'] = df[f'{col}'].str.replace('*','')
        df_dd = pd.DataFrame()
        
        df_dd['DD'] = pd.concat([df['DD1'] ,df['DD2'],
                                 df['DD3'], df['DD4'],
                                 df['DD5']
                                ]
                               )
        
        df_dd['DD'] = pd.to_datetime(df_dd['DD'], format= '%Y-%m-%d')
        
        
        #df_dd.set_index('DD', inplace = True)
        
        #df_dd = df_dd.sort_index()
        
        
        #reading the current QD and DD list
        
        pathlib.Path(os.path.join(working_directory,
                                 'Data',
                                 'Disturbed and Quiet Days',
                                 'Disturbed_Days_list.txt'
                                  )
                    )
        
        df_list = pd.read_csv(pathlib.Path(os.path.join(path_local,
                                                        'Disturbed_Days_list.txt'
                                                        )
                                           )
                              )
        
        df_list['DD'] = pd.to_datetime(df_list['DD'], format= '%Y-%m-%d')
        #df_list.set_index('DD',inplace=True)
    
        df_new = pd.concat([df_list,df_dd], ignore_index=False)
        
        df_new['DD'] = pd.to_datetime(df_new['DD'], infer_datetime_format=True)
        
        df_new = df_new.drop_duplicates()
        
        df_new.set_index('DD', inplace=True)
        
        df_new.dropna().sort_index().to_csv(pathlib.Path(os.path.join(path_local, 'Disturbed_Days_list.txt')), index = True)
        
    if data == 'QD':
        
        df = pd.read_csv(pathlib.Path(os.path.join(path_local, 'qdrecent.txt')),
                          skiprows = 4,
                          sep = '\s+',
                          header = None,
                          usecols = [0, 1, 2,
                                     3, 4, 5,
                                     6, 7, 8,
                                     9, 10, 11
                                     ],
                          names = ['Month', 'Year', 'QD1',
                                   'QD2', 'QD3', 'QD4', 'QD5',
                                   'QD6', 'QD7', 'QD8', 'QD9',
                                   'QD10'
                                  ]
                           )
        
        columns = [f'QD{i}' for i in range(1, 11)]
        
        df['Month'] = pd.to_datetime(df.Month, format='%b').dt.month
        
        for col in columns:
            df[col] = df['Year'].astype(str) + '-' + df['Month'].astype(str) + '-' + df[col].astype(str)
        for col in columns:
            df[col] = df[col].str.replace('A', '')
        for col in columns:
            df[col] = df[col].str.replace('K', '')
        
        df_qd = pd.DataFrame()
        df_qd['QD'] = pd.concat([df['QD1'], df['QD2'], df['QD3'],
                                 df['QD4'], df['QD5'], df['QD6'],
                                 df['QD7'], df['QD8'], df['QD9'],
                                 df['QD10']
                                ]
                               )
        
        df_qd['QD'] = pd.to_datetime(df_qd['QD'], infer_datetime_format=True)
        
        
        #df_qd.set_index('QD', inplace = True)
        
        #df_qd = df_qd.sort_index()
        
        df_list = pd.read_csv(pathlib.Path(os.path.join(path_local,
                                                        'Quiet_Days_list.txt'
                                                        )
                                           )
                              )
        
        df_list['QD'] = pd.to_datetime(df_list['QD'], format= '%Y-%m-%d')
        #df_list.set_index('DD',inplace=True)
    
        df_new = pd.concat([df_list,df_qd], ignore_index=False)
        
        df_new['QD'] = pd.to_datetime(df_new['QD'], infer_datetime_format=True)
        
        df_new = df_new.drop_duplicates()
        
        df_new.set_index('QD', inplace=True)
        
        df_new.dropna().sort_index().to_csv(pathlib.Path(os.path.join(path_local, 'Quiet_Days_list.txt')), index = True)
        
def header_sv_obs_files(station: str,
                        filename: str,
                        data_denoise: str,
                        external_correction: str,
                        chaos_model: str):
    """"
    Function to add a header in the txt outputs of the Function sv_obs.
    
    Output with informations about the used observatory
    and used data processing options.
    
    ------------------------------------------------------------------------------------
    used automatically
    """  
    if utt.IMO.check_existence(station) is False:
        raise ValueError(f'station must be an observatory IAGA CODE!')
    
    working_directory = project_directory()
        
    path = pathlib.Path(os.path.join(working_directory,
                                     'Filtered_data',
                                     f'{station}_data',
                                     f'{station.upper()}_{filename}_preliminary.txt'
                                     )
                        )
    
    path_header = pathlib.Path(os.path.join(working_directory,
                                            'Filtered_data',
                                            f'{station}_data'
                                            )
                               )
    output_path = pathlib.Path(os.path.join(working_directory,
                                             'Filtered_data',
                                             f'{station}_data',
                                             f'{station.upper()}_{filename}.txt'
                                             )
                                )
    
    external_options = {'D': 'Disturbed Days removed',
                        'Q': 'Quiet Days selection', 
                        'NT': 'Night Time selection',
                        'E': 'No'}
    
    denoise_options = {'y': 'Hampel Filter',
                       'n': 'No'
                      }
    
    chaos_options = {'y': 'Yes',
                     'n': 'No'
                    }
    
    #pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

    df_station = pd.read_csv(path,
                             sep = '\s+',
                             index_col = [0]
                             )
    df_station.index = pd.to_datetime(df_station.index, infer_datetime_format=True)
   
    
    Header = (f'MOSFiT package'
              f'\nIAGA CODE {str(station.upper())}'
              f'\nLatitude {str(utt.IMO.latitude(station.upper()))}'
              f'\nLongitude {str(utt.IMO.longitude(station.upper()))}'
              f'\nElevation {str(utt.IMO.elevation(station.upper()))}' 
              f'\nData denoise: {denoise_options[data_denoise]}'
              f'\nExternal field reduction: {external_options[external_correction]}'
              f'\nCHAOS model correction: {chaos_options[chaos_model]}'
              f'\n\n')
    
    with open(os.path.join(f'{path_header}',
                           'header_file.txt'), 'w+') as f2:
        f2.write(Header)
        header = f2.read()
    
    filenames = [os.path.join(f'{path_header}',
                              'header_file.txt'),
                 path
                 ]
    
    with open(output_path, 'w') as outfile:
        for fname in filenames:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)
                    
    
    os.remove(os.path.join(f'{path_header}',
                           'header_file.txt'
                           )
              )
    os.remove(path)
    
def data_type(station: str,
              starttime: str = None,
              endtime: str = None,
              files_path = None):
    '''
    Function to verify the existence of Quasi-definitive data type in the dataset
    
    ----------------------------------------------------------
    Inputs:
    
    station (str) - 3 letters IAGA code 
    
    starttime (str) - first day of the data (format = 'yyyy-mm-dd')
    
    endtime (str) - last day of the data (format = 'yyyy-mm-dd')
    
    files_path (str) - path to the IAGA-2002 intermagnet files or None
                 if None it will use the default path for the files
    
    --------------------------------------------------------------------------------------
    Usage example:
    
    data_type(station = 'VSS',
              starttime = '2000-01-25',
              endtime = '2021-01-25',
              files_path = 'path//to/files')
    
    
    '''
    assert len(station) == 3, 'station must be a IAGA code with 3 letters'
    
    if utt.IMO.check_existence(station) is False:
        raise ValueError(f'station must be an observatory IAGA CODE!')
        
    if not [i for i in (starttime, endtime) if i is None]:
        for i in [starttime, endtime]:
            validate(i)
    else:
        if files_path is None:
            raise ValueError('if starttime and endtime are None, you must inform files_path.')  

    files_station = [] 

    if pd.to_datetime(endtime) >= pd.to_datetime('2018-06-30'):
    
        if files_path is None:
            years_interval = np.arange(int(starttime[0:4]), int(endtime[0:4])+ 1)
            for year in years_interval:
    
                files_station.extend(glob.glob(f'C:\\Users\\marco\\Downloads\\Thesis_notebooks\\Dados OBS\\{str(year)}/*/{station}*'))
                files_station.sort()
        else:
            files_station.extend(glob.glob(os.path.join(f'{files_path}',
                                                        f'{station}*min*'
                                                        )
                                           )
                                 )
            
            files_station.sort()
    
        #d_parser = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S%.f')
        df_data_type = pd.DataFrame()
        df_data_type = pd.concat((pd.read_csv(file,
                                              sep = '\s+',
                                              header = None,                                         
                                              skiprows = 11,
                                              nrows = 20,
                                              usecols = [0,2],
                                              names = ['Date', 'Data_Type'])
                                              for file in files_station))
        
        data_type = df_data_type.loc[[0], ['Data_Type']]
    
        date = df_data_type.loc[[15], ['Date']]
        #date.set_index('Date', inplace = True)
    
        data_type.set_index(date['Date'], inplace = True)
        
        data_type.sort_index()
        #print(df_data_type)
        
        for Date in data_type['Data_Type']: 
            if Date == 'Definitive':
                Date = []
            else:
            
                Date = data_type.loc[data_type['Data_Type'] != 'Definitive'].index[0]
    else:
        Date = []    
        
    return Date

def header_sv_files(station: str,
                    data_denoise: str,
                    external_correction: str,
                    chaos_model:str
                    ):
    """"
    Function to add a header in the txt outputs of the Function sv_obs.
    
    Output with informations about the used observatory
    and used data processing options.
    
    ------------------------------------------------------------------------------------
    used automatically
    """  
    
    #validating inputs
    
    assert len(station) == 3, 'station must be a IAGA code with 3 letters'
    
    if utt.IMO.check_existence(station) is False:
        raise ValueError(f'station must be an observatory IAGA CODE!')
    
    working_directory = project_directory()
        
    path = pathlib.Path(os.path.join(working_directory,
                                     f'SV_update',
                                     f'{station}_data',
                                     f'SV_{station.upper()}_preliminary.txt'
                                     )
                        )
    
    path_header = pathlib.Path(os.path.join(working_directory,
                                            'SV_update',
                                            f'{station}_data'
                                            )
                               )
    
    output_path = pathlib.Path(os.path.join(working_directory,
                                            'SV_update',
                                            f'{station}_data',
                                            f'SV_{station.upper()}.txt'
                                            )
                               )
    
    external_options = {'D': 'Disturbed Days removed',
                        'Q': 'Quiet Days selection', 
                        'NT': 'Night Time selection',
                        None: 'No'
                        }
    
    denoise_options = {True: 'Hampel Filter',
                      False: 'No'
                      }
    
    chaos_options = {True: 'Yes',
                    False: 'No'
                    }
    
    #pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

    df_station = pd.read_csv(path,
                             sep = '\s+',
                             index_col = [0])
    
    df_station.index = pd.to_datetime(df_station.index,
                                      infer_datetime_format = True)
    
    
    header = (f'MOSFiT package'
              f'\nIAGA CODE {str(station.upper())}'
              f'\nLatitude {str(utt.IMO.latitude(station.upper()).round(2))}'
              f'\nLongitude {str(utt.IMO.longitude(station.upper()).round(2))}'
              f'\nElevation {str(utt.IMO.elevation(station.upper()).round(2))}'
              f'\nData denoise: {denoise_options[data_denoise]}'
              f'\nExternal field reduction: {external_options[external_correction]}'
              f'\nCHAOS model correction: {chaos_options[chaos_model]}'
              f'\n\n')
    
    with open(pathlib.Path(os.path.join(f'{path_header}',
                                        'header_file.txt')), 'w+') as f2:
        
        f2.write(header)
        header = f2.read()
    
    filenames = [pathlib.Path(os.path.join(f'{path_header}', 'header_file.txt', path))]
    with open(output_path, 'w') as outfile:
        for fname in filenames:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)
                    
    
    os.remove(pathlib.Path(os.path.join(f'{path_header}', 'header_file.txt')))
    os.remove(path)
       
def date_to_decinal_year_converter(date):
    '''
    Function to convert a datetime to decimal year.
    
    --------------------------------------------------------    
    Inputs:
    
    date - must be a datetime
    
    --------------------------------------------------------
    
    return a decimal year 
    '''
    year_start = datetime(date.year, 1, 1)
    year_end = year_start.replace(year=date.year+1)
    return date.year + ((date - year_start).total_seconds() /  # seconds so far
        float((year_end - year_start).total_seconds()))

def validate(str_date):
    """
    Function to validate input format "YYYY-MM-DD"
    
    """
    try:
        datetime.strptime(str_date, '%Y-%m-%d')
    except ValueError:
        raise ValueError('Incorrect date format, should be YYYY-MM-DD')

def validate_ym(str_date):
    """
    Function to validate input format "YYYY-MM"
    
    """
    try:
        datetime.strptime(str_date, '%Y-%m')
    except ValueError:
        raise ValueError(f'Incorrect {str_date} format, should be YYYY-MM')

class year(object):
    """
    class representing the year
    used to find leap years

    """
    def __init__(self,year):
        self.year = year
    def check_leap_year(year):
        if((year % 400 == 0) or  
           (year % 100 != 0) and  
           (year % 4 == 0)):
            return True
        else:
            return False

def skiprows_detection(files_station):
    """Function to detect the correct number of skiprows
    for each IAGA-2002 file

    Args:
        files_station (list of files): _description_

    Returns:
        list: contains the number of skiprows for each file and the path
    """
    
    skiprows_list = [[], []]
    
    for file in files_station:
        idx = 0
        skiprows = 10
        df_station = pd.read_csv(file,
                                 sep = '\s+',
                                 skiprows = skiprows,
                                 nrows=40,
                                 usecols = [0],
                                 names = ['col']
                                 )
        file = file
        while df_station['col'][idx] != 'DATE':
            skiprows += 1
            idx +=1 
            if df_station['col'][idx] == 'DATE':
                skiprows += 1
                skiprows_list[0].append(skiprows)     
                skiprows_list[1].append(file)
                
    return skiprows_list      

def decimal_year_to_date(date):
    """
    Function to convert from decimal year to date

    Args:
        date (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    decimal_date = float(date)
    year = int(decimal_date)
    rest = decimal_date - year

    base = datetime(year, 1, 1)
    result = base + timedelta(seconds=(base.replace(year=base.year + 1) - base).total_seconds() * rest)
    
    return result.date()      