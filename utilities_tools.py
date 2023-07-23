import pandas as pd
import numpy as np
import glob
import os
import ftplib
import pathlib
import main_functions as mvs
import support_functions as spf
import data_processing_tools as dpt
from os.path import exists
import chaosmagpy as cp
import re


def project_directory():
    return os.getcwd()
            
def download_data_intermagnet(datatype:str,
                              years:list,
                              months:list,
                              obs:str
                              ):
    
    '''
    Download observatory files from Intermagnet FTP server and save in
    a specific directory.
    
    Datatype must be: 'D' for Definitive or 'QD' quasi-definitive
    
    Year must be informed as 2021, for example.
    
    Months must be a list, for example - ['01','02','03']
    
    files must be an obs IAGA code or None, if None, files for all the observatories will
    be downloaded.
    
    example of use - mvs.download_data_intermagnet('QD', '2021', ['07','08','09'], files = None)
    '''
    
    assert datatype in ['QD','D'], 'datatype must be QD or D'
    
    assert isinstance(years, list), 'The input years must be a list'
    
    assert isinstance(months, list), 'The input years must be a list'
    
    for year in years:
        assert isinstance(year, int), 'Each year of the list years must be an integer'
    
    for month in months:
        assert isinstance(year, int), 'Each year of the list years must be an integer'
        
        assert month > 0 and month < 13, 'Each month of the the list months must be greater than 0 and lower than 13'
            
    assert isinstance(obs, str) and len(obs) == 3, 'The input obs must be a string with lenght 3'
    
    assert obs in IMO.database().index, 'obs must be an INTERMAGNET observatory'
    
    working_directory = project_directory()
    
    ftp = ftplib.FTP('seismo.nrcan.gc.ca')
    ftp.login('anonymous', 'email@email.com')
    
    for year in years:    
        for month in months:
        
            directory = os.path.join(working_directory,
                                     'obs_data',
                                     f'{str(year)}',
                                     f'{str(month).zfill(2)}')
            print(directory)
            if datatype == 'QD':
                path = f'intermagnet/minute/quasi-definitive/IAGA2002/{str(year)}/{str(month).zfill(2)}'
                
            if datatype == 'D':
                path = f'intermagnet/minute/definitive/IAGA2002/{str(year)}/{str(month).zfill(2)}'

            ftp = ftplib.FTP('seismo.nrcan.gc.ca')
            ftp.login('anonymous', 'email@email.com')
            ftp.cwd(path)
            filenames = ftp.nlst(obs.lower() + '*') # get filenames within the directory
            filenames.sort()
            
            if not os.path.exists(directory):
                os.makedirs(directory)
             
            for filename in filenames:    
                print('File ' + filename  + ' downloaded!')   
                local_filename = os.path.join(directory, filename)
                file = open(local_filename, 'wb')
                ftp.retrbinary('RETR '+ filename, file.write)
                
    ftp.quit()
    print('Disconnected from INTERMAGNET Ftp server!') 
    
    
def get_solar_zenith(station:str, starttime:str, endtime:str):
    """_summary_

    Args:
        station (str): _description_
        starttime (str): _description_
        endtime (str): _description_
    """
    
    longitude = IMO.longitude(station)
    
    colatitude = 90 - IMO.latitude(station)
    
    date = pd.date_range(starttime, f"{endtime} 23:00:00", freq = 'H')
    jd_time = cp.data_utils.mjd2000(date)
    
    solar_zenith = cp.coordinate_utils.zenith_angle(jd_time, colatitude, longitude)
    
    return solar_zenith
                   
def hdz_to_xyz_conversion(station: str,
                          dataframe: pd.DataFrame(),
                          files_path: str = None) -> pd.DataFrame():
    '''
    Automatically indentify the existence H, D and Z components in the 
    geomagnetic data, and convert to X, Y and Z.
    
    --------------------------------------------------------    
    
    Inputs:
    
    station (str) - 3 letters IAGA code for a INTERMAGNET observatory.
    
    dataframe - a pandas dataframe with geomagnetic data (pd.dataframe).
    
    files_path (str)- path to the IAGA-2002 intermagnet files (str) or None
                 if None it will use the default path for the files
                 
    ----------------------------------------------------------------
    
    Usage example:
    
    hdz_to_xyz_conversion(station = 'VSS',
                          dataframe = name_of_datafrme,
                          files_path = 'files_path')
    
    ------------------------------------------------------------------
    
    Return a dataframe with only X, Y and Z components
    
    '''
    #validating inputs
    
    assert len(station) == 3, 'station must be a three letters IAGA Code'
    
    assert isinstance(dataframe,pd.DataFrame), 'dataframe must be a pandas DataFrame'
    
    if IMO.check_existence(station) == False:
        print(f'Station must be an observatory IAGA CODE!')  
    
    df_station = dataframe
    
    starttime = str(df_station.index[0].date())
    endtime = str(df_station.index[-1].date())
    
    years_interval = np.arange(int(starttime[0:4]), int(endtime[0:4])+ 1)
    
    files_station = []

    
    if files_path is None:
        
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
        start_index = []
        end_index = []
        for file, i in zip(files_station, np.arange(0, len(files_station))):
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
            
    values_list = []
    for file in files_station:
        

        right_line = False

        with open(file, 'r') as f:
            line_count = 1
            for line in f.readlines():
                line_count += 1
                if right_line is True:
                    values_list.append(line[0:10])
                    break
                if re.search ('^DATE', line):
                
                    if line[32:36] == f"{station.upper()}H":
                        right_line = True
                    else:
                        break
    
    
    date_hdz = pd.to_datetime(values_list, infer_datetime_format = True)
    
    for date in date_hdz.year.drop_duplicates():

        D = np.deg2rad(df_station['Y'].loc[str(date)]/60)
        X = df_station['X'].loc[str(date)]*np.cos(D)
        Y = df_station['X'].loc[str(date)]*np.sin(D)
        
        df_station['X'].loc[str(date)] = X
        df_station['Y'].loc[str(date)] = Y
    return df_station

class IMO(object):
    '''
    Class to represent the INTERMAGNET magnet observatory (IMO)
    
    It is possible to check the coordinates stored in the database (Latitude, Longitude and altitude)
    
    It is also possible to add, remove an check the existence of an IMO.
    
    '''
    config = spf.get_config()
    
    working_directory = os.getcwd()
        
    imos_directory = pathlib.Path(os.path.join(config.directory.imos_database,
                                               config.filenames.imos_database
                                               )
                                  )
    
    def database():
        
        return pd.read_csv(IMO.imos_directory,
                           skiprows = 1,
                           sep = '\s+',
                           usecols=[0, 1, 2, 3],
                           names = ['Imos', 'Latitude', 'Longitude', 'Elevation'],
                           index_col= ['Imos'])
    
    def code(station):
        assert isinstance(station, str), 'station must be a string.'
        station = station.upper()
        if IMO.check_existence(station) is True:
            pass
        else:
            raise Exception('station not in the IMOS database, check the IAGA code or add the station.')

        station = IMO.database().loc[station].name
        return station
    
    def latitude(station):
        station = station.upper()
        if IMO.check_existence(station) is True:
            pass
        else:
            raise Exception('station not in the IMOS database, check the IAGA code or add the station.')
            
        return IMO.database().loc[station]['Latitude']
    
    def longitude(station):
        station = station.upper()        
        if IMO.check_existence(station) is True:
            pass
        else:
            raise Exception('station not in the IMOS database, check the IAGA code or add the station.')
            
        return IMO.database().loc[station]['Longitude']
    
    def elevation(station: str):
        station = station.upper()        
        if IMO.check_existence(station) is True:
            pass
        else:
            raise Exception('station not in the IMOS database, check the IAGA code or add the station.')
        
        return IMO.database().loc[station]['Elevation']
    
    def delete(station: str):
        station = station.upper() 
        
        assert station in IMO.database().index, 'station not in database.'
        
        IMO.database().drop(station).to_csv(IMO.imos_directory, sep = '\t')
    
    def check_existence(station):        
        station = station.upper()
        if station not in IMO.database().index:
            return False
        else:
            return True
        
    def add(station: str,
            latitude: float,
            longitude: float,
            elevation: float
            ):
    
        for i in [latitude, longitude, elevation]:
        
            assert isinstance(i, (float, int)), 'station coordinates must be int or float.'
            
        assert len(station) == 3, 'station lenght must be 3'
        
        
        imo_info = {'Imos': [station.upper()],
                    'Latitude': [latitude],
                    'Longitude': [longitude],
                    'Elevation': [elevation]
                    }
        
        df_new_imo = pd.DataFrame(imo_info)
        df_new_imo.set_index('Imos', inplace=True)
        
        database = pd.concat([IMO.database(), df_new_imo])
        
        database.to_csv(IMO.imos_directory, sep = '\t')
 
    
if __name__ == '__main__':
    
    print(get_solar_zenith("NGK", "2010-01-01", "2020-12-31"))   