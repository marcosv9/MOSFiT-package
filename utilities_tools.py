
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
import thesis_functions as mvs
import support_functions as spf
import data_processing_tools as dpt
from os.path import exists


def check_data_availability(station: str):
    '''
    check the available data period, based on the IAGA code.
    
    
    '''
    
    assert len(station) == 3, 'station must be a three letters IAGA Code'
    
    if IMO.check_existence(station) == False:
        print(f'Station must be an observatory IAGA CODE!')
        
    f = []
    f.extend(glob.glob(f'Dados OBS/*/*/{station}*'))
    f.sort()
    print(f'The first available date for {station.upper()} is {f[0][21:29]}')
    print(f'The last available date for {station.upper()} is {f[-1][21:29]}')
            
def download_data_INTERMAGNET(datatype,
                              year,
                              months,
                              files = None
                             ):
    
    '''
    Download observatory files from Intermagnet FTP server and save in
    a specific directory.
    
    Datatype must be: 'D' for Definitive or 'QD' quasi-definitive
    
    Year must be informed as '2021', for example.
    
    Months must be a list, for example - ['01','02','03']
    
    files must be an obs IAGA code or None, if None, files for all the observatories will
    be downloaded.
    
    example of use - mvs.download_data_INTERMAGNET('QD', '2021', ['07','08','09'], files = None)
    '''
    
    assert datatype in ['QD','D'], 'datatype must be QD or D'
    
    df_IMOS = pd.read_csv('Thesis_Marcos/Data/Imos informations/IMOS_INTERMAGNET.txt',
                           skiprows = 1,
                           sep = '\s+',
                           usecols=[0, 1, 2, 3],
                           names = ['Imos', 'Latitude', 'Longitude', 'Elevation'],
                           index_col= ['Imos'])
    
    List_Months = ['01', '02', '03',
                   '04', '05', '06',
                   '07', '08', '09',
                   '10', '11', '12'
                   ]
    
    ftp = ftplib.FTP('seismo.nrcan.gc.ca')
    ftp.login('anonymous', 'email@email.com')
    
    if months == None:
        months = List_Months
        
    for month in months:
        for station in df_IMOS.index[0:150]:
            directory = 'Dados OBS/' + year + '/' + month
            print(directory)
            if datatype == 'QD':
                path = f'intermagnet/minute/quasi-definitive/IAGA2002/{year}/{month}'
                
            if datatype == 'D':
                path = f'intermagnet/minute/definitive/IAGA2002/{year}/{month}'
            
            if files == None:
                ftp = ftplib.FTP('seismo.nrcan.gc.ca')
                ftp.login('anonymous', 'email@email.com')
                ftp.cwd(path)
                filenames = ftp.nlst(f'{station.upper()}*') # get filenames within the directory
                filenames.sort()
                print('List of files that will be downloaded')
                for filename in filenames:
                    print(filename)
                    
                
                while input("Do You Want To Continue? [y/n]") == "y":
                    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)               
                    for filename in filenames:    
                        print('File ' + filename  + ' downloaded!')   
                        local_filename = os.path.join(directory, filename)
                        file = open(local_filename, 'wb')
                        ftp.retrbinary('RETR '+ filename, file.write)
                        
                    ftp.quit()
                    break
    
            if files != None:
                for file in files:
                    ftp = ftplib.FTP('seismo.nrcan.gc.ca')
                    ftp.login('anonymous', 'email@email.com')
                    ftp.cwd(path)
                    filenames = ftp.nlst(file.lower() + '*') # get filenames within the directory
                    filenames.sort()
                    print('List of files that will be downloaded')
                    for filename in filenames:
                        print(filename)
                    
                
                    while input("Do You Want To Continue? [y/n]") == "y":
                        pathlib.Path(directory).mkdir(parents=True, exist_ok=True)               
                        for filename in filenames:    
                            print('File ' + filename  + ' downloaded!')   
                            local_filename = os.path.join(directory, filename)
                            file = open(local_filename, 'wb')
                            ftp.retrbinary('RETR '+ filename, file.write)
                            
                        
                        break
    ftp.quit()
    print('Disconnected from INTERMAGNET Ftp server!') 
                   
def HDZ_to_XYZ_conversion(station: str,
                          dataframe: pd.DataFrame(),
                          files_path: str = None) -> pd.DataFrame():
    '''
    Automatically indentify the existence H, D and Z components in the 
    geomagnetic data, and convert to X, Y and Z.
    
    --------------------------------------------------------    
    
    Inputs:
    
    station - 3 letters IAGA code for a INTERMAGNET observatory (str).
    
    dataframe - a pandas dataframe with geomagnetic data (pd.dataframe).
    
    files_path - path to the IAGA-2002 intermagnet files (str) or None
                 if None it will use the default path for the files
                 
    ----------------------------------------------------------------
    
    Usage example:
    
    HDZ_to_XYZ_conversion(station = 'VSS',
                          dataframe = name_of_datafrme,
                          files_path = files_path)
    
    ------------------------------------------------------------------
    
    Return a dataframe with only X, Y and Z components
    
    '''
    #validating inputs
    
    assert len(station) == 3, 'station must be a three letters IAGA Code'
    
    assert isinstance(dataframe,pd.DataFrame), 'dataframe must be a pandas DataFrame'
    
    if IMO.check_existence(station) == False:
        print(f'Station must be an observatory IAGA CODE!')  
          
    if files_path != None:
        if files_path[-1] == '/':
            pass
        else:
            files_path = files_path + '/'    
    
    df_station = dataframe
    
    starttime = str(df_station.index[0].date())
    endtime = str(df_station.index[-1].date())
    
    years_interval = np.arange(int(starttime[0:4]), int(endtime[0:4])+ 1)
    files_station = []

    
    if files_path == None:
        for year in years_interval:
            files_station.extend(glob.glob(f'Dados OBS\\{str(year)}/*/{station}*'))
            files_station.sort()
    else:
        files_station.extend(glob.glob(f'{files_path}{station}*'))
        files_station.sort()
            
            
    values_list = []
    for file in files_station:
        df_data = pd.read_csv(file,
                        sep = '\s+',
                        skiprows = 12,
                        nrows = 40,
                        usecols = [0, 3],
                        names = ['date', 'col']
                        )
        file = file
        idx = 0
        while df_data['col'][idx] != station.upper() + 'H':
            idx+=1
    
            if df_data['col'][idx] == station.upper() + 'H':
                
                values_list.append(df_data['date'][idx + 1])     
                #values_list[1].append(x['col'][idx])
    
            if df_data['col'][idx] == station.upper() + 'X':
                break
    
    
    date_hdz = pd.to_datetime(values_list, infer_datetime_format = True)
    
    for date in date_hdz.year.drop_duplicates():
    #print(date)
    #Data_HDZ = pd.concat([Data_HDZ,df_HER.loc[str(date)]])
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
     
     
    def __init__(self,
                 station,
                 latitude,
                 longitude,
                 elevation
                ):
        
        df_IMOS = pd.read_csv('Thesis_Marcos/Data/Imos informations/IMOS_INTERMAGNET.txt',
                               skiprows = 1,
                               sep = '\s+',
                               usecols=[0, 1, 2, 3],
                               names = ['Imos', 'Latitude', 'Longitude', 'Elevation'],
                               index_col= ['Imos'])
        

            #self.station = station
        self.station = df_IMOS.loc[station].name
        self.latitude = latitude
        self.longitude = longitude
        self.elevation = elevation
        

    df_IMOS = pd.read_csv('Thesis_Marcos/Data/Imos informations/IMOS_INTERMAGNET.txt',
                               skiprows = 1,
                               sep = '\s+',
                               usecols=[0, 1, 2, 3],
                               names = ['Imos', 'Latitude', 'Longitude', 'Elevation'],
                               index_col= ['Imos'])
    
    def code(station):
        
        if IMO.check_existence(station) == True:
            pass
        else:
            raise Exception('station not in the IMOS database, check the IAGA code or add the station.')

        station = IMO.df_IMOS.loc[station].name
        return station
    
    def latitude(station):
        if IMO.check_existence(station) == True:
            pass
        else:
            raise Exception('station not in the IMOS database, check the IAGA code or add the station.')
            
        return IMO.df_IMOS.loc[station]['Latitude']
    
    def longitude(station):
        
        if IMO.check_existence(station) == True:
            pass
        else:
            raise Exception('station not in the IMOS database, check the IAGA code or add the station.')
            
        return IMO.df_IMOS.loc[station]['Longitude']
    
    def elevation(station: str):
        
        if IMO.check_existence(station) == True:
            pass
        else:
            raise Exception('station not in the IMOS database, check the IAGA code or add the station.')
        
        return IMO.df_IMOS.loc[station]['Elevation']
    
    def delete(station: str):
    
        IMO.df_IMOS.drop(station)
        
        IMO.df_IMOS.to_csv('Thesis_Marcos/Data/Imos informations/IMOS_INTERMAGNET.txt', sep = '\t')
    
    def check_existence(station):
        station = station.upper()
        if station not in IMO.df_IMOS.index:
            return False
        else:
            return True
        
    def add(station: str,
            latitude: float,
            longitude: float,
            elevation: float
            ):
        
        imo_info = {'Imos': [station],
                    'Latitude': [latitude],
                    'Longitude': [longitude],
                    'Elevation': [elevation]
                    }
        
        df_new_imo = pd.DataFrame(imo_info)
        df_new_imo.set_index('Imos', inplace=True)
        
        IMO.df_IMOS = pd.concat([IMO.df_IMOS, df_new_imo])
        
        IMO.df_IMOS.to_csv('Thesis_Marcos/Data/Imos informations/IMOS_INTERMAGNET.txt', sep = '\t')
        
                
def check_duplicate_files(station,
                          start_year,
                          end_year):
    
    end_year = str(int(end_year) + 1)
    
    
    if station != None:
        station = station.lower()
    
    months_list = ['01', '02', '03',
                   '04', '05', '06',
                   '07', '08', '09',
                   '10', '11', '12'
                   ]
    
    year_period = pd.date_range(start_year, end_year, freq = 'Y')
    
    df_IMOS = pd.read_csv('Thesis_Marcos/Data/Imos informations/IMOS_INTERMAGNET.txt',
                          skiprows = 1,
                           sep = '\s+',
                           usecols=[0, 1, 2, 3],
                           names = ['Imos', 'Latitude', 'Longitude', 'Elevation'],
                           index_col= ['Imos'])
    if station == None:
        for station in df_IMOS.index[0:150]:
            for year in year_period.year:
                directory = f'Dados OBS/{year}/*'
                files = glob.glob(f'{directory}/{station}*')
                if len(files) > 366:
                    print(f'{len(files)} in {year} for {station.upper()}')
                    for file in files:                        
                        filename = os.path.basename(file)
                        filename = filename[0:11]
                        if len(glob.glob(f'{directory}/{filename}*')) > 1:
                            
                            try:
                                os.remove(glob.glob(f'{directory}/{filename}qmin*')[0])
                            except:
                                os.remove(glob.glob(f'{directory}/{filename}dmin*')[0])
                            print(f'file {os.path.basename(file)} removed')
                        #print(f'{files} in {year} for {station}.')
    if station != None:
        for year in year_period.year:
            directory = f'Dados OBS/{year}/*'
            files = glob.glob(f'{directory}/{station}*')
            if len(files) > 366:
                print(f'{len(files)} in {year} for {station.upper()}')
                for file in files: 
                    filename = os.path.basename(file)
                    filename = filename[0:11]
                    
                    if len(glob.glob(f'{directory}/{filename}*')) > 1:
                        
                        try:
                            os.remove(glob.glob(f'{directory}/{filename}qmin*')[0])
                        except:
                            os.remove(glob.glob(f'{directory}/{filename}dmin*')[0])
                        print(f'file {os.path.basename(file)} removed')
    #return files

def update_hourly_database(starttime,
                           endtime,
                           type = 'IMO',
                           stations = None):
    '''
    '''
    
    try:
        type.upper() in ['IMO','CHAOS']
    except ValueError:
        raise ValueError('Type must be IMO or CHAOS')     
    
    for i in [starttime, endtime]:
        spf.validate(i)
    
        
    df_imos = pd.read_csv('Thesis_Marcos/Data/Imos informations/IMOS_INTERMAGNET.txt', sep = '\s+', index_col = [0])
    
    if stations == None:
        stations = df_imos.index
    else:
        stations = stations
    #for station in df_imos.index:
    
    if type.upper() == 'IMO':
        for station in stations:
            try:
                df = mvs.load_INTERMAGNET_files(station = station,
                                                starttime = starttime,
                                                endtime = endtime)
                #temporary until solve the bug on HDZ_to_XYZ_conversion for such stations

                if station in ['ESK','VAL','SOD','YKC','BSL','SHU','AIA','VOS','NUR','IRT','HRN','DOU','MAB','NCK','LVV']:
                    pass
                else:
                    df = HDZ_to_XYZ_conversion(station, df)
                    
                df = dpt.resample_obs_data(df, 'H', apply_percentage= False)
                
                if exists(f'hourly_data/{station}_hourly_data.txt') == True:
                    df_base = pd.read_csv(f'hourly_data/{station}_hourly_data.txt', sep = '\t')
                    df_base.index = pd.to_datetime(df_base['Date'], format= '%Y-%m-%d %H:%M:%S.%f')
                    df_base.pop('Date')
                       
                    df_new_imo = pd.concat([df_base,df])
                    df_new_imo.round(2)[~df_new_imo.round(2).index.duplicated(keep='last')].to_csv(f"hourly_data/{station}_hourly_data.txt", sep = '\t')

                else:
                    
                    df.round(2)[~df.round(2).index.duplicated(keep='last')].to_csv(f"hourly_data/{station}_hourly_data.txt", sep = '\t')
            except:
                print('No data found for station: {}'.format(station))
                pass
            
    if type.upper() == 'CHAOS':        
        for station in stations:
            try:
                df_chaos_new = dpt.chaos_model_prediction(station = station,
                                                          starttime = starttime,
                                                          endtime = endtime)
                
                if exists(f"hourly_data/{station.upper()}_chaos_hourly_data.txt") == True:
                    
                    df_chaos = pd.read_csv(f'hourly_data/{station}_chaos_hourly_data.txt', sep = '\t')
                    df_chaos.index = pd.to_datetime(df_chaos['Unnamed: 0'], format= '%Y-%m-%d %H:%M:%S.%f')
                    df_chaos.pop('Unnamed: 0')
        
                    df_new_c = pd.concat([df_chaos, df_chaos_new])
                    df_new_c.round(2)[~df_new_c.round(2).index.duplicated(keep='last')].to_csv(f"hourly_data/{station.upper()}_chaos_hourly_data.txt", sep = '\t')
                else:
                    df_chaos_new.round(2)[~df_chaos_new.round(2).index.duplicated(keep='last')].to_csv(f"hourly_data/{station.upper()}_chaos_hourly_data.txt", sep = '\t')    
            except:
                pass    