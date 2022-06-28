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
import sqlite3
import chaosmagpy as cp
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from Thesis_Marcos import thesis_functions as mvs
from Thesis_Marcos import support_functions as spf
from Thesis_Marcos import data_processing_tools as dpt


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
                              Year,
                              Months,
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
    List_Months = ['01', '02', '03',
                   '04', '05', '06',
                   '07', '08', '09',
                   '10', '11', '12'
                   ]
    
    ftp = ftplib.FTP('seismo.nrcan.gc.ca')
    ftp.login('anonymous', 'email@email.com')
    
    if Months == None:
        Months = List_Months
        
    for month in Months:
        directory = 'Dados OBS/' + Year + '/' + month
        print(directory)
        if datatype == 'QD':
            path = 'intermagnet/minute/quasi-definitive/IAGA2002/' + Year + '/' + month
            
        if datatype == 'D':
            path = 'intermagnet/minute/definitive/IAGA2002/' + Year + '/' + month
        
        if files == None:
            ftp = ftplib.FTP('seismo.nrcan.gc.ca')
            ftp.login('anonymous', 'email@email.com')
            ftp.cwd(path)
            filenames = ftp.nlst('*') # get filenames within the directory
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
        x = pd.read_csv(file,
                        sep = '\s+',
                        skiprows = 12,
                        nrows = 40,
                        usecols = [0, 3],
                        names = ['date', 'col']
                        )
        file = file
        idx = 0
        while x['col'][idx] != station.upper() + 'H':
            idx+=1
    
            if x['col'][idx] == station.upper() + 'H':
                
                values_list.append(x['date'][idx + 1])     
                #values_list[1].append(x['col'][idx])
    
            if x['col'][idx] == station.upper() + 'X':
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