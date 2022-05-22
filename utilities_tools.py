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


def check_data_availability(station):
    '''
    check the available data period, based on the IAGA code.
    
    
    '''
    
    assert len(station) == 3, 'station must be a three letters IAGA Code'
    
    df_IMOS = pd.read_csv('Thesis_Marcos/Data/Imos informations/Imos_INTERMAGNET.txt',
                          skiprows = 1,
                          sep = '\s+',
                          usecols=[0,1,2,3],
                          names = ['Imos','Latitude','Longitude','Elevation'],
                          index_col= ['Imos'])
    
    assert station in df_IMOS.index, 'station must be an INTERMAGNET observatory IAGA code'
    
    f = []
    f.extend(glob.glob('Dados OBS/*/*/' + station + '*'))
    f.sort()
    print('The first available date for ' + station.upper() + ' is ' +  f[0][21:29])
    print('The last available date for '  + station.upper() + ' is ' +  f[-1][21:29])
            
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
                filenames = ftp.nlst(file + '*') # get filenames within the directory
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
                   
def HDZ_to_XYZ_conversion(station,
                          dataframe,
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
                        nrows=40,
                        usecols = [0,3],
                        names = ['date','col'])
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
    

    
    def __init__(self,station,latitude,longitude,elevation):
        

            #self.station = station
        self.station = df_IMOS.loc[station].name
        self.latitude = latitude
        self.longitude = longitude
        self.elevation = elevation
        
    df_IMOS = pd.read_csv('Thesis_Marcos/Data/Imos informations/IMOS_INTERMAGNET.txt',
              skiprows = 1,
              sep = '\s+',
              usecols=[0,1,2,3],
              names = ['Imos','Latitude','Longitude','Elevation'],
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
    
    def elevation(station):
        
        if IMO.check_existence(station) == True:
            pass
        else:
            raise Exception('station not in the IMOS database, check the IAGA code or add the station.')
        
        return IMO.df_IMOS.loc[station]['Elevation']
    
    def delete(station: str):
    
        conn = sqlite3.connect('Imos_database.db')
        c = conn.cursor()
        c.execute('DELETE FROM Imos_informations WHERE Imos =?',(station,))
        conn.commit()
        sql_query = pd.read_sql_query ('''
                                   SELECT
                               *
                               FROM Imos_informations
                               ''', conn, index_col=['Imos'])
    
        sql_query.round(3).to_csv('Thesis_Marcos/Data/Imos informations/IMOS_INTERMAGNET.txt', sep = ' ')
        conn.close()
    
    def check_existence(station):
        
        if station not in IMO.df_IMOS.index:
            return False
        else:
            return True
        
    def add(station: str,latitude: float,longitude: float,elevation: float):
        
        
        conn = sqlite3.connect('Imos_database.db')
        c = conn.cursor()
        c.execute('INSERT INTO Imos_informations VALUES (?, ?, ?, ?)', (station,
                                                                                                       latitude,
                                                                                                       longitude,
                                                                                                       elevation))
        #rows = c.execute("SELECT * FROM Imos_informations").fetchall()
        #print(rows)
        conn.commit()
        sql_query = pd.read_sql_query ('''
                                   SELECT
                                   *
                                   FROM Imos_informations
                                   ''', conn, index_col=['Imos'])
        
        sql_query.round(3).to_csv('Thesis_Marcos/Data/Imos informations/IMOS_INTERMAGNET.txt', sep = ' ')
        conn.close()