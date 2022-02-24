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
from Thesis_Marcos import support_functions as spf
from Thesis_Marcos import data_processing_tools as dpt


def check_data_availability(station):
    '''
    check the available data period, based on the IAGA code.
    
    
    '''
    
    assert len(station) == 3, 'station must be a three letters IAGA Code'
    
    f = []
    f.extend(glob.glob('Dados OBS/*/*/' + station + '*'))
    f.sort()
    print('The first available date for ' + station.upper() + ' is ' +  f[0][21:29])
    print('The last available date for '  + station.upper() + ' is ' +  f[-1][21:29])
            
def download_data_INTERMAGNET(datatype, Year, Months, files = None):
    
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
    List_Months = ['01','02','03','04','05','06','07','08','09','10','11','12']
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
                          starttime,
                          endtime):
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
    
    for i in [starttime,endtime]:
        spf.validate(i)
        
        
    
    df_station = dataframe
    
    year  = []
    Reported = []
    for i in range(int(starttime[0:4]),int(endtime[0:4])+ 1):
        Y = i
        year.append(Y)
    
    Years = []
    Years.extend([str(i) for i in year])
    Years
    #print(starttime)
    files_station = []

    
    
    for Year in Years:

    
        files_station.extend(glob.glob('Dados OBS\\' + Year + '/*/' + station + '*'))
        files_station.sort()

    #d_parser = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S%.f')
    df_reported = pd.DataFrame()
    df_reported = pd.concat( (pd.read_csv(file,sep = '\s+',
                                         header = None,                                         
                                         skiprows = 7,
                                         nrows = 25,
                                         usecols = [0,1],
                                         names = ['Date','Reported'])
                             for file in files_station))
    
    Reported = df_reported.loc[[0],['Reported']]

    date = df_reported.loc[[21],['Date']]
    #date.set_index('Date', inplace = True)

    Reported.set_index(date['Date'], inplace = True)
    
    Reported.sort_index()
    #print(df_data_type)
    
    #Reported_HDZ = Reported.loc[Reported['Reported'] == 'HDZF']
    
    Date_HDZ = pd.to_datetime(Reported.loc[Reported['Reported'] == 'HDZF'].index)
    
    for date in Date_HDZ.year.drop_duplicates():
    #print(date)
    #Data_HDZ = pd.concat([Data_HDZ,df_HER.loc[str(date)]])
        D = np.deg2rad(df_station['Y'].loc[str(date)]/60)
        X = df_station['X'].loc[str(date)]*np.cos(D)
        Y = df_station['X'].loc[str(date)]*np.sin(D)
        
        df_station['X'].loc[str(date)] = X
        df_station['Y'].loc[str(date)] = Y
    
                 #for Reported in df_data_type['Reported']: 
    #    
    #    Reported = df_data_type.loc[df_data_type['Reported'] == 'HDZF']
    #calculating declination
    #D = np.deg2rad(df_station['Y'].loc[(df_station.index >= Reported.index[0] + ' 00:00:00') &
    #                               (df_station.index <= Reported.index[-1] + ' 23:59:59')]/60)
    #
    ##converting H and D to X and Y.
    #
    #x = df_station['X'].loc[(df_station.index >= Reported.index[0] + ' 00:00:00') &
    #                    (df_station.index <= Reported.index[-1] + ' 23:59:59')]*np.cos(D)
    #
    #y = df_station['X'].loc[(df_station.index >= Reported.index[0] + ' 00:00:00') &
    #                    (df_station.index <= Reported.index[-1] + ' 23:59:59')]*np.sin(D)
    #
    #df_station['X'].loc[(df_station.index >= Reported.index[0] + ' 00:00:00') & 
    #                (df_station.index <= Reported.index[-1] + ' 23:59:59')] = x
    #df_station['Y'].loc[(df_station.index >= Reported.index[0] + ' 00:00:00') &
    #                (df_station.index <= Reported.index[-1] + ' 23:59:59')] = y 
#
    
    
    return df_station

