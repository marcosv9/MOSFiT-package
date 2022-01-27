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


def check_data_availability(station):
    '''
    check the available data period, based of the IAGA code.
    
    '''
    f = []
    f.extend(glob.glob('Dados OBS/*/*/' + station + '*'))
    f.sort()
    print('The first available date for ' + station.upper() + ' is ' +  f[0][21:29])
    print('The last available date for '  + station.upper() + ' is ' +  f[-1][21:29])
    
def update_qd_and_dd(data):
    
    
    if data not in ['DD','QD']:
        print('Data must be QD or DD!')
        
    path = 'Dados OBS/Data/Disturbed and Quiet Days/' 
    files = glob.glob(path + 'qd*')
    files.sort()
    
    if data == 'DD':

        df = pd.concat((pd.read_csv(file,skiprows = 4,sep = '\s+',
                    header = None,
                    usecols = [0,1,12,13,14,15,16],
                    names = ['Month','Year','D1','D2','D3','D4','D5'])
                        for file in files),
                       ignore_index=True)
        
         
        columns = ['D1','D2','D3','D4','D5']
        
        df['Month'] = pd.to_datetime(df.Month, format='%b').dt.month
        
        for col in columns:
            df['Test' +  col] = df['Year'].astype(str)+'-'+df['Month'].astype(str)+'-'+df[col].astype(str)
        for col in columns:
            df['Test' + col] = df['Test' + col].str.replace('*','')
        
        df_DD = pd.DataFrame()
        
        df_DD['DD'] = pd.concat([df['TestD1'],df['TestD2'],df['TestD3'],df['TestD4'],df['TestD5']])
        
        df_DD['DD'] = pd.to_datetime(df_DD['DD'],infer_datetime_format=True)
        
        
        df_DD.set_index('DD', inplace = True)
        
        df_DD = df_DD.sort_index()
        
        df_DD.to_csv(path + 'Disturbed_Days_list.txt',index = True)
        
    if data == 'QD':
        df = pd.concat((pd.read_csv(file,
                                    skiprows = 4,
                                    sep = '\s+',
                                    header = None,
                                    usecols = [0,1,2,3,4,5,6,7,8,9,10,11],
                                    names = ['Month','Year','Q1','Q2','Q3','Q4','Q5','Q6','Q7','Q8','Q9','Q10'])
                        for file in files),
                       ignore_index = True)
        
        columns = ['Q1','Q2','Q3','Q4','Q5','Q6','Q7','Q8','Q9','Q10'] 
        
        df['Month'] = pd.to_datetime(df.Month, format='%b').dt.month
        
        for col in columns:
            df['Test' +  col] = df['Year'].astype(str)+'-'+df['Month'].astype(str)+'-'+df[col].astype(str)
        for col in columns:
            df['Test' + col] = df['Test' + col].str.replace('A','')
        for col in columns:
            df['Test' + col] = df['Test' + col].str.replace('K','')
        
        df_QD = pd.DataFrame()
        df_QD['QD'] = pd.concat([df['TestQ1'],df['TestQ2'],df['TestQ3'],df['TestQ4'],df['TestQ5'],df['TestQ6'],df['TestQ7'],df['TestQ8'],df['TestQ9'],df['TestQ10']])
        
        df_QD['QD'] = pd.to_datetime(df_QD['QD'],infer_datetime_format=True)
        
        
        df_QD.set_index('QD', inplace = True)
        
        df_QD = df_QD.sort_index()
        
        df_QD.to_csv(path + 'Quiet_Days_list.txt',index = True)
        
def Header_SV_obs_files(station, filename, data_denoise, external_correction, chaos_model):
    
    #filenames = ['minute_mean','hourly_mean','daily_mean','monthly_mean','annual_mean']
     
    
    path = 'Filtered_data/'+ station +'_data/'+ station.upper() + '_' + filename +'_preliminar.txt'
    path_header = 'Filtered_data/'+ station +'_data'
    destiny_path = 'Filtered_data/'+ station +'_data/'+ station.upper() + '_' + filename +'.txt'
    
    external_options = {'D': 'Disturbed Days removed',
                           'Q': 'Quiet Days selection', 
                           'NT': 'Night Time selection',
                          'E': 'No'}
    
    denoise_options = {'y': 'Hampel Filter','n': 'No'}
    
    chaos_options = {'y': 'Yes','n': 'No'}
    
    #pathlib.Path(destiny_path).mkdir(parents=True, exist_ok=True)

    df_station = pd.read_csv(path,sep = '\s+', index_col = [0])
    df_station.index = pd.to_datetime(df_station.index,infer_datetime_format=True)


    df_IMOS = pd.read_csv('IMOS_INTERMAGNET.txt', sep = '\s+')  
    df_IMOS.set_index('Imos', inplace = True)
    
    
    Header = ('Thesis project provisory header\nIAGA CODE '
              + str(station.upper()) + '\nLatitude ' + 
              str(df_IMOS.loc[station.upper()]['Latitude'].round(2))
              +'\nLongitude ' + 
              str(df_IMOS.loc[station.upper()]['Longitude'].round(2)) +'\nData denoise: ' + denoise_options[data_denoise] +
              '\nExternal field reduction: ' + external_options[external_correction] + '\nCHAOS model correction: ' + chaos_options[chaos_model] +  
              '\n\n')
    
    with open(path_header +'/header_file.txt','w+') as f2:
        f2.write(Header)
        header = f2.read()
    
    filenames = [path_header +'/header_file.txt',path]
    with open(destiny_path, 'w') as outfile:
        for fname in filenames:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)
                    
    
    os.remove(path_header +'/header_file.txt')
    os.remove(path)
    
def data_type(station, starttime, endtime):
    '''
    Function to read and concat observatory data.
    Works with every INTERMAGNET Observatorie, QD and definitive data.
    
    Inputs:
    station - 3 letters IAGA code
    starttime - Begin of the time interest
    endtime - The end of the period.
    
    Usage example:
    load_INTERMAGNET_files(station = 'VSS', starttime = '2006-01-25', endtime = '2006-01-25')
    
    
    '''
    year  = []
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
    df_data_type = pd.DataFrame()
    df_data_type = pd.concat( (pd.read_csv(file,sep = '\s+',
                                         header = None,                                         
                                         skiprows = 11,
                                         nrows = 20,
                                         usecols = [0,2],
                                         names = ['Date','Data_Type'])
                             for file in files_station))
    
    data_type = df_data_type.loc[[0],['Data_Type']]

    date = df_data_type.loc[[15],['Date']]
    #date.set_index('Date', inplace = True)

    data_type.set_index(date['Date'], inplace = True)
    
    data_type.sort_index()
    #print(df_data_type)
    
    for Date in data_type['Data_Type']: 
        if Date == 'Definitive':
            Date = []
        else:
        
            Date = data_type.loc[data_type['Data_Type'] != 'Definitive'].index[0]
    
    
    return Date

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
               
def download_obs_files_from_INTERMAGNET_ftp_server(path, destiny_path, files):
    '''
    Download observatory files from Intermagnet FTP server and save in your
    destiny_path.
    
    Necessary implementations -
    Automatically create the destiny_path if the user set a nonexistent folder
    A more intelligent way to set the files
    
    
    
    Ex 
    
    download_obs_files_from_INTERMAGNET_ftp_server(path = 'intermagnet/minute/quasi-definitive/IAGA2002/2021/07',
                                             destiny_path = 'C:\\test\\07\\',
                                             files = 'thy*07*')
        
    
    '''
    
#path = path
    ftp = ftplib.FTP('seismo.nrcan.gc.ca')
    ftp.login('anonymous', 'email@email.com')
    ftp.cwd(path)


#ftp.login('anonymous', 'email@email.com')
    
    print('Connected to Intermagnet FTP server')
    filenames = ftp.nlst(files) # get filenames within the directory
    filenames.sort()
    print('List of files that will be downloaded')
    for filename in filenames:

        print(filename)
#filename = 'box20200510dmin.min.gz'
    while input("Do You Want To Continue? [y/n]") == "y":
        for filename in filenames:    
            print('File',filename, 'downloaded!')   
            local_filename = os.path.join(destiny_path, filename)
            file = open(local_filename, 'wb')
            ftp.retrbinary('RETR '+ filename, file.write)

            file.close()
        break
    ftp.quit()

    print('Disconnected from INTERMAGNET Ftp server!')  