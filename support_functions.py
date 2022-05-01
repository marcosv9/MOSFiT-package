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
#from Thesis_Marcos import support_functions as spf


def update_qd_and_dd(data):
    '''
    
    '''
    
    if data not in ['DD','QD']:
        print('Data must be QD or DD!')
        
    path = 'Thesis_Marcos/Data/Disturbed and Quiet Days/' 
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
        
def Header_SV_obs_files(station,
                        filename,
                        data_denoise,
                        external_correction,
                        chaos_model):
    '''
    Function to add a header in the txt outputs of the Function SV_obs.
    
    Output with informations about the used observatory
    and used data processing options.
    
    ---------------------------------------------------------------------------------------
    Inputs:
    Station - 
    filename - 
    data_denoise - 
    exernal_correction -
    chaos_model
    
    
    '''  
    df_IMOS = pd.read_csv('Thesis_Marcos/Data/Imos informations/Imos_INTERMAGNET.txt',
                          skiprows = 1,
                          sep = '\s+',
                          usecols=[0,1,2,3],
                          names = ['Imos','Latitude','Longitude','Elevation'],
                          index_col= ['Imos'])
    
    assert station in df_IMOS.index, 'station must be an INTERMAGNET observatory IAGA code'
    
    path = 'Filtered_data/'+ station +'_data/'+ station.upper() + '_' + filename +'_preliminary.txt'
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


    df_IMOS = pd.read_csv('Thesis_Marcos/Data/Imos informations/Imos_INTERMAGNET.txt',
                          skiprows = 1,
                          sep = '\s+',
                          usecols=[0,1,2,3],
                          names = ['Imos','Latitude','Longitude','Elevation'],
                          index_col= ['Imos'])
    
    
    Header = ('Thesis project provisory header\nIAGA CODE '
              + str(station.upper()) + '\nLatitude ' + 
              str(df_IMOS.loc[station.upper()]['Latitude'].round(2))
              +'\nLongitude ' + 
              str(df_IMOS.loc[station.upper()]['Longitude'].round(2)) +
              '\nElevation ' +
              str(df_IMOS.loc[station.upper()]['Elevation'].round(2)) +
              '\nData denoise: ' +
              denoise_options[data_denoise] +
              '\nExternal field reduction: ' + 
              external_options[external_correction] +
              '\nCHAOS model correction: ' +
              chaos_options[chaos_model] +  
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
    
def data_type(station, starttime, endtime, files_path = None):
    '''
    Function to verify the presence of Quasi-definitive data in the dataset
    
    ---------------------------------------------------------------------------------
    Inputs:
    
    station - 3 letters IAGA code
    starttime - Begin of the time interest
    endtime - The end of the period.
    --------------------------------------------------------------------------------------
    Usage example:
    load_INTERMAGNET_files(station = 'VSS', starttime = '2006-01-25', endtime = '2006-01-25')
    
    
    '''
    df_IMOS = pd.read_csv('Thesis_Marcos/Data/Imos informations/Imos_INTERMAGNET.txt',
                          skiprows = 1,
                          sep = '\s+',
                          usecols=[0,1,2,3],
                          names = ['Imos','Latitude','Longitude','Elevation'],
                          index_col= ['Imos'])
    
    assert station in df_IMOS.index, 'station must be an INTERMAGNET observatory IAGA code'
    
    if files_path != None:
        if files_path[-1] == '/':
            pass
        else:
            files_path = files_path + '/'  

    files_station = [] 

    if endtime >= '2018-06-30':

        years_interval = np.arange(int(starttime[0:4]),int(endtime[0:4])+ 1)
    
        if files_path == None:
            for year in years_interval:
    
                files_station.extend(glob.glob('Dados OBS\\' + str(year) + '/*/' + station + '*'))
                files_station.sort()
        else:
            files_station.extend(glob.glob(files_path + station + '*'))
            files_station.sort()
    
        #d_parser = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S%.f')
        df_data_type = pd.DataFrame()
        df_data_type = pd.concat((pd.read_csv(file,sep = '\s+',
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
    else:
        Date = []    
        
    return Date

def Header_SV_files(station, data_denoise, external_correction, chaos_model):
    '''    
    '''
    
    df_IMOS = pd.read_csv('Thesis_Marcos/Data/Imos informations/Imos_INTERMAGNET.txt',
                          skiprows = 1,
                          sep = '\s+',
                          usecols=[0,1,2,3],
                          names = ['Imos','Latitude','Longitude','Elevation'],
                          index_col= ['Imos'])
    
    assert station in df_IMOS.index, 'station must be an INTERMAGNET observatory IAGA code'
    
    path = 'SV_update/'+ station +'_data/SV_'+ station.upper() + '_preliminary.txt'
    path_header = 'SV_update/'+ station +'_data'
    destiny_path = 'SV_update/'+ station +'_data/SV_'+ station.upper() + '.txt'
    
    external_options = {'D': 'Disturbed Days removed',
                           'Q': 'Quiet Days selection', 
                           'NT': 'Night Time selection',
                          None: 'No'}
    
    denoise_options = {True: 'Hampel Filter',False: 'No'}
    
    chaos_options = {True: 'Yes',False: 'No'}
    
    #pathlib.Path(destiny_path).mkdir(parents=True, exist_ok=True)

    df_station = pd.read_csv(path,sep = '\s+', index_col = [0])
    df_station.index = pd.to_datetime(df_station.index,infer_datetime_format=True)


    df_IMOS = pd.read_csv('Thesis_Marcos/Data/Imos informations/Imos_INTERMAGNET.txt',
                          skiprows = 1,
                          sep = '\s+',
                          usecols=[0,1,2,3],
                          names = ['Imos','Latitude','Longitude','Elevation'],
                          index_col= ['Imos'])
    
    
    Header = ('Thesis project provisory header\nIAGA CODE '
              + str(station.upper()) + '\nLatitude ' + 
              str(df_IMOS.loc[station.upper()]['Latitude'].round(2))
              +'\nLongitude ' + 
              str(df_IMOS.loc[station.upper()]['Longitude'].round(2)) +
              '\nElevation ' +
              str(df_IMOS.loc[station.upper()]['Elevation'].round(2))+
              '\nData denoise: ' +
              denoise_options[data_denoise] +
              '\nExternal field reduction: ' +
              external_options[external_correction] +
              '\nCHAOS model correction: ' +
              chaos_options[chaos_model] +  
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
    try:
        datetime.strptime(str_date, '%Y-%m-%d')
    except ValueError:
        raise ValueError('Incorrect date format, should be YYYY-MM-DD')

def validate_YM(str_date):
    try:
        datetime.strptime(str_date, '%Y-%m')
    except ValueError:
        raise ValueError('Incorrect ' + str_date + ' format, should be YYYY-MM')

class year(object):
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
    values_list = [[],[]]
    for file in files_station:
        idx = 0
        skiprows = 12
        x = pd.read_csv(file,sep = '\s+',skiprows = skiprows,nrows=40, usecols = [0], names = ['col'])
        file = file
        while x['col'][idx] != 'DATE':
            skiprows += 1
            idx +=1 
            if x['col'][idx] == 'DATE':
                skiprows += 1
                values_list[0].append(skiprows)     
                values_list[1].append(file)
    return values_list            