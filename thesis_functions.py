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



def load_obs_files(station, path, skiprows):
    '''
    Function to read and concat observatory data xxxx
    
    '''
    files_station = glob.glob(path)
    files_station.sort()
    print('Ther number of files readed is:',len(files_station),'.')
    
    
    #d_parser = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S%.f')
    
    
    df_station = pd.concat( (pd.read_csv(file, sep='\s+',usecols = [0,1,3,4,5], 
                   header = None,skiprows = skiprows,encoding='latin-1', 
                   parse_dates = {'Date': ['date', 'Time']},
                   names = ['date','Time','X','Y','Z']) for file in files_station), 
                   ignore_index = True)
    df_station['Date'] = pd.to_datetime(df_station['Date'],infer_datetime_format=True)     
    #df_station['Hour'] = pd.to_datetime(df_station['Hour'], format = '%H:%M:%S.%f').dt.time               
    df_station.set_index('Date', inplace = True)

    return df_station    


def load_obs_files_OTIMIZADA(station, starttime, endtime):
    '''
    Function to read and concat observatory data.
    Works with every INTERMAGNET Observatorie, QD and definitive data.
    
    Inputs:
    station - 3 letters IAGA code
    starttime - Begin of the time interest
    endtime - The end of the period.
    
    Usage example:
    load_obs_files_OTIMIZADA(station = 'VSS', starttime = '2006-01-25', endtime = '2006-01-25')
    
    
    '''
    for i in [starttime,endtime]:
        validate(i)
    
    print('Reading files from '+ station.upper() +'...')
    year  = []
    for i in range(int(starttime[0:4]),int(endtime[0:4])+ 1):
        Y = i
        year.append(Y)
    
    Years = []
    Years.extend([str(i) for i in year])
    Years
    
    files_station = []
    
    L_27 = ['CZT','DRV','PAF']
    L_26 = ['NGK','MAW','CNB','HAD','TSU','HON','KAK','BOU','KOU','HBK','BMT']
    
    skiprows = 26
    if station.upper() in L_27:
        skiprows = 27
    if station.upper() in L_26:
        skiprows = 26
    
    for Year in Years:

    
        files_station.extend(glob.glob('Dados OBS\\' + Year + '/*/' + station + '*'))
        files_station.sort()

    #d_parser = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S%.f')
    df_station = pd.DataFrame()
    df_station = pd.concat( (pd.read_csv(file, sep='\s+',usecols = [0,1,3,4,5], 
                   header = None,skiprows = skiprows, 
                   parse_dates = {'Date': ['date', 'Time']},
                   names = ['date','Time','X','Y','Z']) for file in files_station), 
                   ignore_index = True)
    df_station['Date'] = pd.to_datetime(df_station['Date'], format = '%Y-%m-%dd %H:%M:%S.%f')     
    #df_station['Hour'] = pd.to_datetime(df_station['Hour'], format = '%H:%M:%S.%f').dt.time               
    df_station.set_index('Date', inplace = True)

    
    df_station.loc[df_station['X'] >= 99999.0, 'X'] = np.nan
    df_station.loc[df_station['Y'] >= 99999.0, 'Y'] = np.nan
    df_station.loc[df_station['Z'] >= 99999.0, 'Z'] = np.nan
    #df_station.loc[df_station['F'] == 99999.00, 'F'] = np.nan
    #df_station = df_station.dropna()
    df_station = df_station.loc[starttime:endtime]
    
    return df_station

#def load_obs_files_OTIMIZADA(station, skiprows, starttime, endtime):
#    '''
#    Function to read and concat observatory data
#    
#    Sample must be H, D, M, Y   
#    
#    '''
#    print('Reading files from '+ station.upper() +'...')
#    year  = []
#    for i in range(int(starttime[0:4]),int(endtime[0:4])+ 1):
#        Y = i
#        year.append(Y)
#    
#    Years = []
#    Years.extend([str(i) for i in year])
#    Years
#    
#    files_station = []
#    
#    for Year in Years:
#        
#        files_station.extend(glob.glob('Dados OBS\\' + Year + '/*/' + station + '*'))
#        files_station.sort()
#        
#        if files_station != []:
#            
#            df_station = pd.concat( (pd.read_csv(file, sep='\s+',usecols = [0,1,3,4,5], 
#                   header = None,skiprows = skiprows, 
#                   parse_dates = {'Date': ['date', 'Time']},
#                   names = ['date','Time','X','Y','Z']) for file in files_station), 
#                   ignore_index = True)
#            df_station['Date'] = pd.to_datetime(df_station['Date'], format = '%Y-%m-%dd %H:%M:%S.%f')     
#    #df_station['Hour'] = pd.to_datetime(df_station['Hour'], format = '%H:%M:%S.%f').dt.time               
#            df_station.set_index('Date', inplace = True)
#
#        else:
#            
#            print('no files to read')
#            
#            break            
#
#    #d_arser = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S%.f')
#
#
#    return df_station

def data_filter_basic(component, vmin, vmax):
    '''
    Filter the data based on informed max and min values..
    
    Too slow for big datasets
    '''
    
    Filtered_comp = np.asarray(component)
    #comp_new = np.asarray(component)
    
    
    Filtered_comp = [number for number in component if number < vmin or number > vmax]
    #print(Filtered_comp)
    component = component.replace(Filtered_comp,np.nan)
    
    return component

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
    
def remove_Disturbed_Days(dataframe, start, end):
    ''' 
    Function created to remove geomagnetic disturbed 
    days from observatory geomagnetic data.

    Ther list of geomagnetic disturbed days used is available
    on https://www.gfz-potsdam.de/en/kp-index/. It is update every month.
    
    The data need to be a dataframe.
    
    August-2021 Fow now, the output is a data without disturbed days in a daily mean. I need to improve it!!!!
    
    Implement to not be required to inform start and end period (only if the user want to)    
    
    Function(dataframe = Name of your dataframe ex: df_VSS, start = date that you want to start removing DD, end = the same thing )
    Ex  - df_NGK2 = mvs.remove_Disturbed_Days(df_NGK, start = '2010-01' , end = '2019-12')
    
    '''
    for i in [start,end]:
        validate(i)
    
    df = pd.DataFrame()
    df = dataframe.resample('D').mean()
    #df = dataframe
    df_d = pd.read_excel('Dados OBS/DIS_DAYS.xls', skiprows = 1, 
                     usecols = [0],
                     names = ['dd'],
                     parse_dates = {'D-Days': ['dd']},
                    )
    df_d['D-Days'] = pd.to_datetime(df_d['D-Days'], format = '%YYYY-%mm-%dd')

    df_d.set_index('D-Days', inplace = True)
    
    df_d = df_d.sort_values('D-Days')
    
    first_day = str(df.index[0].year) + '-' + str(df.index[0].month) + '-' + str(df.index[0].day)
    last_day = str(df.index[-1].year) + '-' + str(df.index[-1].month) + '-' + str(df.index[-1].day)
    
    
    remove = df_d.loc[first_day:last_day]
    
    df = df.drop(remove.index)
    
    
    return df

def keep_Q_Days(dataframe, start, end):
    ''' 
    Function created to keep only geomagnetic quiet 
    days from observatory geomagnetic data.

    Ther list of geomagnetic quiet days used is available
    on https://www.gfz-potsdam.de/en/kp-index/. It is update every month.
    
    The data need to be a pandas DataFrame.
    
    October-2021 Fow now, the output is a data with only quiet days.
    
    Implement to not be required to inform start and end period (only if the user want to)    
    
    Function(dataframe = Name of your dataframe ex: df_VSS, start = date that you want to start removing DD, end = the same thing )
    
    Ex  - df_NGK2 = keep_Q_Days(df_NGK, start = '2010-01' , end = '2019-12')
    '''
    
    for i in [start,end]:
        validate(i)
    
    df = pd.DataFrame()
    df = dataframe.resample('D').mean()
    #df = dataframe
    df_q = pd.read_excel('Dados OBS/Q_DAYS.xls',header = None,skiprows = 1, 
                     usecols = [1],
                     names = ['qd'],
                     parse_dates = {'Q-Days': ['qd']},
                    )
    df_q['Q-Days'] = pd.to_datetime(df_q['Q-Days'], format = '%YYYY-%mm-%dd')
    df_q = df_q.sort_values('Q-Days')
    df_q.set_index('Q-Days', inplace = True)
    
    keep = df_q.loc[start:end]
    #for i in range(df.index):
    df = df[start:end].reindex(df_q.index)
 
        
    #if df.index not in keep.index:
        #df = df.drop(df.index)
    return df

def night_time_selection(dataframe, start, end):
    
    for i in [start,end]:
        validate(i)
    df = pd.DataFrame()
    df = dataframe
    df = df.loc[start:end]
    #df = df.loc[(df.index.hour >= 19)]
    
    df = df.drop(df.loc[(df.index.hour > 5) & (df.index.hour < 23)].index).dropna()
    
    return df

def SV_obs(station, starttime, endtime):
    '''
    Function to read and concat observatory data
    
    Sample must be H, D, M, Y   
    
    '''
    
    df_station = load_obs_files_OTIMIZADA(station, starttime, endtime)
    
    df_station2 = df_station.copy()
    
    while True: 
        inp5 = input("Do You Want To denoise the data based on median absolute deviation? [y/n]: ")
        if inp5 == 'y':
            print('Denoising data...')
            df_station = hampel_filter_denoising(input_series = df_station, window_size = 100, n_sigmas=3)
            break
        if inp5 == 'n':
            print('No data changed!')
            break
        else:
            print('You must type y or n, try again!')
        
    
    options = ['Q','D','NT','E','C']
    while True: 
        inp = str(input("Press Q to use only Quiet Days, D to remove Disturbed Days, NT to use only the night-time, C to correct using Chaos model, or E to Exit without actions [Q/D/NT/C/E]: "))
        
        if all([inp != option for option in options]):
            print('You must type Q, D, NT, C or E')
        else:
            break
    
    if inp == 'Q':
        
        df_station = keep_Q_Days(df_station, starttime, endtime)
        print('Only Quiet Days remains')
    if inp == 'D':
        
        df_station = remove_Disturbed_Days(df_station, starttime, endtime)
        print('Disturbed Days removed')
        
    if inp == 'NT':
        
        df_station = NT_LT(station, df_station, starttime, endtime)
        print('Night-time selected')
        
    if inp == 'E':
        print('No action')
    
    if inp == 'C':
        
        df_station, df_chaos = external_field_correction_chaos_model(station = station,
                                                           starttime = starttime,
                                                           endtime = endtime,
                                                           df_station = df_station,
                                                           df_chaos = None)
        

        
    samples = ['Min','H','D','M','Y']
    directory = 'Filtered_data/'+ station +'_data'
    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)  
    

    df_SV = calculate_SV(df_station, starttime = starttime, endtime = endtime)
    
    while True: 
        inp2 = input("Do You Want To Save a File With the Variation? [y/n]: ")
        if inp2 == 'y':

            print('Saving files...')
            for sample in samples:
                          
                if sample == 'Min':
                
                    file = df_station2[starttime:endtime].resample(sample).mean().round(3)
                    #ax.plot(df_HER['X']['2010-01':'2019-12'].resample(sample).mean().shift(12, freq = 'H'))
                    file.to_csv(directory + '/' + station.upper() + '_from_'
                            + starttime +'_to_' + endtime + '_' + sample + '_mean.zip', sep ='\t', index=True)
            
                if sample == 'H':
                    
                    file = df_station2[starttime:endtime].resample(sample).mean().shift(30, freq = 'min').round(3)
                    #ax.plot(df_HER['X']['2010-01':'2019-12'].resample(sample).mean().shift(12, freq = 'H'))
                    file.to_csv(directory + '/' + station.upper() + '_from_' 
                                + starttime +'_to_' + endtime + '_' + sample + '_mean.zip', sep ='\t', index=True)
                
                if sample == 'D':
                    
                    file = df_station[starttime:endtime].resample(sample).mean().shift(12, freq = 'H').round(3)
                    #ax.plot(df_HER['X']['2010-01':'2019-12'].resample(sample).mean().shift(12, freq = 'H'))
                    file.to_csv(directory + '/' + station.upper() + '_from_' 
                                + starttime +'_to_' + endtime + '_' + sample + '_mean.zip', sep ='\t', index=True)
                    
                if sample == 'M':
                    
                   # ax.plot(df_HER['X']['2010-01':'2019-12'].resample(sample).mean().shift(-15, freq = 'D'))
                    file = df_station[starttime:endtime].resample(sample).mean().shift(-15, freq = 'D').round(3)
                    
                    file_SV = df_SV
                    
                    file.to_csv(directory + '/' + station.upper() + '_from_'
                                + starttime +'_to_' + endtime + '_' + sample + '_mean.zip', sep ='\t', index=True)
                    file_SV.to_csv(directory + '/' + station.upper() + '_from_'
                                + starttime +'_to_' + endtime + '_' + sample + '_SV.zip', sep ='\t', index=True)
                if sample == 'Y':
                    
                    #ax.plot(df_HER['X']['2010-01':'2019-12'].resample(sample).mean().shift(-182.5, freq = 'D'))
                    file = df_station[starttime:endtime].resample(sample).mean().shift(-182.5, freq = 'D').round(3)
                    file.to_csv(directory + '/' + station.upper() + '_from_' 
                                + starttime +'_to_' + endtime + '_' + sample + '_mean.zip', sep ='\t', index=True)
            print('Minute, Hourly, Daily, Monthly, Yearly means and Secular Variation were saved on directory:')
            print(directory)   
            break

        elif inp2 =='n':
            print('No files saved!')
            break
        else:
            print('You musy type y or n.')
        
        
    while True:
       
        inp3 = input("Do You Want To Save Plots of the Variation and SV for X, Y and Z? [y/n]: ")
        if inp3 == 'y':
            directory = 'Filtered_data/'+ station +'_data'
            pathlib.Path(directory).mkdir(parents=True, exist_ok=True)  
            
            #plot to use in the interactive map 
            
            fig, ax = plt.subplots(3,1, figsize = (8,6.5))
            
            ax[0].set_title(station.upper() + ' Secular Variation (ADMM)', fontsize = 12)
    
            ax[0].plot(df_SV['X'], 'o', color  = 'blue')
            ax[0].set_xlim(np.datetime64(starttime),np.datetime64(endtime))
            ax[0].set_ylim(df_SV['X'].min() - 3, df_SV['X'].max() + 3)
            #ax[0,0].set_xlim(np.datetime64('2011-01'),np.datetime64('2021-12'))
            ax[0].set_ylabel('dX/dT(nT/yr)', fontsize = 9)
            ax[0].grid()
            
            ax[1].plot(df_SV['Y'], 'o', color  = 'green')
            ax[1].set_xlim(np.datetime64(starttime),np.datetime64(endtime))
            ax[1].set_ylim(df_SV['Y'].min() - 3, df_SV['Y'].max() + 3)
            ax[1].set_ylabel('dY/dT(nT/yr)', fontsize = 9)
            ax[1].grid()
            
            ax[2].plot(df_SV['Z'], 'o', color  =  'black')
            ax[2].set_xlim(np.datetime64(starttime),np.datetime64(endtime))
            ax[2].set_ylim(df_SV['Z'].min() - 3, df_SV['Z'].max() + 3)
            #ax[0].set_xlim(np.datetime64('2010-01'),np.datetime64('2021-06'))
            ax[2].set_ylabel('dZ/dT(nT/yr)', fontsize = 9)
            ax[2].grid()
            
            plt.tick_params(labelsize=8)
            plt.savefig(directory + '/' + station + '_map_SV.jpeg', bbox_inches='tight')
            #plt.show()
            
            #plot of secular variation and monthly mean
            
            fig, ax = plt.subplots(3,2, figsize = (18,10))
            
            
            ax[0,1].set_title(station.upper() + ' Secular Variation (ADMM)', fontsize = 18)
            ax[0,1].plot(df_SV['X'], 'o', color  = 'blue')
            ax[0,1].set_xlim(np.datetime64(starttime),np.datetime64(endtime))
            ax[0,1].set_ylabel('dX/dT(nT/yr)', fontsize = 12)
            ax[0,1].grid()
            
            ax[1,1].plot(df_SV['Y'], 'o', color  = 'green')
            ax[1,1].set_xlim(np.datetime64(starttime),np.datetime64(endtime))
            ax[1,1].set_ylabel('dY/dT(nT/yr)', fontsize = 12)
            ax[1,1].grid()
            
            ax[2,1].plot(df_SV['Z'], 'o', color  =  'black')
            ax[2,1].set_xlim(np.datetime64(starttime),np.datetime64(endtime))
            ax[2,1].set_ylabel('dZ/dT(nT/yr)', fontsize = 12)
            ax[2,1].grid()
            
            
            ax[0,0].set_title(station.upper() + ' Monthly Mean', fontsize = 18)
            ax[0,0].plot(df_station['X'][starttime:endtime].resample('M').mean().shift(-15, freq = 'D'), color  = 'blue')
            ax[0,0].set_xlim(np.datetime64(starttime),np.datetime64(endtime))
            ax[0,0].set_ylabel('X/nT', fontsize = 14)
            
            ax[0,0].grid()
            
            ax[1,0].plot(df_station['Y'][starttime:endtime].resample('M').mean().shift(-15, freq = 'D'), color  = 'green')
            ax[1,0].set_xlim(np.datetime64(starttime),np.datetime64(endtime))
            ax[1,0].set_ylabel('Y/nT', fontsize = 14)
            
            ax[1,0].grid()
    
    
            ax[2,0].plot(df_station['Z'][starttime:endtime].resample('M').mean().shift(-15, freq = 'D'), color  = 'black')
            ax[2,0].set_xlim(np.datetime64(starttime),np.datetime64(endtime))
            ax[2,0].set_ylabel('Z/nT', fontsize = 14)
            #ax[2,1].set_xlabel('Years', fontsize = 14 )
            #ax[1].set_ylim(-30,30)
            ax[2,0].grid()
            
            plt.savefig(directory + '/' + station + '_Var_SV.jpeg', bbox_inches='tight')
            plt.show()      
            
             #plot of SV alone     
                  
            fig, ax = plt.subplots(3,1, figsize = (16,10))
            
            ax[0].set_title(station.upper() + ' Secular Variation (ADMM)', fontsize = 18)
    
            ax[0].plot(df_SV['X'], 'o', color  = 'blue')
            ax[0].set_xlim(np.datetime64(starttime),np.datetime64(endtime))
            ax[0].set_ylim(df_SV['X'].min() - 3, df_SV['X'].max() + 3)
            #ax[0,0].set_xlim(np.datetime64('2011-01'),np.datetime64('2021-12'))
            ax[0].set_ylabel('dX/dT(nT/yr)', fontsize = 12)
            ax[0].grid()
            
            ax[1].plot(df_SV['Y'], 'o', color  = 'green')
            ax[1].set_xlim(np.datetime64(starttime),np.datetime64(endtime))
            ax[1].set_ylim(df_SV['Y'].min() - 3, df_SV['Y'].max() + 3)
            ax[1].set_ylabel('dY/dT(nT/yr)', fontsize = 12)
            ax[1].grid()
            
            ax[2].plot(df_SV['Z'], 'o', color  =  'black')
            ax[2].set_xlim(np.datetime64(starttime),np.datetime64(endtime))
            ax[2].set_ylim(df_SV['Z'].min() - 3, df_SV['Z'].max() + 3)
            #ax[0].set_xlim(np.datetime64('2010-01'),np.datetime64('2021-06'))
            ax[2].set_ylabel('dZ/dT(nT/yr)', fontsize = 12)
            ax[2].grid()
            
            plt.savefig(directory + '/' + station + '_SV.jpeg', bbox_inches='tight')
            plt.show()
            
            for sample in samples:
                             
                fig, ax = plt.subplots(3,1,figsize = (16,10))
                
                if sample == 'Min':
                    ax[0].set_title(station.upper() + ' Minute Mean', fontsize = 18)
                    ax[0].plot(df_station2['X'].loc[starttime:endtime].resample(sample).mean(), color  = 'blue')
                    ax[1].plot(df_station2['Y'][starttime:endtime].resample(sample).mean(), color  = 'green')
                    ax[2].plot(df_station2['Z'][starttime:endtime].resample(sample).mean(), color  = 'black')
                if sample == 'H':    
                    ax[0].set_title(station.upper() + ' Hourly Mean', fontsize = 18)
                    ax[0].plot(df_station2['X'][starttime:endtime].resample(sample).mean().shift(30, freq = 'Min'), color  = 'blue')
                    ax[1].plot(df_station2['Y'][starttime:endtime].resample(sample).mean().shift(30, freq = 'Min'), color  = 'green')
                    ax[2].plot(df_station2['Z'][starttime:endtime].resample(sample).mean().shift(30, freq = 'Min'), color  = 'black')
                if sample == 'D':
                    ax[0].set_title(station.upper() + ' Daily Mean', fontsize = 18)
                    ax[0].plot(df_station['X'][starttime:endtime].resample(sample).mean().shift(-12, freq = 'H'), color  = 'blue')
                    ax[1].plot(df_station['Y'][starttime:endtime].resample(sample).mean().shift(-12, freq = 'H'), color  = 'green')
                    ax[2].plot(df_station['Z'][starttime:endtime].resample(sample).mean().shift(-12, freq = 'H'), color  = 'black')
                if sample == 'M':
                    ax[0].set_title(station.upper() + ' Monthly Mean', fontsize = 18)
                    ax[0].plot(df_station['X'][starttime:endtime].resample(sample).mean().shift(-15, freq = 'D'), color  = 'blue')
                    ax[1].plot(df_station['Y'][starttime:endtime].resample(sample).mean().shift(-15, freq = 'D'), color  = 'green')
                    ax[2].plot(df_station['Z'][starttime:endtime].resample(sample).mean().shift(-15, freq = 'D'), color  = 'black')           
                if sample == 'Y':
                    ax[0].set_title(station.upper() + ' Yearly Mean', fontsize = 18)
                    ax[0].plot(df_station['X'][starttime:endtime].resample(sample).mean().shift(-182.5, freq = 'D'),'o-', color  = 'blue')
                    ax[1].plot(df_station['Y'][starttime:endtime].resample(sample).mean().shift(-182.5, freq = 'D'),'o-', color  = 'green')
                    ax[2].plot(df_station['Z'][starttime:endtime].resample(sample).mean().shift(-182.5, freq = 'D'),'o-', color  = 'black')    
                    
              
                ax[0].set_ylabel('X (nT)', fontsize = 12)
                ax[0].set_xlim(np.datetime64(starttime),np.datetime64(endtime))
                #ax[0].set_ylim(df_station['X'][starttime:endtime].min()*0.9,df_station['X'][starttime:endtime].max()*1.1)
                ax[0].grid()
                
               
                ax[1].set_ylabel('Y (nT)', fontsize = 12)
                ax[1].set_xlim(np.datetime64(starttime),np.datetime64(endtime))
                #ax[1].set_ylim(df_station['Y'][starttime:endtime].min()*0.9,df_station['Y'][starttime:endtime].max()*1.1)
                ax[1].grid()
                
                
                ax[2].set_ylabel('Z (nT)', fontsize = 12)
                ax[2].set_xlim(np.datetime64(starttime),np.datetime64(endtime))
                #ax[2].set_ylim(df_station['Z'][starttime:endtime].min()*0.9,df_station['Z'][starttime:endtime].max()*1.1)
                ax[2].grid()
                
               
                plt.savefig(directory + '/' + station + '_' + sample + '_mean.jpeg', bbox_inches='tight')
                plt.show()
            print('Plots of Minute, Hourly, Daily, Monthly, Yearly means and Secular Variation were saved on directory:')
            print(directory)    
            break
        elif inp3 == 'n':
            print('No plots saved')
            fig, ax = plt.subplots(3,2, figsize = (18,10))
            
            
            ax[0,1].set_title(station.upper() + ' Secular Variation (ADMM)', fontsize = 18)
            ax[0,1].plot(df_SV['X'], 'o', color  = 'blue')
            ax[0,1].set_ylim(df_SV['X'].min() - 3, df_SV['X'].max() + 3)
            ax[0,1].set_xlim(np.datetime64(starttime),np.datetime64(endtime))
            ax[0,1].set_ylabel('dX/dT(nT/yr)', fontsize = 12)
            ax[0,1].grid()
            
            ax[1,1].plot(df_SV['Y'], 'o', color  = 'green')
            ax[1,1].set_ylim(df_SV['Y'].min() - 3, df_SV['Y'].max() + 3)
            ax[1,1].set_xlim(np.datetime64(starttime),np.datetime64(endtime))
            ax[1,1].set_ylabel('dY/dT(nT/yr)', fontsize = 12)
            ax[1,1].grid()
            
            ax[2,1].plot(df_SV['Z'], 'o', color  =  'black')
            ax[2,1].set_ylim(df_SV['Z'].min() - 3, df_SV['Z'].max() + 3)
            ax[2,1].set_xlim(np.datetime64(starttime),np.datetime64(endtime))
            ax[2,1].set_ylabel('dZ/dT(nT/yr)', fontsize = 12)
            ax[2,1].grid()
            
            
            ax[0,0].set_title(station.upper() + ' Monthly Mean', fontsize = 18)
            ax[0,0].plot(df_station['X'][starttime:endtime].resample('M').mean().shift(-15, freq = 'D'), color  = 'blue')
            ax[0,0].set_xlim(np.datetime64(starttime),np.datetime64(endtime))
            ax[0,0].set_ylabel('X/nT', fontsize = 14)
            
            ax[0,0].grid()
            
            ax[1,0].plot(df_station['Y'][starttime:endtime].resample('M').mean().shift(-15, freq = 'D'), color  = 'green')
            ax[1,0].set_xlim(np.datetime64(starttime),np.datetime64(endtime))
            ax[1,0].set_ylabel('Y/nT', fontsize = 14)
            
            ax[1,0].grid()
    
    
            ax[2,0].plot(df_station['Z'][starttime:endtime].resample('M').mean().shift(-15, freq = 'D'), color  = 'black')
            ax[2,0].set_xlim(np.datetime64(starttime),np.datetime64(endtime))
            ax[2,0].set_ylabel('Z/nT', fontsize = 14)
            #ax[2,1].set_xlabel('Years', fontsize = 14 )
            #ax[1].set_ylim(-30,30)
            ax[2,0].grid()
            
            plt.show()

            
             #plot of SV alone     
                  
            fig, ax = plt.subplots(3,1, figsize = (16,12))
            
            ax[0].set_title(station.upper() + ' Secular Variation (ADMM)', fontsize = 18)
    
            ax[0].plot(df_SV['X'], 'o', color  = 'blue')
            ax[0].set_xlim(np.datetime64(starttime),np.datetime64(endtime))
            ax[0].set_ylim(df_SV['X'].min() - 3, df_SV['X'].max() + 3)
            #ax[0,0].set_xlim(np.datetime64('2011-01'),np.datetime64('2021-12'))
            ax[0].set_ylabel('dX/dT(nT/yr)', fontsize = 12)
            ax[0].grid()
            
            ax[1].plot(df_SV['Y'], 'o', color  = 'green')
            ax[1].set_xlim(np.datetime64(starttime),np.datetime64(endtime))
            ax[1].set_ylim(df_SV['Y'].min() - 3, df_SV['Y'].max() + 3)
            #ax[0].set_xlim(np.datetime64('2010-01'),np.datetime64('2021-06'))
            ax[1].set_ylabel('dY/dT(nT/yr)', fontsize = 12)
            ax[1].grid()
            
            ax[2].plot(df_SV['Z'], 'o', color  =  'black')
            ax[2].set_xlim(np.datetime64(starttime),np.datetime64(endtime))
            ax[2].set_ylim(df_SV['Z'].min() - 3, df_SV['Z'].max() + 3)
            #ax[0].set_xlim(np.datetime64('2010-01'),np.datetime64('2021-06'))
            ax[2].set_ylabel('dZ/dT(nT/yr)', fontsize = 12)
            ax[2].grid()
            
            plt.show()
            
            
            #plt.show()
            
            for sample in samples:
                             
                fig, ax = plt.subplots(3,1,figsize = (16,10))
                
                if sample == 'Min':
                    ax[0].set_title(station.upper() + ' Minute Mean', fontsize = 18)
                    ax[0].plot(df_station2['X'][starttime:endtime].resample(sample).mean(), color  = 'blue')
                    ax[1].plot(df_station2['Y'][starttime:endtime].resample(sample).mean(), color  = 'green')
                    ax[2].plot(df_station2['Z'][starttime:endtime].resample(sample).mean(), color  = 'black')
                if sample == 'H':    
                    ax[0].set_title(station.upper() + ' Hourly Mean', fontsize = 18)
                    ax[0].plot(df_station2['X'][starttime:endtime].resample(sample).mean().shift(-30, freq = 'Min'), color  = 'blue')
                    ax[1].plot(df_station2['Y'][starttime:endtime].resample(sample).mean().shift(-30, freq = 'Min'), color  = 'green')
                    ax[2].plot(df_station2['Z'][starttime:endtime].resample(sample).mean().shift(-30, freq = 'Min'), color  = 'black')
                if sample == 'D':
                    ax[0].set_title(station.upper() + ' Daily Mean', fontsize = 18)
                    ax[0].plot(df_station['X'][starttime:endtime].resample(sample).mean().shift(-12, freq = 'H'), color  = 'blue')
                    ax[1].plot(df_station['Y'][starttime:endtime].resample(sample).mean().shift(-12, freq = 'H'), color  = 'green')
                    ax[2].plot(df_station['Z'][starttime:endtime].resample(sample).mean().shift(-12, freq = 'H'), color  = 'black')
                if sample == 'M':
                    ax[0].set_title(station.upper() + ' Monthly Mean', fontsize = 18)
                    ax[0].plot(df_station['X'][starttime:endtime].resample(sample).mean().shift(-15, freq = 'D'), color  = 'blue')
                    ax[1].plot(df_station['Y'][starttime:endtime].resample(sample).mean().shift(-15, freq = 'D'), color  = 'green')
                    ax[2].plot(df_station['Z'][starttime:endtime].resample(sample).mean().shift(-15, freq = 'D'), color  = 'black')           
                if sample == 'Y':
                    ax[0].set_title(station.upper() + ' Yearly Mean', fontsize = 18)
                    ax[0].plot(df_station['X'][starttime:endtime].resample(sample).mean().shift(-182.5, freq = 'D'),'o-', color  = 'blue')
                    ax[1].plot(df_station['Y'][starttime:endtime].resample(sample).mean().shift(-182.5, freq = 'D'),'o-', color  = 'green')
                    ax[2].plot(df_station['Z'][starttime:endtime].resample(sample).mean().shift(-182.5, freq = 'D'),'o-', color  = 'black')    
                    
              
                ax[0].set_ylabel('X (nT)', fontsize = 12)
                ax[0].set_xlim(np.datetime64(starttime),np.datetime64(endtime))
                #ax[0].set_ylim(df_station['X'][starttime:endtime].min()*0.9,df_station['X'][starttime:endtime].max()*1.1)
                ax[0].grid()
                
               
                ax[1].set_ylabel('Y (nT)', fontsize = 12)
                ax[1].set_xlim(np.datetime64(starttime),np.datetime64(endtime))
                #ax[1].set_ylim(df_station['Y'][starttime:endtime].min()*0.9,df_station['Y'][starttime:endtime].max()*1.1)
                ax[1].grid()
                
                
                ax[2].set_ylabel('Z (nT)', fontsize = 12)
                ax[2].set_xlim(np.datetime64(starttime),np.datetime64(endtime))
                #ax[2].set_ylim(df_station['Z'][starttime:endtime].min()*0.9,df_station['Z'][starttime:endtime].max()*1.1)
                ax[2].grid()
                plt.show()
                
                #plt.show()
                      
            break
        else:
            print('You must type y or n!')
    #file = df_station.round(decimals = 3)
    #file.to_csv('Filtered_data/'+ station + '_from_' + starttime +
    #            '_to_' + endtime + '.txt', sep ='\t', index=True)
    
    while True:
        condition = input("Do You Want To adopt piecewise linear segments for the SV? [y/n]: ")
        pass
        if condition == 'y':
            try:  
                ls = input('Type the number of linear segments that best fit the SV: ')
                list_ls = [int(k) for k in ls.split(" ")]
                
                        #create an option to save or not a plot
                jerk_detection(station, df_station, list_ls, starttime, endtime)
            except:
                print("""This is not the correct format. Please reenter. (correct format: 
                       integers separated by spaces)""")
                continue
            else:
                break
        if condition == 'n':
            print('No linear segments adopted')
            break
        else:
            print('You must type y or n, try again!')
            
    return df_station[starttime:endtime]

def check_data_availability(station):
    '''
    check the available data period, based of the IAGA code.
    
    '''
    f = []
    f.extend(glob.glob('Dados OBS/*/*/' + station + '*'))
    f.sort()
    print('The first available date for ' + station.upper() + ' is ' +  f[0][21:29])
    print('The last available date for '  + station.upper() + ' is ' +  f[-1][21:29])
    
def NT_LT(station, dataframe, start, end):
    f = []
    f.extend(glob.glob('Dados OBS/*/*/' + station + '*'))
    f.sort()
    
    Long = pd.read_csv('Dados OBS/' + f[0][21:25] +'/' + f[0][25:27] + '/' + station + f[0][21:41],
            nrows = 1, 
            sep = ' ', 
            usecols = [7],
            header  = None,
            names = ['Geodetic Longitude'],
            index_col=None,
            skiprows = 5)
    
    Longitude = Long['Geodetic Longitude'][0]
   
    if Longitude > 180:
        Longitude = Longitude - 360 
    print(Longitude)
    dif =  Longitude/15
    print(dif)
    
    df = dataframe
    df_lt = df.shift(round(dif, 3), freq = 'H')
    
    df_NT_lt = night_time_selection(df_lt,start, end)
    df_NT = pd.DataFrame()
    df_NT = df_NT_lt.shift(round(-dif, 3), freq = 'H')
    #mini = 22 + dif
    #maxi = 2 + dif
    #if mini > 24:
    #    mini -= 24
    #elif mini < 0:
    #    mini += 24
    #if maxi > 24:
    #    maxi -= 24
    #elif maxi < 0:
    #    maxi += 24
    #print(mini,'mini')
    #print(maxi,'maxi')
    #df = pd.DataFrame()
    #df = dataframe
    #df = df.loc[start:end]
    ##df = df.loc[(df.index.hour >= 19)]
    #print(df)
    #if mini > maxi:
    #    df = df.drop(df.loc[(df.index.hour > int(maxi)) & (df.index.hour < int(mini))].index).dropna() 
    #else:
    #    df = df.loc[((df.index.hour > int(mini)) & (df.index.hour <= int(maxi)))].dropna()
    
    return df_NT

def jerk_detection(station, dataframe,ls, starttime, endtime):
    
    '''
    adopt piecewise linear segments on the secular variation
    
    Inputs:
    
    station = IAGA code
    dataframe = Pandas dataframe with the secular variation
    from the observatory.
    ls = Number of linear segments, must be a list containing 3 numbers (one for each component)
    starttime = initial period
    endtime = final period
    
    usage example:
    jerk_detection(station = 'VSS', dataframe = df_VSS, ls = [3,4,3],
                   starttime = '2005-01-01', endtime = '2021-09-30')
    
    '''
    #ls = []
    df_station = pd.DataFrame()
    df_station = dataframe
    df_SV = pd.DataFrame()
    #fit your data (x and y)
    components = ['X','Y','Z']
    
    
    df_SV = calculate_SV(df_station, starttime = starttime, endtime = endtime)
    
    X = np.arange(0,len(df_SV.index))
    
    for (component,i) in zip(components,ls):
        
        myPWLF = pwlf.PiecewiseLinFit(X,df_SV[component])
        
        #fit the data for n line segments
        z = myPWLF.fit(i)
        #calculate slopes
        #slopes = myPWLF.calc_slopes()
        
        # predict for the determined points
        xHat = X 
        yHat = myPWLF.predict(xHat)
        
        #calculate statistics
        #p = myPWLF.p_values(method='non-linear', step_size=1e-4) #p-values
        #se = myPWLF.se  # standard errors
        
        df_SV[str(component) + 'p'] = yHat
        
        
    directory = 'Filtered_data/'+ station +'_data'
    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)      
        
    fig, ax = plt.subplots(3,1,figsize = (14,10))


    ax[0].plot(df_SV['X'],'o', color = 'blue')
    ax[0].plot(df_SV['Xp'],'-', color = 'red')
    #ax01].plot(y_poly_pred,'-')
    #ax[0].set_xlim(np.datetime64(starttime),np.datetime64(endtime))
    ax[0].set_ylabel('dX/dT', fontsize = 14)
    ax[0].set_ylim(df_SV['X'].min() - 3,df_SV['X'].max() + 3)
    ax[0].set_title('Automatic detection - ' + station.upper(), fontsize = 16)
    ax[0].grid()
    
    
    ax[1].plot(df_SV['Y'],'o',color = 'green')
    ax[1].plot(df_SV['Yp'],'-', color = 'red')
    #ax11].plot(y_poly_pred,'-')
    #ax[1].set_xlim(np.datetime64(starttime),np.datetime64(endtime))
    ax[1].set_ylabel('dY/dT', fontsize = 14)
    ax[1].set_ylim(df_SV['Y'].min() - 3,df_SV['Y'].max() + 3)
    ax[1].grid()
    
    ax[2].plot(df_SV['Z'],'o',color = 'black')
    ax[2].plot(df_SV['Zp'],'-', color = 'red')
    #ax21].plot(y_poly_pred,'-')
    #ax[2].set_xlim(np.datetime64(starttime),np.datetime64(endtime))
    ax[2].set_ylabel('dZ/dT', fontsize = 14)
    ax[2].set_xlabel('Years', fontsize = 14 )
    ax[2].set_ylim(df_SV['Z'].min() - 3,df_SV['Z'].max() + 3)
    ax[2].grid()
    
    plt.savefig(directory + '/' + station + '_SV_LFit.jpeg', bbox_inches='tight')
    plt.show()
    
def hampel_filter_denoising(input_series, window_size, n_sigmas=3):
    '''
    
    
    
    
    
    '''
    input_series = input_series.resample('H').mean()
    new_series = input_series.copy()
    for column in input_series:
        
        n = len(input_series[column])
        #new_series = input_series.copy()
        k = 1.4826 # scale factor for Gaussian distribution
        
        indices = []
        
        # possibly use np.nanmedian 
        for i in range((window_size),(n - window_size)):
            x0 = np.median(input_series[column][(i - window_size):(i + window_size)])
            S0 = k * np.median(np.abs(input_series[column][(i - window_size):(i + window_size)] - x0))
            if (np.abs(input_series[column][i] - x0) > n_sigmas * S0):
                new_series[column][i] = x0
        
        fig, ax = plt.subplots(figsize = (16,4))
        ax.plot(input_series[column], 'k', label = 'Removed Outliers')
        ax.plot(new_series[column], 'r', label = 'Denoised ' + column)
        ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize = 12)
        plt.grid()
        plt.show()
        
    return new_series

    

def SV_(stations, starttime, endtime, external_reduce = None, file = None, hampel_filter: bool = False, plot_chaos: bool = False):
    '''
    
    
    Functions arguments ----
    
    stations - must be a list of observatories IAGA code or None
    starttime - must be a 'yyyy-mm-dd' format
    endtime - must be a 'yyyy-mm-dd' format
    external_reduce - must be 'QD', 'DD', 'NT', 'C' or None 
               *QD for keep quiet days
               *DD for remove disturbed days
               *NT for Night time selection
               *C use CHAOS Model predicted external field
    hampel_filter - boolean, True or False
    Plot_chaos - boolean, True or False
    
    
    
    '''
    df_imos = pd.read_csv('Imos_INTERMAGNET.txt', sep = '\t')
    
    External_reduce = ['QD','DD','NT','C']
    
    Files = [None,'update','off']
    
    
    if (external_reduce is not None ) and (external_reduce not in External_reduce):
        print('External field Reduction must be QD, DD or NT. No changes applied.')
        pass
    
    if stations == None:
        for station in df_imos['Imos']:

            df_station =  load_obs_files_OTIMIZADA(station, starttime, endtime)
            if hampel_filter == True:
                df_station = hampel_filter_denoising(input_series = df_station, window_size = 12, n_sigmas=3)
            else:
                pass
            if external_reduce == None:
                pass
            
            if external_reduce == 'QD':
    
                df_station = keep_Q_Days(df_station, starttime, endtime)
                print('Only Quiet Days remains')
                
            if external_reduce == 'DD':
            
                df_station = remove_Disturbed_Days(df_station, start = df_station.resample('D').mean().index[0].date(),
                                                   end = df_station.resample('D').mean().index[-1].date())
                print('Disturbed Days removed')
            
            if external_reduce =='NT':
            
                df_station = NT_LT(station, df_station, starttime, endtime)
                print('Night-time selected')
                
            if external_reduce == 'C':
                df_station, df_chaos = external_field_correction_chaos_model(station = station,
                                                                   starttime = starttime,
                                                                   endtime = endtime,
                                                                   df_station = df_station,
                                                                   df_chaos = None)
                
                df_SV = calculate_SV(df_station, starttime = starttime, endtime = endtime)
                df_SV_chaos = calculate_SV(df_chaos, starttime = starttime, endtime = endtime, columns = ['X_int','Y_int','Z_int'])
                
                RMS = rms(df_SV_chaos, df_SV)
                    
            
            df_SV = calculate_SV(df_station, starttime = starttime, endtime = endtime)
            
            for file in Files:

                if file == None:
                    
                    directory = 'SV_update/'+ station +'_data/'
                    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)     
                    df_SV.dropna().to_csv(directory + 'SV_' + station + '.txt', sep ='\t', index=True)
                
                if file == 'update':
                    
                    df1 = pd.read_csv('SV_update/' + station + '_data/SV_'+ station + '.txt', sep = '\t')
                    df1['Date'] = pd.to_datetime(df1['Date'], infer_datetime_format=True)
                    df1.set_index('Date', inplace = True)
                    df2 = pd.concat([df1,df_SV])
                    #df2 = df2.sort_values(by=df2.index)
                    df2.dropna().drop_duplicates().sort_index().to_csv('SV_update/' + station + '_data/SV_'+ station + '.txt', sep = '\t')
                
                
                if file == 'off':    
                    pass
                
                if file not in Files:
                    print('File must be None, update or off!')
            
            
            
            fig, ax = plt.subplots(3,1, figsize = (16,10))
            
            if plot_chaos == True:
                ax[0].plot(df_SV_chaos['X_int'],'o-', label = 'Chaos - rms: ' + str(RMS[0]),linewidth = 3, color = 'red')
                ax[1].plot(df_SV_chaos['Y_int'],'o-', label = 'Chaos - rms: ' + str(RMS[1]),linewidth = 3, color = 'red')
                ax[2].plot(df_SV_chaos['Z_int'],'o-', label = 'Chaos - rms: ' + str(RMS[2]),linewidth = 3, color = 'red')
                ax[0].legend()  
                ax[1].legend()  
                ax[2].legend()  
                
            ax[0].plot(df_SV['X'],'o', color = 'blue')
            ax[0].set_ylim(df_SV['X'].min() - 10, df_SV['X'].max() + 10)
            ax[0].set_title(station.upper() + ' Secular Variation', fontsize = 16)
                      
                       
            ax[1].plot(df_SV['Y'],'o',color = 'green')
            ax[1].set_ylim(df_SV['Y'].min() - 10, df_SV['Y'].max() + 10)
                    
                       
            ax[2].plot(df_SV['Z'],'o',color = 'black')
            ax[2].set_ylim(df_SV['Z'].min() - 10,df_SV['Z'].max() + 10)
            
            plt.show()
    
            directory2 = 'Map_plots/'+ station +'_data'
            pathlib.Path(directory2).mkdir(parents=True, exist_ok=True)  
            
            #plot to use in the interactive map 
            
            
            fig, ax = plt.subplots(3,1, figsize = (8,6.5))
            if plot_chaos == True:
                ax[0].plot(df_SV_chaos['X_int'],'-', label = 'Chaos - rms: ' + str(RMS[0]),linewidth = 3, color  = 'red')
                ax[1].plot(df_SV_chaos['Y_int'],'-', label = 'Chaos - rms: ' + str(RMS[1]),linewidth = 3, color  = 'red')
                ax[2].plot(df_SV_chaos['Z_int'],'-', label = 'Chaos - rms: ' + str(RMS[2]),linewidth = 3, color  = 'red')
                ax[0].legend()  
                ax[1].legend()  
                ax[2].legend() 
            
            ax[0].set_title(station.upper() + ' Secular Variation (ADMM)', fontsize = 12)
    
            ax[0].plot(df_SV['X'], 'o', color  = 'blue')
            ax[0].set_xlim(np.datetime64(starttime),np.datetime64(endtime))
            ax[0].set_ylim(df_SV['X'].min() - 10, df_SV['X'].max() + 10)
            #ax[0,0].set_xlim(np.datetime64('2011-01'),np.datetime64('2021-12'))
            ax[0].set_ylabel('dX/dT(nT/yr)', fontsize = 9)
            ax[0].grid()
            
            ax[1].plot(df_SV['Y'], 'o', color  = 'green')
            ax[1].set_xlim(np.datetime64(starttime),np.datetime64(endtime))
            ax[1].set_ylim(df_SV['Y'].min() - 10, df_SV['Y'].max() + 10)
            ax[1].set_ylabel('dY/dT(nT/yr)', fontsize = 9)
            ax[1].grid()
            
            ax[2].plot(df_SV['Z'], 'o', color  =  'black')
            ax[2].set_xlim(np.datetime64(starttime),np.datetime64(endtime))
            ax[2].set_ylim(df_SV['Z'].min() - 10, df_SV['Z'].max() + 10)
            #ax[0].set_xlim(np.datetime64('2010-01'),np.datetime64('2021-06'))
            ax[2].set_ylabel('dZ/dT(nT/yr)', fontsize = 9)
            ax[2].grid()
            
            plt.tick_params(labelsize=8)
            plt.savefig(directory2 + '/' + station + '_map_SV.jpeg', bbox_inches='tight')
            plt.show()
            
    for station in stations:
            
        df_station =  load_obs_files_OTIMIZADA(station, starttime, endtime)
        
        if hampel_filter == True:
            df_station = hampel_filter_denoising(input_series = df_station, window_size = 12, n_sigmas=3)
        else:
            pass
        if external_reduce == 'QD':

            df_station = keep_Q_Days(df_station, starttime, endtime)
            print('Only Quiet Days remains')
            
        if external_reduce == 'DD':
        
            df_station = remove_Disturbed_Days(df_station, start = df_station.resample('D').mean().index[0].date(),
                                               end = df_station.resample('D').mean().index[-1].date())
            print('Disturbed Days removed')
        
        if external_reduce =='NT':
        
            df_station = NT_LT(station, df_station, starttime, endtime)
            print('Night-time selected')
            
        if external_reduce == 'C':
            df_station, df_chaos = external_field_correction_chaos_model(station = station,
                                                                   starttime = starttime,
                                                                   endtime = endtime,
                                                                   df_station = df_station,
                                                                   df_chaos = None)
            
            df_SV = calculate_SV(df_station, starttime = starttime, endtime = endtime)
            df_SV_chaos = calculate_SV(df_chaos, starttime = starttime, endtime = endtime, columns = ['X_int','Y_int','Z_int'])
            
            RMS = rms(df_SV_chaos, df_SV)
          
        
        df_SV = calculate_SV(df_station, starttime = starttime, endtime = endtime)
        

        if file == None:
    
            directory = 'SV_update/'+ station +'_data/'
            pathlib.Path(directory).mkdir(parents=True, exist_ok=True)     
            df_SV.dropna().to_csv(directory + 'SV_' + station + '.txt', sep ='\t')
            
        if file == 'update':
    
            df1 = pd.read_csv('SV_update/' + station + '_data/SV_'+ station + '.txt', sep = '\t')
            df1['Date'] = pd.to_datetime(df1['Date'], infer_datetime_format=True)
            df1.set_index('Date', inplace = True)
            df2 = pd.concat([df1,df_SV])
            #df2 = df2.sort_values(by=df2.index)
            df2.dropna().drop_duplicates().sort_index().to_csv('SV_update/' + station + '_data/SV_'+ station + '.txt', sep = '\t')
        if file == 'off':
            pass
        
        if file not in Files:
            print('File must be None, update or off!')
            pass    
        
        directory2 = 'Map_plots/'+ station +'_data'
        pathlib.Path(directory2).mkdir(parents=True, exist_ok=True)  
        
        fig, ax = plt.subplots(3,1, figsize = (8,6.5))
        
        if plot_chaos == True:
            
            ax[0].plot(df_SV_chaos['X_int'],'-', label = 'Chaos - rms: ' + str(RMS[0]),linewidth = 3, color = 'red')
            ax[1].plot(df_SV_chaos['Y_int'],'-', label = 'Chaos - rms: ' + str(RMS[1]),linewidth = 3, color = 'red')
            ax[2].plot(df_SV_chaos['Z_int'],'-', label = 'Chaos - rms: ' + str(RMS[2]),linewidth = 3, color = 'red')
            ax[0].legend()  
            ax[1].legend()  
            ax[2].legend() 
        
        ax[0].set_title(station.upper() + ' Secular Variation (ADMM)', fontsize = 12)

        ax[0].plot(df_SV['X'], 'o', color  = 'blue')
        ax[0].set_xlim(np.datetime64(starttime),np.datetime64(endtime))
        ax[0].set_ylim(df_SV['X'].min() - 10, df_SV['X'].max() + 10)
        #ax[0,0].set_xlim(np.datetime64('2011-01'),np.datetime64('2021-12'))
        ax[0].set_ylabel('dX/dT(nT/yr)', fontsize = 9)
        ax[0].grid()
        
        ax[1].plot(df_SV['Y'], 'o', color  = 'green')
        ax[1].set_xlim(np.datetime64(starttime),np.datetime64(endtime))
        ax[1].set_ylim(df_SV['Y'].min() - 10, df_SV['Y'].max() + 10)
        ax[1].set_ylabel('dY/dT(nT/yr)', fontsize = 9)
        ax[1].grid()
        
        ax[2].plot(df_SV['Z'], 'o', color  =  'black')
        ax[2].set_xlim(np.datetime64(starttime),np.datetime64(endtime))
        ax[2].set_ylim(df_SV['Z'].min() - 10, df_SV['Z'].max() + 10)
        #ax[0].set_xlim(np.datetime64('2010-01'),np.datetime64('2021-06'))
        ax[2].set_ylabel('dZ/dT(nT/yr)', fontsize = 9)
        ax[2].grid()
        
        
        plt.tick_params(labelsize=8)
        plt.savefig(directory2 + '/' + station + '_map_SV.jpeg', bbox_inches='tight')
      
        
        fig, ax = plt.subplots(3,1, figsize = (16,10))
        if plot_chaos == True:
            ax[0].plot(df_SV_chaos['X_int'],'-', label = 'Chaos - rms: ' + str(RMS[0]),linewidth = 3, color = 'red')
            ax[1].plot(df_SV_chaos['Y_int'],'-', label = 'Chaos - rms: ' + str(RMS[1]),linewidth = 3, color = 'red')
            ax[2].plot(df_SV_chaos['Z_int'],'-', label = 'Chaos - rms: ' + str(RMS[2]),linewidth = 3, color = 'red')
            ax[0].legend()  
            ax[1].legend()  
            ax[2].legend() 
        
        ax[0].plot(df_SV['X'],'o', color = 'blue')
        ax[0].set_ylim(df_SV['X'].min() - 10, df_SV['X'].max() + 10)
        ax[0].set_title(station.upper() + ' Secular Variation', fontsize = 16)
        ax[1].plot(df_SV['Y'],'o',color = 'green')
        ax[1].set_ylim(df_SV['Y'].min() - 10, df_SV['Y'].max() + 10)
        ax[2].plot(df_SV['Z'],'o',color = 'black')
        ax[2].set_ylim(df_SV['Z'].min() - 10,df_SV['Z'].max() + 10)
        plt.show()
        
def calculate_SV(dataframe, starttime, endtime, info = 'ADMM', columns = None):
    df = pd.DataFrame()
    df = dataframe
    df = df.loc[starttime:endtime]
    
    Info = ['ADMM','YD']
    df_SV = pd.DataFrame()
    
    #if info is not in Info:
    #    print('info must be ADMM or YD')
#   
    df_ADMM = resample_obs_data(dataframe = df, sample = 'M')
    df_YD = resample_obs_data(dataframe = df, sample = 'Y')

    if columns == None:
        columns = ['X','Y','Z']
    else:
        columns = columns
        
    if info not in Info:
        print('Info musdf_SV = pd.DataFrame()t be ADMM or YD')
    
    if info == 'ADMM':
        for col in columns:
            SV = (df_ADMM[col].diff(6) - df_ADMM[col].diff(-6)).round(3).dropna()
            df_SV[col] = SV  
    if info == 'YD':
        for col in columns:
            SV = (df[col].diff().round(3).dropna()
            df_SV[col] = SV 
            
    return df_SV

def update_qd_and_dd(data, file):
    
    
    DATA = ['DD','QD']
    
    if data not in DATA:
        print('Data must be QD or DD!')
        
    path = 'Dados OBS/' + file
    if data == 'DD':
        df = pd.read_csv(path,skiprows = 4,sep = '\s+',
                    header = None,
                    usecols = [0,1,12,13,14,15,16],
                    names = ['Month','Year','D1','D2','D3','D4','D5'])
        
         
        columns = ['D1','D2','D3','D4','D5']
        
        df['Month'] = pd.to_datetime(df.Month, format='%b').dt.month
        
        for col in columns:
            df['Test' +  col] = df['Year'].astype(str)+'-'+df['Month'].astype(str)+'-'+df[col].astype(str)
        for col in columns:
            df['Test' + col] = df['Test' + col].str.replace('*','')
        
        df_DD = pd.DataFrame()
        df_DD['DD'] = pd.concat([df['TestD1'],df['TestD2'],df['TestD3'],df['TestD4'],df['TestD5']])
        
        df_DD['DD'].to_csv('NEW_DD.txt',index = False)
        
    if data == 'QD':
        
        df = pd.read_csv(path,skiprows = 4,sep = '\s+',
                    header = None, 
                    usecols = [0,1,2,3,4,5,6,7,8,9,10,11],
                    names = ['Month','Year','Q1','Q2','Q3','Q4','Q5','Q6','Q7','Q8','Q9','Q10'])
        
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
        
        df_QD['QD'].to_csv('NEW_QD.txt',index = False)

def Kp_index_correction(dataframe, starttime, endtime, kp):
    '''
    '''
    for i in [starttime,endtime]:
        validate(i)
    
    if (kp <=0 ) or (kp>= 9): 
        print('kp must be a number from 0 to 9, try again!')
    df = pd.DataFrame()
    df = dataframe
    
    KP_ = pd.read_csv('Dados OBS/Kp_ap_since_1932.txt', skiprows = 30,
                  header = None,
                  sep = '\s+', 
                  usecols = [7,8],
                  names = ['Kp','Ap'],
                 )
    Date = pd.date_range('1932-01-01 00:00:00','2021-12-08 21:00:00', freq = '3H')    
    KP_.index = Date
    
    x=pd.DataFrame()
    x['Date'] = KP_[starttime:endtime].loc[KP_['Kp'] > kp].index.date
    x['Date'] = x['Date'].drop_duplicates()
    x.index = x['Date']
    x =x.dropna()
    
    dataframe = dataframe.resample('D').mean().drop(x.index)
    return dataframe


def chaos_model_provisory(station, starttime, endtime):
    '''
    '''
    
    for i in [starttime,endtime]:
        validate(i)
        
        
    model = cp.load_CHAOS_matfile('chaosmagpy_package_0.8/data/CHAOS-7.9.mat')
    
    station = station.upper()
    df_IMOS = pd.read_csv('IMOS_INTERMAGNET.txt', sep = '\s+')
    df_IMOS.index = df_IMOS['Imos'] 
    
    
    R_REF = 6371.2
    
    if station not in df_IMOS.index:
        print('Station must be an observatory IAGA CODE!')
        
    f = []
    f.extend(glob.glob('Dados OBS/*/*/' + station + '*'))
    f.sort()
    
    
    Longitude = pd.read_csv('Dados OBS/' + f[0][21:25] +'/' + f[0][25:27] + '/' + station + f[0][21:41],
            nrows = 1, 
            sep = ' ', 
            usecols = [7],
            header  = None,
            names = ['Geodetic Longitude'],
            index_col=None,
            skiprows = 5)
    
    Longitude = Longitude['Geodetic Longitude'][0]
    
    Latitude = pd.read_csv('Dados OBS/' + f[0][21:25] +'/' + f[0][25:27] + '/' + station + f[0][21:41],
            nrows = 1, 
            sep = ' ', 
            usecols = [8],
            header  = None,
            names = ['Geodetic Latitude'],
            index_col=None,
            skiprows = 4)
    
    Latitude = 90 - Latitude['Geodetic Latitude'][0]
    
    Elevation = pd.read_csv('Dados OBS/' + f[0][21:25] +'/' + f[0][25:27] + '/' + station + f[0][21:41],
        nrows = 1, 
        sep = ' ', 
        usecols = [15],
        header  = None,
        names = ['Elevation'],
        index_col=None,
        skiprows = 6)   
    
    Elevation = Elevation['Elevation'][0]/1000 + R_REF
    
    #if Latitude < 0:
    #    Latitude = 90 - Latitude
    #else:
    #    Latitude = Latitude
    
    
    #start = cp.data_utils.mjd2000(starttime[0:4],starttime[5:7],starttime[8:10])
    #end = cp.data_utils.mjd2000(endtime[0:4],endtime[5:7],endtime[8:10])        
    
    #print(end)
    Date = pd.date_range(starttime,endtime, freq = 'H')
    Time =cp.data_utils.mjd2000(Date)
    
    # Internal field
    print('Initiating geomagnetic field computation for ' + station.upper() +'.')
    print('Computing core field.')
    B_core = model.synth_values_tdep(time = Time,
                                     radius = Elevation,
                                     theta = Latitude ,
                                     phi = Longitude)

    print('Computing crustal field up to degree 70.')
    B_crust = model.synth_values_static(radius = Elevation,
                                        theta = Latitude,
                                        phi = Longitude,
                                        nmax=70)
    
    # complete internal contribution
    B_radius_int = B_core[0] + B_crust[0]
    B_theta_int = B_core[1] + B_crust[1]
    B_phi_int = B_core[2] + B_crust[2]
    
    print('Computing field due to external sources, incl. induced field: GSM.')
    B_gsm = model.synth_values_gsm(time = Time,
                                   radius = Elevation, 
                                   theta = Latitude,
                                   phi = Longitude, 
                                   source='all')

    print('Computing field due to external sources, incl. induced field: SM.')
    B_sm = model.synth_values_sm(time = Time,
                                 radius = Elevation,
                                 theta = Latitude,
                                 phi = Longitude,
                                 source='all')

    # complete external field contribution
    B_radius_ext = B_gsm[0] + B_sm[0]
    B_theta_ext = B_gsm[1] + B_sm[1]
    B_phi_ext = B_gsm[2] + B_sm[2]

    # complete forward computation
    B_radius = B_radius_int + B_radius_ext
    B_theta = B_theta_int + B_theta_ext
    B_phi = B_phi_int + B_phi_ext
    

    df_station = pd.DataFrame()
    df_station.index = Date
    
    df_station['X_tot'] = B_theta.round(3)*-1
    df_station['Y_tot'] = B_phi.round(3)
    df_station['Z_tot'] = B_radius.round(3)*-1    
    
    df_station['X_int'] = B_theta_int.round(3)*-1
    df_station['Y_int'] = B_phi_int.round(3)
    df_station['Z_int'] = B_radius_int.round(3)*-1

    df_station['X_ext_gsm'] = B_gsm[1].round(3)*-1
    df_station['Y_ext_gsm'] = B_gsm[2].round(3)
    df_station['Z_ext_gsm'] = B_gsm[0].round(3)*-1
    
    df_station['X_ext_sm'] = B_sm[1].round(3)*-1
    df_station['Y_ext_sm'] = B_sm[2].round(3)
    df_station['Z_ext_sm'] = B_sm[0].round(3)*-1
    
    return df_station 

def INTERMAGNET_AND_CHAOS_COMPARISION(station, dataframe_Chaos, starttime, endtime, dataframe_intermagnet = None):
    
    directory = 'Chaos_comparison/'+ station +'_data'
    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
    
    Dataframe_intermagnet = pd.DataFrame()
    Dataframe_intermagnet = dataframe_intermagnet
    
    Dataframe_Chaos = pd.DataFrame()
    Dataframe_Chaos = dataframe_Chaos
    
    Dataframe_Chaos['X_ext'] = Dataframe_Chaos['X_ext_gsm'] + Dataframe_Chaos['X_ext_sm']
    Dataframe_Chaos['Y_ext'] = Dataframe_Chaos['Y_ext_gsm'] + Dataframe_Chaos['Y_ext_sm']
    Dataframe_Chaos['Z_ext'] = Dataframe_Chaos['Z_ext_gsm'] + Dataframe_Chaos['Z_ext_sm']
    
    cols_tot = ['X_tot','Y_tot','Z_tot']
    colors = ['blue','green','black']
    
    #plotting total field for chaos data
    
    fig, axes = plt.subplots(3,1, figsize = (14,10))
    axes[0].set_title('Predicted Chaos Data - Total Field', fontsize = 16)
    for col, ax, color in zip(cols_tot, axes.flatten(), colors):
        ax.plot(Dataframe_Chaos[col].loc[starttime:endtime], color  = color)
        ax.set_ylabel(col.upper() + '/nT', fontsize = 12)
    plt.savefig(directory + '/' + station + '_Chaos_Total_field.jpeg', bbox_inches='tight')
    plt.show()
    #plotting internal field for Chaos data
    
    cols_int = ['X_int','Y_int','Z_int']
    fig, axes = plt.subplots(3,1, figsize = (14,10))
    axes[0].set_title('Predicted Chaos Data - Internal Field', fontsize = 16)
    for col, ax, color in zip(cols_int, axes.flatten(), colors):
        ax.plot(Dataframe_Chaos[col].loc[starttime:endtime], color  = color)
        ax.set_ylabel(col.upper() + '/nT', fontsize = 12)
    plt.savefig(directory + '/' + station + '_Chaos_Internal_field.jpeg', bbox_inches='tight')
    plt.show()
    # plotting external field
    
    cols_ext = ['X_ext','Y_ext','Z_ext']
    fig, axes = plt.subplots(3,1, figsize = (14,10))
    axes[0].set_title('Predicted Chaos Data External Field', fontsize = 16)
    for col, ax, color in zip(cols_ext, axes.flatten(), colors):
        
        ax.plot(Dataframe_Chaos[col][starttime:endtime], color  = color)
        ax.set_ylabel(col.upper() + '/nT', fontsize = 12)
    plt.savefig(directory + '/' + station + '_Chaos_External_field.jpeg', bbox_inches='tight')
    plt.show()
        
    if dataframe_intermagnet is not None:
        
        cols_intermagnet = ['X','Y','Z']
        
        #comparing SV - Total field
        fig, axes = plt.subplots(3,1, figsize = (14,10))
        axes[0].set_title(station.upper() + ' - INTERMAGNET and CHAOS SV comparison (Total Field)', fontsize = 16)
        for col, ax, color,column in zip(cols_tot, axes.flatten(), colors, cols_intermagnet):
            ax.plot((Dataframe_Chaos[col].loc[starttime:endtime].resample('M').mean().diff(6) - 
                    Dataframe_Chaos[col].loc[starttime:endtime].resample('M').mean().diff(-6)) ,'o-', color  = 'red')
            ax.plot((Dataframe_intermagnet[column].loc[starttime:endtime].resample('M').mean().diff(6) - 
                    Dataframe_intermagnet[column].loc[starttime:endtime].resample('M').mean().diff(-6)) ,'o-', color  = color)
        plt.savefig(directory + '/' + station + '_Chaos_Total_field_SV_comp.jpeg', bbox_inches='tight') 
        plt.show()
        
        #comparing SV - Internal field
        fig, axes = plt.subplots(3,1, figsize = (14,10))
        axes[0].set_title(station.upper() + ' - INTERMAGNET and CHAOS SV comparison (Internal Field)', fontsize = 16)
        for col, ax, color,column in zip(cols_int, axes.flatten(), colors, cols_intermagnet):
            ax.plot((Dataframe_Chaos[col].loc[starttime:endtime].resample('M').mean().diff(6) - 
                    Dataframe_Chaos[col].loc[starttime:endtime].resample('M').mean().diff(-6)) ,'o-', color  = 'red')
            ax.plot((Dataframe_intermagnet[column].loc[starttime:endtime].resample('M').mean().diff(6) - 
                    Dataframe_intermagnet[column].loc[starttime:endtime].resample('M').mean().diff(-6)) ,'o-', color  = color)                          
            
            ax.set_ylabel(col.upper() + '/nT', fontsize = 12)
        plt.savefig(directory + '/' + station + '_Chaos_internal_field_SV_comp.jpeg', bbox_inches='tight')
        plt.show()
        
        #comparing intermagnet SV corrected for external field 
        
        fig, axes = plt.subplots(3,1, figsize = (14,10))
        axes[0].set_title(station.upper() + ' - INTERMAGNET (corrected for external field) and CHAOS SV comparison', fontsize = 16)
        for col, cols, ax, color,column in zip(cols_int,cols_ext, axes.flatten(), colors, cols_intermagnet):
            ax.plot((Dataframe_Chaos[col].loc[starttime:endtime].resample('M').mean().diff(6) - 
                    Dataframe_Chaos[col].loc[starttime:endtime].resample('M').mean().diff(-6)) ,'o-', color  = 'red')
            ax.plot(((Dataframe_intermagnet[column].loc[starttime:endtime].resample('H').mean() - Dataframe_Chaos[cols]).resample('M').mean().diff(6) - 
                    (Dataframe_intermagnet[column].loc[starttime:endtime].resample('H').mean() - Dataframe_Chaos[cols]).resample('M').mean().diff(-6)) ,'o-', color  = color)                          
            
            ax.set_ylabel(col.upper() + '/nT', fontsize = 12)
        plt.savefig(directory + '/' + station + '_SV_comp_external_corrected.jpeg', bbox_inches='tight')
        plt.show()
        
def external_field_correction_chaos_model(station, starttime, endtime,df_station = None, df_chaos = None):
    
    '''
    '''
    
    station = station.upper()
    df_IMOS = pd.read_csv('IMOS_INTERMAGNET.txt', sep = '\s+')
    df_IMOS.index = df_IMOS['Imos'] 
    if station not in df_IMOS.index:
        print('Station must be an observatory IAGA CODE!')
        
    
    
    if df_chaos is not None:
        df_chaos = df_chaos
        df_chaos.loc[starttime:endtime] = df_chaos
    
    else:
        
        df_chaos = chaos_model_provisory(station = station,
                                         starttime = starttime,
                                         endtime = endtime)
        
        
    df_chaos['X_ext'] = df_chaos['X_ext_gsm'] + df_chaos['X_ext_sm']
    df_chaos['Y_ext'] = df_chaos['Y_ext_gsm'] + df_chaos['Y_ext_sm']
    df_chaos['Z_ext'] = df_chaos['Z_ext_gsm'] + df_chaos['Z_ext_sm']
    
    if df_station is not None:
        
        df_station = df_station
        df_station.loc[starttime:endtime] = df_station
    else:
        df_station = load_obs_files_OTIMIZADA(station = station,
                                                  starttime = starttime,
                                                  endtime = endtime)
        df_station = df_station.loc[starttime:endtime]
    df_station = df_station.resample('H').mean()
        
    df_station['X'] = df_station['X'] - df_chaos['X_ext']
    df_station['Y'] = df_station['Y'] - df_chaos['Y_ext']
    df_station['Z'] = df_station['Z'] - df_chaos['Z_ext']    
        
    return df_station, df_chaos    

def rms(predictions, real_data):
    columns = ['X_int','Y_int','Z_int']
    x = []
    for col,cols in zip(columns,real_data.columns):
        y = (real_data[cols].resample('M').mean().diff(6) - real_data[cols].resample('M').mean().diff(-6)).dropna()
        ypred = (predictions[col].resample('M').mean().diff(6) - predictions[col].resample('M').mean().diff(-6)).dropna()
        ypred = ypred.reindex(y.index)
        rms = np.sqrt(((ypred - y) ** 2).mean()).round(3)
        x.append(rms)
        #print('the rmse for ' + str(cols) + ' component is ' + str(rms) + '.')
    return x

def validate(str_date):
    try:
        datetime.strptime(str_date, '%Y-%m-%d')
    except ValueError:
        raise ValueError('Incorrect ' + str_date + ' format, should be YYYY-MM-DD')
        
        
def polynomial_jerk_detection(station, window_start, 
                              window_end, 
                              starttime, 
                              endtime,
                              df_station = None,
                              plot_detection: bool = True,
                              CHAOS_correction: bool = True):
    
    station = station
    window_start = window_start
    window_end = window_end
    starttime = starttime
    endtime = endtime
    
    for i in [starttime,endtime,window_start,window_end]:
        validate(i)
        
    if df_station is not None:
        
        df_station = df_station
        
    else:
        df_station = load_obs_files_OTIMIZADA(station = station, starttime = starttime, endtime = endtime)
        
    if CHAOS_correction == True:
        
        df_station, df_chaos = external_field_correction_chaos_model(station = station,
                                                  starttime = starttime,
                                                  endtime = endtime,
                                                  df_station = df_station,
                                                  df_chaos = None)
    else:
        pass
    
    #calculating SV
    df_SV = calculate_SV(dataframe = df_station, starttime = starttime, endtime = endtime, info = 'ADMM', columns = None)
    
    #starting with polynomial jerk detection
    
    t = np.arange(0,df_SV.loc[window_start:window_end].size/3).reshape(-1,1)
    jerk_prediction = pd.DataFrame()
    jerk_prediction.index = df_SV.loc[window_start:window_end].index
    
    for column in df_SV:

        polynomial_features= PolynomialFeatures(degree=3)
        x_poly = polynomial_features.fit_transform(t)
        
        model = LinearRegression()
        model.fit(x_poly, df_SV[column].loc[window_start:window_end])
        jerk_prediction[column] = model.predict(x_poly)
        
    if plot_detection == True:
        colors = ['blue','green','black']
        fig, axes = plt.subplots(3,1,figsize = (10,8))
        plt.suptitle(station.upper() +' secular variation', fontsize = 14, y = 0.92)
        plt.xlabel('Date (Years)', fontsize = 12)
        
        for col, ax, color in zip(df_SV.columns, axes.flatten(), colors):
            ax.plot(df_SV[col],'o-',color = color)
            ax.plot(df_SV[col].loc[window_start:window_end].index,
                    jerk_prediction[col],color = 'red', linewidth = 3, label = '3rd order polynomial')
            ax.set_ylabel('d' + col.upper() +'/dt (nT)', fontsize = 12)
            ax.legend()
        
        
        fig, axes = plt.subplots(1,3,figsize = (16,4))
        plt.suptitle(station.upper() +' secular variation', fontsize = 14, y = 0.93)
        fig.text(0.5, 0.04, 'Date (Years)', ha='center')
     
        upper_limit = int(str(datetime.strptime(window_end ,'%Y-%m-%d'))[0:4]) +1
        lower_limit = int(str(datetime.strptime(window_start ,'%Y-%m-%d'))[0:4]) -1

        for col, ax, color in zip(df_SV.columns, axes.flatten(), colors):
            ax.plot(df_SV[col].loc[str(lower_limit):str(upper_limit)],'o-',color = color)
            ax.plot(jerk_prediction[col],color = 'red', linewidth = 3)
            ax.set_ylabel('d' + col.upper() +'/dt (nT)', fontsize = 12)
            
        
        
    else:
        pass
        
    return df_SV, jerk_prediction 

def read_txt_SV(station, starttime, endtime):
    path = 'SV_update/'+ station.upper() +'_data/SV_' + station.upper() + '.txt'

    df_SV = pd.read_csv(path,sep = '\s+', index_col = [0])
    df_SV.index = pd.to_datetime(df_SV.index,infer_datetime_format=True)
    df_SV = df_SV.loc[starttime:endtime]
    
    return df_SV

def resample_obs_data(dataframe, sample):
    
    df_station = pd.DataFrame()
    df_station = dataframe
    
    samples = ['min','H','D','M','Y']
    
    
    if sample not in samples:
        print('sample must be min, H, D, M or Y!')
    else:
        
        if sample == 'min':
            
            df_station = df_station
            
        if sample == 'H':
            
            df_station = df_station.resample('H').mean()
            df_station.index = df_station.index + to_offset('30min')
            
        if sample == 'D':
            
            df_station = df_station.resample('D').mean()
            df_station.index = df_station.index + to_offset('12H')
            
        if sample == 'M':
            
            df_station = df_station.resample('M').mean()
            df_station.index = df_station.index + to_offset('-1M') + to_offset('15D')
            
            
        if sample == 'Y':
            
            df_station = df_station.resample('Y').mean()
            df_station.index = df_station.index + to_offset('-6M') + to_offset('-15D')
            
    return df_station