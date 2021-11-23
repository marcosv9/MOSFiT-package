import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import glob
import os
import ftplib
import pathlib
import matplotlib.gridspec as gridspec
from datetime import datetime
import pwlf



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


def load_obs_files_OTIMIZADA(station, skiprows, starttime, endtime):
    '''
    Function to read and concat observatory data
    
    Sample must be H, D, M, Y   
    
    '''
    print('Reading files from '+ station.upper() +'...')
    year  = []
    for i in range(int(starttime[0:4]),int(endtime[0:4])+ 1):
        Y = i
        year.append(Y)
    
    Years = []
    Years.extend([str(i) for i in year])
    Years
    
    files_station = []
    
    for Year in Years:

    
        files_station.extend(glob.glob('Dados OBS\\' + Year + '/*/' + station + '*'))
        files_station.sort()

    #d_parser = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S%.f')
    
    df_station = pd.concat( (pd.read_csv(file, sep='\s+',usecols = [0,1,3,4,5], 
                   header = None,skiprows = skiprows, 
                   parse_dates = {'Date': ['date', 'Time']},
                   names = ['date','Time','X','Y','Z']) for file in files_station), 
                   ignore_index = True)
    df_station['Date'] = pd.to_datetime(df_station['Date'], format = '%Y-%m-%dd %H:%M:%S.%f')     
    #df_station['Hour'] = pd.to_datetime(df_station['Hour'], format = '%H:%M:%S.%f').dt.time               
    df_station.set_index('Date', inplace = True)

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
    
    August-2021 Fow now, the output is a data without disturbed days in a monthly mean
    resolution. I need to improve it!!!!
    
    Implement to not be required to inform start and end period (only if the user want to)    
    
    Function(dataframe = Name of your dataframe ex: df_VSS, start = date that you want to start removing DD, end = the same thing )
    Ex  - df_NGK2 = mvs.remove_Disturbed_Days(df_NGK, start = '2010-01' , end = '2019-12')
    
    '''
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
    
    
    remove = df_d.loc[start:end]
    
    df = df.drop(remove.index)
    
    
    return df

def keep_Q_Days(dataframe, start, end):
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
    L_27 = ['CZT','DRV','PAF']
    L_26 = ['NGK','DRV','MAW','CNB','HAD','TSU','HON','KAK','BOU','KOU','HBK']
    
    skiprows = 25
    if station.upper() in L_27:
        skiprows = 27
    if station.upper() in L_26:
        skiprows = 26
    
    df_station = load_obs_files_OTIMIZADA(station, skiprows, starttime, endtime)
    
    
    df_station.loc[df_station['X'] == 99999.0, 'X'] = np.nan
    df_station.loc[df_station['Y'] == 99999.0, 'Y'] = np.nan
    df_station.loc[df_station['Z'] == 99999.0, 'Z'] = np.nan
    df_station2 = df_station.copy()
    
    while True: 
        inp5 = input("Do You Want To denoise the data based on median absolute deviation? [y/n]: ")
        if inp5 == 'y':
            print('Denoising data...')
            df_station = hampel_filter_denoising(input_series = df_station, window_size = 12, n_sigmas=3)
            break
        if inp5 == 'n':
            print('No data changed!')
            break
        else:
            print('You must type y or n, try again!')
        
    
    options = ['Q','D','NT','E']
    while True: 
        inp = str(input("Press Q to use only Quiet Days, D to remove Disturbed Days, NT to use only the night-time or E to Exit without actions [Q/D/NT/E]: "))
        
        if all([inp != option for option in options]):
            print('You must type Q, D, NT or E')
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
        

        
    samples = ['Min','H','D','M','Y']
    directory = 'Filtered_data/'+ station +'_data'
    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)  
    
    X_SV = (df_station['X'][starttime:endtime].resample('M').mean().diff(6) - df_station['X'] [starttime:endtime].resample('M').mean().diff(-6)).shift(-15, freq = 'D').round(3)
    Y_SV = (df_station['Y'][starttime:endtime].resample('M').mean().diff(6) - df_station['Y'] [starttime:endtime].resample('M').mean().diff(-6)).shift(-15, freq = 'D').round(3)
    Z_SV = (df_station['Z'][starttime:endtime].resample('M').mean().diff(6) - df_station['Z'] [starttime:endtime].resample('M').mean().diff(-6)).shift(-15, freq = 'D').round(3)
                
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
                    
                    file = df_station2[starttime:endtime].resample(sample).mean().shift(-30, freq = 'min').round(3)
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
                    
                    file_SV = (df_station[starttime:endtime].resample(sample).mean().diff(6) -
                    df_station[starttime:endtime].resample(sample).mean().diff(-6)).shift(-15, freq = 'D').round(3)
                    
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
    
            ax[0].plot(X_SV, 'o', color  = 'blue')
            ax[0].set_xlim(np.datetime64(starttime),np.datetime64(endtime))
            ax[0].set_ylim(X_SV.min() - 10, X_SV.max() + 10)
            #ax[0,0].set_xlim(np.datetime64('2011-01'),np.datetime64('2021-12'))
            ax[0].set_ylabel('dX/dT(nT/yr)', fontsize = 9)
            ax[0].grid()
            
            ax[1].plot(Y_SV, 'o', color  = 'green')
            ax[1].set_xlim(np.datetime64(starttime),np.datetime64(endtime))
            ax[1].set_ylim(Y_SV.min() - 10, Y_SV.max() + 10)
            ax[1].set_ylabel('dY/dT(nT/yr)', fontsize = 9)
            ax[1].grid()
            
            ax[2].plot(Z_SV, 'o', color  =  'black')
            ax[2].set_xlim(np.datetime64(starttime),np.datetime64(endtime))
            ax[2].set_ylim(Z_SV.min() - 10, Z_SV.max() + 10)
            #ax[0].set_xlim(np.datetime64('2010-01'),np.datetime64('2021-06'))
            ax[2].set_ylabel('dZ/dT(nT/yr)', fontsize = 9)
            ax[2].grid()
            
            plt.tick_params(labelsize=8)
            plt.savefig(directory + '/' + station + '_map_SV.jpeg', bbox_inches='tight')
            #plt.show()
            
            #plot of secular variation and monthly mean
            
            fig, ax = plt.subplots(3,2, figsize = (18,10))
            
            
            ax[0,1].set_title(station.upper() + ' Secular Variation (ADMM)', fontsize = 18)
            ax[0,1].plot(X_SV, 'o', color  = 'blue')
            ax[0,1].set_xlim(np.datetime64(starttime),np.datetime64(endtime))
            ax[0,1].set_ylabel('dX/dT(nT/yr)', fontsize = 12)
            ax[0,1].grid()
            
            ax[1,1].plot(Y_SV, 'o', color  = 'green')
            ax[1,1].set_xlim(np.datetime64(starttime),np.datetime64(endtime))
            ax[1,1].set_ylabel('dY/dT(nT/yr)', fontsize = 12)
            ax[1,1].grid()
            
            ax[2,1].plot(Z_SV, 'o', color  =  'black')
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
    
            ax[0].plot(X_SV, 'o', color  = 'blue')
            ax[0].set_xlim(np.datetime64(starttime),np.datetime64(endtime))
            ax[0].set_ylim(X_SV.min() - 10, X_SV.max() + 10)
            #ax[0,0].set_xlim(np.datetime64('2011-01'),np.datetime64('2021-12'))
            ax[0].set_ylabel('dX/dT(nT/yr)', fontsize = 12)
            ax[0].grid()
            
            ax[1].plot(Y_SV, 'o', color  = 'green')
            ax[1].set_xlim(np.datetime64(starttime),np.datetime64(endtime))
            ax[1].set_ylim(Y_SV.min() - 10, Y_SV.max() + 10)
            ax[1].set_ylabel('dY/dT(nT/yr)', fontsize = 12)
            ax[1].grid()
            
            ax[2].plot(Z_SV, 'o', color  =  'black')
            ax[2].set_xlim(np.datetime64(starttime),np.datetime64(endtime))
            ax[2].set_ylim(Z_SV.min() - 10, Z_SV.max() + 10)
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
                
               
                plt.savefig(directory + '/' + station + '_' + sample + '_mean.jpeg', bbox_inches='tight')
                plt.show()
            print('Plots of Minute, Hourly, Daily, Monthly, Yearly means and Secular Variation were saved on directory:')
            print(directory)    
            break
        elif inp3 == 'n':
            print('No plots saved')
            fig, ax = plt.subplots(3,2, figsize = (18,10))
            
            
            ax[0,1].set_title(station.upper() + ' Secular Variation (ADMM)', fontsize = 18)
            ax[0,1].plot(X_SV, 'o', color  = 'blue')
            ax[0,1].set_ylim(X_SV.min() - 10, X_SV.max() + 10)
            ax[0,1].set_xlim(np.datetime64(starttime),np.datetime64(endtime))
            ax[0,1].set_ylabel('dX/dT(nT/yr)', fontsize = 12)
            ax[0,1].grid()
            
            ax[1,1].plot(Y_SV, 'o', color  = 'green')
            ax[1,1].set_ylim(Y_SV.min() - 10, Y_SV.max() + 10)
            ax[1,1].set_xlim(np.datetime64(starttime),np.datetime64(endtime))
            ax[1,1].set_ylabel('dY/dT(nT/yr)', fontsize = 12)
            ax[1,1].grid()
            
            ax[2,1].plot(Z_SV, 'o', color  =  'black')
            ax[2,1].set_ylim(Z_SV.min() - 10, Z_SV.max() + 10)
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
    
            ax[0].plot(X_SV, 'o', color  = 'blue')
            ax[0].set_xlim(np.datetime64(starttime),np.datetime64(endtime))
            ax[0].set_ylim(X_SV.min() - 10, X_SV.max() + 10)
            #ax[0,0].set_xlim(np.datetime64('2011-01'),np.datetime64('2021-12'))
            ax[0].set_ylabel('dX/dT(nT/yr)', fontsize = 12)
            ax[0].grid()
            
            ax[1].plot(Y_SV, 'o', color  = 'green')
            ax[1].set_xlim(np.datetime64(starttime),np.datetime64(endtime))
            ax[1].set_ylim(Y_SV.min() - 10, Y_SV.max() + 10)
            #ax[0].set_xlim(np.datetime64('2010-01'),np.datetime64('2021-06'))
            ax[1].set_ylabel('dY/dT(nT/yr)', fontsize = 12)
            ax[1].grid()
            
            ax[2].plot(Z_SV, 'o', color  =  'black')
            ax[2].set_xlim(np.datetime64(starttime),np.datetime64(endtime))
            ax[2].set_ylim(Z_SV.min() - 10, Z_SV.max() + 10)
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
        inp4 = input("Do You Want To adopt piecewise linear segments for the SV? [y/n]: ")
        if inp4 == 'y':
            try:
                ls = input('Type the number of linear segments that best fit the SV: ')
                #create an option to save or not a plot
                
                
                jerk_detection(station, df_station, ls, starttime, endtime)
            except ValueError:
                #it is not working for short periods, must be corrected
                print('You must type a number!')
                continue
            break        
        elif inp4 == 'n':
            print('No linear segments adopted')
            break
    return df_station[starttime:endtime]

def check_data_availability(station):
    '''
    Function to read and concat observatory data
    
    Sample must be H, D, M, Y   
    
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
    '''
    
    df_station = pd.DataFrame()
    df_station = dataframe
    df_station.loc[df_station['X'] == 99999.0, 'X'] = np.nan
    df_station.loc[df_station['Y'] == 99999.0, 'Y'] = np.nan
    df_station.loc[df_station['Z'] == 99999.0, 'Z'] = np.nan
    df_SV = pd.DataFrame()
    #fit your data (x and y)
    components = ['X','Y','Z']
    df_SV['X'] = (df_station['X'][starttime:endtime].resample('M').mean().diff(6) - df_station['X'] [starttime:endtime].resample('M').mean().diff(-6)).shift(-15, freq = 'D').round(3)
    df_SV['Y'] = (df_station['Y'][starttime:endtime].resample('M').mean().diff(6) - df_station['Y'] [starttime:endtime].resample('M').mean().diff(-6)).shift(-15, freq = 'D').round(3)
    df_SV['Z'] = (df_station['Z'][starttime:endtime].resample('M').mean().diff(6) - df_station['Z'] [starttime:endtime].resample('M').mean().diff(-6)).shift(-15, freq = 'D').round(3)
    df_SV = df_SV.dropna()
    
    X = np.arange(0,len(df_SV.index))
    
    for component in components:
        
        myPWLF = pwlf.PiecewiseLinFit(X,df_SV[component].dropna())
        
        #fit the data for n line segments
        z = myPWLF.fit(ls)
        
        #calculate slopes
        slopes = myPWLF.calc_slopes()
        
        # predict for the determined points
        xHat = X 
        yHat = myPWLF.predict(xHat)
        
        #calculate statistics
        p = myPWLF.p_values(method='non-linear', step_size=1e-4) #p-values
        se = myPWLF.se  # standard errors
        
        df_SV[component + 'p'] = yHat
        
        
    directory = 'Filtered_data/'+ station +'_data'
    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)      
        
    fig, ax = plt.subplots(3,1,figsize = (14,10))


    ax[0].plot(df_SV['X'],'o', color = 'blue')
    ax[0].plot(df_SV['Xp'],'-', color = 'red')
    #ax01].plot(y_poly_pred,'-')
    #ax[0].set_xlim(np.datetime64(starttime),np.datetime64(endtime))
    ax[0].set_ylabel('dX/dT', fontsize = 14)
    ax[0].set_ylim(df_SV['X'].min() - 10,df_SV['X'].max() + 10)
    ax[0].set_title('Automatic Jerk detection - ' + station.upper(), fontsize = 16)
    ax[0].grid()
    
    
    ax[1].plot(df_SV['Y'],'o',color = 'green')
    ax[1].plot(df_SV['Yp'],'-', color = 'red')
    #ax11].plot(y_poly_pred,'-')
    #ax[1].set_xlim(np.datetime64(starttime),np.datetime64(endtime))
    ax[1].set_ylabel('dY/dT', fontsize = 14)
    ax[1].set_ylim(df_SV['Y'].min() - 10,df_SV['Y'].max() + 10)
    ax[1].grid()
    
    ax[2].plot(df_SV['Z'],'o',color = 'black')
    ax[2].plot(df_SV['Zp'],'-', color = 'red')
    #ax21].plot(y_poly_pred,'-')
    #ax[2].set_xlim(np.datetime64(starttime),np.datetime64(endtime))
    ax[2].set_ylabel('dZ/dT', fontsize = 14)
    ax[2].set_xlabel('Years', fontsize = 14 )
    ax[2].set_ylim(df_SV['Z'].min() - 10,df_SV['Z'].max() + 10)
    ax[2].grid()
    
    plt.savefig(directory + '/' + station + '_SV_LFit.jpeg', bbox_inches='tight')
    plt.show()
    
def hampel_filter_denoising(input_series, window_size, n_sigmas=3):
    input_series = input_series.resample('D').mean()
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

    

def SV_(stations, starttime, endtime, file = None):
    df_imos = pd.read_csv('Imos_INTERMAGNET.txt', sep = '\t')
    
    L_27 = ['CZT','DRV','PAF']
    L_26 = ['NGK','DRV','MAW','HAD']
    
    Files = [None,'update','off']
    if stations == None:

        for station in df_imos['Imos']:
            skiprows = 25
            if station in L_27:
                skiprows = 27
            elif station == 'ngk' or 'cnb ':
                skiprows = 26
            
            
            
            
            df_station =  load_obs_files_OTIMIZADA(station, skiprows , starttime, endtime)
            df_station.loc[df_station['X'] == 99999.0, 'X'] = np.nan
            df_station.loc[df_station['Y'] == 99999.0, 'Y'] = np.nan
            df_station.loc[df_station['Z'] == 99999.0, 'Z'] = np.nan
            
            X_SV_station = (df_station['X'][starttime:endtime].resample('M').mean().diff(6) - df_station['X'] [starttime:endtime].resample('M').mean().diff(-6)).shift(-15, freq = 'D').round(3)
            Y_SV_station = (df_station['Y'][starttime:endtime].resample('M').mean().diff(6) - df_station['Y'] [starttime:endtime].resample('M').mean().diff(-6)).shift(-15, freq = 'D').round(3)
            Z_SV_station = (df_station['Z'][starttime:endtime].resample('M').mean().diff(6) - df_station['Z'] [starttime:endtime].resample('M').mean().diff(-6)).shift(-15, freq = 'D').round(3)
            
            
            df = pd.DataFrame()
            df['X_SV'] = X_SV_station
            df['Y_SV'] = Y_SV_station
            df['Z_SV'] = Z_SV_station
            for file in Files:
                if file == None:
                    
                    directory = 'SV_update/'+ station +'_data/'
                    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)     
                    df.dropna().to_csv(directory + 'SV_' + station + '.txt', sep ='\t', index=True)
                
                if file == 'update':
                    
                    df1 = pd.read_csv('SV_update/' + station + '_data/SV_'+ station + '.txt', sep = '\t')
                    df2 = pd.concat([df1,df],)
                    df2.dropna().drop_duplicates().to_csv('SV_update/' + station + '_data/SV_'+ station + '.txt', sep = '\t',index = True)
                
                if file == 'off':    
                    pass
                if file not in Files:
                    print('File must be None, update or off!')
                    
            fig, ax = plt.subplots(3,1, figsize = (16,10))
             
            #if df['X_SV'].mean() < 0:
            #    ax[0].set_ylim(df['X_SV'].max() + 10, df['X_SV'].min() - 10)
            #else:
            #    ax[0].set_ylim( df['X_SV'].min() - 10,df['X_SV'].max() + 10)
            #    
            #if df['Y_SV'].mean() < 0:
            #    ax[1].set_ylim(df['Y_SV'].max() + 10, df['Y_SV'].min() - 10)
            #else:
            #    ax[1].set_ylim( df['Y_SV'].min() - 10,df['Y_SV'].max() + 10)
            #    
            #if df['Z_SV'].mean() < 0:
            #    ax[2].set_ylim(df['Z_SV'].max() + 10, df['Z_SV'].min() - 10)
            #else:
            #    ax[2].set_ylim( df['Z_SV'].min() - 10,df['Z_SV'].max() + 10)
                
            
            ax[0].plot(X_SV_station,'o', color = 'blue')
            ax[0].set_ylim(df['X_SV'].min() - 10, df['X_SV'].max() + 10)
            ax[0].set_title(station.upper() + ' Secular Variation', fontsize = 16)
            ax[1].plot(Y_SV_station,'o',color = 'green')
            ax[1].set_ylim(df['Y_SV'].min() - 10, df['Y_SV'].max() + 10)
            ax[2].plot(Z_SV_station,'o',color = 'black')
            ax[2].set_ylim( df['Z_SV'].min() - 10,df['Z_SV'].max() + 10)
            
            #ax[0].autoscale()
            #ax[0].set_ylim(df['X_SV'].max() + 10, df['X_SV'].min() - 10)
            #ax[1].set_ylim(l_inf_y,l_sup_y)
            #ax[2].set_ylim(l_inf_z,l_sup_z)
            plt.show()
    
            
            directory2 = 'Map_plots/'+ station +'_data'
            pathlib.Path(directory2).mkdir(parents=True, exist_ok=True)  
            
            #plot to use in the interactive map 
            
            fig, ax = plt.subplots(3,1, figsize = (8,6.5))
            
            ax[0].set_title(station.upper() + ' Secular Variation (ADMM)', fontsize = 12)
    
            ax[0].plot(X_SV_station, 'o', color  = 'blue')
            ax[0].set_xlim(np.datetime64(starttime),np.datetime64(endtime))
            ax[0].set_ylim(X_SV_station.min() - 10, X_SV_station.max() + 10)
            #ax[0,0].set_xlim(np.datetime64('2011-01'),np.datetime64('2021-12'))
            ax[0].set_ylabel('dX/dT(nT/yr)', fontsize = 9)
            ax[0].grid()
            
            ax[1].plot(Y_SV_station, 'o', color  = 'green')
            ax[1].set_xlim(np.datetime64(starttime),np.datetime64(endtime))
            ax[1].set_ylim(Y_SV_station.min() - 10, Y_SV_station.max() + 10)
            ax[1].set_ylabel('dY/dT(nT/yr)', fontsize = 9)
            ax[1].grid()
            
            ax[2].plot(Z_SV_station, 'o', color  =  'black')
            ax[2].set_xlim(np.datetime64(starttime),np.datetime64(endtime))
            ax[2].set_ylim(Z_SV_station.min() - 10, Z_SV_station.max() + 10)
            #ax[0].set_xlim(np.datetime64('2010-01'),np.datetime64('2021-06'))
            ax[2].set_ylabel('dZ/dT(nT/yr)', fontsize = 9)
            ax[2].grid()
            
            plt.tick_params(labelsize=8)
            plt.savefig(directory2 + '/' + station + '_map_SV.jpeg', bbox_inches='tight')
            
    for station in stations:
        skiprows = 25
        if station in L_27:
            skiprows = 27
        if station == 'ngk' or 'cnb':
            skiprows = 26
            
        df_station =  load_obs_files_OTIMIZADA(station, skiprows , starttime, endtime)
        df_station.loc[df_station['X'] == 99999.0, 'X'] = np.nan
        df_station.loc[df_station['Y'] == 99999.0, 'Y'] = np.nan
        df_station.loc[df_station['Z'] == 99999.0, 'Z'] = np.nan
    
        X_SV_station = (df_station['X'][starttime:endtime].resample('M').mean().diff(6) - df_station['X'] [starttime:endtime].resample('M').mean().diff(-6)).shift(-15, freq = 'D').round(3)
        Y_SV_station = (df_station['Y'][starttime:endtime].resample('M').mean().diff(6) - df_station['Y'] [starttime:endtime].resample('M').mean().diff(-6)).shift(-15, freq = 'D').round(3)
        Z_SV_station = (df_station['Z'][starttime:endtime].resample('M').mean().diff(6) - df_station['Z'] [starttime:endtime].resample('M').mean().diff(-6)).shift(-15, freq = 'D').round(3)

        df = pd.DataFrame()
        df['X_SV'] = X_SV_station
        df['Y_SV'] = Y_SV_station
        df['Z_SV'] = Z_SV_station
        df = df.dropna()
        if file == None:
    
            directory = 'SV_update/'+ station +'_data/'
            pathlib.Path(directory).mkdir(parents=True, exist_ok=True)     
            df.dropna().to_csv(directory + 'SV_' + station + '.txt', sep ='\t')
            
        if file == 'update':
    
            df1 = pd.read_csv('SV_update/' + station + '_data/SV_'+ station + '.txt', sep = '\t')
            df1['Date'] = pd.to_datetime(df1['Date'], infer_datetime_format=True)
            df1.set_index('Date', inplace = True)
            df2 = pd.concat([df1,df])
            df2.dropna().drop_duplicates().to_csv('SV_update/' + station + '_data/SV_'+ station + '.txt', sep = '\t')
        if file == 'off':
            pass
        
        if file not in Files:
            print('File must be None, update or off!')
            pass    
        
        
        directory2 = 'Map_plots/'+ station +'_data'
        pathlib.Path(directory2).mkdir(parents=True, exist_ok=True)  
        
        fig, ax = plt.subplots(3,1, figsize = (8,6.5))
        
        ax[0].set_title(station.upper() + ' Secular Variation (ADMM)', fontsize = 12)

        ax[0].plot(X_SV_station, 'o', color  = 'blue')
        ax[0].set_xlim(np.datetime64(starttime),np.datetime64(endtime))
        ax[0].set_ylim(X_SV_station.min() - 10, X_SV_station.max() + 10)
        #ax[0,0].set_xlim(np.datetime64('2011-01'),np.datetime64('2021-12'))
        ax[0].set_ylabel('dX/dT(nT/yr)', fontsize = 9)
        ax[0].grid()
        
        ax[1].plot(Y_SV_station, 'o', color  = 'green')
        ax[1].set_xlim(np.datetime64(starttime),np.datetime64(endtime))
        ax[1].set_ylim(Y_SV_station.min() - 10, Y_SV_station.max() + 10)
        ax[1].set_ylabel('dY/dT(nT/yr)', fontsize = 9)
        ax[1].grid()
        
        ax[2].plot(Z_SV_station, 'o', color  =  'black')
        ax[2].set_xlim(np.datetime64(starttime),np.datetime64(endtime))
        ax[2].set_ylim(Z_SV_station.min() - 10, Z_SV_station.max() + 10)
        #ax[0].set_xlim(np.datetime64('2010-01'),np.datetime64('2021-06'))
        ax[2].set_ylabel('dZ/dT(nT/yr)', fontsize = 9)
        ax[2].grid()
        plt.tick_params(labelsize=8)
        plt.savefig(directory2 + '/' + station + '_map_SV.jpeg', bbox_inches='tight')
      
        
        fig, ax = plt.subplots(3,1, figsize = (16,10))
        
        #if df['X_SV'].mean() < 0:
        #    ax[0].set_ylim(df['X_SV'].min() - 10, df['X_SV'].max() + 10)
        #else:
        #    ax[0].set_ylim( df['X_SV'].min() - 10,df['X_SV'].max() + 10)
        #    
        #if df['Y_SV'].mean() < 0:
        #    ax[1].set_ylim(df['Y_SV'].min() - 10, df['Y_SV'].max() + 10)
        #else:
        #    ax[1].set_ylim(df['Y_SV'].min() - 10,df['Y_SV'].max() + 10)
        #    
        #if df['Z_SV'].mean() < 0:
        #    ax[2].set_ylim(df['Z_SV'].min() - 10, df['Z_SV'].max() + 10)
        #else:
        #    ax[2].set_ylim( df['Z_SV'].min() - 10,df['Z_SV'].max() + 10)
        
        ax[0].plot(X_SV_station,'o', color = 'blue')
        ax[0].set_ylim(df['X_SV'].min() - 10, df['X_SV'].max() + 10)
        ax[0].set_title(station.upper() + ' Secular Variation', fontsize = 16)
        ax[1].plot(Y_SV_station,'o',color = 'green')
        ax[1].set_ylim(df['Y_SV'].min() - 10, df['Y_SV'].max() + 10)
        ax[2].plot(Z_SV_station,'o',color = 'black')
        ax[2].set_ylim( df['Z_SV'].min() - 10,df['Z_SV'].max() + 10)
        #ax[0].set_ylim(l_inf,l_sup)
        #ax[1].set_ylim(l_inf,l_sup)
        #ax[2].set_ylim(l_inf,l_sup)
        plt.show()
