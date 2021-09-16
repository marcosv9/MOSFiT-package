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



def load_obs_files(station, path, skiprows):
    '''
    Function to read and concat observatory data
    
    '''
    files_station = glob.glob(path)
    files_station.sort()
    print('Ther number of files readed is:',len(files_station),'.')
    
    
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


def load_obs_files_OTIMIZADA(station, skiprows, starttime, endtime):
    '''
    Function to read and concat observatory data
    
    Sample must be H, D, M, Y   
    
    '''
    
    Years  = []
    for i in range(int(starttime[0:4]),int(endtime[0:4])):
        Y = i
        Years.append(Y)
    Years.extend([str(i) for i in Years])
    
    files_station = []
    files_station.extend([glob.glob('Dados OBS\\' + Year + '/*/vss*') for Year in Years])
    files_station.sort()
    print('Reading files...')
    
    
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
    df_q = pd.read_excel('Dados OBS/Q_DAYS.xls', skiprows = 1, 
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
    df = df.loc[(df.index.hour >= 19)]
    
    return df

def SV_obs(station, skiprows, starttime, endtime):
    '''
    Function to read and concat observatory data
    
    Sample must be H, D, M, Y   
    
    '''
    files_station = glob.glob('Dados OBS/20*/*/' + station + '*')
    files_station.sort()
    print('Reading files...')
    
    
    #d_parser = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S%.f')
    
    
    df_station = pd.concat( (pd.read_csv(file, sep='\s+',usecols = [0,1,3,4,5], 
                   header = None,skiprows = skiprows, 
                   parse_dates = {'Date': ['date', 'Time']},
                   names = ['date','Time','X','Y','Z']) for file in files_station), 
                   ignore_index = True)
    df_station['Date'] = pd.to_datetime(df_station['Date'], format = '%Y-%m-%dd %H:%M:%S.%f')     
    #df_station['Hour'] = pd.to_datetime(df_station['Hour'], format = '%H:%M:%S.%f').dt.time               
    df_station.set_index('Date', inplace = True)
    
    
    df_station.loc[df_station['X'] == 99999.0, 'X'] = np.nan
    df_station.loc[df_station['Y'] == 99999.0, 'Y'] = np.nan
    df_station.loc[df_station['Z'] == 99999.0, 'Z'] = np.nan
    df_station2 = df_station
    
    inp = input("Press Q to use only Quiet Days, D to remove Disturbed Days or E to Exit without actions [Q/D/E]")
    
    if inp == 'Q':
        
        df_station = keep_Q_Days(df_station, starttime, endtime)
        print('Only Quiet Days remains')
    if inp == 'D':
        
        df_station = remove_Disturbed_Days(df_station, starttime, endtime)
        print('Disturbed Days removed')
    if inp == 'E':
        print('No action')
        

        
    samples = ['Min','H','D','M','Y']
    directory = 'Filtered_data/'+ station +'_data'
    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)  
    
    X_SV = (df_station['X'][starttime:endtime].resample('M').mean().diff(6) - df_station['X'] [starttime:endtime].resample('M').mean().diff(-6)).shift(-15, freq = 'D').round(3)
    Y_SV = (df_station['Y'][starttime:endtime].resample('M').mean().diff(6) - df_station['Y'] [starttime:endtime].resample('M').mean().diff(-6)).shift(-15, freq = 'D').round(3)
    Z_SV = (df_station['Z'][starttime:endtime].resample('M').mean().diff(6) - df_station['Z'] [starttime:endtime].resample('M').mean().diff(-6)).shift(-15, freq = 'D').round(3)
                
    while input("Do You Want To Save a File With the Variation? [y/n]") == "y":
        print('Saving files...')
        for sample in samples:
            #
            #if (starttime or endtime == None) and sample == 'Min':
    #
            #    file = df_station.copy().resample(sample).mean().round(3)
            #    file.to_csv(directory + '/' + station + '_' + sample + '_mean.txt', sep ='\t', index=True)
            #
            #
            #if (starttime or endtime == None) and sample == 'H':
    #
            #    file = df_station.copy().resample(sample).mean().shift(-30, freq = 'Min').round(3)
            #    file.to_csv(directory + '/' + station + '_' + sample + '_mean.txt', sep ='\t', index=True)
            #
            #if (starttime or endtime == None) and sample == 'D':
    #
            #    file = df_station.copy().resample(sample).mean().shift(12, freq = 'H').round(decimals = 3)
            #    file.to_csv(directory + '/' + station + '_' + sample + '_mean.txt', sep ='\t', index=True)
        #
            #if (starttime or endtime == None) and sample == 'M':
    #
            #    file = df_station.copy().resample(sample).mean().shift(-15, freq = 'D').round(3)
            #    file.to_csv(directory + '/' + station + '_' + sample + '_mean.txt', sep ='\t', index=True)
    #
            #if (starttime or endtime == None) and sample == 'Y':
    #
            #    file = df_station.copy().resample(sample).mean().shift(-182.5, freq = 'D').round(3)
            #    file.to_csv(directory + '/' + station + '_' + sample + '_mean.txt', sep ='\t', index=True) 
            
    #df_station.copy() == df_station.resample(sample).mean()
    
    #if sample == 'H':                      
        
            #file = df_station.copy().resample(sample).mean().round(decimals = 3)   
            #file.to_csv(directory + '/' + station + '_' + sample + '_mean.txt', sep ='\t', index=True)
                      
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
        
        
        break
        
      
    
    #if starttime == None and endtime == None:
    #    df_station == df_station
    #df_station = df_station.loc[starttime:endtime]
    #df_station = np.where(df_station > 80000, np.nan, df_station)
    #if df_station.max() > 200000:
        #df_station = df_station.replace(df_station.max(), np.nan)
        
    #f= open(+ station + '_from_' + stattime + '_to_' + endtime + '.txt","w+")
    #f.write(df_station)
    #while input("Do You Want To Save a File With the Variation? [y/n]") == "y":
    #    directory = 'Filtered_data/'+ station +'_data'
    #    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)     
    #    file = df_station.round(decimals = 3)
    #    
    #    file.to_csv(directory + '/' + station + '_from_' + starttime +
    #            '_to_' + endtime + '.txt', sep ='\t', index=True)
    #    break
        
    while input("Do You Want To Save Plot of the Variation and SV for X, Y and Z? [y/n]") == "y":
        
        
        pathlib.Path(directory).mkdir(parents=True, exist_ok=True) 
        
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
        ax[0,0].plot(df_station['X'][starttime:endtime].resample('M').mean(), color  = 'blue')
        ax[0,0].set_xlim(np.datetime64(starttime),np.datetime64(endtime))
        ax[0,0].set_ylabel('X/nT', fontsize = 14)
        
        ax[0,0].grid()
        
        ax[1,0].plot(df_station['Y'][starttime:endtime].resample('M').mean(), color  = 'green')
        ax[1,0].set_xlim(np.datetime64(starttime),np.datetime64(endtime))
        ax[1,0].set_ylabel('Y/nT', fontsize = 14)
        
        ax[1,0].grid()


        ax[2,0].plot(df_station['Z'][starttime:endtime].resample('M').mean(), color  = 'black')
        ax[2,0].set_xlim(np.datetime64(starttime),np.datetime64(endtime))
        ax[2,0].set_ylabel('Z/nT', fontsize = 14)
        #ax[2,1].set_xlabel('Years', fontsize = 14 )
        #ax[1].set_ylim(-30,30)
        ax[2,0].grid()
        
        plt.savefig(directory + '/' + station + '_Var_SV.jpeg', bbox_inches='tight')
              
        
         #plot of SV alone     
              
        fig, ax = plt.subplots(3,1, figsize = (16,12))
        
        ax[0].set_title(station.upper() + ' Secular Variation (ADMM)', fontsize = 18)

        ax[0].plot(X_SV, 'o', color  = 'blue')
        ax[0].set_xlim(np.datetime64(starttime),np.datetime64(endtime))
        #ax[0].set_ylim(X_SV.min()*0.9,X_SV.max()*1.1)
        #ax[0,0].set_xlim(np.datetime64('2011-01'),np.datetime64('2021-12'))
        ax[0].set_ylabel('dX/dT(nT/yr)', fontsize = 12)
        ax[0].grid()
        
        ax[1].plot(Y_SV, 'o', color  = 'green')
        ax[1].set_xlim(np.datetime64(starttime),np.datetime64(endtime))
        #ax[1].set_ylim(Y_SV.min()*0.9,Y_SV.max()*1.1)
        #ax[0].set_xlim(np.datetime64('2010-01'),np.datetime64('2021-06'))
        ax[1].set_ylabel('dY/dT(nT/yr)', fontsize = 12)
        ax[1].grid()
        
        ax[2].plot(Z_SV, 'o', color  =  'black')
        ax[2].set_xlim(np.datetime64(starttime),np.datetime64(endtime))
        #ax[2].set_ylim(Z_SV.min()*0.9,Z_SV.max()*1.1)
        #ax[0].set_xlim(np.datetime64('2010-01'),np.datetime64('2021-06'))
        ax[2].set_ylabel('dZ/dT(nT/yr)', fontsize = 12)
        ax[2].grid()
        
        plt.savefig(directory + '/' + station + '_SV.jpeg', bbox_inches='tight')
        #plt.show()
        
        for sample in samples:
            
            #if starttime == None and endtime == None:
            #   
            #    fig, ax = plt.subplots(3,1,figsize = (16,10))
        #
            #    ax[0].plot(df_station['X'].resample(sample).mean(), color  = 'blue')
            #    ax[0].set_ylabel('X (nT)', fontsize = 12)
            #    ax[0].grid()
            #    
            #    ax[1].plot(df_station['Y'].resample(sample).mean(), color  = 'green')
            #    ax[1].set_ylabel('Y (nT)', fontsize = 12)
            #    ax[1].grid()
            #    
            #    ax[2].plot(df_station['Z'].resample(sample).mean(), color  =  'black')
            #    ax[2].set_ylabel('Z (nT)', fontsize = 12)
            #    ax[2].grid()
            #    
            #    #plt.show()
            #    plt.savefig(directory + '/' + station + '_' + sample + '_mean.jpeg', bbox_inches='tight')
                
                         
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
                ax[0].plot(df_station['X'][starttime:endtime].resample(sample).mean().shift(-182.5, freq = 'D'), color  = 'blue')
                ax[1].plot(df_station['Y'][starttime:endtime].resample(sample).mean().shift(-182.5, freq = 'D'), color  = 'green')
                ax[2].plot(df_station['Z'][starttime:endtime].resample(sample).mean().shift(-182.5, freq = 'D'), color  = 'black')    
                
          
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
            
            #plt.show()
            plt.savefig(directory + '/' + station + '_' + sample + '_mean.jpeg', bbox_inches='tight')
            

        break
    #file = df_station.round(decimals = 3)
    #file.to_csv('Filtered_data/'+ station + '_from_' + starttime +
    #            '_to_' + endtime + '.txt', sep ='\t', index=True)
    
        
    
    return df_station[starttime:endtime]  
    
