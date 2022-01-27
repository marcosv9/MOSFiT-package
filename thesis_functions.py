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
from Thesis_Marcos import utilities_tools as utt
from Thesis_Marcos import data_processing_tools as dpt
from sklearn.preprocessing import PolynomialFeatures



def load_obs_files_not_in_use(station, path, skiprows):
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

def load_INTERMAGNET_files(station, starttime, endtime):
    '''
    Function to read and concat observatory data.
    Works with every INTERMAGNET Observatory.
    
    Data types:
    
    Quasi-definitive and definitive data.
    
    Inputs:
    
    station - 3 letters IAGA code
    starttime - Begin of the time interest
    endtime - The end of the period.
    
    Usage example:
    load_INTERMAGNET_files(station = 'VSS', starttime = '2006-01-25', endtime = '2006-01-25')
    
    
    Return a pd.DataFrame of all readed data.
    '''
    assert len(station) == 3, 'station must be a IAGA code with 3 letters'
    
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

    df_station = pd.DataFrame()
    df_station = pd.concat( (pd.read_csv(file, sep='\s+',usecols = [0,1,3,4,5], 
                   header = None,skiprows = skiprows, 
                   parse_dates = {'Date': ['date', 'Time']},
                   names = ['date','Time','X','Y','Z']) for file in files_station), 
                   ignore_index = True)
    df_station['Date'] = pd.to_datetime(df_station['Date'], format = '%Y-%m-%dd %H:%M:%S.%f')     
              
    df_station.set_index('Date', inplace = True)

    
    df_station.loc[df_station['X'] >= 99999.0, 'X'] = np.nan
    df_station.loc[df_station['Y'] >= 99999.0, 'Y'] = np.nan
    df_station.loc[df_station['Z'] >= 99999.0, 'Z'] = np.nan

    df_station = df_station.loc[starttime:endtime]
    
    return df_station

def SV_obs(station, starttime, endtime, plot_chaos: bool = False):
    '''
    Interactive function for INTERMAGNET data processing
    
    data processing option:
    
    Denoising filter
    External field reduction - Disturned Days, Quiet days or night time selection
    CHAOS-7 model correction
    
    Jerk detection based on linear segments
    
    Return plots of the data

    
    '''
    
    df_station = load_INTERMAGNET_files(station,
                                          starttime,
                                          endtime)
    
    df_station2 = df_station.copy()
    
    First_QD_data = utt.data_type(station = station,
                              starttime = starttime,
                              endtime = endtime)
    
    while True: 
        inp5 = input("Do You Want To denoise the data based on median absolute deviation? [y/n]: ")
        if inp5 == 'y':
            print('Denoising data...')
            df_station = dpt.hampel_filter_denoising(input_series = df_station, window_size = 100, n_sigmas=3)
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
        
        df_station = dpt.keep_Q_Days(df_station, starttime, endtime)
        print('Only Quiet Days remains')
    if inp == 'D':
        
        df_station = dpt.remove_Disturbed_Days(df_station, starttime, endtime)
        print('Disturbed Days removed')
        
    if inp == 'NT':
        
        df_station = dpt.night_time_selection(station, df_station, starttime, endtime)
        print('Night-time selected')
        
    if inp == 'E':
        print('No action')
        
    while True:
    
        input_chaos = input("Do You want to correct the external field using the CHAOS model? [y/n]: ")
        if input_chaos == 'y':
            df_station, df_chaos = dpt.external_field_correction_chaos_model(station = station,
                                                           starttime = starttime,
                                                           endtime = endtime,
                                                           df_station = df_station,
                                                           df_chaos = None)
            print('External field corrected using CHAOS!')
            break
        if input_chaos == 'n':
            print('Correction using CHAOS was not applied.')
            break
        else:
            print('You must type y or n, try again!')
        

    directory = 'Filtered_data/'+ station +'_data'
    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)  
    

    df_SV = dpt.calculate_SV(df_station, starttime = starttime, endtime = endtime)
    
    df_SV_not_corrected = dpt.calculate_SV(df_station2, starttime = starttime, endtime = endtime)
    
    if input_chaos == 'y':
        
        df_chaos_SV = dpt.calculate_SV(df_chaos, starttime = starttime, endtime = endtime, columns = ['X_int','Y_int','Z_int'])   
    else:
        
        pass
    
    while True: 
        inp2 = input("Do You Want To Save a File With the Variation? [y/n]: ")
        if inp2 == 'y':

            print('Saving files...')

            for sample in ['Min','H','D', 'M', 'Y']:
                          
                if sample == 'Min':
                
                    file = df_station[starttime:endtime].resample(sample).mean().round(3)
                    #ax.plot(df_HER['X']['2010-01':'2019-12'].resample(sample).mean().shift(12, freq = 'H'))
                    file.to_csv(directory + '/' + station.upper() + '_minute_mean_preliminar.zip', sep ='\t', index=True)
            
                if sample == 'H':
                    
                    file = dpt.resample_obs_data(df_station, 'H').round(3)
                    #ax.plot(df_HER['X']['2010-01':'2019-12'].resample(sample).mean().shift(12, freq = 'H'))
                    file.to_csv(directory + '/' + station.upper() + '_hourly_mean_preliminar.txt', sep ='\t', index=True)
                
                    utt.Header_SV_obs_files(station = station,
                                        filename = 'hourly_mean',
                                        data_denoise = inp5,
                                        external_correction = inp,
                                        chaos_model = input_chaos)               
                if sample == 'D':
                    
                    file = dpt.resample_obs_data(df_station, 'D').round(3)
                    #ax.plot(df_HER['X']['2010-01':'2019-12'].resample(sample).mean().shift(12, freq = 'H'))
                    file.to_csv(directory + '/' + station.upper() + '_daily_mean_preliminar.txt', sep ='\t', index=True)
                    
                    utt.Header_SV_obs_files(station = station,
                                        filename = 'daily_mean',
                                        data_denoise = inp5,
                                        external_correction = inp,
                                        chaos_model = input_chaos) 
                if sample == 'M':
                    
                   # ax.plot(df_HER['X']['2010-01':'2019-12'].resample(sample).mean().shift(-15, freq = 'D'))
                    file = dpt.resample_obs_data(df_station, 'M').round(3)
                    
                    file_SV = df_SV
                    
                    file.to_csv(directory + '/' + station.upper() +'_monthly_mean_preliminar.txt', sep ='\t', index=True)
                    
                    utt.Header_SV_obs_files(station = station,
                                        filename = 'monthly_mean',
                                        data_denoise = inp5,
                                        external_correction = inp,
                                        chaos_model = input_chaos) 
                    
                    file_SV.to_csv(directory + '/' + station.upper() +'_secular_variation_preliminar.txt', sep ='\t', index=True)
                    
                    utt.Header_SV_obs_files(station = station,
                                        filename = 'secular_variation',
                                        data_denoise = inp5,
                                        external_correction = inp,
                                        chaos_model = input_chaos) 
                if sample == 'Y':
                    
                    #ax.plot(df_HER['X']['2010-01':'2019-12'].resample(sample).mean().shift(-182.5, freq = 'D'))
                    file = dpt.resample_obs_data(df_station, 'Y').round(3)
                    file.to_csv(directory + '/' + station.upper() + '_annual_mean_preliminar.txt', sep ='\t', index=True)
                    
                    utt.Header_SV_obs_files(station = station,
                                        filename = 'annual_mean',
                                        data_denoise = inp5,
                                        external_correction = inp,
                                        chaos_model = input_chaos) 
                    
            print('Minute, Hourly, Daily, Monthly, Annual means and Secular Variation were saved on directory:')
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
            
            #plot minute mean
            if input_chaos == 'y' or inp5 == 'y':
                fig, ax = plt.subplots(3,1, figsize = (16,10))
                
                ax[0].set_title(station.upper() + ' minute mean', fontsize = 18)
                ax[0].plot(df_station2['X'], color  = 'blue')
                ax[0].set_xlim(df_station2['X'].index[0],df_station2['X'].index[-1])
                ax[0].set_ylabel('X(nT)', fontsize = 12)
                ax[0].grid()
                
                ax[1].plot(df_station2['Y'], color  = 'green')
                ax[1].set_xlim(df_station2['Y'].index[0],df_station2['Y'].index[-1])
                ax[1].set_ylabel('Y(nT)', fontsize = 12)
                ax[1].grid()
                
                ax[2].plot(df_station2['Z'], color  =  'black')
                ax[2].set_xlim(df_station2['Z'].index[0],df_station2['Z'].index[-1])
                ax[2].set_ylabel('Z(nT)', fontsize = 12)
                ax[2].grid()
                
                if First_QD_data != []:
                    ax[0].plot(df_station2.loc[df_station2.index > First_QD_data]['X'], color = 'red',label = 'Quasi-definitive data')
                    ax[1].plot(df_station2.loc[df_station2.index > First_QD_data]['Y'], color = 'red',label = 'Quasi-definitive data')
                    ax[2].plot(df_station2.loc[df_station2.index > First_QD_data]['Z'], color = 'red',label = 'Quasi-definitive data')
                    ax[0].legend()
                    ax[1].legend()
                    ax[2].legend()
                
                plt.savefig(directory + '/' + station + '_minute_mean.jpeg', bbox_inches='tight')
                plt.show()
                
            else:
                
                fig, ax = plt.subplots(3,1, figsize = (16,10))
                
                ax[0].set_title(station.upper() + ' minute mean', fontsize = 18)
                ax[0].plot(df_station['X'], 'o', color  = 'blue')
                ax[0].set_xlim(df_station['X'].index[0],df_station['X'].index[-1])
                ax[0].set_ylim(df_station['X'].min(), df_station['X'].max())
                #ax[0,0].set_xlim(np.datetime64('2011-01'),np.datetime64('2021-12'))
                ax[0].set_ylabel('dX/dT(nT/yr)', fontsize = 12)
                ax[0].grid()
                
                ax[1].plot(df_station['Y'], 'o', color  = 'green')
                ax[1].set_xlim(df_station['Y'].index[0],df_station['Y'].index[-1])
                ax[1].set_ylim(df_station['Y'].min(), df_station['Y'].max())
                ax[1].set_ylabel('dY/dT(nT/yr)', fontsize = 12)
                ax[1].grid()
                
                ax[2].plot(df_station['Z'], 'o', color  =  'black')
                ax[2].set_xlim(df_station['Z'].index[0],df_station['Z'].index[-1])
                ax[2].set_ylim(df_station['Z'].min(), df_station['Z'].max())
                #ax[0].set_xlim(np.datetime64('2010-01'),np.datetime64('2021-06'))
                ax[2].set_ylabel('dZ/dT(nT/yr)', fontsize = 12)
                ax[2].grid()
                
                if First_QD_data != []:
                    ax[0].plot(df_station.loc[df_station.index > First_QD_data]['X'], color = 'red',label = 'Quasi-definitive data')
                    ax[1].plot(df_station.loc[df_station.index > First_QD_data]['Y'], color = 'red',label = 'Quasi-definitive data')
                    ax[2].plot(df_station.loc[df_station.index > First_QD_data]['Z'], color = 'red',label = 'Quasi-definitive data')
                    ax[0].legend()
                    ax[1].legend()
                    ax[2].legend()
                
                plt.savefig(directory + '/' + station + '_minute_mean.jpeg', bbox_inches='tight')
                plt.show()
                    
            
            if First_QD_data != []:
            
                plot_samples(station = station, dataframe = df_station, save_plots = True, plot_data_type = First_QD_data)
            else:
                plot_samples(station = station, dataframe = df_station, save_plots = True, plot_data_type = None)
            
            #plot of secular variation and monthly mean
            
            fig, ax = plt.subplots(3,2, figsize = (18,10))    
            
            ax[0,1].set_title(station.upper() + ' Secular Variation (ADMM)', fontsize = 18)
            ax[0,1].plot(df_SV['X'], 'o', color  = 'blue')
            ax[0,1].set_xlim(df_SV['X'].index[0],df_SV['X'].index[-1])
            ax[0,1].set_ylabel('dX/dT(nT/yr)', fontsize = 12)
            ax[0,1].grid()
            
            ax[1,1].plot(df_SV['Y'], 'o', color  = 'green')
            ax[1,1].set_xlim(df_SV['Y'].index[0],df_SV['Y'].index[-1])
            ax[1,1].set_ylabel('dY/dT(nT/yr)', fontsize = 12)
            ax[1,1].grid()
            
            ax[2,1].plot(df_SV['Z'], 'o', color  =  'black')
            ax[2,1].set_xlim(df_SV['Z'].index[0],df_SV['Z'].index[-1])
            ax[2,1].set_ylabel('dZ/dT(nT/yr)', fontsize = 12)
            ax[2,1].grid()
            
            
            ax[0,0].set_title(station.upper() + ' Monthly Mean', fontsize = 18)
            ax[0,0].plot(df_station['X'][starttime:endtime].resample('M').mean().shift(-15, freq = 'D'), color  = 'blue')
            ax[0,0].set_xlim(df_station['X'].index[0],df_station['X'].index[-1])
            ax[0,0].set_ylabel('X/nT', fontsize = 14)   
            ax[0,0].grid()
            
            ax[1,0].plot(df_station['Y'][starttime:endtime].resample('M').mean().shift(-15, freq = 'D'), color  = 'green')
            ax[1,0].set_xlim(df_station['Y'].index[0],df_station['Y'].index[-1])
            ax[1,0].set_ylabel('Y/nT', fontsize = 14)           
            ax[1,0].grid()
    
    
            ax[2,0].plot(df_station['Z'][starttime:endtime].resample('M').mean().shift(-15, freq = 'D'), color  = 'black')
            ax[2,0].set_xlim(df_station['Z'].index[0],df_station['Z'].index[-1])
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
            ax[0].set_xlim(df_SV['X'].index[0],df_SV['X'].index[-1])
            ax[0].set_ylim(df_SV['X'].min() - 3, df_SV['X'].max() + 3)
            #ax[0,0].set_xlim(np.datetime64('2011-01'),np.datetime64('2021-12'))
            ax[0].set_ylabel('dX/dT(nT/yr)', fontsize = 12)
            ax[0].grid()
            
            ax[1].plot(df_SV['Y'], 'o', color  = 'green')
            ax[1].set_xlim(df_SV['Y'].index[0],df_SV['Y'].index[-1])
            ax[1].set_ylim(df_SV['Y'].min() - 3, df_SV['Y'].max() + 3)
            ax[1].set_ylabel('dY/dT(nT/yr)', fontsize = 12)
            ax[1].grid()
            
            ax[2].plot(df_SV['Z'], 'o', color  =  'black')
            ax[2].set_xlim(df_SV['Z'].index[0],df_SV['Z'].index[-1])
            ax[2].set_ylim(df_SV['Z'].min() - 3, df_SV['Z'].max() + 3)
            #ax[0].set_xlim(np.datetime64('2010-01'),np.datetime64('2021-06'))
            ax[2].set_ylabel('dZ/dT(nT/yr)', fontsize = 12)
            ax[2].grid()
            
            plt.savefig(directory + '/' + station + '_SV.jpeg', bbox_inches='tight')
            plt.show()
            

            if input_chaos == 'y' and plot_chaos == True:
                
                #plotting real SV and corrected SV comparison
                
                fig, ax = plt.subplots(3,1, figsize = (16,10))
                
                ax[0].set_title(station.upper() + ' Secular Variation (ADMM)', fontsize = 18)
                ax[0].plot(df_SV_not_corrected['X'], 'o', color  = 'red', label = 'real data')
                ax[0].plot(df_SV['X'], 'o', color  = 'blue', label = 'CHAOS correction')
                ax[0].set_xlim(df_SV['X'].index[0],df_SV['X'].index[-1])
                ax[0].set_ylim(df_SV_not_corrected['X'].min() - 5, df_SV_not_corrected['X'].max() + 5)
                #ax[0,0].set_xlim(np.datetime64('2011-01'),np.datetime64('2021-12'))
                ax[0].set_ylabel('dX/dT(nT/yr)', fontsize = 12)
                ax[0].legend()
                ax[0].grid()
                
                
                ax[1].plot(df_SV_not_corrected['Y'], 'o', color  = 'red', label = 'real data')
                ax[1].plot(df_SV['Y'], 'o', color  = 'green', label = 'CHAOS correction')
                ax[1].set_xlim(df_SV['Y'].index[0],df_SV['Y'].index[-1])
                ax[1].set_ylim(df_SV_not_corrected['Y'].min() - 5, df_SV_not_corrected['Y'].max() + 5)
                ax[1].set_ylabel('dY/dT(nT/yr)', fontsize = 12)
                ax[1].legend()
                ax[1].grid()
                
                
                ax[2].plot(df_SV_not_corrected['Z'], 'o', color  = 'red', label = 'real data')
                ax[2].plot(df_SV['Z'], 'o', color  =  'black', label = 'CHAOS correction')
                ax[2].set_xlim(df_SV['Z'].index[0],df_SV['Z'].index[-1])
                ax[2].set_ylim(df_SV_not_corrected['Z'].min() - 5, df_SV_not_corrected['Z'].max() + 5)
                #ax[0].set_xlim(np.datetime64('2010-01'),np.datetime64('2021-06'))
                ax[2].set_ylabel('dZ/dT(nT/yr)', fontsize = 12)
                ax[2].legend()
                ax[2].grid()
                
                plt.savefig(directory + '/' + station + '_SV_correction_comparison.jpeg', bbox_inches='tight')
                plt.show()
                
                #plotting chaos predicted and corrected SV
                
                fig, ax = plt.subplots(3,1, figsize = (16,10))
                
                ax[0].set_title(station.upper() + ' Secular Variation (ADMM)', fontsize = 18)
                ax[0].plot(df_chaos_SV['X_int'], 'o', color  = 'red', label = 'Chaos prediction SV')
                ax[0].plot(df_SV['X'], 'o', color  = 'blue')
                ax[0].set_xlim(df_SV['X'].index[0],df_SV['X'].index[-1])
                ax[0].set_ylim(df_SV['X'].min() - 5, df_SV['X'].max() + 5)
                ax[0].set_ylabel('dX/dT(nT/yr)', fontsize = 12)
                ax[0].legend()
                ax[0].grid()
                
                
                ax[1].plot(df_chaos_SV['Y_int'], 'o', color  = 'red', label = 'Chaos prediction SV')
                ax[1].plot(df_SV['Y'], 'o', color  = 'green', label = 'CHAOS correction')
                ax[1].set_xlim(df_SV['Y'].index[0],df_SV['Y'].index[-1])
                ax[1].set_ylim(df_SV['Y'].min() - 5, df_SV['Y'].max() + 5)
                ax[1].set_ylabel('dY/dT(nT/yr)', fontsize = 12)
                ax[1].legend()
                ax[1].grid()
                
                
                ax[2].plot(df_chaos_SV['Z_int'], 'o', color  = 'red', label = 'Chaos prediction SV')
                ax[2].plot(df_SV['Z'], 'o', color  =  'black')
                ax[2].set_xlim(df_SV['Z'].index[0],df_SV['Z'].index[-1])
                ax[2].set_ylim(df_SV['Z'].min() - 5, df_SV['Z'].max() + 5)
                #ax[0].set_xlim(np.datetime64('2010-01'),np.datetime64('2021-06'))
                ax[2].set_ylabel('dZ/dT(nT/yr)', fontsize = 12)
                ax[2].legend()
                ax[2].grid()
                
                plt.savefig(directory + '/' + station + '_SV_predicted_and_correction_comparison.jpeg', bbox_inches='tight')
                plt.show()
                
            
            print('Plots of Minute, Hourly, Daily, Monthly, Yearly means and Secular Variation were saved on directory:')
            print(directory)    
            
            break
        elif inp3 == 'n':
            print('No plots saved')
            
            #plot minute mean
            
            if input_chaos == 'y' or inp5 == 'y':
                fig, ax = plt.subplots(3,1, figsize = (16,10))
                
                ax[0].set_title(station.upper() + ' minute mean', fontsize = 18)
                ax[0].plot(df_station2['X'], color  = 'blue')
  
                ax[0].set_xlim(df_station2['X'].index[0],df_station2['X'].index[-1])
                ax[0].set_ylabel('X(nT)', fontsize = 12)
                ax[0].grid()
                
                ax[1].plot(df_station2['Y'], color  = 'green')
                ax[1].set_xlim(df_station2['Y'].index[0],df_station2['Y'].index[-1])
                ax[1].set_ylabel('Y(nT)', fontsize = 12)
                ax[1].grid()
                
                ax[2].plot(df_station2['Z'], color  =  'black')
                ax[2].set_xlim(df_station2['Z'].index[0],df_station2['Z'].index[-1])
                ax[2].set_ylim(df_station2['Z'].min() - 3, df_station2['Z'].max() + 3)
                #ax[0].set_xlim(np.datetime64('2010-01'),np.datetime64('2021-06'))
                ax[2].set_ylabel('Z(nT)', fontsize = 12)
                ax[2].grid()
                
                if First_QD_data != []:
                    ax[0].plot(df_station2.loc[df_station2.index > First_QD_data]['X'], color = 'red',label = 'Quasi-definitive data')
                    ax[1].plot(df_station2.loc[df_station2.index > First_QD_data]['Y'], color = 'red',label = 'Quasi-definitive data')
                    ax[2].plot(df_station2.loc[df_station2.index > First_QD_data]['Z'], color = 'red',label = 'Quasi-definitive data')
                    ax[0].legend()
                    ax[1].legend()
                    ax[2].legend()
                
                plt.show()
                
            else:
                
                fig, ax = plt.subplots(3,1, figsize = (16,10))
                
                ax[0].set_title(station.upper() + ' minute mean', fontsize = 18)
                ax[0].plot(df_station['X'], color  = 'blue')
                ax[0].set_xlim(df_station['X'].index[0],df_station['X'].index[-1])
                ax[0].set_ylabel('dX/dT(nT/yr)', fontsize = 12)
                ax[0].grid()
                
                ax[1].plot(df_station['Y'], color  = 'green')
                ax[1].set_xlim(df_station['Y'].index[0],df_station['Y'].index[-1])
                ax[1].set_ylabel('dY/dT(nT/yr)', fontsize = 12)
                ax[1].grid()
                
                ax[2].plot(df_station['Z'], color  =  'black')
                ax[2].set_xlim(df_station['Z'].index[0],df_station['Z'].index[-1])
                ax[2].set_ylabel('dZ/dT(nT/yr)', fontsize = 12)
                ax[2].grid()
                
                if First_QD_data != []:
                    ax[0].plot(df_station.loc[df_station.index > First_QD_data]['X'], color = 'red',label = 'Quasi-definitive data')
                    ax[1].plot(df_station.loc[df_station.index > First_QD_data]['Y'], color = 'red',label = 'Quasi-definitive data')
                    ax[2].plot(df_station.loc[df_station.index > First_QD_data]['Z'], color = 'red',label = 'Quasi-definitive data')
                    ax[0].legend()
                    ax[1].legend()
                    ax[2].legend()
                
                plt.show()
                
            if First_QD_data != []:
            
                plot_samples(station = station, dataframe = df_station, save_plots = False, plot_data_type = First_QD_data)
            else:
                plot_samples(station = station, dataframe = df_station, save_plots = False, plot_data_type = None)
            
            #plot of secular variation and monthly mean
            
            fig, ax = plt.subplots(3,2, figsize = (18,10))    
            
            ax[0,1].set_title(station.upper() + ' Secular Variation (ADMM)', fontsize = 18)
            ax[0,1].plot(df_SV['X'], 'o', color  = 'blue')
            ax[0,1].set_xlim(df_SV['X'].index[0],df_SV['X'].index[-1])
            ax[0,1].set_ylabel('dX/dT(nT/yr)', fontsize = 12)
            ax[0,1].grid()
            
            ax[1,1].plot(df_SV['Y'], 'o', color  = 'green')
            ax[1,1].set_xlim(df_SV['Y'].index[0],df_SV['Y'].index[-1])
            ax[1,1].set_ylabel('dY/dT(nT/yr)', fontsize = 12)
            ax[1,1].grid()
            
            ax[2,1].plot(df_SV['Z'], 'o', color  =  'black')
            ax[2,1].set_xlim(df_SV['Z'].index[0],df_SV['Z'].index[-1])
            ax[2,1].set_ylabel('dZ/dT(nT/yr)', fontsize = 12)
            ax[2,1].grid()
            
            
            ax[0,0].set_title(station.upper() + ' Monthly Mean', fontsize = 18)
            ax[0,0].plot(df_station['X'][starttime:endtime].resample('M').mean().shift(-15, freq = 'D'), color  = 'blue')
            ax[0,0].set_xlim(df_station['X'].index[0],df_station['X'].index[-1])
            ax[0,0].set_ylabel('X/nT', fontsize = 14)   
            ax[0,0].grid()
            
            ax[1,0].plot(df_station['Y'][starttime:endtime].resample('M').mean().shift(-15, freq = 'D'), color  = 'green')
            ax[1,0].set_xlim(df_station['Y'].index[0],df_station['Y'].index[-1])
            ax[1,0].set_ylabel('Y/nT', fontsize = 14)           
            ax[1,0].grid()
    
    
            ax[2,0].plot(df_station['Z'][starttime:endtime].resample('M').mean().shift(-15, freq = 'D'), color  = 'black')
            ax[2,0].set_xlim(df_station['Z'].index[0],df_station['Z'].index[-1])
            ax[2,0].set_ylabel('Z/nT', fontsize = 14)
            #ax[2,1].set_xlabel('Years', fontsize = 14 )
            #ax[1].set_ylim(-30,30)
            ax[2,0].grid()
            
            plt.show()      
            
             #plot of SV alone     
                  
            fig, ax = plt.subplots(3,1, figsize = (16,10))
            
            ax[0].set_title(station.upper() + ' Secular Variation (ADMM)', fontsize = 18)
    
            ax[0].plot(df_SV['X'], 'o', color  = 'blue')
            ax[0].set_xlim(df_SV['X'].index[0],df_SV['X'].index[-1])
            ax[0].set_ylim(df_SV['X'].min() - 3, df_SV['X'].max() + 3)
            #ax[0,0].set_xlim(np.datetime64('2011-01'),np.datetime64('2021-12'))
            ax[0].set_ylabel('dX/dT(nT/yr)', fontsize = 12)
            ax[0].grid()
            
            ax[1].plot(df_SV['Y'], 'o', color  = 'green')
            ax[1].set_xlim(df_SV['Y'].index[0],df_SV['Y'].index[-1])
            ax[1].set_ylim(df_SV['Y'].min() - 3, df_SV['Y'].max() + 3)
            ax[1].set_ylabel('dY/dT(nT/yr)', fontsize = 12)
            ax[1].grid()
            
            ax[2].plot(df_SV['Z'], 'o', color  =  'black')
            ax[2].set_xlim(df_SV['Z'].index[0],df_SV['Z'].index[-1])
            ax[2].set_ylim(df_SV['Z'].min() - 3, df_SV['Z'].max() + 3)
            #ax[0].set_xlim(np.datetime64('2010-01'),np.datetime64('2021-06'))
            ax[2].set_ylabel('dZ/dT(nT/yr)', fontsize = 12)
            ax[2].grid()
            
            plt.show()
            
            
            if input_chaos == 'y' and plot_chaos == True:
                
                #plotting real SV and corrected SV comparison
                
                fig, ax = plt.subplots(3,1, figsize = (16,10))
                
                ax[0].set_title(station.upper() + ' Secular Variation (ADMM)', fontsize = 18)
                ax[0].plot(df_SV_not_corrected['X'], 'o', color  = 'red', label = 'real data')
                ax[0].plot(df_SV['X'], 'o', color  = 'blue', label = 'CHAOS correction')
                ax[0].set_xlim(df_SV['X'].index[0],df_SV['X'].index[-1])
                ax[0].set_ylim(df_SV_not_corrected['X'].min() - 5, df_SV_not_corrected['X'].max() + 5)
                #ax[0,0].set_xlim(np.datetime64('2011-01'),np.datetime64('2021-12'))
                ax[0].set_ylabel('dX/dT(nT/yr)', fontsize = 12)
                ax[0].legend()
                ax[0].grid()
                
                
                ax[1].plot(df_SV_not_corrected['Y'], 'o', color  = 'red', label = 'real data')
                ax[1].plot(df_SV['Y'], 'o', color  = 'green', label = 'CHAOS correction')
                ax[1].set_xlim(df_SV['Y'].index[0],df_SV['Y'].index[-1])
                ax[1].set_ylim(df_SV_not_corrected['Y'].min() - 5, df_SV_not_corrected['Y'].max() + 5)
                ax[1].set_ylabel('dY/dT(nT/yr)', fontsize = 12)
                ax[1].legend()
                ax[1].grid()
                
                
                ax[2].plot(df_SV_not_corrected['Z'], 'o', color  = 'red', label = 'real data')
                ax[2].plot(df_SV['Z'], 'o', color  =  'black', label = 'CHAOS correction')
                ax[2].set_xlim(df_SV['Z'].index[0],df_SV['Z'].index[-1])
                ax[2].set_ylim(df_SV_not_corrected['Z'].min() - 5, df_SV_not_corrected['Z'].max() + 5)
                #ax[0].set_xlim(np.datetime64('2010-01'),np.datetime64('2021-06'))
                ax[2].set_ylabel('dZ/dT(nT/yr)', fontsize = 12)
                ax[2].legend()
                ax[2].grid()
                
                plt.show()
                
                #plotting chaos predicted and corrected SV
                
                fig, ax = plt.subplots(3,1, figsize = (16,10))
                
                ax[0].set_title(station.upper() + ' Secular Variation (ADMM)', fontsize = 18)
                ax[0].plot(df_chaos_SV['X_int'], 'o', color  = 'red', label = 'Chaos prediction SV')
                ax[0].plot(df_SV['X'], 'o', color  = 'blue')
                ax[0].set_xlim(df_SV['X'].index[0],df_SV['X'].index[-1])
                ax[0].set_ylim(df_SV['X'].min() - 5, df_SV['X'].max() + 5)
                ax[0].set_ylabel('dX/dT(nT/yr)', fontsize = 12)
                ax[0].legend()
                ax[0].grid()
                
                
                ax[1].plot(df_chaos_SV['Y_int'], 'o', color  = 'red', label = 'Chaos prediction SV')
                ax[1].plot(df_SV['Y'], 'o', color  = 'green', label = 'CHAOS correction')
                ax[1].set_xlim(df_SV['Y'].index[0],df_SV['Y'].index[-1])
                ax[1].set_ylim(df_SV['Y'].min() - 5, df_SV['Y'].max() + 5)
                ax[1].set_ylabel('dY/dT(nT/yr)', fontsize = 12)
                ax[1].legend()
                ax[1].grid()
                
                
                ax[2].plot(df_chaos_SV['Z_int'], 'o', color  = 'red', label = 'Chaos prediction SV')
                ax[2].plot(df_SV['Z'], 'o', color  =  'black')
                ax[2].set_xlim(df_SV['Z'].index[0],df_SV['Z'].index[-1])
                ax[2].set_ylim(df_SV['Z'].min() - 5, df_SV['Z'].max() + 5)
                #ax[0].set_xlim(np.datetime64('2010-01'),np.datetime64('2021-06'))
                ax[2].set_ylabel('dZ/dT(nT/yr)', fontsize = 12)
                ax[2].legend()
                ax[2].grid()
                
                plt.show()
         
            break

                   
        else:
            print('You must type y or n!')
    
    while True:
        condition = input("Do You Want To adopt piecewise linear segments for the SV? [y/n]: ")
        pass
        if condition == 'y':
            try:  
                linear_segments = input('Type the number of linear segments that best fit the SV: ')
                list_ls = [int(k) for k in linear_segments.split(" ")]
                
                        #create an option to save or not a plot
                dpt.jerk_detection(station, df_station, list_ls, starttime, endtime)
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

            df_station =  load_INTERMAGNET_files(station, starttime, endtime)
            if hampel_filter == True:
                df_station = dpt.hampel_filter_denoising(input_series = df_station, window_size = 12, n_sigmas=3)
            else:
                pass
            if external_reduce == None:
                pass
            
            if external_reduce == 'QD':
    
                df_station = dpt.keep_Q_Days(df_station, starttime, endtime)
                print('Only Quiet Days remains')
                
            if external_reduce == 'DD':
            
                df_station = dpt.remove_Disturbed_Days(df_station, start = df_station.resample('D').mean().index[0].date(),
                                                   end = df_station.resample('D').mean().index[-1].date())
                print('Disturbed Days removed')
            
            if external_reduce =='NT':
            
                df_station = dpt.NT_LT(station, df_station, starttime, endtime)
                print('Night-time selected')
                
            if external_reduce == 'C':
                df_station, df_chaos = dpt.external_field_correction_chaos_model(station = station,
                                                                   starttime = starttime,
                                                                   endtime = endtime,
                                                                   df_station = df_station,
                                                                   df_chaos = None)
                
                df_SV = dpt.calculate_SV(df_station, starttime = starttime, endtime = endtime)
                df_SV_chaos = dpt.calculate_SV(df_chaos, starttime = starttime, endtime = endtime, columns = ['X_int','Y_int','Z_int'])
                
                RMS = dpt.rms(df_SV_chaos, df_SV)
                    
            
            df_SV = dpt.calculate_SV(df_station, starttime = starttime, endtime = endtime)
            
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
            
        df_station =  load_INTERMAGNET_files(station, starttime, endtime)
        
        if hampel_filter == True:
            df_station = dpt.hampel_filter_denoising(input_series = df_station, window_size = 12, n_sigmas=3)
        else:
            pass
        if external_reduce == 'QD':

            df_station = dpt.keep_Q_Days(df_station, starttime, endtime)
            print('Only Quiet Days remains')
            
        if external_reduce == 'DD':
        
            df_station = dpt.remove_Disturbed_Days(df_station, start = df_station.resample('D').mean().index[0].date(),
                                               end = df_station.resample('D').mean().index[-1].date())
            print('Disturbed Days removed')
        
        if external_reduce =='NT':
        
            df_station = dpt.NT_LT(station, df_station, starttime, endtime)
            print('Night-time selected')
            
        if external_reduce == 'C':
            df_station, df_chaos = dpt.external_field_correction_chaos_model(station = station,
                                                                   starttime = starttime,
                                                                   endtime = endtime,
                                                                   df_station = df_station,
                                                                   df_chaos = None)
            
            df_SV = dpt.calculate_SV(df_station, starttime = starttime, endtime = endtime)
            df_SV_chaos = dpt.calculate_SV(df_chaos, starttime = starttime, endtime = endtime, columns = ['X_int','Y_int','Z_int'])
            
            RMS = dpt.rms(df_SV_chaos, df_SV)
          
        
        df_SV = dpt.calculate_SV(df_station, starttime = starttime, endtime = endtime)
        

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
        
def validate(str_date):
    try:
        datetime.strptime(str_date, '%Y-%m-%d')
    except ValueError:
        raise ValueError('Incorrect ' + str_date + ' format, should be YYYY-MM-DD')
                
def read_txt_SV(station, starttime, endtime):
    path = 'SV_update/'+ station.upper() +'_data/SV_' + station.upper() + '.txt'

    df_SV = pd.read_csv(path,sep = '\s+', index_col = [0])
    df_SV.index = pd.to_datetime(df_SV.index,infer_datetime_format=True)
    df_SV = df_SV.loc[starttime:endtime]
    
    return df_SV

def plot_samples(station, dataframe, save_plots:bool = False, plot_data_type = None):
    '''
    '''
    if save_plots == False and plot_data_type == None:
    
        samples = ['H','D','M','Y']
        colors = ['blue','green','black']
        
        
        for sample, title in zip(samples, ['hourly','daily','monthly','annual']):
            
            df_station = dpt.resample_obs_data(dataframe = dataframe, sample = sample)
            fig, axes = plt.subplots(3,1,figsize = (16,10))
            plt.suptitle(station.upper() + ' ' + title + ' mean', fontsize = 18, y = 0.92)
            plt.xlabel('Date(Years)', fontsize = 12)
                          
            for col, ax, color in zip(df_station.columns, axes.flatten(), colors):
            
                ax.plot(df_station[col],color = color)
                ax.set_ylabel(col.upper() +' (nT)', fontsize = 12)
                ax.set_xlim(df_station[col].index[0],df_station[col].index[-1])
                ax.grid()

            plt.show()
            
    if save_plots == False and plot_data_type != None:

        samples = ['H','D','M','Y']
        colors = ['blue','green','black']
        First_QD_data = plot_data_type
    
        for sample, title in zip(samples, ['hourly','daily','monthly','annual']):
            
            df_station = dpt.resample_obs_data(dataframe = dataframe, sample = sample)
            fig, axes = plt.subplots(3,1,figsize = (16,10))
            plt.suptitle(station.upper() + ' ' + title + ' mean', fontsize = 16, y = 0.92)
            plt.xlabel('Date(Years)', fontsize = 12)
                          
            for col, ax, color in zip(df_station.columns, axes.flatten(), colors):
            
                ax.plot(df_station[col],'o-',color = color)
                ax.set_ylabel(col.upper() +' (nT)', fontsize = 12)
                ax.set_xlim(df_station[col].index[0],df_station[col].index[-1])
                ax.grid()
                ax.plot(df_station.loc[df_station.index > First_QD_data][col],'o-', color = 'red', label = 'Quasi-definitive data')
                ax.legend()
                
            plt.show()
                       
    if save_plots == True and plot_data_type == None:
        directory = 'Filtered_data/'+ station +'_data'
        pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
        samples = ['H','D','M','Y']
        colors = ['blue','green','black']
        
        
        for sample, title in zip(samples, ['hourly','daily','monthly','annual']):
            
            df_station = dpt.resample_obs_data(dataframe = dataframe, sample = sample)
            fig, axes = plt.subplots(3,1,figsize = (16,10))
            plt.suptitle(station.upper() + ' ' + title + ' mean', fontsize = 18, y = 0.92)
            plt.xlabel('Date(Years)', fontsize = 12)
                          
            for col, ax, color in zip(df_station.columns, axes.flatten(), colors):
            
                ax.plot(df_station[col],color = color)
                ax.set_ylabel(col.upper() +' (nT)', fontsize = 12)
                ax.set_xlim(df_station[col].index[0],df_station[col].index[-1])
                ax.grid()
                
            plt.savefig(directory + '/' + station + '_' + title + '_mean.jpeg', bbox_inches='tight')
            plt.show()
    if save_plots == True and plot_data_type != None:
        
        First_QD_data = plot_data_type
        directory = 'Filtered_data/'+ station +'_data'
        pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
        samples = ['H','D','M','Y']
        colors = ['blue','green','black']
        
        
        for sample, title in zip(samples, ['hourly','daily','monthly','annual']):
            
            df_station = dpt.resample_obs_data(dataframe = dataframe, sample = sample)
            fig, axes = plt.subplots(3,1,figsize = (16,10))
            plt.suptitle(station.upper() + ' ' + title + ' mean', fontsize = 16, y = 0.92)
            plt.xlabel('Date(Years)', fontsize = 14)
                          
            for col, ax, color in zip(df_station.columns, axes.flatten(), colors):
            
                ax.plot(df_station[col],color = color)
                ax.set_ylabel(col.upper() +' (nT)', fontsize = 12)
                ax.set_xlim(df_station[col].index[0],df_station[col].index[-1])
                ax.plot(df_station.loc[df_station.index > First_QD_data][col], color = 'red', label = 'Quasi-definitive data')
                ax.legend()
                ax.grid()
                
            plt.savefig(directory + '/' + station + '_' + title + '_mean.jpeg', bbox_inches='tight')
            plt.show()
            
    