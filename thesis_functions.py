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
from Thesis_Marcos import support_functions as spf
from sklearn.preprocessing import PolynomialFeatures

 

def load_INTERMAGNET_files(station, starttime, endtime):
    '''
    
    Function to read and concat observatory data.
    Works with every INTERMAGNET Observatory.
    ----------------------------------------------------------
    Data types:
    
    Quasi-definitive and definitive data.
    ----------------------------------------------------------
    Inputs:
    
    station - 3 letters IAGA code
    
    starttime - first day of the data (format = 'yyyy-mm-dd)
    
    endtime - last day of the data (format = 'yyyy-mm-dd)
    
    ----------------------------------------------------------
    Example of use:
    
    load_INTERMAGNET_files(station = 'VSS', starttime = '2000-01-25', endtime = '2021-12-31')
    
    ------------------------------------------------------------------------------------------
    
    Return a pandas DataFrame of all readed data with X, Y and Z components.
    
    '''
    assert len(station) == 3, 'station must be a IAGA code with 3 letters'
    
    for i in [starttime,endtime]:
        spf.validate(i)
    
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
        
    #reading and concatenating the files

    df_station = pd.DataFrame()
    df_station = pd.concat( (pd.read_csv(file, sep='\s+',usecols = [0,1,3,4,5], 
                   header = None,skiprows = skiprows, 
                   parse_dates = {'Date': ['date', 'Time']},
                   names = ['date','Time','X','Y','Z']) for file in files_station), 
                   ignore_index = True)
    
    df_station['Date'] = pd.to_datetime(df_station['Date'], format = '%Y-%m-%dd %H:%M:%S.%f')     
              
    df_station.set_index('Date', inplace = True)

    # replacing 99999.0 values
    
    df_station.loc[df_station['X'] >= 99999.0, 'X'] = np.nan
    df_station.loc[df_station['Y'] >= 99999.0, 'Y'] = np.nan
    df_station.loc[df_station['Z'] >= 99999.0, 'Z'] = np.nan

    df_station = df_station.sort_index().loc[starttime:endtime]
    
    try:
        df_station.pop('date')
        df_station.pop('Time')
    except:
        pass
    
    return df_station

def SV_obs(station,
           starttime,
           endtime,
           plot_chaos: bool = False,
           convert_HDZ_to_XYZ:bool = False):
    '''
    Interactive function for INTERMAGNET observatories secular variation data processing
    
    --------------------------------------------
    
    data processing option:
    
    *Denoising filter
    
    *External field reduction - Disturned Days, Quiet days or night time selection
    
    *CHAOS-7 model correction
    
    *Jerk detection based on linear segments
    
    -------------------------------------------
    
    Option to save plot and text files of minute, hourly, daily, monthly, annual means and secular variation.
    
    --------------------------------------------
    
    inputs:
        
    station - 3 letters IAGA code
    
    starttime - first day of the data (format = 'yyyy-mm-dd)
    
    endtime - last day of the data (format = 'yyyy-mm-dd)
    
    plot_chaos - boolean (True or False). If the CHAOS model prediction was computed, Will be plotted the comparisons.
    
    convert_HDZ_to_XYZ - boolean (True or False). 
    
    -----------------------------------------------
    
    Return a pandas dataframe (processed)
    
    '''
    #reading the files
    
    df_station = load_INTERMAGNET_files(station,
                                          starttime,
                                          endtime)
    

    #detecting different data types
    
    First_QD_data = spf.data_type(station = station,
                              starttime = starttime,
                              endtime = endtime)
    
    
    
    # HDZ to XYZ conversion
    
    if convert_HDZ_to_XYZ == True:
        df_station =  utt.HDZ_to_XYZ_conversion(station = station,
                                  dataframe = df_station,
                                  starttime = str(df_station.index[0].date()),
                                  endtime = str(df_station.index[-1].date()))
    else:
        pass
    
    df_station2 = df_station.copy()
    
    # Hampel filter interaction
    
    while True: 
        inp5 = input("Do You Want To denoise the data based on median absolute deviation? [y/n]: ")
        if inp5 == 'y':
            print('Denoising data...')
            df_station = dpt.hampel_filter_denoising(dataframe = df_station,
                                                     window_size = 100,
                                                     n_sigmas=3,
                                                     plot_figure = True)
            break
        if inp5 == 'n':
            print('No data changed.')
            break
        else:
            print('You must type y or n, try again!')
        
    
    
    # external field reduction interaction
    
    options = ['Q','D','NT','E']
    while True: 
        inp = str(input("Press Q to use only Quiet Days, D to remove Disturbed Days, NT to use only the night-time or E to Exit without actions [Q/D/NT/E]: "))
        
        if all([inp != option for option in options]):
            print('You must type Q, D, NT or E')
        else:
            break
    
    if inp == 'Q':
        
        df_station = dpt.keep_Q_Days(df_station,
                                     starttime,
                                     endtime)
        
    if inp == 'D':
        
        df_station = dpt.remove_Disturbed_Days(df_station,
                                               starttime,
                                               endtime)
        
    if inp == 'NT':
        
        df_station = dpt.night_time_selection(station,
                                              df_station,
                                              starttime,
                                              endtime)
        
    if inp == 'E':
        print('No action')
        
    # condition for data resampling - not in use
    resample_condition = []
    if inp not in ['Q','D','NT']:
        resample_condition = True
    else:
        resample_condition = False    
        
    #CHAOS model correction interaction
    
    while True:
    
        input_chaos = input("Do You want to correct the external field using the CHAOS model? [y/n]: ")
        if input_chaos == 'y':
            
            df_station, df_chaos = dpt.external_field_correction_chaos_model(station = station,
                                                           starttime = starttime,
                                                           endtime = endtime,
                                                           df_station = df_station,
                                                           df_chaos = None, apply_percentage = False)
            break
        if input_chaos == 'n':
            print('Correction using CHAOS was not applied.')
            break
        else:
            print('You must type y or n, try again!')
        

    
    directory = 'Filtered_data/'+ station +'_data'
    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)  
    

    df_SV = dpt.calculate_SV(df_station,
                             starttime = starttime,
                             endtime = endtime,
                             apply_percentage = False)
    
    df_SV_not_corrected = dpt.calculate_SV(df_station2,
                                           starttime = starttime,
                                           endtime = endtime,
                                           apply_percentage = False)
    
    if input_chaos == 'y':
        
        df_chaos_SV = dpt.calculate_SV(df_chaos,
                                       starttime = starttime,
                                       endtime = endtime,
                                       columns = ['X_int','Y_int','Z_int'],
                                       apply_percentage = False)   
    else:
        
        pass    
        
    
    #option to save txt and plot files
    
    while True: 
        inp2 = input("Do You Want To Save a File With the Variation? [y/n]: ")
        if inp2 == 'y':

            print('Saving files...')

            for sample in ['Min','H','D', 'M', 'Y']:
                          
                if sample == 'Min':
                
                    file = df_station[starttime:endtime].resample(sample).mean().round(3).replace(np.NaN,99999.0)
                    
                    file.to_csv(directory + '/' + station.upper() + '_minute_mean_preliminary.zip', sep ='\t', index=True)
            
                if sample == 'H':
                    
                    file = dpt.resample_obs_data(df_station, 'H',apply_percentage = False).round(3).replace(np.NaN,99999.0)
                    file.to_csv(directory + '/' + station.upper() + '_hourly_mean_preliminary.txt', sep ='\t', index=True)
                
                    spf.Header_SV_obs_files(station = station,
                                        filename = 'hourly_mean',
                                        data_denoise = inp5,
                                        external_correction = inp,
                                        chaos_model = input_chaos)               
                if sample == 'D':
                    
                    file = dpt.resample_obs_data(df_station, 'D',apply_percentage = False).round(3).replace(np.NaN,99999.0)
                    file.to_csv(directory + '/' + station.upper() + '_daily_mean_preliminary.txt', sep ='\t', index=True)
                    
                    spf.Header_SV_obs_files(station = station,
                                        filename = 'daily_mean',
                                        data_denoise = inp5,
                                        external_correction = inp,
                                        chaos_model = input_chaos) 
                if sample == 'M':
                    
                    file = dpt.resample_obs_data(df_station, 'M', apply_percentage = False).round(3).replace(np.NaN,99999.0)
                    
                    file_SV = df_SV.replace(np.NaN,99999.0)
                    
                    file.to_csv(directory + '/' + station.upper() +'_monthly_mean_preliminary.txt', sep ='\t', index=True)
                    
                    spf.Header_SV_obs_files(station = station,
                                        filename = 'monthly_mean',
                                        data_denoise = inp5,
                                        external_correction = inp,
                                        chaos_model = input_chaos) 
                    
                    file_SV.to_csv(directory + '/' + station.upper() +'_secular_variation_preliminary.txt', sep ='\t', index=True)
                    
                    spf.Header_SV_obs_files(station = station,
                                        filename = 'secular_variation',
                                        data_denoise = inp5,
                                        external_correction = inp,
                                        chaos_model = input_chaos) 
                if sample == 'Y':
                    
                    file = dpt.resample_obs_data(df_station, 'Y', apply_percentage = resample_condition).round(3).replace(np.NaN,99999.0)
                    
                    file.to_csv(directory + '/' + station.upper() + '_annual_mean_preliminary.txt', sep ='\t', index=True)
                    
                    spf.Header_SV_obs_files(station = station,
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
                ax[2].set_ylabel('dZ/dT(nT/yr)', fontsize = 12)
                ax[2].grid()
                
                if First_QD_data != []:
                    ax[0].plot(df_station.loc[df_station.index > First_QD_data]['X'],'o-', color = 'red',label = 'Quasi-definitive data')
                    ax[1].plot(df_station.loc[df_station.index > First_QD_data]['Y'],'o-', color = 'red',label = 'Quasi-definitive data')
                    ax[2].plot(df_station.loc[df_station.index > First_QD_data]['Z'],'o-', color = 'red',label = 'Quasi-definitive data')
                    ax[0].legend()
                    ax[1].legend()
                    ax[2].legend()
                
                plt.savefig(directory + '/' + station + '_minute_mean.jpeg', bbox_inches='tight')
                plt.show()
                    
            
            if First_QD_data != []:
            
                plot_samples(station = station,
                             dataframe = df_station,
                             save_plots = True,
                             plot_data_type = First_QD_data,
                             apply_percentage = False)
            else:
                plot_samples(station = station,
                             dataframe = df_station,
                             save_plots = True,
                             plot_data_type = None,
                             apply_percentage = False)
            
            #plot of secular variation and monthly mean
            
            #calculating dataframe with minthly mean
            df_monthly_mean = dpt.resample_obs_data(df_station,
                                               sample = 'M',
                                               apply_percentage = False)
            
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
            ax[0,0].plot(df_monthly_mean['X'][starttime:endtime], color  = 'blue')
            ax[0,0].set_xlim(df_station['X'].index[0],df_station['X'].index[-1])
            ax[0,0].set_ylabel('X/nT', fontsize = 14)   
            ax[0,0].grid()
            
            ax[1,0].plot(df_monthly_mean['Y'][starttime:endtime], color  = 'green')
            ax[1,0].set_xlim(df_station['Y'].index[0],df_station['Y'].index[-1])
            ax[1,0].set_ylabel('Y/nT', fontsize = 14)           
            ax[1,0].grid()
    
    
            ax[2,0].plot(df_monthly_mean['Z'][starttime:endtime], color  = 'black')
            ax[2,0].set_xlim(df_station['Z'].index[0],df_station['Z'].index[-1])
            ax[2,0].set_ylabel('Z/nT', fontsize = 14)
            ax[2,0].grid()
            
            plt.savefig(directory + '/' + station + '_Var_SV.jpeg', bbox_inches='tight')
            plt.show()      
            
             #plot of SV alone     
                  
            fig, ax = plt.subplots(3,1, figsize = (16,10))
            
            ax[0].set_title(station.upper() + ' Secular Variation (ADMM)', fontsize = 18)
    
            ax[0].plot(df_SV['X'], 'o', color  = 'blue')
            ax[0].set_xlim(df_SV['X'].index[0],df_SV['X'].index[-1])
            ax[0].set_ylim(df_SV['X'].min() - 3, df_SV['X'].max() + 3)
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
            ax[2].set_ylabel('dZ/dT(nT/yr)', fontsize = 12)
            ax[2].grid()
            
            plt.savefig(directory + '/' + station + '_SV.jpeg', bbox_inches='tight')
            plt.show()
            

            if input_chaos == 'y' and plot_chaos == True:
                
                #plotting real SV and corrected SV comparison
                
                fig, ax = plt.subplots(3,1, figsize = (16,10))
                
                ax[0].set_title(station.upper() + ' Secular Variation (ADMM) - Real data SV x corrected data SV (CHAOS)', fontsize = 18)
                ax[0].plot(df_SV_not_corrected['X'], 'o', color  = 'red', label = 'Real data SV')
                ax[0].plot(df_SV['X'], 'o', color  = 'blue', label = 'Corrected SV')
                ax[0].set_xlim(df_SV['X'].index[0],df_SV['X'].index[-1])
                ax[0].set_ylim(df_SV_not_corrected['X'].min() - 5, df_SV_not_corrected['X'].max() + 5)
                ax[0].set_ylabel('dX/dT(nT/yr)', fontsize = 12)
                ax[0].legend()
                ax[0].grid()
                
                
                ax[1].plot(df_SV_not_corrected['Y'], 'o', color  = 'red', label = 'Real data SV')
                ax[1].plot(df_SV['Y'], 'o', color  = 'green', label = 'Corrected SV')
                ax[1].set_xlim(df_SV['Y'].index[0],df_SV['Y'].index[-1])
                ax[1].set_ylim(df_SV_not_corrected['Y'].min() - 5, df_SV_not_corrected['Y'].max() + 5)
                ax[1].set_ylabel('dY/dT(nT/yr)', fontsize = 12)
                ax[1].legend()
                ax[1].grid()
                
                
                ax[2].plot(df_SV_not_corrected['Z'], 'o', color  = 'red', label = 'Real data SV')
                ax[2].plot(df_SV['Z'], 'o', color  =  'black', label = 'Corrected SV')
                ax[2].set_xlim(df_SV['Z'].index[0],df_SV['Z'].index[-1])
                ax[2].set_ylim(df_SV_not_corrected['Z'].min() - 5, df_SV_not_corrected['Z'].max() + 5)
                ax[2].set_ylabel('dZ/dT(nT/yr)', fontsize = 12)
                ax[2].legend()
                ax[2].grid()
                
                plt.savefig(directory + '/' + station + '_SV_correction_comparison.jpeg', bbox_inches='tight')
                plt.show()
                
                #plotting chaos predicted and corrected SV
                
                fig, ax = plt.subplots(3,1, figsize = (16,10))
                
                ax[0].set_title(station.upper() + ' Secular Variation (ADMM) - Corrected SV x CHAOS predicted SV (Internal field)', fontsize = 18)
                ax[0].plot(df_chaos_SV['X_int'], 'o', color  = 'red', label = 'Chaos prediction SV')
                ax[0].plot(df_SV['X'], 'o', color  = 'blue', label = 'Corrected SV')
                ax[0].set_xlim(df_SV['X'].index[0],df_SV['X'].index[-1])
                ax[0].set_ylim(df_SV['X'].min() - 5, df_SV['X'].max() + 5)
                ax[0].set_ylabel('dX/dT(nT/yr)', fontsize = 12)
                ax[0].legend()
                ax[0].grid()
                
                
                ax[1].plot(df_chaos_SV['Y_int'], 'o', color  = 'red', label = 'Chaos prediction SV')
                ax[1].plot(df_SV['Y'], 'o', color  = 'green', label = 'Corrected SV')
                ax[1].set_xlim(df_SV['Y'].index[0],df_SV['Y'].index[-1])
                ax[1].set_ylim(df_SV['Y'].min() - 5, df_SV['Y'].max() + 5)
                ax[1].set_ylabel('dY/dT(nT/yr)', fontsize = 12)
                ax[1].legend()
                ax[1].grid()
                
                
                ax[2].plot(df_chaos_SV['Z_int'], 'o', color  = 'red', label = 'Chaos prediction SV')
                ax[2].plot(df_SV['Z'], 'o', color  =  'black', label = 'Corrected SV')
                ax[2].set_xlim(df_SV['Z'].index[0],df_SV['Z'].index[-1])
                ax[2].set_ylim(df_SV['Z'].min() - 5, df_SV['Z'].max() + 5)
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
                ax[2].set_ylabel('Z(nT)', fontsize = 12)
                ax[2].grid()
                
                if First_QD_data != []:
                    ax[0].plot(df_station2.loc[df_station2.index > First_QD_data]['X'],'-', color = 'red',label = 'Quasi-definitive data')
                    ax[1].plot(df_station2.loc[df_station2.index > First_QD_data]['Y'],'-', color = 'red',label = 'Quasi-definitive data')
                    ax[2].plot(df_station2.loc[df_station2.index > First_QD_data]['Z'],'-', color = 'red',label = 'Quasi-definitive data')
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
                    ax[0].plot(df_station.loc[df_station.index > First_QD_data]['X'],'o-', color = 'red',label = 'Quasi-definitive data')
                    ax[1].plot(df_station.loc[df_station.index > First_QD_data]['Y'],'o-', color = 'red',label = 'Quasi-definitive data')
                    ax[2].plot(df_station.loc[df_station.index > First_QD_data]['Z'],'o-', color = 'red',label = 'Quasi-definitive data')
                    ax[0].legend()
                    ax[1].legend()
                    ax[2].legend()
                
                plt.show()
                
            if First_QD_data != []:
            
                plot_samples(station = station,
                             dataframe = df_station,
                             save_plots = False,
                             plot_data_type = First_QD_data,
                             apply_percentage = resample_condition)
            else:
                plot_samples(station = station,
                             dataframe = df_station,
                             save_plots = False,
                             plot_data_type = None, 
                             apply_percentage = resample_condition)
            
            #plot of secular variation and monthly mean
            
            #calculating monthly mean dataframe
            
            df_monthly_mean = dpt.resample_obs_data(df_station,
                                               sample = 'M',
                                               apply_percentage = resample_condition)
            
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
            ax[0,0].plot(df_monthly_mean['X'][starttime:endtime], color  = 'blue')
            ax[0,0].set_xlim(df_station['X'].index[0],df_station['X'].index[-1])
            ax[0,0].set_ylabel('X/nT', fontsize = 14)   
            ax[0,0].grid()
            
            ax[1,0].plot(df_monthly_mean['Y'][starttime:endtime], color  = 'green')
            ax[1,0].set_xlim(df_station['Y'].index[0],df_station['Y'].index[-1])
            ax[1,0].set_ylabel('Y/nT', fontsize = 14)           
            ax[1,0].grid()
    
    
            ax[2,0].plot(df_monthly_mean['Z'][starttime:endtime], color  = 'black')
            ax[2,0].set_xlim(df_station['Z'].index[0],df_station['Z'].index[-1])
            ax[2,0].set_ylabel('Z/nT', fontsize = 14)
            ax[2,0].grid()
            
            plt.show()      
            
             #plot of SV alone     
                  
            fig, ax = plt.subplots(3,1, figsize = (16,10))
            
            ax[0].set_title(station.upper() + ' Secular Variation (ADMM)', fontsize = 18)
    
            ax[0].plot(df_SV['X'], 'o', color  = 'blue')
            ax[0].set_xlim(df_SV['X'].index[0],df_SV['X'].index[-1])
            ax[0].set_ylim(df_SV['X'].min() - 3, df_SV['X'].max() + 3)
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
            ax[2].set_ylabel('dZ/dT(nT/yr)', fontsize = 12)
            ax[2].grid()
            
            plt.show()
            
            
            if input_chaos == 'y' and plot_chaos == True:
                
                #plotting real SV and corrected SV comparison
                
                fig, ax = plt.subplots(3,1, figsize = (16,10))
                
                ax[0].set_title(station.upper() + ' Secular Variation (ADMM) - Real data SV x corrected data SV (CHAOS)', fontsize = 18)
                ax[0].plot(df_SV_not_corrected['X'], 'o', color  = 'red', label = 'Real data SV')
                ax[0].plot(df_SV['X'], 'o', color  = 'blue', label = 'Corrected SV')
                ax[0].set_xlim(df_SV['X'].index[0],df_SV['X'].index[-1])
                ax[0].set_ylim(df_SV_not_corrected['X'].min() - 5, df_SV_not_corrected['X'].max() + 5)
                ax[0].set_ylabel('dX/dT(nT/yr)', fontsize = 12)
                ax[0].legend()
                ax[0].grid()
                
                
                ax[1].plot(df_SV_not_corrected['Y'], 'o', color  = 'red', label = 'Real data SV')
                ax[1].plot(df_SV['Y'], 'o', color  = 'green', label = 'Corrected SV')
                ax[1].set_xlim(df_SV['Y'].index[0],df_SV['Y'].index[-1])
                ax[1].set_ylim(df_SV_not_corrected['Y'].min() - 5, df_SV_not_corrected['Y'].max() + 5)
                ax[1].set_ylabel('dY/dT(nT/yr)', fontsize = 12)
                ax[1].legend()
                ax[1].grid()
                
                
                ax[2].plot(df_SV_not_corrected['Z'], 'o', color  = 'red', label = 'Real data SV')
                ax[2].plot(df_SV['Z'], 'o', color  =  'black', label = 'Corrected SV')
                ax[2].set_xlim(df_SV['Z'].index[0],df_SV['Z'].index[-1])
                ax[2].set_ylim(df_SV_not_corrected['Z'].min() - 5, df_SV_not_corrected['Z'].max() + 5)
                ax[2].set_ylabel('dZ/dT(nT/yr)', fontsize = 12)
                ax[2].legend()
                ax[2].grid()
                
                plt.show()
                
                #plotting chaos predicted and corrected SV
                
                fig, ax = plt.subplots(3,1, figsize = (16,10))
                
                ax[0].set_title(station.upper() + ' Secular Variation (ADMM) - Corrected SV x CHAOS predicted SV (Internal field)', fontsize = 18)
                ax[0].plot(df_chaos_SV['X_int'], 'o', color  = 'red', label = 'Chaos prediction SV')
                ax[0].plot(df_SV['X'], 'o', color  = 'blue', label = 'Corrected SV')
                ax[0].set_xlim(df_SV['X'].index[0],df_SV['X'].index[-1])
                ax[0].set_ylim(df_SV['X'].min() - 5, df_SV['X'].max() + 5)
                ax[0].set_ylabel('dX/dT(nT/yr)', fontsize = 12)
                ax[0].legend()
                ax[0].grid()
                
                
                ax[1].plot(df_chaos_SV['Y_int'], 'o', color  = 'red', label = 'Chaos prediction SV')
                ax[1].plot(df_SV['Y'], 'o', color  = 'green', label = 'Corrected SV')
                ax[1].set_xlim(df_SV['Y'].index[0],df_SV['Y'].index[-1])
                ax[1].set_ylim(df_SV['Y'].min() - 5, df_SV['Y'].max() + 5)
                ax[1].set_ylabel('dY/dT(nT/yr)', fontsize = 12)
                ax[1].legend()
                ax[1].grid()
                
                
                ax[2].plot(df_chaos_SV['Z_int'], 'o', color  = 'red', label = 'Chaos prediction SV')
                ax[2].plot(df_SV['Z'], 'o', color  =  'black', label = 'Corrected SV')
                ax[2].set_xlim(df_SV['Z'].index[0],df_SV['Z'].index[-1])
                ax[2].set_ylim(df_SV['Z'].min() - 5, df_SV['Z'].max() + 5)
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
        
def SV_(stations, 
        starttime,
        endtime,
        jerk_start_window = None,
        jerk_end_window = None,
        external_reduction = None,
        CHAOS_correction:bool = False,
        file = 'off',
        hampel_filter: bool = False,
        plot_chaos: bool = False,
        convert_HDZ_to_XYZ:bool = False
       ):
    
    '''
    
    ---------------------------------------------------------------------
    Inputs:
    
    stations - must be a list of observatories IAGA code or None
    
    starttime - must be a date, 'yyyy-mm-dd' format
    
    endtime - must be a date, 'yyyy-mm-dd' format
    
    external_reduction - must be 'QD', 'DD', 'NT', 'C' or None 
               *QD for keep quiet days
               *DD for remove disturbed days
               *NT for Night time selection
               *C use CHAOS Model predicted external field
    
    CHAOS_correction - boolean, option to correct the geomagnetic data with the CHAOS-model.
    
    file - must be 'save', 'update' or 'off'.
          *save - will save in a specific directory a text file of the SV for the selected period
          *off - No file will be saved.
          *update - If there are new data, Will update the current file.
    
    hampel_filter - boolean, True or False
    
    Plot_chaos - boolean (True or False). Option to plot the CHAOS-model SV prediction
    
    convert_HDZ_to_XYZ - boolean (True or False). If true, will identify the existence of H, D and Z
    component in the dataset and convert to X, Y and Z.
    
    -------------------------------------------------------------------------------------
    Use example:
    
    SV_(stations = ['VSS','NGK','TTB'],
        starttime = '2000-01-01',
        endtime = '2021-06-30',
        jerk_start_window= None,
        jerk_end_window= None,
        file = 'save',
        external_reduction = None,
        CHAOS_correction = True,
        hampel_filter = False,
        plot_chaos = True,
        convert_HDZ_to_XYZ=True)
        
    ---------------------------------------------------------------------------------------    
    
    '''
    
    
    df_imos = pd.read_csv('Thesis_Marcos/Data/Imos informations/Imos_INTERMAGNET.txt',
                          skiprows = 1,
                          sep = '\s+',
                          usecols=[0,1,2,3],
                          names = ['Imos','Latitude','Longitude','Elevation'],
                          index_col= ['Imos'])
    
    #external_reduction_options = [None,'QD','DD','NT']
    
    Files = ['save','update','off']
    
    
    if external_reduction not in [None,'QD','DD','NT']:
        print('External field Reduction must be QD, DD or NT. No changes applied.')
    
    if stations == None:
        for station in df_imos.index:
            
            #computing dataframe for the selected period
            try:
                df_station =  load_INTERMAGNET_files(station,
                                                 starttime,
                                                 endtime)
            except:
                print('No files for ' + station.upper() + ' in the selected period')
                continue
            #converting hdz data to xyz
            if convert_HDZ_to_XYZ == True:
                
                df_station =  utt.HDZ_to_XYZ_conversion(station = station,
                                  dataframe = df_station,
                                  starttime = str(df_station.index[0].date()),
                                  endtime = str(df_station.index[-1].date()))
            else:
                pass            
            
            if hampel_filter == True:
                df_station = dpt.hampel_filter_denoising(dataframe = df_station,
                                                         window_size = 100,
                                                         n_sigmas=3,
                                                         plot_figure = False)
            else:
                pass
            if external_reduction == None:
                pass
            
            if external_reduction == 'QD':
    
                df_station = dpt.keep_Q_Days(df_station,
                                             starttime,
                                             endtime)

                
            if external_reduction == 'DD':
            
                df_station = dpt.remove_Disturbed_Days(df_station,
                                                       starttime,
                                                   endtime)

            
            if external_reduction =='NT':
            
                df_station = dpt.night_time_selection(station,
                                                      df_station,
                                                      starttime,
                                                      endtime)

            
            
            df_station_2 = df_station.copy()
            if CHAOS_correction == True:
                df_station, df_chaos = dpt.external_field_correction_chaos_model(station = station,
                                                                   starttime = starttime,
                                                                   endtime = endtime,
                                                                   df_station = df_station,
                                                                   df_chaos = None)
                
                df_SV = dpt.calculate_SV(df_station,
                                         starttime = starttime,
                                         endtime = endtime)
                
                df_SV_chaos = dpt.calculate_SV(df_chaos,
                                               starttime = starttime,
                                               endtime = endtime,
                                               columns = ['X_int','Y_int','Z_int'])
                
                #RMS = dpt.rms(df_SV_chaos, df_SV)
            else:
                    
            
                df_SV = dpt.calculate_SV(df_station,
                                         starttime = starttime,
                                         endtime = endtime)
            
            

            if file == 'save':
                
                directory = 'SV_update/'+ station +'_data/'
                
                pathlib.Path(directory).mkdir(parents=True, exist_ok=True)    
                
                df_SV.replace(np.NaN,99999.0).to_csv(directory + 'SV_' + station + '_preliminary.txt',
                                                     sep ='\t', index=True)
                
                spf.Header_SV_files(station = station,
                                    data_denoise = hampel_filter,
                                    external_correction = external_reduction,
                                    chaos_model = CHAOS_correction)
                    
                
            if file == 'update':
                
                df1 = pd.read_csv('SV_update/' + station + '_data/SV_'+ station + '.txt', sep = '\t', skiprows = 9)
                df1['Date'] = pd.to_datetime(df1['Date'], infer_datetime_format=True)
                df1.set_index('Date', inplace = True)
                df2 = pd.concat([df1,df_SV])
                #df2 = df2.sort_values(by=df2.index)
                df2.replace(np.NaN,99999.0).drop_duplicates().sort_index().to_csv('SV_update/' + station + '_data/SV_'+ station + '_preliminary.txt', sep = '\t')
                
                spf.Header_SV_files(station = station,
                                        data_denoise = hampel_filter,
                                        external_correction = external_reduction,
                                        chaos_model = CHAOS_correction)
                
                
            if file == 'off':
                
                pass
            
            if file not in ['save','update','off']:
                print('File must be None, update or off!')
            
            
            
            fig, ax = plt.subplots(3,1, figsize = (16,10))
            
            if plot_chaos == True and CHAOS_correction == True:
                ax[0].plot(df_SV_chaos['X_int'],'o-',color = 'red') #label = 'Chaos - rms: ' + str(RMS[0]),linewidth = 3, color = 'red')
                ax[1].plot(df_SV_chaos['Y_int'],'o-',color = 'red') #label = 'Chaos - rms: ' + str(RMS[1]),linewidth = 3, color = 'red')
                ax[2].plot(df_SV_chaos['Z_int'],'o-',color = 'red') #label = 'Chaos - rms: ' + str(RMS[2]),linewidth = 3, color = 'red')
                ax[0].set_ylim(df_SV_chaos['X_int'].min() - 10, df_SV_chaos['X_int'].max() + 10)
                ax[1].set_ylim(df_SV_chaos['Y_int'].min() - 10, df_SV_chaos['Y_int'].max() + 10)
                ax[2].set_ylim(df_SV_chaos['Z_int'].min() - 10, df_SV_chaos['Z_int'].max() + 10)
                ax[0].set_xlim(df_SV_chaos['X_int'].index[0],df_SV_chaos['X_int'].index[-1])
                ax[1].set_xlim(df_SV_chaos['Y_int'].index[0],df_SV_chaos['Y_int'].index[-1])
                ax[2].set_xlim(df_SV_chaos['Z_int'].index[0],df_SV_chaos['Z_int'].index[-1])
                ax[0].legend()  
                ax[1].legend()  
                ax[2].legend()  
                
            ax[0].plot(df_SV['X'],'o', color = 'blue')
            ax[0].set_ylim(df_SV['X'].min() - 10, df_SV['X'].max() + 10)
            ax[0].set_title(station.upper() + ' Secular Variation', fontsize = 16)
            ax[0].set_ylabel('dX/dT(nT/yr)', fontsize = 12)          
                       
            ax[1].plot(df_SV['Y'],'o',color = 'green')
            ax[1].set_ylim(df_SV['Y'].min() - 10, df_SV['Y'].max() + 10)
            ax[1].set_ylabel('dY/dT(nT/yr)', fontsize = 12)        
                       
            ax[2].plot(df_SV['Z'],'o',color = 'black')
            ax[2].set_ylim(df_SV['Z'].min() - 10,df_SV['Z'].max() + 10)
            ax[2].set_ylabel('dZ/dT(nT/yr)', fontsize = 12)
            
            ax[0].set_xlim(df_SV['X'].index[0],df_SV['X'].index[-1])
            ax[1].set_xlim(df_SV['Y'].index[0],df_SV['Y'].index[-1])
            ax[2].set_xlim(df_SV['Z'].index[0],df_SV['Z'].index[-1])
            
            plt.show()
    
            directory2 = 'Map_plots/'+ station +'_data'
            pathlib.Path(directory2).mkdir(parents=True, exist_ok=True)  
            
            #plot to use in the interactive map 
            
            
            fig, ax = plt.subplots(3,1, figsize = (8,6.5))
            if plot_chaos == True and CHAOS_correction == True:
                ax[0].plot(df_SV_chaos['X_int'],'-',color = 'red') #label = 'Chaos - rms: ' + str(RMS[0]),linewidth = 3, color  = 'red')
                ax[1].plot(df_SV_chaos['Y_int'],'-',color = 'red') #label = 'Chaos - rms: ' + str(RMS[1]),linewidth = 3, color  = 'red')
                ax[2].plot(df_SV_chaos['Z_int'],'-',color = 'red') #label = 'Chaos - rms: ' + str(RMS[2]),linewidth = 3, color  = 'red')
                ax[0].set_ylim(df_SV_chaos['X_int'].min() - 10, df_SV_chaos['X_int'].max() + 10)
                ax[1].set_ylim(df_SV_chaos['Y_int'].min() - 10, df_SV_chaos['Y_int'].max() + 10)
                ax[2].set_ylim(df_SV_chaos['Z_int'].min() - 10, df_SV_chaos['Z_int'].max() + 10)
                ax[0].set_xlim(df_SV_chaos['X_int'].index[0],df_SV_chaos['X_int'].index[-1])
                ax[1].set_xlim(df_SV_chaos['Y_int'].index[0],df_SV_chaos['Y_int'].index[-1])
                ax[2].set_xlim(df_SV_chaos['Z_int'].index[0],df_SV_chaos['Z_int'].index[-1])
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
            
            ax[0].set_xlim(df_SV['X'].index[0],df_SV['X'].index[-1])
            ax[1].set_xlim(df_SV['Y'].index[0],df_SV['Y'].index[-1])
            ax[2].set_xlim(df_SV['Z'].index[0],df_SV['Z'].index[-1])
            
            plt.tick_params(labelsize=8)
            plt.savefig(directory2 + '/' + station + '_map_SV.jpeg', bbox_inches='tight')
            plt.close(fig)
            
        if jerk_start_window != None and jerk_end_window != None and CHAOS_correction == False:
            
            
            dpt.jerk_detection_window(station = station,
                                      window_start = jerk_start_window, 
                                      window_end = jerk_end_window, 
                                      starttime = starttime, 
                                      endtime = endtime,
                                      df_station = df_station_2,
                                      df_CHAOS = None,
                                      plot_detection = True,
                                      CHAOS_correction = False,
                                      plot_CHAOS_prediction = False)
            
        if jerk_start_window != None and jerk_end_window != None and CHAOS_correction  == True:
            
            dpt.jerk_detection_window(station = station,
                                      window_start = jerk_start_window, 
                                      window_end = jerk_end_window, 
                                      starttime = starttime, 
                                      endtime = endtime,
                                      df_station = df_station_2,
                                      df_CHAOS = df_chaos,
                                      plot_detection = True,
                                      CHAOS_correction = True,
                                      plot_CHAOS_prediction = True)
                    

            
    for station in stations:
            
        df_station =  load_INTERMAGNET_files(station,
                                             starttime,
                                             endtime)
        
        if convert_HDZ_to_XYZ == True:
                
            df_station =  utt.HDZ_to_XYZ_conversion(station = station,
                                  dataframe = df_station,
                                  starttime = str(df_station.index[0].date()),
                                  endtime = str(df_station.index[-1].date()))
        else:
            pass  
        
        if hampel_filter == True:
            
            df_station = dpt.hampel_filter_denoising(dataframe = df_station,
                                                     window_size = 100,
                                                     n_sigmas=3,
                                                     plot_figure = False)
        else:
            pass
        
        if external_reduction == 'QD':

            df_station = dpt.keep_Q_Days(df_station,
                                         starttime,
                                         endtime)
            
        if external_reduction == 'DD':
        
            df_station = dpt.remove_Disturbed_Days(df_station,
                                                   starttime,
                                               endtime)

        
        if external_reduction =='NT':
        
            df_station = dpt.night_time_selection(station,
                                                  df_station,
                                                  starttime,
                                                  endtime)

        
        df_station_2 = df_station.copy()
        if CHAOS_correction == True:
            df_station, df_chaos = dpt.external_field_correction_chaos_model(station = station,
                                                                   starttime = starttime,
                                                                   endtime = endtime,
                                                                   df_station = df_station,
                                                                   df_chaos = None)
            
            df_SV = dpt.calculate_SV(df_station,
                                     starttime = starttime,
                                     endtime = endtime)
            
            df_SV_chaos = dpt.calculate_SV(df_chaos,
                                           starttime = starttime,
                                           endtime = endtime,
                                           columns = ['X_int','Y_int','Z_int'])
            
            #RMS = dpt.rms(df_SV_chaos,
             #             df_SV)
        else:
            
        
            df_SV = dpt.calculate_SV(df_station,
                                     starttime = starttime,
                                     endtime = endtime)
        

        if file == 'save':
    
            directory = 'SV_update/'+ station +'_data/'
            pathlib.Path(directory).mkdir(parents=True, exist_ok=True)     
            df_SV.replace(np.NaN,99999.0).to_csv(directory + 'SV_' + station + '_preliminary.txt', sep ='\t')
            
            spf.Header_SV_files(station = station,
                                        data_denoise = hampel_filter,
                                        external_correction = external_reduction,
                                        chaos_model = CHAOS_correction)
            
        if file == 'update':
    
            df1 = pd.read_csv('SV_update/' + station + '_data/SV_'+ station + '.txt', sep = '\t',skiprows = 9)
            df1['Date'] = pd.to_datetime(df1['Date'], infer_datetime_format=True)
            df1.set_index('Date', inplace = True)
            df2 = pd.concat([df1,df_SV])
            #df2 = df2.sort_values(by=df2.index)
            df2.replace(np.NaN,99999.0).drop_duplicates().sort_index().to_csv('SV_update/' + station + '_data/SV_'+ station + '_preliminary.txt', sep = '\t')
            
            spf.Header_SV_files(station = station,
                                        data_denoise = hampel_filter,
                                        external_correction = external_reduction,
                                        chaos_model = CHAOS_correction)
        if file == 'off':
            pass
        
        if file not in Files:
            print('File must be None, update or off!')
            pass    
        
        directory2 = 'Map_plots/'+ station +'_data'
        pathlib.Path(directory2).mkdir(parents=True, exist_ok=True)  
        
        fig, ax = plt.subplots(3,1, figsize = (8,6.5),sharex = True)
        
        if plot_chaos == True and CHAOS_correction == True:
            
            ax[0].plot(df_SV_chaos['X_int'],'-',color = 'red') #label = 'Chaos - rms: ' + str(RMS[0]),linewidth = 3, color = 'red')
            ax[1].plot(df_SV_chaos['Y_int'],'-',color = 'red') #label = 'Chaos - rms: ' + str(RMS[1]),linewidth = 3, color = 'red')
            ax[2].plot(df_SV_chaos['Z_int'],'-',color = 'red') #label = 'Chaos - rms: ' + str(RMS[2]),linewidth = 3, color = 'red')
            ax[0].set_ylim(df_SV_chaos['X_int'].min() - 10, df_SV_chaos['X_int'].max() + 10)
            ax[1].set_ylim(df_SV_chaos['Y_int'].min() - 10, df_SV_chaos['Y_int'].max() + 10)
            ax[2].set_ylim(df_SV_chaos['Z_int'].min() - 10, df_SV_chaos['Z_int'].max() + 10)
            
            ax[0].set_xlim(df_SV_chaos['X_int'].index[0],df_SV_chaos['X_int'].index[-1])
            ax[1].set_xlim(df_SV_chaos['Y_int'].index[0],df_SV_chaos['Y_int'].index[-1])
            ax[2].set_xlim(df_SV_chaos['Z_int'].index[0],df_SV_chaos['Z_int'].index[-1])
            
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
        
        ax[0].set_xlim(df_SV['X'].index[0],df_SV['X'].index[-1])
        ax[1].set_xlim(df_SV['Y'].index[0],df_SV['Y'].index[-1])
        ax[2].set_xlim(df_SV['Z'].index[0],df_SV['Z'].index[-1])
        
        
        plt.tick_params(labelsize=8)
        plt.savefig(directory2 + '/' + station + '_map_SV.jpeg', bbox_inches='tight')
        plt.close(fig)
      
        
        fig, ax = plt.subplots(3,1, figsize = (16,10))
        if plot_chaos == True and CHAOS_correction == True:
            ax[0].plot(df_SV_chaos['X_int'],'-',color = 'red')  #label = 'Chaos - rms: ' + str(RMS[0]),linewidth = 3, color = 'red')
            ax[1].plot(df_SV_chaos['Y_int'],'-',color = 'red')  #label = 'Chaos - rms: ' + str(RMS[1]),linewidth = 3, color = 'red')
            ax[2].plot(df_SV_chaos['Z_int'],'-',color = 'red')  #label = 'Chaos - rms: ' + str(RMS[2]),linewidth = 3, color = 'red')
            ax[0].set_ylim(df_SV_chaos['X_int'].min() - 10, df_SV_chaos['X_int'].max() + 10)
            ax[1].set_ylim(df_SV_chaos['Y_int'].min() - 10, df_SV_chaos['Y_int'].max() + 10)
            ax[2].set_ylim(df_SV_chaos['Z_int'].min() - 10, df_SV_chaos['Z_int'].max() + 10)
            ax[0].set_xlim(df_SV_chaos['X_int'].index[0],df_SV_chaos['X_int'].index[-1])
            ax[1].set_xlim(df_SV_chaos['Y_int'].index[0],df_SV_chaos['Y_int'].index[-1])
            ax[2].set_xlim(df_SV_chaos['Z_int'].index[0],df_SV_chaos['Z_int'].index[-1])
            
            ax[0].legend()  
            ax[1].legend()  
            ax[2].legend() 
        
        ax[0].plot(df_SV['X'],'o', color = 'blue')
        ax[0].set_ylim(df_SV['X'].min() - 10, df_SV['X'].max() + 10)
        ax[0].set_ylabel('dZ/dT(nT/yr)', fontsize = 12)
        ax[1].set_ylabel('dZ/dT(nT/yr)', fontsize = 12)
        ax[2].set_ylabel('dZ/dT(nT/yr)', fontsize = 12)
        ax[0].set_title(station.upper() + ' Secular Variation', fontsize = 16)
        ax[1].plot(df_SV['Y'],'o',color = 'green')
        ax[1].set_ylim(df_SV['Y'].min() - 10, df_SV['Y'].max() + 10)
        ax[2].plot(df_SV['Z'],'o',color = 'black')
        ax[2].set_ylim(df_SV['Z'].min() - 10,df_SV['Z'].max() + 10)
        plt.show()
        
        
        if jerk_start_window != None and jerk_end_window != None and CHAOS_correction == False:
            
            
            dpt.jerk_detection_window(station = station,
                                      window_start = jerk_start_window, 
                                      window_end = jerk_end_window, 
                                      starttime = starttime, 
                                      endtime = endtime,
                                      df_station = df_station_2,
                                      df_CHAOS = None,
                                      plot_detection = True,
                                      CHAOS_correction = False,
                                      plot_CHAOS_prediction = False)
            
        elif jerk_start_window != None and jerk_end_window != None and CHAOS_correction  == True:
            
            dpt.jerk_detection_window(station = station,
                                      window_start = jerk_start_window, 
                                      window_end = jerk_end_window, 
                                      starttime = starttime, 
                                      endtime = endtime,
                                      df_station = df_station_2,
                                      df_CHAOS = df_chaos,
                                      plot_detection = True,
                                      CHAOS_correction = True,
                                      plot_CHAOS_prediction = True)
        

                
def read_txt_SV(station, starttime, endtime):
    path = 'SV_update/'+ station.upper() +'_data/SV_' + station.upper() + '.txt'

    df_SV = pd.read_csv(path,sep = '\s+', index_col = [0])
    df_SV.index = pd.to_datetime(df_SV.index,infer_datetime_format=True)
    df_SV = df_SV.loc[starttime:endtime]
    
    return df_SV

def plot_samples(station, dataframe, save_plots:bool = False, plot_data_type = None, apply_percentage:bool = False):
    '''
    '''
    if save_plots == False and plot_data_type == None:
    
        samples = ['H','D','M','Y']
        colors = ['blue','green','black']
        
        
        for sample, title in zip(samples, ['hourly','daily','monthly','annual']):
            
            df_station = dpt.resample_obs_data(dataframe = dataframe, sample = sample,apply_percentage = apply_percentage)
            fig, axes = plt.subplots(3,1,figsize = (16,10))
            plt.suptitle(station.upper() + ' ' + title + ' mean', fontsize = 18, y = 0.92)
            plt.xlabel('Date(Years)', fontsize = 12)
                          
            for col, ax, color in zip(df_station.columns, axes.flatten(), colors):
            
                ax.plot(df_station[col],'o-',color = color)
                ax.set_ylabel(col.upper() +' (nT)', fontsize = 12)
                ax.set_xlim(df_station[col].index[0],df_station[col].index[-1])
                ax.grid()

            plt.show()
            
    if save_plots == False and plot_data_type != None:

        samples = ['H','D','M','Y']
        colors = ['blue','green','black']
        First_QD_data = plot_data_type
    
        for sample, title in zip(samples, ['hourly','daily','monthly','annual']):
            
            df_station = dpt.resample_obs_data(dataframe = dataframe, sample = sample,apply_percentage = apply_percentage)
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
            
            df_station = dpt.resample_obs_data(dataframe = dataframe, sample = sample,apply_percentage = apply_percentage)
            fig, axes = plt.subplots(3,1,figsize = (16,10))
            plt.suptitle(station.upper() + ' ' + title + ' mean', fontsize = 18, y = 0.92)
            plt.xlabel('Date(Years)', fontsize = 12)
                          
            for col, ax, color in zip(df_station.columns, axes.flatten(), colors):
            
                ax.plot(df_station[col],'o-',color = color)
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
            
            df_station = dpt.resample_obs_data(dataframe = dataframe, sample = sample,apply_percentage = apply_percentage)
            fig, axes = plt.subplots(3,1,figsize = (16,10))
            plt.suptitle(station.upper() + ' ' + title + ' mean', fontsize = 16, y = 0.92)
            plt.xlabel('Date(Years)', fontsize = 14)
                          
            for col, ax, color in zip(df_station.columns, axes.flatten(), colors):
            
                ax.plot(df_station[col],'o-',color = color)
                ax.set_ylabel(col.upper() +' (nT)', fontsize = 12)
                ax.set_xlim(df_station[col].index[0],df_station[col].index[-1])
                ax.plot(df_station.loc[df_station.index > First_QD_data][col],'o-', color = 'red', label = 'Quasi-definitive data')
                ax.legend()
                ax.grid()
                
            plt.savefig(directory + '/' + station + '_' + title + '_mean.jpeg', bbox_inches='tight')
            plt.show()
            
    