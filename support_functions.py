import pandas as pd
import numpy as np
import glob
import os
import ftplib
import pathlib
from datetime import datetime, timedelta
import utilities_tools as utt
import requests
import re
import shutil
from bs4 import BeautifulSoup
import yaml
from box import Box

def get_config() -> None:
    with open(os.path.join(os.getcwd(),"config","main_cfg.yaml"), "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config_box = Box(config)
    #config = ruamel.yaml.safe_load(open("C://Users//marcos//Documents//leomagnetics//qnl-code-1//config//main_cfg.yaml"))
    #config_dict = ruamel.yaml.comments.CommentedMap(config)
    #config = yaml.safe_load(open("C://Users//marcos//Documents//leomagnetics//qnl-code-1//config//main_cfg.yaml"))
    # Access the configuration parameters
    return config_box

def get_chaos_latest_release():
    
    config = get_config()
    
    url = config.url.chaos_model
    
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check if the request was successful
    except requests.exceptions.RequestException as e:
        print("Error fetching the webpage:", e)
        return None
    
    soup = BeautifulSoup(response.text, 'html.parser')
    h2_elements = soup.find_all('h2')
    last_release = str(h2_elements[3])[7:17]
    
    return last_release

def check_chaos_local_version():
    
    config = get_config()
    
    chaos_path = glob.glob(os.path.join(config.directory.chaos_model,
                                        'data',
                                        'CHAOS*'
                                        )
                           ) 
    
    filename = os.path.basename(chaos_path[0])
    local_version = filename[0:10]
    
    lastest_version = get_chaos_latest_release()
    
    if lastest_version != local_version:
        print("Local CHAOS model version is not the latest version.")
        print("Downloading latest version.")
        download_chaos_latest_release()
        print("Old version deleted.")

    return

def unzip_chaos_model():

# Assuming you have already downloaded the file and have its filename
    zipfile = glob.glob(os.path.join(os.getcwd(),"chaosmagpy_package_*.zip"))[0]

    # Specify the directory where you want to extract the contents
    extract_dir = "./chaosmagpy_package"

    # Create the extraction directory if it doesn't exist
    os.makedirs(extract_dir, exist_ok=True)

    # Open the zip file in read mode
    shutil.unpack_archive(zipfile, extract_dir)
    os.remove(zipfile)

def download_chaos_latest_release():
    
    config = get_config()
    
    url = config.url.chaos_model    
    
    # Send a GET request to the URL and get the HTML content
    response = requests.get(url)
    html_content = response.text

    # Use BeautifulSoup to parse the HTML content
    soup = BeautifulSoup(html_content, "html.parser")

    # Find the <a> tag with the href containing "chaosmagpy_package" (you can modify the regex pattern accordingly)
    pattern = re.compile("chaosmagpy_package_.*\.zip")
    link = soup.find("a", href=pattern)

    if link:
        download_url = url + link["href"]

        # Download the file using requests
        response = requests.get(download_url)
        if response.status_code == 200:
            # Save the file to a local directory
            filename = download_url.split("/")[-1]
            with open(filename, "wb") as file:
                file.write(response.content)
            print(f"File '{filename}' downloaded successfully.")
        else:
            print(f"Failed to download the file.")
    else:
        print("File link not found.")
    unzip_chaos_model()
    #os.remove(filename)


def gg_to_geo(h, gdcolat):
    """
    Function from pyIGRF
    
    Compute geocentric colatitude and radius from geodetic colatitude and
    height.

    Parameters
    ----------
    h : ndarray, shape (...)
        Altitude in kilometers.
    gdcolat : ndarray, shape (...)
        Geodetic colatitude

    Returns
    -------
    radius : ndarray, shape (...)
        Geocentric radius in kilometers.
    theta : ndarray, shape (...)
        Geocentric colatitude in degrees.
    
    sd : ndarray shape (...) 
        rotate B_X to gd_lat 
    cd :  ndarray shape (...) 
        rotate B_Z to gd_lat 

    References
    ----------
    Equations (51)-(53) from "The main field" (chapter 4) by Langel, R. A. in:
    "Geomagnetism", Volume 1, Jacobs, J. A., Academic Press, 1987.
    
    Malin, S.R.C. and Barraclough, D.R., 1981. An algorithm for synthesizing 
    the geomagnetic field. Computers & Geosciences, 7(4), pp.401-405.

    """
    # Use WGS-84 ellipsoid parameters

    eqrad = 6378.137 # equatorial radius
    flat  = 1/298.257223563
    plrad = eqrad*(1-flat) # polar radius
    ctgd  = np.cos(np.deg2rad(gdcolat))
    stgd  = np.sin(np.deg2rad(gdcolat))
    a2    = eqrad*eqrad
    a4    = a2*a2
    b2    = plrad*plrad
    b4    = b2*b2
    c2    = ctgd*ctgd
    s2    = 1-c2
    rho   = np.sqrt(a2*s2 + b2*c2)
    
    rad   = np.sqrt(h*(h+2*rho) + (a4*s2+b4*c2)/rho**2)

    cd    = (h+rho)/rad
    sd    = (a2-b2)*ctgd*stgd/(rho*rad)
    
    cthc  = ctgd*cd - stgd*sd           # Also: sthc = stgd*cd + ctgd*sd
    thc   = np.rad2deg(np.arccos(cthc)) # arccos returns values in [0, pi]
    
    return rad, thc, sd, cd

def download_new_iaga(station:str, starttime:str, endtime:str):
    """_summary_

    Args:
        station (str): _description_
        starttime (str): _description_
        endtime (str): _description_
    """

    if not os.path.exists(os.path.join(os.getcwd(), station.upper())):
        os.makedirs(os.path.join(os.getcwd(), station.upper()))
        
    starttime = datetime.strptime(starttime, '%Y-%m-%d')
    endtime = datetime.strptime(endtime, '%Y-%m-%d')
    
    while starttime <= endtime:
        
        specific_directory = f"https://imag-data.bgs.ac.uk/GIN_V1/GINServices?Request=GetData&format=IAGA2002&testObsys=0&observatoryIagaCode={station.upper()}&samplesPerDay=1440&publicationState=adj-or-rep&recordTermination=UNIX&dataStartDate=[{starttime}]&dataDuration=1"    


        url = specific_directory    

    #url = "https://imag-data.bgs.ac.uk/GIN_V1/GINForms2?dlC=AAE&downloadDataStartDate=2022-05-12&downloadPublicationState=Best+available&downloadDataDuration=10&downloadDurationType=Days&downloadSamplesPerDay=minute&downloadFormat=IAGA2002&downloadScriptTarget=Windows&downloadFolderOptions=YearThenObservatory&downloadProxyAddress=&request=BulkDownloadOptions&submitValue=Done"
    #req = requests.get(url)
#
    #content_url = req.content
    #
    #data_string = content_url.decode('utf-8')
    

        response = requests.get(f"https://imag-data.bgs.ac.uk/GIN_V1/GINServices?Request=GetData&format=IAGA2002&testObsys=0&observatoryIagaCode={station.upper()}&samplesPerDay=1440&publicationState=adj-or-rep&recordTermination=UNIX&dataStartDate={starttime}&dataDuration=1")
        local_file = os.path.join(os.getcwd(), station.lower(), f"{station.lower()}{starttime.strftime('%Y%m%d')}min.min")

        with open(local_file, 'wb') as file:

            file.write(response.content)

        with open(local_file, 'r') as f:
            for line in f.readlines():
                if re.search ('^ Data Type', line):
                    dt = line[24:25].lower()

        new_file = os.path.join(os.getcwd(), station.lower(), f"{station.lower()}{starttime.strftime('%Y%m%d')}{dt}min.min")
        with open(new_file, 'wb') as file:

            file.write(response.content)

        os.remove(local_file)
        
        print(f"File {station.lower()}{starttime.strftime('%Y%m%d')}{dt}min.min downloaded!")
        
        starttime += timedelta(days=1)

def project_directory():
    '''
    Get the project directory 
    '''
    return os.getcwd()

def update_qd_and_dd(data:str):
    """
    Update list of quiet and disturbed days.
    
    Used automatically when function to remove disturbed days
    or keep quiet days are used.
    """

    #validating input parameter
    
    assert data in ['DD', 'QD'], 'data must be QD or DD!'
    
    config = get_config()
        
    #connecting to the ftp server 
    ftp = ftplib.FTP('ftp.gfz-potsdam.de')
    ftp.login('anonymous', 'email@email.com')
    
    ##path to read the already stored QD and DD
    
    path_local = config.directory.qd_dd
    
    ##path inside ftp server
    path_ftp = f'/pub/home/obs/kp-ap/quietdst'
    
    #getting the file
    ftp.cwd(path_ftp)
    
    filenames = ftp.nlst('qdrecen*')
    
    pathlib.Path(path_local).mkdir(parents=True, exist_ok=True)
    #writting file in the local computer
    for filename in filenames:
        local_filename = os.path.join(path_local, filename)
        file = open(local_filename, 'wb')
        ftp.retrbinary('RETR ' + filename, file.write)
        file.close()
    ftp.quit()
    
    if data == 'DD':
        
        df = pd.read_csv(os.path.join(path_local,
                                      config.filenames.recent_qd
                                      ),
                         skiprows = 4,
                         sep = '\s+',
                         header = None,
                         usecols = [0,1,12,13,14,15,16],
                         names = ['Month', 'Year', 'DD1',
                                  'DD2', 'DD3', 'DD4', 'DD5'
                                  ]
                         )
                          
        columns = ['DD1','DD2',
                   'DD3','DD4',
                   'DD5'
                  ]
        
        df['Month'] = pd.to_datetime(df.Month, format='%b').dt.month
        
        for col in columns:
            df[f'{col}'] = df['Year'].astype(str)+'-'+df['Month'].astype(str)+'-'+df[col].astype(str)
        for col in columns:
            df[f'{col}'] = df[f'{col}'].str.replace('*','')
        df_dd = pd.DataFrame()
        
        df_dd['DD'] = pd.concat([df['DD1'] ,df['DD2'],
                                 df['DD3'], df['DD4'],
                                 df['DD5']
                                ]
                               )
        
        df_dd['DD'] = pd.to_datetime(df_dd['DD'], format= '%Y-%m-%d')
        
        df_list = pd.read_csv(pathlib.Path(os.path.join(path_local,
                                                        config.filenames.disturbed_days
                                                        )
                                           )
                              )
        
        df_list['DD'] = pd.to_datetime(df_list['DD'], format= '%Y-%m-%d')
    
        df_new = pd.concat([df_list,df_dd], ignore_index=False)
        
        df_new['DD'] = pd.to_datetime(df_new['DD'], infer_datetime_format=True)
        
        df_new = df_new.drop_duplicates()
        
        df_new.set_index('DD', inplace=True)
        
        df_new.dropna().sort_index().to_csv(pathlib.Path(os.path.join(path_local, config.filenames.disturbed_days)), index = True)
        
    if data == 'QD':
        
        df = pd.read_csv(pathlib.Path(os.path.join(path_local, config.filenames.recent_qd)),
                          skiprows = 4,
                          sep = '\s+',
                          header = None,
                          usecols = [0, 1, 2,
                                     3, 4, 5,
                                     6, 7, 8,
                                     9, 10, 11
                                     ],
                          names = ['Month', 'Year', 'QD1',
                                   'QD2', 'QD3', 'QD4', 'QD5',
                                   'QD6', 'QD7', 'QD8', 'QD9',
                                   'QD10'
                                  ]
                           )
        
        columns = [f'QD{i}' for i in range(1, 11)]
        
        df['Month'] = pd.to_datetime(df.Month, format='%b').dt.month
        
        for col in columns:
            df[col] = df['Year'].astype(str) + '-' + df['Month'].astype(str) + '-' + df[col].astype(str)
        for col in columns:
            df[col] = df[col].str.replace('A', '')
        for col in columns:
            df[col] = df[col].str.replace('K', '')
        
        df_qd = pd.DataFrame()
        df_qd['QD'] = pd.concat([df['QD1'], df['QD2'], df['QD3'],
                                 df['QD4'], df['QD5'], df['QD6'],
                                 df['QD7'], df['QD8'], df['QD9'],
                                 df['QD10']
                                ]
                               )
        
        df_qd['QD'] = pd.to_datetime(df_qd['QD'], infer_datetime_format=True)
        
        df_list = pd.read_csv(pathlib.Path(os.path.join(path_local,
                                                        config.filenames.quiet_days
                                                        )
                                           )
                              )
        
        df_list['QD'] = pd.to_datetime(df_list['QD'], format= '%Y-%m-%d')
    
        df_new = pd.concat([df_list,df_qd], ignore_index=False)
        
        df_new['QD'] = pd.to_datetime(df_new['QD'], infer_datetime_format=True)
        
        df_new = df_new.drop_duplicates()
        
        df_new.set_index('QD', inplace=True)
        
        df_new.dropna().sort_index().to_csv(pathlib.Path(os.path.join(path_local, config.filenames.quiet_days)), index = True)

        
def header_sv_obs_files(station: str,
                        filename: str,
                        data_denoise: str,
                        external_correction: str,
                        chaos_model: str):
    """"
    Function to add a header in the txt outputs of the Function sv_obs.
    
    Output with informations about the used observatory
    and used data processing options.
    
    ------------------------------------------------------------------------------------
    used automatically
    """  
    if utt.IMO.check_existence(station) is False:
        raise ValueError(f'station must be an observatory IAGA CODE!')
    
    working_directory = project_directory()
    
    config = get_config()
        
    path = pathlib.Path(os.path.join(config.directory.filtered_data,
                                     f'{station}_data',
                                     f'{station.upper()}_{filename}_preliminary.txt'
                                     )
                        )
    
    path_header = pathlib.Path(os.path.join(config.directory.filtered_data,
                                            f'{station}_data'
                                            )
                               )
    output_path = pathlib.Path(os.path.join(config.directory.filtered_data,
                                             f'{station}_data',
                                             f'{station.upper()}_{filename}.txt'
                                             )
                                )
    
    external_options = {'D': 'Disturbed Days removed',
                        'Q': 'Quiet Days selection', 
                        'NT': 'Night Time selection',
                        'E': 'No'}
    
    denoise_options = {'y': 'Hampel Filter',
                       'n': 'No'
                      }
    
    chaos_options = {'y': 'Yes',
                     'n': 'No'
                    }

    df_station = pd.read_csv(path,
                             sep = '\s+',
                             index_col = [0]
                             )
    df_station.index = pd.to_datetime(df_station.index, infer_datetime_format=True)
   
    
    Header = (f'MOSFiT package'
              f'\nIAGA CODE {str(station.upper())}'
              f'\nLatitude {str(utt.IMO.latitude(station.upper()))}'
              f'\nLongitude {str(utt.IMO.longitude(station.upper()))}'
              f'\nElevation {str(utt.IMO.elevation(station.upper()))}' 
              f'\nData denoise: {denoise_options[data_denoise]}'
              f'\nExternal field reduction: {external_options[external_correction]}'
              f'\nCHAOS model correction: {chaos_options[chaos_model]}'
              f'\n\n')
    
    with open(os.path.join(f'{path_header}',
                           config.filenames.header_file), 'w+') as f2:
        f2.write(Header)
        header = f2.read()
    
    filenames = [os.path.join(f'{path_header}',
                              config.filenames.header_file),
                 path
                 ]
    
    with open(output_path, 'w') as outfile:
        for fname in filenames:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)
                    
    
    os.remove(os.path.join(f'{path_header}',
                           config.filenames.header_file
                           )
              )
    os.remove(path)
    
def data_type(station: str,
              starttime: str = None,
              endtime: str = None,
              files_path = None):
    '''
    Function to verify the existence of Quasi-definitive data type in the dataset
    
    ----------------------------------------------------------
    Inputs:
    
    station (str) - 3 letters IAGA code 
    
    starttime (str) - first day of the data (format = 'yyyy-mm-dd')
    
    endtime (str) - last day of the data (format = 'yyyy-mm-dd')
    
    files_path (str) - path to the IAGA-2002 intermagnet files or None
                 if None it will use the default path for the files
    
    --------------------------------------------------------------------------------------
    Usage example:
    
    data_type(station = 'VSS',
              starttime = '2000-01-25',
              endtime = '2021-01-25',
              files_path = 'path//to/files')
    
    
    '''
    assert len(station) == 3, 'station must be a IAGA code with 3 letters'
    
    if utt.IMO.check_existence(station) is False:
        raise ValueError(f'station must be an observatory IAGA CODE!')
        
    if not [i for i in (starttime, endtime) if i is None]:
        for i in [starttime, endtime]:
            validate(i)
    else:
        if files_path is None:
            raise ValueError('if starttime and endtime are None, you must inform files_path.')  

    files_station = [] 

    if pd.to_datetime(endtime) >= pd.to_datetime('2018-06-30'):
    
        if files_path is None:
            years_interval = np.arange(int(starttime[0:4]), int(endtime[0:4])+ 1)
            for year in years_interval:
    
                files_station.extend(glob.glob(f'C:\\Users\\marco\\Downloads\\Thesis_notebooks\\Dados OBS\\{str(year)}/*/{station}*'))
                files_station.sort()
        else:
            files_station.extend(glob.glob(os.path.join(f'{files_path}',
                                                        f'{station}*min*'
                                                        )
                                           )
                                 )
            
            files_station.sort()
    
        df_data_type = pd.DataFrame()
        df_data_type = pd.concat((pd.read_csv(file,
                                              sep = '\s+',
                                              header = None,                                         
                                              skiprows = 11,
                                              nrows = 20,
                                              usecols = [0,2],
                                              names = ['Date', 'Data_Type'])
                                              for file in files_station))
        
        data_type = df_data_type.loc[[0], ['Data_Type']]
    
        date = df_data_type.loc[[15], ['Date']]
    
        data_type.set_index(date['Date'], inplace = True)
        
        data_type.sort_index()
        
        for Date in data_type['Data_Type']: 
            if Date == 'Definitive':
                Date = []
            else:
            
                Date = data_type.loc[data_type['Data_Type'] != 'Definitive'].index[0]
    else:
        Date = []    
        
    return Date

def header_sv_files(station: str,
                    data_denoise: str,
                    external_correction: str,
                    chaos_model:str
                    ):
    """"
    Function to add a header in the txt outputs of the Function sv_obs.
    
    Output with informations about the used observatory
    and used data processing options.
    
    ------------------------------------------------------------------------------------
    used automatically
    """  
    
    #validating inputs
    
    assert len(station) == 3, 'station must be a IAGA code with 3 letters'
    
    if utt.IMO.check_existence(station) is False:
        raise ValueError(f'station must be an observatory IAGA CODE!')
    
    working_directory = project_directory()
        
    path = pathlib.Path(os.path.join(working_directory,
                                     f'SV_update',
                                     f'{station}_data',
                                     f'SV_{station.upper()}_preliminary.txt'
                                     )
                        )
    
    path_header = pathlib.Path(os.path.join(working_directory,
                                            'SV_update',
                                            f'{station}_data'
                                            )
                               )
    
    output_path = pathlib.Path(os.path.join(working_directory,
                                            'SV_update',
                                            f'{station}_data',
                                            f'SV_{station.upper()}.txt'
                                            )
                               )
    
    external_options = {'D': 'Disturbed Days removed',
                        'Q': 'Quiet Days selection', 
                        'NT': 'Night Time selection',
                        None: 'No'
                        }
    
    denoise_options = {True: 'Hampel Filter',
                      False: 'No'
                      }
    
    chaos_options = {True: 'Yes',
                    False: 'No'
                    }

    df_station = pd.read_csv(path,
                             sep = '\s+',
                             index_col = [0])
    
    df_station.index = pd.to_datetime(df_station.index,
                                      infer_datetime_format = True)
    
    
    header = (f'MOSFiT package'
              f'\nIAGA CODE {str(station.upper())}'
              f'\nLatitude {str(utt.IMO.latitude(station.upper()).round(2))}'
              f'\nLongitude {str(utt.IMO.longitude(station.upper()).round(2))}'
              f'\nElevation {str(utt.IMO.elevation(station.upper()).round(2))}'
              f'\nData denoise: {denoise_options[data_denoise]}'
              f'\nExternal field reduction: {external_options[external_correction]}'
              f'\nCHAOS model correction: {chaos_options[chaos_model]}'
              f'\n\n')
    
    with open(pathlib.Path(os.path.join(f'{path_header}',
                                        'header_file.txt')), 'w+') as f2:
        
        f2.write(header)
        header = f2.read()
    
    filenames = [pathlib.Path(os.path.join(f'{path_header}', 'header_file.txt', path))]
    with open(output_path, 'w') as outfile:
        for fname in filenames:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)
                    
    
    os.remove(pathlib.Path(os.path.join(f'{path_header}', 'header_file.txt')))
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
    return date.year + ((date - year_start).total_seconds() / 
        float((year_end - year_start).total_seconds()))

def validate(str_date):
    """
    Function to validate input format "YYYY-MM-DD"
    
    """
    try:
        datetime.strptime(str_date, '%Y-%m-%d')
    except ValueError:
        raise ValueError('Incorrect date format, should be YYYY-MM-DD')

def validate_ym(str_date):
    """
    Function to validate input format "YYYY-MM"
    
    """
    try:
        datetime.strptime(str_date, '%Y-%m')
    except ValueError:
        raise ValueError(f'Incorrect {str_date} format, should be YYYY-MM')

class year(object):
    """
    class representing the year
    used to find leap years

    """
    def __init__(self,year):
        self.year = year
    def check_leap_year(year):
        if((year % 400 == 0) or  
           (year % 100 != 0) and  
           (year % 4 == 0)):
            return True
        else:
            return False

def skiprows_detection_v2(file_path) -> int:
    """Function to detect the correct number of skiprows
        for each IAGA-2002 file
    Args:
        file_path (str): _description_
    Returns:
        int: The number of rows to skip before the data
    """
    
    idx = 0
    skiprows = 10
    df_station = pd.read_csv(file_path,
                             sep = '\s+',
                             skiprows = skiprows,
                             nrows=40,
                             usecols = [0],
                             names = ['col']
                             )

    while df_station['col'][idx] != 'DATE':
        skiprows += 1
        idx +=1 
        if df_station['col'][idx] == 'DATE':
            skiprows += 1
                
    return skiprows 

def skiprows_detection(files_station):
    """Function to detect the correct number of skiprows
    for each IAGA-2002 file

    Args:
        files_station (list of files): _description_

    Returns:
        list: contains the number of skiprows for each file and the path
    """
    
    skiprows_list = [[], []]
    
    for file in files_station:
        idx = 0
        skiprows = 10
        df_station = pd.read_csv(file,
                                 sep = '\s+',
                                 skiprows = skiprows,
                                 nrows=30,
                                 usecols = [0],
                                 names = ['col']
                                 )
        file = file
        while df_station['col'][idx] != 'DATE':
            skiprows += 1
            idx +=1 
            if df_station['col'][idx] == 'DATE':
                skiprows += 1
                skiprows_list[0].append(skiprows)     
                skiprows_list[1].append(file)
                
    return skiprows_list      

def decimal_year_to_date(date):
    """
    Function to convert from decimal year to date

    Args:
        date (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    decimal_date = float(date)
    year = int(decimal_date)
    rest = decimal_date - year

    base = datetime(year, 1, 1)
    result = base + timedelta(seconds=(base.replace(year=base.year + 1) - base).total_seconds() * rest)
    
    return result.date()      