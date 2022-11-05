##  About Me
I'm a Marcos Vinicius Silva...


# Fast geomagnetic jerk detection 

My Master project in geomagnetism.

A python tool to study the secular variation of the geomagnetic field and accelerate the geomagnetic jerk detection.

Actually works for every INTERMAGNET geomagnetic observatory (definitive and quasi-definitive IAGA-2002 data).

There are functions to reduce the external field contribution, calculate the secular variation, detect geomagnetic jerks...

The package is separetade in modules called data_processing_tools, utility_tools, support_functions and main_functions. 

All the functions have a docstring with the inputs explanation and an usage example.

## Setting up MOSFiT package

The MOSFiT python package was built to work with INTERMAGNET minute mean data in the IAGA-2002 format, in order to analyse the SV and check INTERMAGNET Magnetic Observatory (IMO) data quality. The definitive and quasi-definitive data are mainly used because of higher quality and reliability, especially for SV studies. However, we can also apply MOSFiT to others types of IAGA-2002 data, i.e. provisional data).

Step 1: Requirements and software installation.

MOSFiT is developed in the Python 3 language. The package can be compiled in the command window or in a “jupyter notebook enviroment”.

You can download MOSFiT in: https://github.com/marcosv9/Thesis-Marcos. In this same link, there is a documentation of how to use the package functions, with some examples.


Step 2: Downloading IAGA-2002 data.

In order to use MOSFiT, it is necessary to have the data stored in a local computer.

This data can be downloaded from the INTERMAGNET website (https://www.intermagnet.org/), directly from the INTERMAGNET ftp server (ftp://ftp.seismo.nrcan.gc.ca/intermagnet/) or by using the MOSFiT function called “download INTERMAGNET file” (by choosing datatype, year, month and the list of observatories).

MOSFiT will only read filenames in the same format of INTERMAG-NET IAGA-2002 2. After the data is downloaded, the user may organize all files from different observatories in a single or multiple folders.


## Package modules import suggestion

To use the developed funtions, is necessary to import them. I suggest to use the following statements to import the modules.

```bash
  import data_processing_tools as dpt
  import utilities_tools as utt
  import main_functions as mvs
  import support_functions as spf
```

## load_intermagnet_files

This function is the most important, since it reads any geomagnetic data following the IAGA-2002 format.

The output is a pandas DataFrame indexed by time and the columns are the X, Y and Z geomagnetic components.

Its output is used as input is most of the data processing functions.
 

```python
load_intermagnet_files(station = 'XXX', starttime = 'yyyy-mm-dd', endtime = 'yyyy-mm-dd', files_path = 'path//to//files')
```
The returned dataframe can be manipulated by the user or processed with the others functions of the package. 

You can set a specific directory or just use the default (automatically created when the files are downloaded using download_data_INTERMAGNET function).

## data_processing_tools functions

Here I explain the principal function of the data_processing_tools module.

As the name says such functions are responsable for the data processing.

Most of them are methods to reduce the external field contribution from the observatory data. In order to investigate the SV.


### resample_observatory_data

This function allows the user to resample geomagnetic observatory

data into different samples (hourly, daily, monthly and annual).

```python
import data_processing_tools as dpt
dpt.resample_obs_data(dataframe = df_name, sample = 'H', apply_percentage = True)
```
Example of different data samples calculated using MOSFiT.

![](figures/resample_obs_data.jpeg)


### kp_index_correction

The function removes periods with Kp index values above user input limit from the geomagnetic components 

Find the index on https://kp.gfz-potsdam.de/en/

```python
import data_processing_tools as dpt
dpt.kp_index_correction(dataframe = df_name, kp = 2)
```
![](figures/kp_index_ex.jpeg)


### keep_quiet_days

The function select only the top 10 international quiet days from each month

Find the list on https://kp.gfz-potsdam.de/en/

```python
import data_processing_tools as dpt
dpt.keep_quiet_days(dataframe = df_name)
```
Example of SV calculate using normal data and selecting quiet days for each monthly.
![](figures/quiet_days_ex.jpeg)

### remove_disturbed_days

The function remove the top 5 international disturbed days from each month

Find the list on https://kp.gfz-potsdam.de/en/

```python
import data_processing_tools as dpt
dpt.remove_disturbed_days(dataframe = df_name)
```
Example of SV calculate using normal data and removing the top 5 disturbed days from each month.
![](figures/disturbed_days_ex.jpeg)

### night_time_selection

The function remove the top 5 international disturbed days from each month

Find the list on https://kp.gfz-potsdam.de/en/

```python
import data_processing_tools as dpt
dpt.night_time_selection(station = 'XXX', dataframe = df_name)
```
Example of SV calculate using normal data and selecting only nighttime period.
![](figures/nighttime_ex.jpeg)

### calculate_sv

Calculate geomagnetic secular variation using monthly means or annual means

```python
import data_processing_tools as dpt
dpt.calculate_sv(dataframe = df_name, method = 'ADMM')
```
Example of SV calculate from VSS monthly means using MOSFiT.
![](figures/VSS_SV.jpeg)

## SV_OBS Usage

sv_obs is a function that includes the most important data processing options.

The processing according to the figure is already implemented in a dedicated function.

However the user can combine any of the processing steps in any possible order or combination

It allows the user to process the geomagnetic data in a interactive workflow,

 using most of the available data_processing functions.

![](figures/worflow.png)

```python

SV_OBS(station = 'VSS', stattime = '2000-01-01', endtime = '2021-06-30', files_path = 'path//to//files', plot_chaos = True)
```

