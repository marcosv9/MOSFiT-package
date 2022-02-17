
##  About Me
I'm a Marcos Vinicius Silva...


# Fast geomagnetic jerk detection 
My Thesis project in geomagnetism.

A python tool to study the secular variation of the geomagnetic field and accelerate the geomagnetic jerk detection.

Actually works for every INTERMAGNET geomagnetic observatory (definitive and quasi-definitive data).

There are functions to reduce the external field contribution, calculate the secular variation, detect geomagnetic jerks...

The functions are separetade in data processing, utility, support and main functions.



## Import function suggestion

To use the developed funtions, is necessary to import them. I suggest to use the following statements.

```bash
  from Thesis_Marcos import data_processing_tools as dpt
  from Thesis_Marcos import utilities_tools as utt
  from Thesis_Marcos import thesis_functions as mvs
  from Thesis_Marcos import support_functions as spf
```


## SV_OBS Usage

SV_OBS is the main funcion, allowing the user to process the geomagnetic data in a interactive workflow, using most of the available data_processing functions.

![](figures/worflow.png)

```python

SV_OBS(station = 'VSS',
stattime = '2000-01-01,
endtime = '2021-06-30')}
```
