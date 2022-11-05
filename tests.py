import pytest
import main_functions as mvs
import data_processing_tools as dpt
import pandas as pd
import random
import numpy as np


def test_resample_condition():
    
    index = pd.date_range('2010-01-01 00:00',
                          '2010-01-01 01:00',
                          freq = 'min')
    
    data = random.sample(range(1, 50), 60)
    
    df = pd.DataFrame(data = data, index = index)
    
    df[0:7] = np.nan
    
    df = dpt.resample_obs_data(df, sample='H', apply_percentage=True)
    
    assert df[0][0] is np.nan