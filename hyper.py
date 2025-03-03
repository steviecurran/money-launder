#!/usr/bin/python3
import numpy as np
import pandas as pd
from shutil import get_terminal_size
pd.set_option('display.width', get_terminal_size()[0]) 
pd.set_option('display.max_columns', None)

df = pd.read_csv("N=10.csv-101-runs.csv"); print(df)
print(df.sort_values(by=['Mean']))
