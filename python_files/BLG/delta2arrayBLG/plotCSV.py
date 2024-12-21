import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt
import sys, os

path_to_csv = './csvresults/'

NUMGS = 1
csv_names = [f'f{i}_complete_grid_results.csv' for i in range(NUMGS)]

dflist = []
for i, name in enumerate(csv_names):
    try:
        df = pd.read_csv(os.path.join(path_to_csv,name))
    except FileNotFoundError:
        print(f'csv file {name} not found')
    dflist.append(df)

df = dflist[0]
omegavals = df.index
rvals = df.columns
print(f'omegavals = {omegavals}')
print(f'rvals  = {rvals}')

print(df.head)
