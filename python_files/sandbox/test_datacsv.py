import os, sys
sys.path.insert(0,'../')
import numpy as np
import pandas as pd
from utils.h5_handler import *

data = np.array([[2, np.nan, 3.2], [3.6, np.inf, np.pi], [2.7,7.7,9]])
df = pd.DataFrame(data)
print(df)
df.to_csv(f'csv_with_nansi.csv')
