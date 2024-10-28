import numpy as np 

a = 10.
xstart = 5.
ystart = 14.


firstrowx = xstart + np.arange(0,4) * 2 * a 
firstcolumny = ystart + np.arange(0,4) * 2 * a 

print(f'firstrowx = {firstrowx}')
print(f'firstcolumny = {firstcolumny}')

