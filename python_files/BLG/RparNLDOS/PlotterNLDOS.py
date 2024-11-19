import sys
import os
sys.path.insert(0,'..')
sys.path.insert(0,'../..')
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from utils.h5_handler import *
# path_to_dump = '../Output/BLG/solveLDOStest/'
path_to_dump = '/Users/aravindhswaminathan/Documents/GitHub/RKKYTightBinding/python_files/Output/nldos'
if not os.path.exists(path_to_dump): 
	raise exception('path to dump not found')
	exit(1)

fig,ax = plt.subplots(2)

for job_idx in np.arange(0,20, dtype=int):
    filename = f'results_r_{job_idx}.h5'
    try:
        load_dict = h52dict(os.path.join(path_to_dump,filename))
    except FileNotFoundError:
        print('Load file not found: ', os.path.join(path_to_dump,filename))
        # exit(1)
        continue

    print('keys = ', load_dict.keys()) if __debug__ else 0
    omegavals = load_dict['omegavals']
    NLDOS = np.array(load_dict['NLDOS'])
    rval = load_dict['r']
    print(load_dict['INFO']) if __debug__ else 0
    ax[0].plot(omegavals, NLDOS, '.-', label = f'r = {rval}')
    ax[1].loglog(omegavals[omegavals<1],NLDOS[omegavals<1], label = f'r = {rval}')

ax[0].set_xlabel('omega')
ax[0].set_title('graphene NLDOS A site , ')
ax[0].legend()
ax[1].set_xlabel('omega')
ax[1].legend()

plt.show()
