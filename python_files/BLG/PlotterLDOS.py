import sys
import os
sys.path.insert(0,'..')
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from utils.h5_handler import *
# path_to_dump = '../Output/BLG/solveLDOStest/'
path_to_dump = '../Output/BLG/CUHREsolveLDOS/'
if not os.path.exists(path_to_dump): 
	raise Exception('Path to dump not found')
	exit(1)

# filename = 'v3BLG_LDOS_00_2439941.h5'
# filename = 'v3BLG_LDOS_11_2441853.h5'
# filename = 'BLGsolveRGF_3393566.h5'
# filename = 'BLGsolveRGF_3400662.h5'
# filename = 'CUHREBLGsolveRGF_3421462.h5'
filename = 'Om6CUHREBLGsolveRGF_3450657.h5'

try:
	load_dict = h52dict(os.path.join(path_to_dump,filename))
except FileNotFoundError:
	print('Load file not found: ', os.path.join(path_to_dump,filename))
	exit(1)


omegavals = load_dict['omegavals']
LDOS = np.array(load_dict['LDOS'])
print(load_dict['INFO'])
fig,ax = plt.subplots(2)
ax[0].plot(omegavals, LDOS, '.-', label = 'LDOS')
ax[0].axvline(1e-3, ls='--', c='grey')
ax[0].set_xlabel('omega')
ax[0].set_title('BLG LDOS A site ' + str(load_dict['INFO']))

ax[1].loglog(omegavals[omegavals<1],LDOS[omegavals<1])
ax[1].set_xlabel('omega')

plt.show()
