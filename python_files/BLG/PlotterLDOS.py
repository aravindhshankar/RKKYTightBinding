import sys
import os
sys.path.insert(0,'..')
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from h5_handler import *
path_to_dump = '../Outputs/BLG'
if not os.path.exists(path_to_dump): 
	raise Exception('Path to dump not found')
	exit(1)

filename = 'v2BLG_LDOS_00_4053680.h5'
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
ax[0].axvline(1., ls='--', c='grey')
ax[0].set_xlabel('omega')
ax[0].set_title(f'Graphene LDOS A site with $ ')

ax[1].loglog(omegavals[omegavals<1],LDOS[omegavals<1])
ax[1].set_xlabel('omega')

plt.show()
