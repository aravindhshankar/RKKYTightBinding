import sys
import os
sys.path.insert(0,'..')
sys.path.insert(0,'../..')
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from utils.h5_handler import *
from scipy.special import j0
path_to_dump = '/Users/aravindhswaminathan/Documents/GitHub/RKKYTightBinding/python_files/Output/anaBLG/'
path_to_fig = '/Users/aravindhswaminathan/Documents/GitHub/RKKYTightBinding/python_files/Figures/anaBLG'
# path_to_dump = '/Users/aravindhswaminathan/Documents/GitHub/RKKYTightBinding/python_files/Output/BLGnldos'
# path_to_fig = '/Users/aravindhswaminathan/Documents/GitHub/RKKYTightBinding/python_files/Figures/BLGnldosfigs'
if not os.path.exists(path_to_dump):
	raise exception('path to dump not found')
	exit(1)

if not os.path.exists(path_to_fig):
    print('Figures directory does not exist, creating .....')
    os.makedirs(path_to_fig, exist_ok = True)

NUMGS = 1 #number of dictionary entries in the load file

figlist = [plt.figure() for i in range(NUMGS)]
axlist = [figlist[i].subplots(2) for i in range(NUMGS)]

# jobarray = np.arange(0,13,1,dtype=int)
jobarray = [0,5,12,15,20]
# jobarray = [5,]
for i, job_idx in enumerate(jobarray):
    col = 'C' + str(i)
    filename = f'results_g3_{job_idx}.h5'
    try:
        load_dict = h52dict(os.path.join(path_to_dump,filename))
    except FileNotFoundError:
        print('Load file not found: ', os.path.join(path_to_dump,filename))
        # exit(1)
        continue

    print('keys = ', load_dict.keys()) if __debug__ else 0
    omegavals = load_dict['omegavals']
    LDOS = np.array(load_dict['LDOS'])
    g3val = load_dict['gamma3']
    print(load_dict['INFO']) if __debug__ else 0
    for j, ax in enumerate(axlist):
        ax[0].plot(omegavals, LDOS, '.-', c=col, label = f'g3 = {g3val}')
        # ax[0].plot(omegavals, analytic, '--', c=col)
        ax[1].loglog(omegavals[omegavals<1],LDOS[omegavals<1],c=col, label = f'g3 = {g3val}')

        ax[0].set_xlabel('omega')
        ax[0].set_title(f'BLG scan gamma3 LDOS site Analytical two band ')
        ax[0].legend()
        ax[0].set_ylabel(r'G($\omega$)')
        ax[1].set_ylabel(r'G($\omega$)')
        ax[1].set_xlabel('omega')
        ax[1].legend()

for j in range(NUMGS):
    savefigname = f'BLGNLDOS_{j,j}.pdf'
    figlist[j].savefig(os.path.join(path_to_fig, savefigname))

plt.show()
