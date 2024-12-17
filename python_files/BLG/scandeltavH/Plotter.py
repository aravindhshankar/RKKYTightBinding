###### Learning : for BLG, delta = 5e-3 * omega should be sufficient
import sys
import os
sys.path.insert(0,'..')
sys.path.insert(0,'../..')
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from utils.h5_handler import *
from scipy.special import j0
from scipy.integrate import simpson
# path_to_dump = '/Users/aravindhswaminathan/Documents/GitHub/RKKYTightBinding/python_files/Output/scang3/NewBLGTdag/'
# path_to_fig = '/Users/aravindhswaminathan/Documents/GitHub/RKKYTightBinding/python_files/Figures/scang3/NewBLGTdag/'
path_to_dump = '/Users/aravindhswaminathan/Documents/GitHub/RKKYTightBinding/python_files/Output/scandeltaVH/'
path_to_fig = '/Users/aravindhswaminathan/Documents/GitHub/RKKYTightBinding/python_files/Figures/scandeltaVH/'
if not os.path.exists(path_to_dump): 
	raise exception('path to dump not found')
	exit(1)

if not os.path.exists(path_to_fig): 
    print("Fig directory not found, creating,....")
    os.makedirs(path_to_fig)


NUMGS = 4 #number of dictionary entries in the load file

figlist = [plt.figure() for i in range(NUMGS)]
axlist = [figlist[i].subplots(2) for i in range(NUMGS)]

jobarray = np.arange(0,10,1,dtype=int)
# jobarray = [0,5,12,15,20]
# jobarray = [5,]
areas = np.array([np.zeros_like(jobarray,dtype=np.float64) for i in range(NUMGS)]).T
for i, job_idx in enumerate(jobarray):
    col = 'C' + str(i)
    filename = f'results_delta_{job_idx}.h5'
    try:
        load_dict = h52dict(os.path.join(path_to_dump,filename))
    except FileNotFoundError:
        print('Load file not found: ', os.path.join(path_to_dump,filename))
        # exit(1)
        continue

    print('keys = ', load_dict.keys()) if __debug__ else 0
    omegavals = load_dict['omegavals']
    LDOS = np.array(load_dict['LDOS'])
    deltaval = load_dict['delta']
    ####### Total spectral weight ######

    print(load_dict['INFO']) if __debug__ else 0
    for j, ax in enumerate(axlist):
        np.testing.assert_equal(len(LDOS.T[j]), len(omegavals))
        areas[i][j] = simpson(y = LDOS.T[j], x = omegavals)
        print(simpson(y = LDOS.T[j], x = omegavals))
        print(areas[i][j])
        ax[0].plot(omegavals, LDOS.T[j], '.-', c=col, label = f'delta = {deltaval}')
        # ax[0].plot(omegavals, analytic, '--', c=col)
        ax[1].loglog(omegavals[omegavals<1],LDOS.T[j][omegavals<1],c=col, label = f'delta = {deltaval}')

        ax[0].set_xlabel('omega')
        ax[0].set_title(f'BLG scan delta LDOS site [{j},{j}] ')
        ax[0].legend()
        ax[0].set_ylabel(r'G($\omega$)')
        ax[1].set_ylabel(r'G($\omega$)')
        ax[1].set_xlabel('omega')
        ax[1].legend()

for j in range(NUMGS):
    savefigname = f'BLGNLDOS_{j,j}.pdf'
    figlist[j].savefig(os.path.join(path_to_fig, savefigname))


print(areas)
print(f'Areas = ', areas[:,0])
print(f'combined areas = ', np.sum(areas,axis=1))




plt.show()
