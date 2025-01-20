import sys
import os
sys.path.insert(0,'..')
sys.path.insert(0,'../..')
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from utils.h5_handler import *
from scipy.special import j0
# path_to_dump = '/Users/aravindhswaminathan/Documents/GitHub/RKKYTightBinding/python_files/Output/NewBLGturnoffg4LDOS/'
# path_to_fig = '/Users/aravindhswaminathan/Documents/GitHub/RKKYTightBinding/python_files/Figures/NewBLGturnoffg4LDOS'
path_to_dump = '../../Output/NEWBLGnldos/NegEnergies/'
path_to_fig = '../../Figures/NEWBLGnldos/NegEnergies/'
if not os.path.exists(path_to_dump): 
	raise exception('path to dump not found')
	exit(1)

if not os.path.exists(path_to_fig):
    print('Path to fig not found, creating ......')
    os.makedirs(path_to_fig)
    print('Created fig directory at ', path_to_fig)

def analyticNLDOS_graphene(omega,r,vF):
    ''' 
    Later put this in the class definition once factors of pi are resolved 
    '''
    pref = np.pi * omega / (vF**2)
    return pref * j0(np.pi*r*omega/vF) / np.pi #Spurious factors of pi as usual


NUMGS = 4 #number of dictionary entries in the load file

figlist = [plt.figure() for i in range(NUMGS)]
axlist = [figlist[i].subplots(3) for i in range(NUMGS)]

# jobarray = np.arange(0,20,4,dtype=int)
# jobarray = [0,5,12,15,20]
jobarray = [0,]
for i, job_idx in enumerate(jobarray):
    col = 'C' + str(i)
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
    # analytic = np.array([analyticNLDOS_graphene(omegaval,rval,vF) for omegaval in omegavals])
    print(load_dict['INFO']) if __debug__ else 0
    for j, ax in enumerate(axlist):
        ax[0].plot(omegavals, NLDOS.T[j], '.-', c=col, label = f'r = {rval}')
        # ax[0].plot(omegavals, analytic, '--', c=col)
        ax[1].loglog(omegavals[omegavals<1],NLDOS.T[j][omegavals<1],c=col, label = f'r = {rval}')
        ax[2].loglog(np.abs(omegavals)[omegavals<0],NLDOS.T[j][omegavals<0],c=col, label = f'r = {rval}')

        ax[0].set_xlabel('omega')
        ax[0].set_title(f'BLG NLDOS site [{j},{j}] ')
        ax[0].legend()
        ax[0].set_ylabel(r'G(r,$\omega$)')
        ax[1].set_ylabel(r'G(r,$\omega$)')
        ax[1].set_xlabel('omega')
        ax[1].legend()
        ax[2].set_ylabel(r'G(r,$\omega$)')
        ax[2].set_xlabel('-omega')
        ax[2].legend()

# for j in range(NUMGS):
    # savefigname = f'BLGNLDOS_{j,j}.pdf'
    # figlist[j].savefig(os.path.join(path_to_fig, savefigname))

plt.show()
