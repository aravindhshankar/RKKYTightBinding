import sys
import os
sys.path.insert(0,'..')
sys.path.insert(0,'../..')
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from utils.h5_handler import *
from scipy.special import j0
# path_to_dump = '../Output/BLG/solveLDOStest/'
path_to_dump = '/Users/aravindhswaminathan/Documents/GitHub/RKKYTightBinding/python_files/Output/nldos'
if not os.path.exists(path_to_dump): 
	raise exception('path to dump not found')
	exit(1)


def analyticNLDOS_graphene(omega,r,vF):
    ''' 
    Later put this in the class definition once factors of pi are resolved 
    '''
    pref = np.pi * omega / (vF**2)
    return pref * j0(np.pi*r*omega/vF) / np.pi #Spurious factors of pi as usual





fig,ax = plt.subplots(2)

# jobarray = np.arange(0,20, dtype=int)
jobarray = np.arange(0,20,4,dtype=int)
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
    vF = 1.5
    analytic = np.array([analyticNLDOS_graphene(omegaval,rval,vF) for omegaval in omegavals])
    print(load_dict['INFO']) if __debug__ else 0
    ax[0].plot(omegavals, NLDOS, '.-', c=col, label = f'r = {rval}')
    ax[0].plot(omegavals, analytic, '--', c=col)
    ax[1].loglog(omegavals[omegavals<1],NLDOS[omegavals<1],c=col, label = f'r = {rval}')

ax[0].set_xlabel('omega')
ax[0].set_title('graphene NLDOS A site , ')
ax[0].legend()
ax[0].set_ylabel(r'G(r,$\omega$)')
ax[1].set_ylabel(r'G(r,$\omega$)')
ax[1].set_xlabel('omega')
ax[1].legend()

plt.show()
