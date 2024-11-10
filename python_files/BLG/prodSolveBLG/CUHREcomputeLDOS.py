import sys 
import os
sys.path.insert(0,'..')
sys.path.insert(0,'../../')
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import time 
from FastRGF.solveRGF import MOMfastrecDOSfull
from dask.distributed import Client
from utils.h5_handler import *
from utils.decorators import cubify
import pycuba 

savename = 'default_savename'
path_to_output = '../../Output/BLG/CUHREsolveLDOS'
if not os.path.exists(path_to_output): 
	os.makedirs(path_to_output)
	print('Output directory created at ', path_to_output)

if len(sys.argv) > 1:
    savename = str(sys.argv[1])

print = partial(print, flush=True) #lol

## Maybe you want to rewrite this section as data members of a class that you can inherit from
## Or even add these data quantities to a separate module which can be imported

## energy quantities in units of eV
epsA1 = 0 
deltaprime = 0.022
epsA2 = deltaprime
epsB1 = deltaprime
epsB2 = 0
gamma0 = 3.16
gamma1 = 0.381
gamma3 = 0.38
gamma4 = 0.14



def ret_H0(kx):
	Tx = np.zeros((8,8),dtype=np.cdouble)
	Tx[0,5] = -gamma3
	Tx[0,6] = -gamma0
	Tx[0,7] = gamma4
	Tx[2,5] = gamma4
	Tx[3,5] = -gamma0
	Tx = Tx * np.exp(-1j*kx)
	#P = Tx + Tx.conj().T

	#Horrible code ahead : please don't judge

	M = np.zeros((8,8),dtype=np.cdouble)
	M[0,0] = epsB2
	M[0,1] = -gamma3
	M[0,2] = -gamma0
	M[0,3] = gamma1
	M[1,1] = epsA1
	M[1,2] = gamma4
	M[1,3] = -gamma0
	M[1,4] = -gamma3
	M[1,6] = gamma4
	M[1,7] = -gamma0
	M[2,2] = epsA2
	M[2,3] = gamma1
	M[2,4] = -gamma0
	M[3,3] = epsB1
	M[3,4] = gamma4
	M[4,4] = epsB2
	M[4,5] = -gamma3
	M[4,6] = -gamma0
	M[4,7] = gamma4
	M[5,5] = epsA1
	M[5,6] = gamma4
	M[5,7] = -gamma0
	M[6,6] = epsA2
	M[6,7] = gamma1
	M[7,7] = epsB1
	M = M + Tx
	M = M + M.conj().T - np.diag(np.diag(M))

	return M 


def ret_Ty(kx):
	# non-hermitian : only couples along -ve y direction
	Ty = np.zeros((8,8),dtype = np.cdouble)
	Ty[0,2] = -gamma0
	Ty[0,3] = gamma4
	Ty[0,5] = -gamma3 * np.exp(-1j*kx)
	Ty[1,2] = gamma4
	Ty[1,3] = -gamma0
	Ty[1,4] = -gamma3
	Ty[6,4] = -gamma0
	Ty[6,5] = gamma4
	Ty[7,4] = gamma4
	Ty[7,5] = -gamma0

	return Ty


def helper_LDOS_mp(omega):
	idx_x, idx_y = 0,0
	dochecks = False
	delta = 5e-3 * omega
	RECURSIONS = 30
	dimH = 8
	
	##### initialize cubify ######
	cubify.set_limits(-np.pi,np.pi)
	NDIM = 2
	KEY = 0
	MAXEVAL = int(1e6)
	VERBOSE = 0
	EPSREL = 1e-5
	
	@cubify.Cubify
	def call_int(kx): 
 		return MOMfastrecDOSfull(omega,ret_H0(kx),ret_Ty(kx),RECURSIONS,delta,dochecks)[idx_x,idx_y]

	################### starting integration #################
	start_time = time.perf_counter() if __debug__ else 0.0
	CUHREdict = pycuba.Cuhre(call_int, NDIM, key=KEY, maxeval=MAXEVAL, verbose=VERBOSE,epsrel=EPSREL) 
	# CUHREdict=  {'neval': 19955, 'fail': 0, 'comp': 0, 'nregions': 154, 'results': [{'integral': 0.002789452157117059, 'error': 2.63535369804456e-07, 'prob': 0.0}]}
	intval = CUHREdict['results'][0]['integral']
	if __debug__: 
		elapsed = time.perf_counter() - start_time
		print(f"Finished integration for omega = {omega:.6f} in {elapsed} sec(s) with fail = {CUHREdict['fail']}, neval = {CUHREdict['neval']}.")

	return intval

def dask_LDOS():
	# omegavals = np.logspace(np.log10(1e-5), np.log10(1e-1), num = int(2040))
	# omegavals = np.sort(np.concatenate((np.logspace(np.log10(1e-4),np.log10(1e-2),500),np.linspace(1e-2+eps,5e-1,50))))
	omegavals = [0.0003,0.003,0.03,0.3]

	PROCESSES = int(os.environ.get('SLURM_NTASKS','2'))
	print(f'PROCESSES = {PROCESSES}')
	client = Client(threads_per_worker=1, n_workers=PROCESSES)

	startmp = time.perf_counter()
	LDOS = client.gather(client.map(helper_LDOS_mp,omegavals))
	stopmp = time.perf_counter()

	elapsedmp = stopmp-startmp
	print(f'DASK parrallelization with {PROCESSES} processes finished in time {elapsedmp} seconds')
	
	savedict = {'omegavals' : omegavals,
				'LDOS' : LDOS,
				'INFO' : '[0,0] site of -1/pi Im G, delta = 5e-3 * omega, with EPSREL = {1e-5}'
				}
	savefileoutput = savename + '.h5'
	dict2h5(savedict,os.path.join(path_to_output,savefileoutput), verbose=True)


if __name__ == '__main__': 
	dask_LDOS()






