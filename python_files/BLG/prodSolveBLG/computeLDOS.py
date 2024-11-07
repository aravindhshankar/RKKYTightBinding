import sys 
import os
sys.path.insert(0,'..')
sys.path.insert(0,'../../')
import numpy as np
# from scipy.linalg import eigvals, eigvalsh
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
# from scipy.linalg import norm
from scipy.integrate import simpson, quad
from functools import partial
import time 
import multiprocessing as mp
from FastRGF.solveRGF import MOMfastrecDOSfull
from h5_handler import *
# import concurrent.futures
from dask.distributed import Client

savename = 'default_savename'
# path_to_dump = '../../Dump/BLG/solveLDOS'
path_to_output = '../../Output/BLG/solveLDOStest'
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

def generate_grid_with_peaks(a, b, peaks, peak_spacing=0.01, num_pp = 200, num_uniform=1000):
    # Sort the peaks list
    assert a < b , "a should be less than b" 
    peaks = sorted(peaks)
    
    # Generate grid around peaks
    peak_grid = np.concatenate([np.linspace(max(a, peak - peak_spacing), min(b, peak + peak_spacing), num = num_pp, dtype=np.double)
                                for peak in peaks])

    # Generate uniform grid for the remaining region
    # uniform_grid = np.linspace(max(a, min(peaks, default=a) + peak_spacing),
    #                            min(b, max(peaks, default=b) - peak_spacing), num=int((b - a) / uniform_spacing))
    uniform_grid = np.linspace(a,b,num=num_uniform,dtype=np.double)

    # Concatenate the peak and uniform grids
    grid = np.sort(np.concatenate([peak_grid, uniform_grid]))

    return grid


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
	kxvals = np.sort(np.concatenate((np.linspace(-0.05,0.05,5000,dtype=np.double), np.linspace(-np.pi,np.pi,1000,dtype=np.double))))
	dochecks = False
	delta = 5e-3 * omega
	RECURSIONS = 30
	dimH = 8
	start_time = time.perf_counter() if __debug__ else 0.0
	kDOS = np.array([MOMfastrecDOSfull(omega,ret_H0(kx),ret_Ty(kx),RECURSIONS,delta,dochecks) \
				for kx in kxvals],dtype=np.longdouble).reshape(len(kxvals),dimH,dimH) 
	if __debug__: 
		elapsed = time.perf_counter() - start_time
		print(f'Finished calculating kDOS for omega = {omega:.6f} in {elapsed} sec(s).')

	peaks = find_peaks(kDOS[:,idx_x,idx_y],prominence=0.01*np.max(kDOS[:,idx_x,idx_y]))[0]
	
	if __debug__:
		print(f'Peaks found on sparse grid : {len(peaks)}')

	peakvals = [kxvals[peak] for peak in peaks]
	call_int = lambda kx : MOMfastrecDOSfull(omega,ret_H0(kx),ret_Ty(kx),RECURSIONS,delta,dochecks)[idx_x,idx_y]

	################### starting integration after findind peaks #################
	start_time = time.perf_counter() if __debug__ else 0.0
	start,stop = -np.pi,np.pi
	ranges = []
	eta = 1e-5
	current = start
	for peak in peakvals:
		ranges += [(current, peak-eta)]
		ranges += [(peak-eta, peak+eta)]
		current = peak+eta
	ranges += [(current, stop)]		
	intlist = [quad(call_int,window[0],window[1],limit=5000,epsabs=0.01*delta)[0] for window in ranges]
	intval = np.sum(intlist)
	if __debug__: 
		elapsed = time.perf_counter() - start_time
		print(f'Finished integration for omega = {omega:.6f} in {elapsed} sec(s).')

	# print(f'intval = {intval:.6}')
	return intval

def dask_LDOS():
	# omegavals = np.logspace(np.log10(1e-5), np.log10(1e-1), num = int(2040))
	eps=1e-4
	# omegavals = (2e-2,2e-4)
	omegavals = np.sort(np.concatenate((np.logspace(np.log10(1e-4),np.log10(1e-2),500),np.linspace(1e-2+eps,5e-1,50))))
	# PROCESSES = mp.cpu_count()
	# PROCESSES = int(os.environ.get('SLURM_CPUS_PER_TASK','2'))
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
				'INFO' : '[0,0] site of -1/pi Im G, delta = 5e-3 * omega'
				}
	# dict2h5(savedict,'BLGAsiteLDOS.h5', verbose=True)
	savefileoutput = savename + '.h5'
	dict2h5(savedict,os.path.join(path_to_output,savefileoutput), verbose=True)


if __name__ == '__main__': 
	dask_LDOS()






