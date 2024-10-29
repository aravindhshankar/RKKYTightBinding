import sys 
import os
sys.path.insert(0,'..')
import numpy as np
# from scipy.linalg import eigvals, eigvalsh
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
# from scipy.linalg import norm
from scipy.integrate import simpson, quad
from functools import partial
import time 
import multiprocessing as mp
# from FastRGF import MOMfastrecDOSfull
from FastRGF.solveRGF import MOMfastrecDOSfull
from h5_handler import *
# import concurrent.futures
from dask.distributed import Client

savename = 'default_savename'
path_to_output = '../Outputs/BLG/'
path_to_dump = '../Dump/BLG'
if not os.path.exists(path_to_dump): 
	os.makedirs(path_to_dump)
	print('Dump directory created at ', path_to_dump)

if not os.path.exists(path_to_output):
    os.makedirs(path_to_output)
    print("Outputs directory created at ", path_to_output)

if len(sys.argv) > 1:
    savename = str(sys.argv[1])

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

def test_Ginfkx():
	# omega = 1 - 1e-2
	omega = 2e-3 
	# omega = 0.000444967
	# omega = 0.000740532
	# omega = 2e-2
	omegavals = (omega,)
	# kxvals = np.linspace(-np.pi,np.pi,10000,dtype=np.double)
	kxvals = np.linspace(-0.05,0.05,5000,dtype=np.double)
	# delta = min(1e-4,0.01*omega)
	# delta = 1e-4 if omega>1e-3 else 1e-6
	# delta = 1e-5
	dochecks = False
	delta = 5e-3 * omega
	# delta = 0.01*omega
	RECURSIONS = 30
	dimH = 8
	num_pp = 2000
	start, stop = kxvals[0], kxvals[-1]
	start_time = time.perf_counter()
	kDOS = np.array([MOMfastrecDOSfull(omega,ret_H0(kx),ret_Ty(kx),RECURSIONS,delta,dochecks)
					for omega in omegavals for kx in kxvals],dtype=np.longdouble).reshape((len(omegavals),len(kxvals),dimH,dimH))
	elapsed = time.perf_counter() - start_time
	print(f'Finished calculating kDOS in {elapsed} sec(s).')
	# peaks = find_peaks(kDOS[0,:,0,0],prominence=0.01*np.max(kDOS[0,:,0,0]))[0]
	peaks = find_peaks(kDOS[0,:,0,0],prominence=0.1*np.max(kDOS[0,:,0,0]))[0]
	print(f'Peaks found on sparse grid : {len(peaks)}')
	peakvals = [kxvals[peak] for peak in peaks]
	print("creating fine grid")
	# adaptive_kxgrid = generate_grid_with_peaks(start,stop,peakvals,peak_spacing=0.01,num_uniform=1000,num_pp=num_pp)
	# fine_integrand = np.array([MOMfastrecDOSfull(omega,ret_H0(kx),ret_Ty(kx),RECURSIONS,delta)[0,0] for kx in adaptive_kxgrid],dtype=np.double)
	# 
	# print('finding additional peaks')
	# # new_peaks = find_peaks(fine_integrand,prominence=0.01*np.max(fine_integrand))[0]
	# new_peaks = find_peaks(fine_integrand,prominence=0.1*np.max(kDOS[0,:,0,0]))[0]
	# new_peakvals = [adaptive_kxgrid[peak] for peak in new_peaks]
	# new_peakvals = sorted(new_peakvals)

	fig, ax = plt.subplots(1)
	ax.plot(kxvals, kDOS[0,:,0,0])
	ax.set_xlabel(r'$k_x$')
	ax.set_title(f'$\\omega$ = {omega:.3}, \\delta = {delta:.6}, RECURSIONS = {RECURSIONS}')
	# ax.vlines(kxvals[peaks], ls = '--', c = 'grey')
	# for peak in new_peakvals:
		# ax.axvline(peak,ls='--',c='gray')

	# ax.legend()
	# print('Started quad integrate without peaks')
	# start_time = time.perf_counter()
	# intval = quad(lambda kx : MOMfastrecDOSfull(omega,ret_H0(kx),ret_Ty(kx),RECURSIONS,delta,)[0,0], 
	# 					-np.pi,np.pi)[0]
	# elapsed = time.perf_counter() - start_time
	# print(f'Finished quad integrator with delta = {delta:.6} and {RECURSIONS} recursions in {elapsed} sec(s).')
	# print(f'intval = {intval:.5}')

	# print(f'Total number of peaks found = {len(new_peakvals)}')

	# print('Started quad integrate WITH peaks')
	# start_time = time.perf_counter()
	# call_int = lambda kx : MOMfastrecDOSfull(omega,ret_H0(kx),ret_Ty(kx),RECURSIONS,delta)[0,0]

	# start,stop = -np.pi,np.pi
	# ranges = []
	# eta = 0.5*delta
	# # eta = 1e-3
	# current = start
	# for peak in new_peakvals:
	# 	ranges += [(current, peak-eta)]
	# 	current = peak+eta
	# ranges += [(current, stop)]		
	# intlist = [quad(call_int,window[0],window[1],limit=500,epsabs=0.1*delta)[0] for window in ranges]
	# for word in list(zip(ranges,intlist)):
	# 	print(word)
	# intval = np.sum(intlist)

	# # intval = quad(call_int, -np.pi,np.pi, points = [kxvals[peak] for peak in peaks])[0]
	# elapsed = time.perf_counter() - start_time
	# print(f'Finished quad integrator with delta = {delta:.6} and {RECURSIONS} recursions in {elapsed} sec(s).')
	# print(f'intval = {intval:.5}')

	# print('STARTED SIMPSON INTEGRATION')
	# start_time = time.perf_counter()
	# simpson_intval = simpson(fine_integrand,adaptive_kxgrid)
	# elapsed = time.perf_counter() - start_time
	# print(f'Finished simpson integrator with delta = {delta:.6} and {RECURSIONS} recursions in {elapsed} sec(s).')
	# print(f'intval = {simpson_intval:.8}')



	# for num_pp in [2000]: #checking convergence
	# 	print('Started simpson integrate WITH peaks')
	# 	peak_spacing = 0.01
	# 	print(f'num_pp = {num_pp}, peak_spacing = {peak_spacing:.4}')
	# 	start_time = time.perf_counter()
	# 	adaptive_kxgrid = generate_grid_with_peaks(-np.pi,np.pi,new_peakvals,peak_spacing=0.1,num_uniform=10000,num_pp=num_pp)
	# 	fine_integrand = np.array([MOMfastrecDOSfull(omega,ret_H0(kx),ret_Ty(kx),RECURSIONS,delta)[0,0] for kx in adaptive_kxgrid],dtype=np.double)
	# 	simpson_intval = simpson(fine_integrand,adaptive_kxgrid)
	# 	elapsed = time.perf_counter() - start_time
	# 	print(f'Finished simpson integrator with delta = {delta:.6} and {RECURSIONS} recursions in {elapsed} sec(s).')
	# 	print(f'intval = {simpson_intval:.8}')
	# print(type(fine_integrand[0]))
	# ax.plot(adaptive_kxgrid,fine_integrand,'.',c='red')
	plt.show()
	# print(adaptive_kxgrid[fine_integrand<0])


def helper_LDOS_mp(omega):
	RECURSIONS = 20
	delta = 1e-4 if omega>1e-3 else 1e-6
	num_pp = 1000
	call_int = lambda kx : MOMfastrecDOSfull(omega,ret_H0(kx),ret_Ty(kx),RECURSIONS,delta)[1,1]
	start_time = time.perf_counter()
	print(f'omega = {omega:.6} entered',flush=True)
	kxgrid = np.linspace(-np.pi,np.pi,10000,dtype=np.double)
	# sparseLDOS = np.array([MOMfastrecDOSfull(omega,ret_H0(kx),ret_Ty(kx),RECURSIONS,delta)
	# 				for kx in kxvals],dtype=np.longdouble).reshape((len(kxvals),dimH,dimH))
	sparseLDOS = np.array([call_int(kx) for kx in kxgrid],dtype=np.double)
	peaks = find_peaks(sparseLDOS,prominence=0.01)[0]
	peakvals = [kxgrid[peak] for peak in peaks] #peakvals
	peakvals = sorted(peakvals)

	if len(peakvals) > 20:
		return float('NaN')

	adaptive_kxgrid = generate_grid_with_peaks(-np.pi,np.pi,peakvals,peak_spacing=0.01,num_uniform=10000,num_pp=num_pp)
	fine_integrand = np.array([call_int(kx) for kx in adaptive_kxgrid],dtype=np.double)
	new_peaks = find_peaks(fine_integrand,prominence=0.01)[0]
	new_peakvals = [adaptive_kxgrid[peak] for peak in new_peaks]
	new_peakvals = sorted(new_peakvals)

	if len(new_peakvals) > 25:
		return float('NaN')

	start,stop = -np.pi,np.pi
	ranges = []
	# eta = 0.5*delta
	eta = 0.5*delta
	current = start
	for peak in new_peakvals:
		ranges += [(current, peak-eta)]
		current = peak+eta
	ranges += [(current, stop)]		
	intlist = [quad(call_int,window[0],window[1],limit=500,epsabs=0.1*delta)[0] for window in ranges]
	LDOS = np.sum(intlist)
	elapsed = time.perf_counter() - start_time
	print(f'omega = {omega:.6} finished in {elapsed} seconds.',flush=True)
	return LDOS

def dask_LDOS():
	RECURSIONS = 20
	delta = 1e-4
	omegavals = np.logspace(np.log10(1e-5), np.log10(1e-1), num = int(2040))

	# PROCESSES = mp.cpu_count()
	# PROCESSES = int(os.environ['SLURM_CPUS_PER_TASK'])
	# PROCESSES = int(32)
	PROCESSES = 24

	print(f'PROCESSES = {PROCESSES}')
	client = Client(threads_per_worker=1, n_workers=PROCESSES)

	startmp = time.perf_counter()
	LDOS = client.gather(client.map(helper_LDOS_mp,omegavals))
	stopmp = time.perf_counter()

	elapsedmp = stopmp-startmp
	print(f'DASK parrallelization with {PROCESSES} processes finished in time {elapsedmp} seconds')
	
	savedict = {'omegavals' : omegavals,
				'LDOS' : LDOS,
				'INFO' : '[1,1] site of -1/pi Im G, delta = 1e-4 if omega>1e-3 else 1e-6, RECURSIONS = 25'
				}
	# dict2h5(savedict,'BLGAsiteLDOS.h5', verbose=True)
	savefileoutput = savename + '.h5'
	dict2h5(savedict,os.path.join(path_to_output,savefileoutput), verbose=True)

def test_LDOS_mp():
	'''
	Use scipy.quad for this
	'''
	RECURSIONS = 25
	delta = 1e-4
	omegavals = np.logspace(np.log10(1e-6), np.log10(1e0), num = int(8))

	# PROCESSES = mp.cpu_count()
	PROCESSES = int(os.environ['SLURM_CPUS_PER_TASK'])
	print(f'PROCESSES = {PROCESSES}')
	startmp = time.perf_counter()
	with mp.Pool(PROCESSES) as pool:
			LDOS = pool.map(partial(helper_LDOS_mp,delta=delta,RECURSIONS=RECURSIONS), omegavals)
			# r = pool.map_async(partial(helper_LDOS_mp,delta=delta,RECURSIONS=RECURSIONS,analyze=True,method='adaptive'), omegavals)
			# LDOS = r.get()
	stopmp = time.perf_counter()
	elapsedmp = stopmp-startmp
	print(f'Parallel computation with {PROCESSES} processes finished in time {elapsedmp} seconds')
	
	savedict = {'omegavals' : omegavals,
				'LDOS' : LDOS,
				'INFO' : '[0,0] site of -1/pi Im G, delta = 1e-4 if omega>1e-3 else 1e-6, RECURSIONS = 25'
				}
	# dict2h5(savedict,'BLGAsiteLDOS.h5', verbose=True)
	savefileoutput = savename + '.h5'
	# dict2h5(savedict,os.path.join(path_to_output,savefileoutput), verbose=True)

	# fig,ax = plt.subplots(1)
	# ax.plot(omegavals, LDOS, '.-', label = 'quad LDOS')
	# # ax.axvline(1., ls='--', c='grey')
	# ax.set_xlabel('omega')
	# ax.set_title(f'Bilayer Graphene LDOS A site with $\\delta = $ {delta:.6}')
	# plt.show()



def test_Gk_single():
	# kx =  0.0008039 
	kx =  0.00073387
	omega = 2e-3
	RECURSIONS = 30
	delta = 1e-6
	DOSkx = MOMfastrecDOSfull(omega,ret_H0(kx),ret_Ty(kx),RECURSIONS,delta)[0,0]
	print(DOSkx)

	
if __name__ == '__main__': 
	test_Ginfkx()
	# dask_LDOS()
	# test_LDOS_mp()
	# test_Gk_single()









