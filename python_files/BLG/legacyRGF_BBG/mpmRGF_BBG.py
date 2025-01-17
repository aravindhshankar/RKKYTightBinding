import sys 
import os
sys.path.insert(0,'..')
import numpy as np
import mpmath as mpm 
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.linalg import norm
from scipy.integrate import simpson, quad
from functools import partial
import time 
import multiprocessing as mp
# from FastRGF import MOMfastrecDOSfull
from FastRGF.mpmRGF import MOMfastrecDOSfull
from h5_handler import *
import concurrent.futures
path_to_dump = '../Dump/BLG'
if not os.path.exists(path_to_dump): 
	os.makedirs(path_to_dump)
	print('Dump directory created at ', path_to_dump)
mpm.mp.dps = 9
## Maybe you want to rewrite this section as data members of a class that you can inherit from
## Or even add these data quantities to a separate module which can be imported

## energy quantities in units of eV
epsA1 = mpm.mpf('0') 
deltaprime = mpm.mpf('0.022')
epsA2 = deltaprime
epsB1 = deltaprime
epsB2 = mpm.mpf('0')
gamma0 = mpm.mpf('3.16')
gamma1 = mpm.mpf('0.381')
gamma3 = mpm.mpf('0.38')
gamma4 = mpm.mpf('0.14')

kwargs = {  'epsA1': epsA1, 
			'epsA1': epsA1,
			'epsA2': epsA2,
			'epsB1': epsB1, 
			'epsB2': epsB2,
			'gamma0': gamma0, 
			'gamma1': gamma1, 
			'gamma3': gamma3, 
			'gamma4': gamma4
			}


def f(k):
	t1 = np.exp(1j*k[1]/np.sqrt(3))
	t2 = 2 * np.exp(-1j * k[1]/(2*np.sqrt(3))) * np.cos(k[0]/2.)
	return t1 + t2

def Ham_BLG(k): 
	ham = [ [epsA1, -gamma0*f(k), gamma4*f(k), -gamma3*np.conj(f(k))],
			[-gamma0*np.conj(f(k)), epsB1, gamma1, gamma4*f(k)],
			[gamma4*np.conj(f(k)), gamma1, epsA2, -gamma0*f(k)],
			[-gamma3*f(k), gamma4*np.conj(f(k)), -gamma0*np.conj(f(k)), epsB2] ]
	return np.array(ham)

def ret_H0(kx):
	Tx = mpm.zeros(8)
	Tx[0,5] = -gamma3
	Tx[0,6] = -gamma0
	Tx[0,7] = gamma4
	Tx[2,5] = gamma4
	Tx[3,5] = -gamma0
	Tx = Tx * mpm.exp(-1j*kx)

	M = mpm.zeros(8)
	M[0,1] = -gamma3
	M[0,2] = -gamma0
	M[0,3] = gamma1
	M[1,2] = gamma4
	M[1,3] = -gamma0
	M[1,4] = -gamma3
	M[1,6] = gamma4
	M[1,7] = -gamma0
	M[2,3] = gamma1
	M[2,4] = -gamma0
	M[3,4] = gamma4
	M[4,5] = -gamma3
	M[4,6] = -gamma0
	M[4,7] = gamma4
	M[5,6] = gamma4
	M[5,7] = -gamma0
	M[6,7] = gamma1
	M = M + Tx
	M = M + M.transpose_conj()
	M[0,0] = epsB2
	M[1,1] = epsA1
	M[2,2] = epsA2
	M[3,3] = epsB1
	M[4,4] = epsB2
	M[5,5] = epsA1
	M[6,6] = epsA2
	M[7,7] = epsB1
	
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
	Ty = mpm.zeros(8)
	Ty[0,2] = -gamma0
	Ty[0,3] = gamma4
	Ty[0,5] = -gamma3 * mpm.exp(-1j*kx)
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
	omega = mpm.mpf('2e-3')
	# omega = 2e-2
	omegavals = (omega,)
	kxvals = mpm.linspace(-mpm.pi,mpm.pi,10)
	delta = mpm.mpf('1e-6')
	RECURSIONS = 20
	dimH = 8
	# kDOS = mpm.matrix([MOMfastrecDOSfull(omega,ret_H0(kx),ret_Ty(kx),RECURSIONS,delta)
	# 				for omega in omegavals for kx in kxvals]).reshape((len(omegavals),len(kxvals),dimH,dimH))
	print("Started computing kDOS")
	start_time = time.perf_counter()
	kDOS = [MOMfastrecDOSfull(omega,ret_H0(kx),ret_Ty(kx),RECURSIONS,delta)[0,0] for kx in kxvals]
	elapsed = time.perf_counter() - start_time
	print(f"Finished computing list comprehension kDOS in {elapsed} sec(s). ")
	peaks = find_peaks(kDOS,prominence=0.1*max(kDOS))[0]
	peakvals = [kxvals[peak] for peak in peaks]
	fig, ax = plt.subplots(1)
	ax.plot(kxvals, kDOS)
	ax.set_xlabel(r'$k_x$')
	# ax.set_title(f'$\\omega$ = {omega:.3}, \\delta = {delta:.6}, RECURSIONS = {RECURSIONS}')
	# ax.vlines(kxvals[peaks], ls = '--', c = 'grey')
	for peak in peakvals:
		ax.axvline(peak,ls='--',c='gray')
	# ax.legend()
	print('Started MPM quad integrate without peaks')
	start_time = time.perf_counter()
	intval = mpm.quad(lambda kx : MOMfastrecDOSfull(omega,ret_H0(kx),ret_Ty(kx),RECURSIONS,delta,)[0,0], 
						[-mpm.pi,mpm.pi],error=True)
	elapsed = time.perf_counter() - start_time
	print(f'Finished quad integrator with delta = {delta:.6} and {RECURSIONS} recursions in {elapsed} sec(s).')
	# print(f'intval = {intval:.5}')


	# print('Started quad integrate WITH peaks')
	# start_time = time.perf_counter()
	# intval = quad(lambda kx : MOMfastrecDOSfull(omega,ret_H0(kx),ret_Ty(kx),RECURSIONS,delta,)[0,0], 
	# 					-np.pi,np.pi, points = [kxvals[peak] for peak in peaks])[0]
	# elapsed = time.perf_counter() - start_time
	# print(f'Finished quad integrator with delta = {delta:.6} and {RECURSIONS} recursions in {elapsed} sec(s).')
	# print(f'intval = {intval:.5}')

	# for num_pp in [1000]: #checking convergence
	# 	print('Started simpson integrate WITH peaks')
	# 	peak_spacing = 0.05
	# 	print(f'num_pp = {num_pp}, peak_spacing = {peak_spacing:.4}')
	# 	start_time = time.perf_counter()
	# 	adaptive_kxgrid = generate_grid_with_peaks(-np.pi,np.pi,peakvals,peak_spacing=0.01,num_uniform=10000,num_pp=num_pp)
	# 	fine_integrand = np.array([MOMfastrecDOSfull(omega,ret_H0(kx),ret_Ty(kx),RECURSIONS,delta)[0,0] for kx in adaptive_kxgrid],dtype=np.double)
	# 	simpson_intval = simpson(fine_integrand,adaptive_kxgrid)
	# 	elapsed = time.perf_counter() - start_time
	# 	print(f'Finished simpson integrator with delta = {delta:.6} and {RECURSIONS} recursions in {elapsed} sec(s).')
	# 	print(f'intval = {simpson_intval:.8}')
	# 	ax.plot(adaptive_kxgrid,fine_integrand,'.',c='red')
	# print(type(fine_integrand[0]))
	plt.show()


def helper_LDOS_mp(omega,delta,RECURSIONS,analyze=False,method = 'adaptive'):
	callintegrand = lambda kx: MOMfastrecDOSfull(omega,ret_H0(kx),ret_Ty(kx),RECURSIONS,delta)[0,0]
	if analyze == True:
		kxgrid = np.linspace(-np.pi,np.pi,1000)
		sparseLDOS = np.array([callintegrand(kx) for kx in kxgrid])
		peaks = find_peaks(sparseLDOS,prominence=0.1*np.max(sparseLDOS))[0]
		breakpoints = [kxgrid[peak] for peak in peaks] #peakvals
		if method == 'quad':
			LDOS = quad(callintegrand,-np.pi,np.pi,limit=100,points=breakpoints,epsabs=delta)[0] 
		elif method == 'adaptive': 
			adaptive_kxgrid = generate_grid_with_peaks(-np.pi,np.pi,breakpoints,peak_spacing=0.005,num_uniform=500,num_pp=300)
			fine_integrand = [MOMfastrecDOSfull(omega,ret_H0(kx),ret_Ty(kx),RECURSIONS,delta,)[0,0] for kx in adaptive_kxgrid]
			LDOS = simpson(fine_integrand,adaptive_kxgrid)
		else: 
			raise Exception('Unkown method for integration')
			exit(1)
	else: 
		LDOS = quad(callintegrand,-np.pi,np.pi,limit=50)[0] 
	return LDOS



def test_LDOS_mp():
	'''
	Use scipy.quad for this
	'''
	RECURSIONS = 25
	delta = 1e-6
	# omegavals = np.linspace(0,3.1,512)
	# omegavals = np.linspace(0,3.1,100)
	# omegavals = make_omega_grid()
	omegavals = np.logspace(np.log10(1e-6), np.log10(1e0), num = 100)

	PROCESSES = mp.cpu_count()
	startmp = time.perf_counter()
	# with mp.Pool(PROCESSES) as pool:
	# 	LDOS = pool.map(helper_LDOS_mp, omegavals)
	with mp.Pool(PROCESSES) as pool:
			LDOS = pool.map(partial(helper_LDOS_mp,delta=delta,RECURSIONS=RECURSIONS,analyze=True,method='adaptive'), omegavals)
	stopmp = time.perf_counter()
	elapsedmp = stopmp-startmp
	print(f'Parallel computation with {PROCESSES} processes finished in time {elapsedmp} seconds')
	
	savedict = {'omegavals' : omegavals,
				'LDOS' : LDOS,
				'INFO' : '[0,0] site of -1/pi Im G'
				}
	# dict2h5(savedict,'BLGAsiteLDOS.h5', verbose=True)

	fig,ax = plt.subplots(1)
	ax.plot(omegavals, LDOS, '.-', label = 'quad LDOS')
	# ax.axvline(1., ls='--', c='grey')
	ax.set_xlabel('omega')
	ax.set_title(f'Bilayer Graphene LDOS A site with $\\delta = $ {delta:.6}')
	plt.show()






if __name__ == '__main__': 
	test_Ginfkx()
	# test_LDOS_mp()










