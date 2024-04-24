import sys
import os
sys.path.insert(0,'..')
import numpy as np
from scipy.linalg import eigvals, eigvalsh
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
# from scipy.linalg import norm
from scipy.integrate import simpson, quad
from functools import partial
import time 
import multiprocessing as mp
# from FastRGF import MOMfastrecDOSfull
from FastRGF.RGF import MOMfastrecDOSfull
from h5_handler import *
import concurrent.futures
from scipy.fft import ifft, fftshift
path_to_dump = '../Dump/Graphene'
if not os.path.exists(path_to_dump): 
	os.makedirs(path_to_dump)
	print('Dump directory created at ', path_to_dump)

epsB = 0.
epsA = 0.
t = 1. 
a = 1. 



def ret_H0(kx): 
	P = np.array([[0,0,0,-np.conj(t)*np.exp(-1j*kx*a)],[0,0,0,0],[0,0,0,0],[-t*np.exp(1j*kx*a),0,0,0]]) 
	# np.testing.assert_almost_equal(P,P.conj().T)
	M = np.array([[epsB,-np.conj(t),0,0],[-t,epsA,-t,0],[0,-np.conj(t),epsB,-t],[0,0,-np.conj(t),epsA]])
	return M + P


def ret_Ty():
	return np.array([[0,-t,0,0],[0,0,0,0],[0,0,0,0],[0,0,-t,0]]) #Right hopping matrix along Y


def make_omega_grid():
	# Logarithmic spacing from 1e-4 to 5e-1
	log_space1 = np.logspace(np.log10(1e-6), np.log10(5e-1), num=20)

	# Linear spacing from 5e-1 to 9e-1
	linear_space1 = np.linspace(5e-1, 9e-1, num=20)

	# Logarithmic spacing from 9e-1 to 1.1
	log_space2 = np.logspace(np.log10(9e-1), np.log10(1.1), num=30)

	# Linear spacing from 1.1 to 3
	linear_space2 = np.linspace(1.1, 3, num=30)

	# Concatenating the arrays
	grid = np.concatenate((log_space1, linear_space1, log_space2, linear_space2))

	return grid

def test_omega_grid():
	omegavals = make_omega_grid()
	print(f'size of grid = {len(omegavals)}')
	print(omegavals)




def test_Ginfomega():
	kx = 0.5
	kxvals = (0.5,)
	delta = 0.0001
	omegavals = np.linspace(-3.1,3.1,1000) 
	dimH = 4
	G = np.array([fastrecGfull(omega,kx) 
					for omega in omegavals for kx in kxvals]).reshape((len(omegavals),len(kxvals),dimH,dimH))
	DOS = np.array([-1./np.pi * G[:,0,i,i].imag for i in range(dimH)])

	fig, ax = plt.subplots(1)
	for i in range(dimH):
		ax.plot(omegavals,DOS[i], label=f'index {i}')
	ax.set_xlabel(r'$\omega$')
	ax.set_title(f'$k_x$ = {kx:.3}')
	ax.legend()

	plt.show()

def test_Ginfkx():
	omega = 1 - 1e-2
	omegavals = (omega,)
	kxvals = np.linspace(-np.pi,np.pi,1000)
	delta = 1e-8
	RECURSIONS = 30
	dimH = 4
	kDOS = np.array([MOMfastrecDOSfull(omega,ret_H0(kx),ret_Ty(),RECURSIONS,delta)
					for omega in omegavals for kx in kxvals]).reshape((len(omegavals),len(kxvals),dimH,dimH))
	peaks = find_peaks(kDOS[0,:,0,0],prominence=0.1*np.max(kDOS[0,:,0,0]))[0]
	fig, ax = plt.subplots(1)
	ax.plot(kxvals, kDOS[0,:,0,0])
	ax.set_xlabel(r'$k_x$')
	ax.set_title(f'$\\omega$ = {omega:.3}, \\delta = {delta:.6}, RECURSIONS = {RECURSIONS}')
	# ax.vlines(kxvals[peaks], ls = '--', c = 'grey')
	for peak in peaks:
		ax.axvline(kxvals[peak],ls='--',c='gray')
	# ax.legend()

	print('Started quad integrate without peaks')
	start_time = time.perf_counter()
	intval = quad(lambda kx : MOMfastrecDOSfull(omega,ret_H0(kx),ret_Ty(),RECURSIONS,delta,)[0,0], 
						-np.pi,np.pi)[0]
	elapsed = time.perf_counter() - start_time
	print(f'Finished quad integrator with delta = {delta:.6} and {RECURSIONS} recursions in {elapsed} sec(s).')
	print(f'intval = {intval:.5}')


	print('Started quad integrate WITH peaks')
	start_time = time.perf_counter()
	intval = quad(lambda kx : MOMfastrecDOSfull(omega,ret_H0(kx),ret_Ty(),RECURSIONS,delta,)[0,0], 
						-np.pi,np.pi, points = [kxvals[peak] for peak in peaks])[0]
	elapsed = time.perf_counter() - start_time
	print(f'Finished quad integrator with delta = {delta:.6} and {RECURSIONS} recursions in {elapsed} sec(s).')
	print(f'intval = {intval:.5}')

	plt.show()


def compare_integrate():
	omega = 1.
	omegavals = (omega,)
	kxvals = np.linspace(-np.pi,np.pi,1000)
	delta = 0.0001
	dimH = 4
	startsimps = time.time()
	G = np.array([fastrecGfull(omega,kx) 
					for omega in omegavals for kx in kxvals]).reshape((len(omegavals),len(kxvals),dimH,dimH))
	DOS = np.array([-1./np.pi * G[0,:,i,i].imag for i in range(dimH)]) 
	integrand = DOS[0] #-ImG(kx), we're doing now the LDOS
	simpsint = simpson(integrand,kxvals)
	stopsimps = time.time()

	startquad = time.time()
	callintegrand = lambda kx: -1./np.pi * fastrecGfull(omega,kx)[0,0].imag
	quadint = quad(callintegrand, -np.pi,np.pi)
	stopquad = time.time()

	print(f'simpson with {len(kxvals)} points = {simpsint:.8} finished in {(stopsimps-startsimps):.8} sec')
	# print(f'quadint = {quadint:.5}')
	print('quad = ', quadint, f' finished in {(stopquad - startquad):.8} sec')


def helper_LDOS_mp(omega,delta,RECURSIONS,analyze=False):
	callintegrand = lambda kx: MOMfastrecDOSfull(omega,ret_H0(kx),ret_Ty(),RECURSIONS,delta)[0,0]
	if analyze == True:
		kxgrid = np.linspace(-np.pi,np.pi,1000)
		sparseLDOS = np.array([callintegrand(kx) for kx in kxgrid])
		peaks = find_peaks(sparseLDOS,prominence=0.1*np.max(sparseLDOS))[0]
		breakpoints = [kxgrid[peak] for peak in peaks]
		LDOS = quad(callintegrand,-np.pi,np.pi,limit=100,points=breakpoints,epsabs=delta)[0] 
	else: 
		LDOS = quad(callintegrand,-np.pi,np.pi,limit=50)[0] 
	return LDOS



def test_LDOS_mp():
	'''
	Use scipy.quad for this
	'''
	RECURSIONS = 20
	delta = 1e-4
	# omegavals = np.linspace(0,3.1,512)
	# omegavals = np.linspace(0,3.1,100)
	omegavals = make_omega_grid()

	PROCESSES = mp.cpu_count()
	startmp = time.perf_counter()
	# with mp.Pool(PROCESSES) as pool:
	# 	LDOS = pool.map(helper_LDOS_mp, omegavals)
	with mp.Pool(PROCESSES) as pool:
			LDOS = pool.map(partial(helper_LDOS_mp,delta=delta,RECURSIONS=RECURSIONS,analyze=False), omegavals)
	stopmp = time.perf_counter()
	elapsedmp = stopmp-startmp
	print(f'Parallel computation with {PROCESSES} processes finished in time {elapsedmp} seconds')
	
	savedict = {'omegavals' : omegavals,
				'LDOS' : LDOS,
				'RECURSIONS' : RECURSIONS, 
				'delta' : delta,
				'INFO' : '[0,0] site of -1/pi Im G'
				}
	savepath = os.path.join(path_to_dump, 'FunkGridGrapheneAsiteLDOS.h5')
	dict2h5(savedict,savepath, verbose=True)

	fig,ax = plt.subplots(1)
	ax.plot(omegavals, LDOS, '.-', label = 'quad LDOS')
	ax.axvline(1., ls='--', c='grey')
	ax.set_xlabel('omega')
	ax.set_title(f'Graphene LDOS A site with $\\delta = $ {delta:.6}')
	plt.show()



def test_LDOS_LC():
	RECURSIONS = 40
	delta = 1e-6
	# omegavals = np.linspace(0,3.1,10)
	# DOS = np.array([MOMfastrecDOSfull(omega,ret_H0(kx),ret_Ty(),RECURSIONS,delta) 
	# 				for omega in omegavals for kx in kxvals]).resize((len(omegavals),len(kxvals),dimH,dimH))
	start_time = time.perf_counter()
	LDOS = [quad(lambda kx: MOMfastrecDOSfull(omega,ret_H0(kx),ret_Ty(),RECURSIONS,delta)[0,0],
						 -np.pi,np.pi)[0] for omega in omegavals]
	elapsed = time.perf_counter() - start_time
	print(f'List Comprehension with {len(omegavals)} points finished in {elapsed} sec(s). ')

	fig, ax =plt.subplots(1)
	ax.plot(omegavals, LDOS, '.-') 
	ax.set_xlabel('$\\omega$')
	ax.set_ylabel('LDOS')
	ax.set_title(f'List Comprehension with {len(omegavals)} points finished in {elapsed} sec(s).') 
	plt.show()




def test_LDOS_threads():
	'''
	Even slower lol
	'''
	omegavals = np.linspace(0,3.1,20)
	# callintegrand = lambda kx, omega: -1./np.pi * fastrecGfull(omega,kx)[0,0].imag
	# LDOS = quad(partial(callintegrand,omega=om),-np.pi,np.pi)[0] for om in omegavals
	startmp = time.time()
	with concurrent.futures.ThreadPoolExecutor() as pool:
		LDOS = list(pool.map(helper_LDOS_mp, omegavals))
	stopmp = time.time()
	elapsedmp = stopmp-startmp
	print(f'Parallel computation with Threading finished in time {elapsedmp} seconds')
	
	savedict = {'omegavals' : omegavals,
				'LDOS' : LDOS,
				'INFO' : '[0,0] site of -1/pi Im G'
				}

	fig,ax = plt.subplots(1)
	ax.plot(omegavals, LDOS, label = 'quad LDOS')
	ax.axvline(1., ls='--', c='grey')
	ax.set_xlabel('omega')
	ax.set_title('Graphene LDOS A site')
	plt.show()


def testFFT():
	return None


if __name__ == '__main__': 
	# main()
	# test_Ginfkx() #show lifshitz transition in spectral weight
	test_LDOS_mp()
	# test_omega_grid()









