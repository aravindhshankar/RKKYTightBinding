import numpy as np
from scipy.linalg import eigvals, eigvalsh
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.linalg import norm
from scipy.integrate import simpson, quad
from functools import partial
import time 
import multiprocessing as mp
from h5_handler import *
import concurrent.futures

from Sources import fastrecDOSfull

epsB = 0.
epsA = 0.
t = 1. 
a = 1. 

def ret_H0(kx): 
	P = np.array([[0,0,0,-np.conj(t)*np.exp(-1j*kx*a)],[0,0,0,0],[0,0,0,0],[-t*np.exp(1j*kx*a),0,0,0]]) 
	# np.testing.assert_almost_equal(P,P.conj().T)
	M = np.array([[epsB,-np.conj(t),0,0],[-t,epsA,-t,0],[0,-np.conj(t),epsB,-t],[0,0,-np.conj(t),epsA]])
	return M + P


def ret_T():
	return = np.array([[0,-t,0,0],[0,0,0,0],[0,0,0,0],[0,0,-t,0]]) #Right hopping matrix along Y



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
	omega = 0.8
	omegavals = (omega,)
	kxvals = np.linspace(-np.pi,np.pi,1000)
	delta = 0.0001
	dimH = 4
	G = np.array([fastrecDOSfull(omega) 
					for omega in omegavals for kx in kxvals]).reshape((len(omegavals),len(kxvals),dimH,dimH))
	DOS = np.array([-1./np.pi * G[0,:,i,i].imag for i in range(dimH)])

	fig, ax = plt.subplots(1)
	for i in range(dimH):
		ax.plot(kxvals,DOS[i], label=f'index {i}')
	ax.set_xlabel(r'$k_x$')
	ax.set_title(f'$\\omega$ = {omega:.3}')
	ax.legend()

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


def helper_LDOS_mp(omega):
	callintegrand = lambda kx: -1./np.pi * fastrecGfull(omega,kx)[0,0].imag
	LDOS = quad(callintegrand,-np.pi,np.pi)[0] 
	return LDOS


def test_LDOS_mp():
	'''
	Use scipy.quad for this
	'''
	omegavals = np.linspace(0,3.1,512)
	# callintegrand = lambda kx, omega: -1./np.pi * fastrecGfull(omega,kx)[0,0].imag
	# LDOS = quad(partial(callintegrand,omega=om),-np.pi,np.pi)[0] for om in omegavals
	# PROCESSES = 10
	PROCESSES = mp.cpu_count()
	startmp = time.time()
	with mp.Pool(PROCESSES) as pool:
		LDOS = pool.map(helper_LDOS_mp, omegavals)
	stopmp = time.time()
	elapsedmp = stopmp-startmp
	print(f'Parallel computation with {PROCESSES} processes finished in time {elapsedmp} seconds')
	
	savedict = {'omegavals' : omegavals,
				'LDOS' : LDOS,
				'INFO' : '[0,0] site of -1/pi Im G'
				}
	# dict2h5(savedict,'GrapheneAsiteLDOS.h5', verbose=True)

	fig,ax = plt.subplots(1)
	ax.plot(omegavals, LDOS, label = 'quad LDOS')
	ax.axvline(1., ls='--', c='grey')
	ax.set_xlabel('omega')
	ax.set_title('Graphene LDOS A site')
	plt.show()



def test_LDOS_LC():
	omegavals = np.linspace(0,3.1,20)
	DOS = np.array([fastrecDOSfull(omega,ret_H0(kx),ret_Ty(),RECURSIONS,delta) 
					for omega in omegavals for kx in kxvals]).resize((len(omegavals),len(kxvals),dimH,dimH))
	print(DOS)

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

if __name__ == '__main__': 
	# main()
	# test_Ginfkx() #show lifshitz transition in spectral weight
	test_LDOS_LC()









