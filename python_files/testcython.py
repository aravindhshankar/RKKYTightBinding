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
from numba import njit


epsB = 0.
epsA = 0.
t = 1. 
a = 1. 
kwargs = {  't':t, 
			'a':a,
			'epsA': 2,
			'epsB': 5 
			}



@njit
def ret_H0(kx): 
	P = np.array([[0,0,0,-np.conj(t)*np.exp(-1j*kx*a)],[0,0,0,0],[0,0,0,0],[-t*np.exp(1j*kx*a),0,0,0]]) 
	# np.testing.assert_almost_equal(P,P.conj().T)
	M = np.array([[epsB,-np.conj(t),0,0],[-t,epsA,-t,0],[0,-np.conj(t),epsB,-t],[0,0,-np.conj(t),epsA]])
	return M + P

@njit
def ret_Ty():
	Ty = np.array([[0,-t,0,0],[0,0,0,0],[0,0,0,0],[0,0,-t,0]]) #Right hopping matrix along Y
	return Ty

def fastrecG(omega,kx):
	'''
	Takes a single omega and a single kx 
	'''
	dimH = 4
	RECURSIONS=20
	delta=0.001
	G0inv = (omega+1j*delta)*np.eye(dimH) - ret_H0(kx)
	G = np.linalg.inv(G0inv) #Initialize G to G0
	Ty = ret_Ty()
	Tydag = Ty.conj().T

	tnf = G@Tydag
	tnb = G@Ty
	T = tnf
	tprod = tnb
	for itern in range(RECURSIONS):
		tmp = np.linalg.inv(np.eye(dimH) - tnf@tnb - tnb@tnf)
		tnf = tmp@tnf@tnf
		tnb = tmp@tnb@tnb
		T = T + tprod@tnf
		tprod = tprod@tnb
	Gn = np.linalg.inv(G0inv - Ty@T)
	G = Gn
	# G = np.linalg.inv(np.linalg.inv(G) - Ty@G@Tydag)
	#G = np.linalg.inv(np.linalg.inv(G) - Tydag@G@Ty) #couple other way around

	return G

@njit
def fastrecGfwd(omega,kx):
	'''
	Takes a single omega and a single kx 
	'''
	dimH = 4
	RECURSIONS=20
	delta=0.001
	G0inv = (omega+1j*delta)*np.eye(dimH) - ret_H0(kx)
	G = np.linalg.inv(G0inv) #Initialize G to G0
	Ty = ret_Ty()
	Tydag = Ty.conj().T

	tnf = G@Tydag
	tnb = G@Ty
	T = tnf
	tprod = tnb
	for itern in range(RECURSIONS):
		tmp = np.linalg.inv(np.eye(dimH) - tnf@tnb - tnb@tnf)
		tnf = tmp@tnf@tnf
		tnb = tmp@tnb@tnb
		T = T + tprod@tnf
		tprod = tprod@tnb
	Gn = np.linalg.inv(G0inv - Ty@T)
	G = Gn
	# G = np.linalg.inv(np.linalg.inv(G) - Ty@G@Tydag)
	#G = np.linalg.inv(np.linalg.inv(G) - Tydag@G@Ty) #couple other way around

	return G

@njit
def fastrecGrev(omega,kx):
	'''
	Takes a single omega and a single kx 
	'''
	dimH = 4
	RECURSIONS=20
	delta=0.001
	G0inv = (omega+1j*delta)*np.eye(dimH) - ret_H0(kx)
	G = np.linalg.inv(G0inv) #Initialize G to G0
	Tydag = ret_Ty()
	Ty = Tydag.conj().T

	tnf = G@Tydag
	tnb = G@Ty
	T = tnf
	tprod = tnb
	for itern in range(RECURSIONS):
		tmp = np.linalg.inv(np.eye(dimH) - tnf@tnb - tnb@tnf)
		tnf = tmp@tnf@tnf
		tnb = tmp@tnb@tnb
		T = T + tprod@tnf
		tprod = tprod@tnb
	Gn = np.linalg.inv(G0inv - Ty@T)
	G = Gn
	# G = np.linalg.inv(np.linalg.inv(G) - Ty@G@Tydag)
	#G = np.linalg.inv(np.linalg.inv(G) - Tydag@G@Ty) #couple other way around

	return G


@njit
def fastrecGfull(omega,kx):
	Ty = ret_Ty()
	Tydag = Ty.conj().T
	Gfwd = fastrecGfwd(omega,kx)
	Grev = fastrecGrev(omega,kx)
	G = np.linalg.inv(np.linalg.inv(Grev) - Ty@Gfwd@Tydag)
	return G

def fastrecAndrew(omega,kx):
	dimH = 4
	RECURSIONS=20
	delta=0.001
	Pauli1 = np.array([[0,1],[1,0]])
	G0inv2x2 = (omega+1j*delta)*np.eye(2) + t*Pauli1
	G0inv = (omega+1j*delta)*np.eye(dimH) - ret_H0(kx)
	G = np.linalg.inv(G0inv) #Initialize G to G0
	Ty = ret_Ty()
	Tydag = Ty.conj().T

	tnf = G@Tydag
	tnb = G@Ty
	T = tnf
	tprod = tnb
	for itern in range(RECURSIONS):
		tmp = np.linalg.inv(np.eye(dimH) - tnf@tnb - tnb@tnf)
		tnf = tmp@tnf@tnf
		tnb = tmp@tnb@tnb
		T = T + tprod@tnf
		tprod = tprod@tnb
	Gn = np.linalg.inv(G0inv - Ty@T)
	G = np.linalg.inv(G0inv2x2 - 2*(t**2)*Gn[2:3, 2:3])

	# G = Inverse[(\[Omega] + I \[Delta]) IdentityMatrix[2] - 
	#    t PauliMatrix[1] - 2 t^2 Gn[[1 ;; 2, 1 ;; 2]]]; -(1/\[Pi]) Im[
	#   G[[1, 1]]]
	return G



def main():
	print(ret_H0(0))
	kx = 0.5/a
	# kxvals = np.linspace(0,2*np.pi/a,1000)
	kxvals = np.linspace(-np.pi/a,np.pi/a,1000)
	# kxvals = (kx,)
	omega = 0.4
	delta = 0.0001
	# omegavals = np.linspace(-3.1,3.1,2000) 
	# omegavals = np.linspace(-1.5,1.5, 200)
	#omegavals = [0.001,0.005,0.01,0.05,0.1,0.5,1,1.5]
	omegavals = (omega,)
	# omegavals = np.linspace(0.001,2,4)
	# omegavals = (0.002,0.01,0.3,0.9,1.4)
	omegavals = (0.01,0.4)
	err = 1e-1
	dimH = 4
	#R = np.arange(0,500)
	R = (0,)
	G = np.array([fastrecGfull(omega,kx) 
					for omega in omegavals for kx in kxvals]).reshape((len(omegavals),len(kxvals),dimH,dimH))



	DOS0 = (-1./np.pi) * G[:,:,0,0].imag
	DOS1 = (-1./np.pi) * G[:,:,1,1].imag
	DOS2 = (-1./np.pi) * G[:,:,2,2].imag
	DOS3 = (-1./np.pi) * G[:,:,3,3].imag

	# np.testing.assert_equal(len(kxvals),len(DOSend0))
	# np.testing.assert_equal(len(kxvals),len(DOSfull1))

	peaks = [find_peaks(DOS0[i,:],prominence=0.1*np.max(DOS0[i,:]))[0] for i in range(len(omegavals))]
	print(len(peaks))
	print(peaks)
	# GR0 = np.array([(0.5/np.pi)*simpson(np.exp(-1j * kxvals * R[0])*DOSfull0[i,:], kxvals) for i in range(len(omegavals))])
	# GR0 = np.array([0.5/np.pi*simpson(np.exp(-1j*0)*DOS0, kxvals])

	# print(GR0.size)


	#################PLOTTING######################
	NUMPLOTS = len(omegavals)
	assert NUMPLOTS < 6 , "TOO MANY PLOTS"
	fig, ax = plt.subplots(NUMPLOTS)
	#fig.suptitle(f'Recursions = {RECURSIONS}, kx = {kx:.3}')
	for i,omega in enumerate(omegavals):
		ax[i].plot(kxvals,DOS0[i,:],label='index 0',ls = '--' )
		ax[i].plot(kxvals,DOS1[i,:],label='index 1',ls = '--')
		ax[i].plot(kxvals,DOS2[i,:],label='index 2',ls = '--')
		ax[i].plot(kxvals,DOS3[i,:],label='index 3',ls = '--')
		#ax[i].set_ylim(0,2.2)
		ax[i].set_title(f'$\\omega$ = {omega:.3}')
		ax[i].set_xlabel('$k_x$')
		for peak in peaks[i]:
			ax[i].axvline(kxvals[peak],ls='--',c='k')
		ax[i].legend()


	# fig.tight_layout()
	
	# fig, ax = plt.subplots(1)
	# ax.plot(R,GR0,'.-')
	# ax.set_xlabel('R')

	# fig,ax = plt.subplots(1)
	# ax.plot(omegavals, GR0)
	# ax.set_xlabel('$\\omega$')

	plt.show()



def test_kwargs():
	H0 = ret_H0(0.1, )
	print(H0)


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
	G = np.array([fastrecGfull(omega,kx) 
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


def helper_LDOS(omega):
	callintegrand = lambda kx: -1./np.pi * fastrecGfull(omega,kx)[0,0].imag
	LDOS = quad(callintegrand,-np.pi,np.pi)[0] 
	return LDOS


def test_LDOS():
	'''
	Use scipy.quad for this
	'''
	# omegavals = np.linspace(0,3.1,512)
	omegavals = np.linspace(0,3.1,256)
	# callintegrand = lambda kx, omega: -1./np.pi * fastrecGfull(omega,kx)[0,0].imag
	# LDOS = quad(partial(callintegrand,omega=om),-np.pi,np.pi)[0] for om in omegavals
	# PROCESSES = 10
	PROCESSES = mp.cpu_count()
	print(f'Number of CPUs = {PROCESSES}')
	print(f'Number of discretized omegas = {len(omegavals)}')
	startmp = time.time()
	with mp.Pool(PROCESSES) as pool:
		LDOS = pool.map(helper_LDOS, omegavals)
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


def test_LDOS_threads():
	'''
	Even slower lol
	'''
	omegavals = np.linspace(0,3.1,16)
	# callintegrand = lambda kx, omega: -1./np.pi * fastrecGfull(omega,kx)[0,0].imag
	# LDOS = quad(partial(callintegrand,omega=om),-np.pi,np.pi)[0] for om in omegavals
	startmp = time.time()
	with concurrent.futures.ThreadPoolExecutor() as pool:
		LDOS = list(pool.map(helper_LDOS, omegavals))
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
	# test_LDOS()
	# test_LDOS_threads()
	compare_integrate()









