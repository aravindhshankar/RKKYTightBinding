import sys 
sys.path.insert(0,'..')
import numpy as np
from scipy.linalg import eigvals, eigvalsh
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.linalg import norm
from scipy.integrate import simpson, quad
from functools import partial
import time 
import multiprocessing as mp
# from FastRGF import MOMfastrecDOSfull
from FastRGF.RGF import MOMfastrecDOSfull
from h5_handler import *
import concurrent.futures

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
	Tx = np.zeros((8,8))
	Tx[0,5] = -gamma3
	Tx[0,6] = -gamma0
	Tx[0,7] = gamma4
	Tx[2,5] = gamma4
	Tx[3,5] = -gamma0
	Tx = Tx * np.exp(-1j*kx)
	#P = Tx + Tx.conj().T

	#Horrible code ahead : please don't judge

	M = np.zeros((8,8))
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
	Ty = np.zeros((8,8),dtype = complex)
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



def helper_LDOS_mp(omega,delta,RECURSIONS,analyze=False):
	callintegrand = lambda kx: MOMfastrecDOSfull(omega,ret_H0(kx),ret_Ty(kx),RECURSIONS,delta)[0,0]
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
	RECURSIONS = 25
	delta = 1e-4
	# omegavals = np.linspace(0,3.1,512)
	# omegavals = np.linspace(0,3.1,100)
	# omegavals = make_omega_grid()
	omegavals = np.logspace(np.log10(1e-5), np.log10(1e-1), num = 50)

	PROCESSES = mp.cpu_count()
	startmp = time.perf_counter()
	# with mp.Pool(PROCESSES) as pool:
	# 	LDOS = pool.map(helper_LDOS_mp, omegavals)
	with mp.Pool(PROCESSES) as pool:
			LDOS = pool.map(partial(helper_LDOS_mp,delta=delta,RECURSIONS=RECURSIONS,analyze=True), omegavals)
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
	test_LDOS_mp()










