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

path_to_figdir = '../Figures/Lifshitz'
if not os.path.exists(path_to_figdir): 
	os.makedirs(path_to_figdir)
	print('figdir directory created at ', path_to_figdir)

epsB = 0.
epsA = 0.
t = 1. 
a = 1. 



def ret_H0(kx): 
	P = np.array([[0,0,0,-np.conj(t)*np.exp(-1j*kx*a)],[0,0,0,0],[0,0,0,0],[-t*np.exp(1j*kx*a),0,0,0]]) 
	# np.testing.assert_almost_equal(P,P.conj().T)
	M = np.array([[epsB,-np.conj(t),0,0],[-t,epsA,-t,0],[0,-np.conj(t),epsB,-t],[0,0,-np.conj(t),epsA]])
	return M + P


def ret_Ty(kx=0):
	return np.array([[0,-t,0,0],[0,0,0,0],[0,0,0,0],[0,0,-t,0]]) #Right hopping matrix along Y

def generate_grid_with_peaks(a, b, peaks, peak_spacing=0.01, num_pp = 200, num_uniform=1000):
    # Sort the peaks list
    assert a < b , "a should be less than b" 
    peaks = sorted(peaks)
    
    # Generate grid around peaks
    peak_grid = np.concatenate([np.linspace(max(a, peak - peak_spacing), min(b, peak + peak_spacing), num = num_pp)
                                for peak in peaks])

    # Generate uniform grid for the remaining region
    # uniform_grid = np.linspace(max(a, min(peaks, default=a) + peak_spacing),
    #                            min(b, max(peaks, default=b) - peak_spacing), num=int((b - a) / uniform_spacing))
    uniform_grid = np.linspace(a,b,num=num_uniform)

    # Concatenate the peak and uniform grids
    grid = np.sort(np.concatenate([peak_grid, uniform_grid]))

    return grid


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
	omega = 1.15
	# omega = 0.15
	# omegavals = (omega,)
	omegavals = (0.2,0.8,1.2,1.8)
	kxvals = np.linspace(-np.pi,np.pi,1000)
	delta = 1e-5
	RECURSIONS = 25
	dimH = 4
	kDOS = np.array([MOMfastrecDOSfull(omega,ret_H0(kx),ret_Ty(),RECURSIONS,delta)
					for omega in omegavals for kx in kxvals]).reshape((len(omegavals),len(kxvals),dimH,dimH))
	peaks = find_peaks(kDOS[0,:,0,0],prominence=0.1*np.max(kDOS[0,:,0,0]))[0]
	peakvals = [kxvals[peak] for peak in peaks]

	fig, ax = plt.subplots(1)
	for i, omegaval in enumerate(omegavals):
		ax.plot(kxvals, kDOS[i,:,0,0], label = f'$\\omega$ = {omegaval:.2f}')
	ax.plot(kxvals, 0.0*kDOS[0,:,0,0], "k--") #just to show x axis
	ax.set_xlabel(r'$k_x$')
	# ax.set_title(f'$\\omega$ = {omega:.3}, \\delta = {delta:.6}, RECURSIONS = {RECURSIONS}')
	ax.set_title(f'\\delta = {delta:.6}, RECURSIONS = {RECURSIONS}')
	fig.suptitle('Monolayer Graphene')
	ax.legend()
	# ax.vlines(kxvals[peaks], ls = '--', c = 'grey')
	# for peak in peaks:
		# ax.axvline(kxvals[peak],ls='--',c='gray')
	# fig.savefig(os.path.join(path_to_figdir, 'BelowLifshitzGraphene.pdf'))
	# fig.savefig(os.path.join(path_to_figdir, 'AboveLifshitzGraphene.pdf'))
	fig.savefig(os.path.join(path_to_figdir, 'ShowLifshitz.pdf'))
	# ax.legend()

	# print('Started quad integrate without peaks')
	# start_time = time.perf_counter()
	# intval = quad(lambda kx : MOMfastrecDOSfull(omega,ret_H0(kx),ret_Ty(),RECURSIONS,delta,)[0,0], 
	# 					-np.pi,np.pi)[0]
	# elapsed = time.perf_counter() - start_time
	# print(f'Finished quad integrator with delta = {delta:.6} and {RECURSIONS} recursions in {elapsed} sec(s).')
	# print(f'intval = {intval:.5}')


	# print('Started quad integrate WITH peaks')
	# start_time = time.perf_counter()
	# intval = quad(lambda kx : MOMfastrecDOSfull(omega,ret_H0(kx),ret_Ty(),RECURSIONS,delta,)[0,0], 
	# 					-np.pi,np.pi, points = [kxvals[peak] for peak in peaks])[0]
	# elapsed = time.perf_counter() - start_time
	# print(f'Finished quad integrator with delta = {delta:.6} and {RECURSIONS} recursions in {elapsed} sec(s).')
	# print(f'intval = {intval:.5}')


	# for num_pp in [1000,]: #checking convergence
		# print('Started simpson integrate WITH peaks')
		# peak_spacing = 0.005
		# num_uniform = 500
		# print(f'num_pp = {num_pp}, peak_spacing = {2.*peak_spacing/num_pp:.5}, lin_spacing = {2.*np.pi/num_uniform:.5}')
		# start_time = time.perf_counter()
		# adaptive_kxgrid = generate_grid_with_peaks(-np.pi,np.pi,peakvals,peak_spacing=peak_spacing,num_uniform=num_uniform,num_pp=num_pp)
		# fine_integrand = np.array([MOMfastrecDOSfull(omega,ret_H0(kx),ret_Ty(kx),RECURSIONS,delta,)[0,0] for kx in adaptive_kxgrid])
		# simpson_intval = simpson(fine_integrand,adaptive_kxgrid)
		# elapsed = time.perf_counter() - start_time
		# print(f'Finished simpson integrator with delta = {delta:.6} and {RECURSIONS} recursions in {elapsed} sec(s).')
		# print(f'intval = {simpson_intval:.8}')
		# ax.plot(adaptive_kxgrid,fine_integrand,'.',c='red')
		# coeff = np.max(fine_integrand) / np.max(delta/np.abs(adaptive_kxgrid))
		# ax.plot(adaptive_kxgrid,coeff*delta/np.abs(adaptive_kxgrid),ls = '--')
	# 
	# new_peak_idx = find_peaks(fine_integrand,prominence=0.01*np.max(fine_integrand))[0][1]
	# new_peak_val = adaptive_kxgrid[new_peak_idx]
	# ax.axvline(new_peak_val,ls='--')
	# print(f'New peak idx = {new_peak_idx}')
	# fig,ax = plt.subplots(1)
	# fitslice = slice(new_peak_idx-10,new_peak_idx-2)
	# fitslice = slice(new_peak_idx+3,new_peak_idx+55)
	# ax.loglog(new_peak_val - adaptive_kxgrid, fine_integrand,'.-')
	# m,c = np.polyfit(np.log(new_peak_val - adaptive_kxgrid[fitslice]),np.log(fine_integrand[fitslice]),1)
	# m,c = np.polyfit(np.log(adaptive_kxgrid[new_peak_idx+1:new_peak_idx+20]),np.log(fine_integrand[new_peak_idx+1:new_peak_idx+20]),1)
	# ax.loglog(new_peak_val - adaptive_kxgrid[fitslice], np.exp(c)*adaptive_kxgrid[fitslice]**m, label=f'Fit with slope {m:.4}')
	# ax.legend()
	# ax.set_xlabel(r'$k_x$')
	# ax.set_title(f'$\\omega$ = {omega:.3}, \\delta = {delta:.6}, RECURSIONS = {RECURSIONS}')
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
	RECURSIONS = 22
	delta = 1e-6
	# omegavals = np.linspace(0,3.1,512)
	# omegavals = np.linspace(0,3.1,100)
	omegavals = make_omega_grid()
	print(f'Total number of points on the omega grid = {len(omegavals)}')

	PROCESSES = mp.cpu_count()
	startmp = time.perf_counter()
	# with mp.Pool(PROCESSES) as pool:
	# 	LDOS = pool.map(helper_LDOS_mp, omegavals)
	with mp.Pool(PROCESSES) as pool:
			LDOS = pool.map(partial(helper_LDOS_mp,delta=delta,RECURSIONS=RECURSIONS,analyze=True,method='adaptive'), omegavals)
	stopmp = time.perf_counter()
	elapsedmp = stopmp-startmp
	print(f'Parallel computation with {PROCESSES} processes finished in time {elapsedmp} seconds')
	LDOS = np.array(LDOS)
	savedict = {'omegavals' : omegavals,
				'LDOS' : LDOS,
				'RECURSIONS' : RECURSIONS, 
				'delta' : delta,
				'INFO' : '[0,0] site of -1/pi Im G'
				}
	savepath = os.path.join(path_to_dump, 'AdaptiveIntegrnGrapheneAsiteLDOS.h5')
	dict2h5(savedict,savepath, verbose=True)

	fig,ax = plt.subplots(2)
	ax[0].plot(omegavals, LDOS, '.-', label = 'quad LDOS')
	ax[0].axvline(1., ls='--', c='grey')
	ax[0].set_xlabel('omega')
	ax[0].set_title(f'Graphene LDOS A site with $\\delta = $ {delta:.6}')

	ax[1].loglog(omegavals[omegavals<1],LDOS[omegavals<1])
	ax[1].set_xlabel('omega')

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
	test_Ginfkx() #show lifshitz transition in spectral weight
	# test_LDOS_mp()
	# test_omega_grid()









