import sys 
import os
sys.path.insert(0,'..')
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.integrate import simpson, quad
from functools import partial
import time 
from FastRGF.solveRGF import MOMfastrecDOSfull, MOMfastrecNLDOSfull
from utils.h5_handler import *
from utils.decorators import cubify
from utils.models import BLG,Graphene
import pycuba
from dask.distributed import Client

savename = 'default_savename'
path_to_output = '../Outputs/BLG/'

if not os.path.exists(path_to_output):
    os.makedirs(path_to_output)
    print("Outputs directory created at ", path_to_output)

if len(sys.argv) > 1:
    savename = str(sys.argv[1])




def test_Ginfkx():
	# omega = 1 - 1e-2
	# omega = 2e-4 
	# omega = 0.000444967
	# omega = 0.000740532
	# omega = 2e-2
	# blg = BLG()	
	blg = Graphene()	
	ret_H0 = blg.ret_H0
	ret_Ty = blg.ret_Ty
	omega = 3e-4
	r = 20
	omegavals = (omega,)
	kxvals = np.sort(np.concatenate((np.linspace(-0.05,0.05,5000,dtype=np.double), np.linspace(-np.pi,np.pi,500,dtype=np.double))))
	dochecks = False
	# delta = 5e-3 * omega
	delta = 5e-2 * omega
	RECURSIONS = 30
	dimH = blg.dimH
	num_pp = 2000
	start, stop = kxvals[0], kxvals[-1]
	start_time = time.perf_counter()
	kDOS = np.array([MOMfastrecNLDOSfull(omega,r,kx,ret_H0(kx),ret_Ty(kx),RECURSIONS,delta,dochecks)
					for omega in omegavals for kx in kxvals],dtype=np.longdouble).reshape((len(omegavals),len(kxvals),dimH,dimH))
	elapsed = time.perf_counter() - start_time
	print(f'Finished calculating kDOS in {elapsed} sec(s).')
	peaks = find_peaks(kDOS[0,:,0,0],prominence=0.01*np.max(kDOS[0,:,0,0]))[0]
	print(f'Peaks found on sparse grid : {len(peaks)}')
	peakvals = [kxvals[peak] for peak in peaks]
	fig, ax = plt.subplots(1)
	ax.plot(kxvals, kDOS[0,:,0,0])
	ax.set_xlabel(r'$k_x$')
	ax.set_title(f'$\\omega$ = {omega:.3}, \\delta = {delta:.6}, RECURSIONS = {RECURSIONS}')
	# ax.vlines(kxvals[peaks], ls = '--', c = 'grey')
	# for peak in new_peakvals:
		# ax.axvline(peak,ls='--',c='gray')

	call_int = lambda kx : MOMfastrecNLDOSfull(omega,r,kx,ret_H0(kx),ret_Ty(kx),RECURSIONS,delta)[0,0]

	start,stop = -np.pi,np.pi
	ranges = []
	eta = 1e-5
	current = start
	for peak in peakvals:
		ranges += [(current, peak-eta)]
		ranges += [(peak-eta, peak+eta)]
		current = peak+eta
	ranges += [(current, stop)]		
	intlist = [quad(call_int,window[0],window[1],limit=5000,epsabs=0.01*delta,full_output=True) for window in ranges]
	for word in list(zip(ranges,[(ival[0],ival[1],ival[2]['neval'],ival[2]['last']) for ival in intlist])):
		print(word)
	intval = np.sum([ilist[0] for ilist in intlist])

	# elapsed = time.perf_counter() - start_time
	print(f'Finished quad integrator with delta = {delta:.6} and {RECURSIONS} recursions in {elapsed} sec(s).')
	print(f'intval = {intval:.8}')

	print('STARTED CUHRE')
	##### initialize cubify ######
	cubify.set_limits(-np.pi,np.pi)
	Integrand = cubify.Cubify(call_int)
	NDIM = 2
	KEY = 0
	MAXEVAL = int(1e6)
	verbose = 0
	start_time = time.perf_counter()
	CUHREdict = pycuba.Cuhre(Integrand, NDIM, key=KEY, maxeval=MAXEVAL, verbose=verbose,epsrel=1e-4) 
	elapsed = time.perf_counter() - start_time
	print(f'Finished CUHRE in {elapsed} sec(s).')
	print('CUHRE intval = ', CUHREdict)

	# print('STARTED SIMPSON INTEGRATION')
	# start_time = time.perf_counter()
	# simpson_intval = simpson(fine_integrand,adaptive_kxgrid)
	# elapsed = time.perf_counter() - start_time
	# print(f'Finished simpson integrator with delta = {delta:.6} and {RECURSIONS} recursions in {elapsed} sec(s).')
	# print(f'intval = {simpson_intval:.8}')

	for peak in peakvals:
		ax.axvline(peak, c='r', ls = '--')
		ax.axvline(peak-eta, c='gray', ls = '--',alpha=0.2)
		ax.axvline(peak+eta, c='gray', ls = '--',alpha=0.2)

	# for num_pp in [2000]: #checking convergence
	# 	print('Started simpson integrate WITH peaks')
	# 	peak_spacing = 0.01
	# 	print(f'num_pp = {num_pp}, peak_spacing = {peak_spacing:.4}')
	# 	start_time = time.perf_counter()
	# 	adaptive_kxgrid = generate_grid_with_peaks(-np.pi,np.pi,new_peakvals,peak_spacing=0.1,num_uniform=10000,num_pp=num_pp)
	# 	fine_integrand = np.array([MOMfastrecNLDOSfull(omega,r,kx,ret_H0(kx),ret_Ty(kx),RECURSIONS,delta)[0,0] for kx in adaptive_kxgrid],dtype=np.double)
	# 	simpson_intval = simpson(fine_integrand,adaptive_kxgrid)
	# 	elapsed = time.perf_counter() - start_time
	# 	print(f'Finished simpson integrator with delta = {delta:.6} and {RECURSIONS} recursions in {elapsed} sec(s).')
	# 	print(f'intval = {simpson_intval:.8}')
	# print(type(fine_integrand[0]))
	# ax.plot(adaptive_kxgrid,fine_integrand,'.',c='red')
	plt.show()
	# print(adaptive_kxgrid[fine_integrand<0])


def helper_LDOS_mp(omega):
	idx_x, idx_y = 0,0
	dochecks = False
	# delta = 5e-3 * omega # for BLG 
	delta = 5e-2 * omega # for graphene 
	RECURSIONS = 30
	dimH = blg.dimH
	
	##### initialize cubify ######
	cubify.set_limits(-np.pi,np.pi)
	NDIM = 2
	KEY = 0
	MAXEVAL = int(1e6)
	VERBOSE = 0
	EPSREL = 1e-5
	
	@cubify.Cubify
	def call_int(kx): 
 		return MOMfastrecNLDOSfull(omega,r,kx,ret_H0(kx),ret_Ty(kx),RECURSIONS,delta,dochecks)[idx_x,idx_y]

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
				'INFO' : '[0,0] site of -1/pi Im G, delta = 5e-3 * omega'
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
	DOSkx = MOMfastrecNLDOSfull(omega,r,kx,ret_H0(kx),ret_Ty(kx),RECURSIONS,delta)[0,0]
	print(DOSkx)

	
if __name__ == '__main__': 
	test_Ginfkx()
	# dask_LDOS()
	# test_LDOS_mp()
	# test_Gk_single()









