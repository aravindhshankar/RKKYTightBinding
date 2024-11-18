import sys 
import os
sys.path.insert(0,'..')
sys.path.insert(0,'../..')
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.integrate import simpson, quad
from functools import partial
import time 
from FastRGF.solveRGF import MOMfastrecNLDOSfull
from utils.h5_handler import *
from utils.decorators import cubify
from utils.models import BLG,Graphene
import pycuba
from dask.distributed import Client, LocalCluster
from dask import delayed

path_to_output = '../../Output/nldos/'

if not os.path.exists(path_to_output):
    os.makedirs(path_to_output)
    print("Outputs directory created at ", path_to_output)

if len(sys.argv) > 1:
    savename = str(sys.argv[1])


def helper_mp(omega,r):
	idx_x, idx_y = 0,0
	dochecks = False
	# delta = 5e-3 * omega # for BLG 
	delta = 5e-2 * omega # for graphene 
	RECURSIONS = 30
	blg = Graphene()
	dimH = blg.dimH
	ret_H0 = blg.ret_H0
	ret_Ty = blg.ret_Ty
	
	##### initialize cubify ######
	cubify.set_limits(-np.pi,np.pi)
	NDIM = 2
	KEY = 0
	MAXEVAL = int(5e5)
	VERBOSE = 0
	EPSREL = 1e-4
	
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


def process_r(r_index):
	r_values = np.linspace(1, 20, 20)  # Example r values, replace with your own
	r = r_values[r_index]
	# omegavals = [0.0003,0.003,0.03,0.3]
	eps = 1e-5
	omegavals = np.sort(np.concatenate((np.logspace(np.log10(1e-4),np.log10(1e-2),500),np.linspace(1e-2+eps,5e-1,50))))

	# PROCESSES = int(os.environ.get('SLURM_NTASKS','2'))
	# client = Client(threads_per_worker=1, n_workers=PROCESSES)
	PROCESSES = int(os.environ.get('SLURM_CPUS_PER_TASK','2'))
	# print(f'PROCESSES = {PROCESSES}')
	start_time = time.perf_counter()
	# Initialize Dask cluster
	cluster = LocalCluster(n_workers=PROCESSES, processes=True)
	client = Client(cluster)

	# Create Dask tasks
	tasks = [delayed(helper_mp)(omega, r) for omega in omegavals] #returns the NLDOS as function of omega for a given r
	results = client.compute(tasks)
	results = client.gather(results)

	client.close()
	elapsed = time.perf_counter() - start_time
	print(f'Dask process with {PROCESSES} processes for r = {r} finished in {elapsed} sec(s).')
	# Return a dictionary that needs to be saved 
	savedict = {'omegavals' : omegavals,
		'r' : r,
		'NLDOS' : results,
		'INFO' : '[0,0] site , delta = 5e-2 * omega'
		}
	return savedict

if __name__ == "__main__":
	# Get the Slurm task ID
	task_id = int(sys.argv[1])  # Passed by Slurm as an argument
	save_dict = process_r(task_id)
	savename = f"results_r_{task_id}"
	savefileoutput = savename + '.h5'
	dict2h5(save_dict,os.path.join(path_to_output,savefileoutput), verbose=True)


