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

path_to_output = '../../Output/scandeltaVH/'

if not os.path.exists(path_to_output):
    os.makedirs(path_to_output)
    print("Outputs directory created at ", path_to_output)


print = partial(print, flush=True) #To see output at each step in alice

def helper_mp(omega,delta):
    idx_x, idx_y = 0,0
    dochecks = False
    # delta = 5e-3 * omega # for BLG 
    # delta = 5e-3 * omega if omega > 1e-3 else 5e-2 * omega
    # match omega:
        # case omega if np.abs(omega) > 1e-3:
            # delta = 5e-3 * np.abs(omega)
        # case omega if np.abs(omega) > 5e-5:
            # delta = 5e-2 * np.abs(omega)
        # case _ :
            # delta = 1e-1 * np.abs(omega)

    RECURSIONS = 30
    blg = BLG() #intialize model
    dimH = blg.dimH
    ret_H0 = blg.ret_H0
    ret_Ty = blg.ret_Ty

    ##### initialize cubify ######
    cubify.set_limits(-np.pi,np.pi)
    NDIM = 2
    KEY = 0
    NCOMP = 4
    MAXEVAL = int(5e5)
    VERBOSE = 0
    EPSREL = 1e-4

    @cubify.VECCubify
    def call_int(kx) : 
        Gkx = MOMfastrecNLDOSfull(omega,0,kx,ret_H0(kx),ret_Ty(kx).conj().T,RECURSIONS,delta) #adding the right hopping matrix as needed for fast recursion in v2
        return np.array((Gkx[0,0], Gkx[1,1], Gkx[2,2], Gkx[3,3]))

    ################### starting integration #################
    start_time = time.perf_counter() if __debug__ else 0.0
    CUHREdict = pycuba.Cuhre(call_int, NDIM, ncomp = NCOMP, key=KEY, maxeval=MAXEVAL, verbose=VERBOSE, epsrel=EPSREL) 
    intarr = np.array([CUHREdict['results'][i]['integral'] for i in range(NCOMP)], dtype = np.float64)
    if __debug__: 
        elapsed = time.perf_counter() - start_time
        print(f"Finished integration for omega = {omega:.6f} in {elapsed} sec(s) with fail = {CUHREdict['fail']}, neval = {CUHREdict['neval']}.")

    return intarr


def process_delta(delta_index):
    delta_values = np.logspace(np.log(1e-6),np.log(1e-1),10)
    delta = delta_values[delta_index]
    # omegavals = np.array([0.0003,0.003,0.03,0.3])
    eps = 1e-5
    # omegavals = np.sort(np.concatenate((np.logspace(np.log10(1e-6),np.log10(1e-2),300),np.linspace(1e-2+eps,5e-1,50))))
    # omegavals = np.concatenate((-1.*omegavals[::-1], omegavals))
    omegavals = np.logspace(np.log10(1e-4),np.log10(1e-3),400)

    PROCESSES = int(os.environ.get('SLURM_CPUS_PER_TASK','2'))
    start_time = time.perf_counter()
    # Initialize Dask cluster
    # scheduler_port = 8786 + delta_index
    # local_dir = f"/tmp/dask-task-{delta_index}-{os.getpid()}"
    # cluster = LocalCluster(n_workers=PROCESSES, processes=True, scheduler_port=scheduler_port, local_directory=local_dir) 
    cluster = LocalCluster(n_workers=PROCESSES, processes=True) 
    client = Client(cluster)

    # Create Dask tasks
    tasks = [delayed(helper_mp)(omega, delta) for omega in omegavals] #returns the NLDOS as function of omega for a given r
    results = client.compute(tasks)
    results = client.gather(results)

    client.close()
    elapsed = time.perf_counter() - start_time
    print(f'Dask process with {PROCESSES} processes for delta = {delta} finished in {elapsed} sec(s).')
    # Return a dictionary that needs to be saved 
    savedict = {'omegavals' : omegavals,
                'delta' : delta,
                'LDOS' : results,
                'INFO' : '[0,1,2,3] sites of the default BLG model , with v2 model and correct right hopping matrix'
                }
    return savedict

if __name__ == "__main__":
    # Get the Slurm task ID
    for task_id in range(11):
        print(f"started task_id {task_id}")
        save_dict = process_delta(task_id)
        savename = f"results_delta_{task_id}"
        savefileoutput = savename + '.h5'
        dict2h5(save_dict,os.path.join(path_to_output,savefileoutput), verbose=True)


