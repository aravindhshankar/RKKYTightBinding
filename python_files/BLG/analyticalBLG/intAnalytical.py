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
from utils.twoBandReducedBlg import Gkx as anaGkx 
import pycuba
from dask.distributed import Client, LocalCluster
from dask import delayed

path_to_output = '../../Output/anaBLG/'

if not os.path.exists(path_to_output):
    os.makedirs(path_to_output)
    print("Outputs directory created at ", path_to_output)


# def Gkx(px, omega, eta, m, v3):

print = partial(print, flush=True) #To see output at each step in alice

def helper_mp(omega,g3):
    # delta = 5e-3 * omega # for BLG 
    delta = 5e-2 * omega # for turning off gamma 4 this is better 
    default_blg = BLG()
    gamma0 = default_blg.gamma0
    gamma1 = default_blg.gamma1
    m = (3.*gamma0**2)/(4.*gamma1) + (g3)/(8.)
    v3 = np.sqrt(3.) * g3 * 0.5
    ##### initialize cubify ######
    cubify.set_limits(-np.pi,np.pi) #not sure about this - you should integrate from -inf to inf right? 
    NDIM = 2
    KEY = 0
    NCOMP = 1
    MAXEVAL = int(5e5)
    VERBOSE = 0
    EPSREL = 1e-3

    @cubify.VECCubify
    def call_int(kx) : 
        return anaGkx(kx, omega, delta, m, v3)

    ################### starting integration #################
    start_time = time.perf_counter() if __debug__ else 0.0
    CUHREdict = pycuba.Cuhre(call_int, NDIM, ncomp = NCOMP, key=KEY, maxeval=MAXEVAL, verbose=VERBOSE, epsrel=EPSREL) 
    intarr = np.array([CUHREdict['results'][i]['integral'] for i in range(NCOMP)], dtype = np.float64)
    if __debug__: 
        elapsed = time.perf_counter() - start_time
        print(f"Finished integration for omega = {omega:.6f} in {elapsed} sec(s) with fail = {CUHREdict['fail']}, neval = {CUHREdict['neval']}.")

    return intarr


def process_g3(g3_index):
    g3_values = np.array([0.01,0.1,0.2,0.3,0.36,0.38,0.4,0.5,0.6,0.7,0.8,1.0,3.0]) # list of gamma3 values to scan , 5th 0 based index is the expt one
    if g3_index > len(g3_values) - 1 :
        raise(Exception('g3_index OUT OF BOUNDS!!!!!! aborting......'))
        exit(1)
    g3 = g3_values[g3_index]
    # omegavals = [0.0003,0.003,0.03,0.3]
    eps = 1e-5
    omegavals = np.sort(np.concatenate((np.logspace(np.log10(1e-6),np.log10(1e-2),300),np.linspace(1e-2+eps,1.2,60))))

    PROCESSES = int(os.environ.get('SLURM_CPUS_PER_TASK','2'))
    start_time = time.perf_counter()
    # Initialize Dask cluster
    # scheduler_port = 8786 + r_index
    # local_dir = f"/tmp/dask-task-{r_index}-{os.getpid()}"
    # cluster = LocalCluster(n_workers=PROCESSES, processes=True, scheduler_port=scheduler_port, local_directory=local_dir) 
    cluster = LocalCluster(n_workers=PROCESSES, processes=True) 
    client = Client(cluster)

    # Create Dask tasks
    tasks = [delayed(helper_mp)(omega, g3) for omega in omegavals] #returns the NLDOS as function of omega for a given r
    results = client.compute(tasks)
    results = client.gather(results)

    client.close()
    elapsed = time.perf_counter() - start_time
    print(f'Dask process with {PROCESSES} processes for gamma3  = {g3} finished in {elapsed} sec(s).')
    # Return a dictionary that needs to be saved 
    savedict = {'omegavals' : omegavals,
                'gamma3' : g3,
                'LDOS' : results,
                'INFO' : 'two band model based on mccann-koshino with only h0 and hw , delta = 5e-2 * omega, deltprime=gamma4=0'
                }
    return savedict

if __name__ == "__main__":
    # Get the Slurm task ID
    task_id = int(sys.argv[1])  # Passed by Slurm as an argument
    save_dict = process_g3(task_id)
    savename = f"results_g3_{task_id}"
    savefileoutput = savename + '.h5'
    dict2h5(save_dict,os.path.join(path_to_output,savefileoutput), verbose=True)


