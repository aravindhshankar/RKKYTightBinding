#NOTE currently supported only for LDOS, i.e only pass 0 as command line argument
import sys 
import os
sys.path.insert(0,'..')
sys.path.insert(0,'../..')
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import time 
from FastRGF.solveRGF import MOMfastrecNLDOSfull
from utils.h5_handler import *
from utils.decorators import cubify
from utils.models import BLG,Graphene
from utils.twoBandReducedBlg import Gkx as anaGkx # returns -1/pi Im G(kx,omega)
import pycuba
from dask.distributed import Client, LocalCluster
from dask import delayed

path_to_output = '../../Output/anaBLG/'

if not os.path.exists(path_to_output):
    os.makedirs(path_to_output)
    print("Outputs directory created at ", path_to_output)

print = partial(print, flush=True) #To see output at each step in alice

def helper_mp(omega,r):
    ''' 
    This should be Dasked - delayed(helper_mp, omegavals)
    Currently there is only support for r = 0!!!!
    '''
    # delta = 5e-3 * omega # for BLG 
    delta = 5e-2 * omega # for turning off gamma 4 this is better 
    default_blg = BLG()
    gamma0 = default_blg.gamma0
    gamma1 = default_blg.gamma1
    g3 = default_blg.gamma3
    m = (3.*gamma0**2)/(4.*gamma1) + (g3)/(8.)
    v3 = np.sqrt(3.) * g3 * 0.5
    ##### initialize cubify ######
    cubify.set_limits(-np.pi,np.pi) 
    NDIM = 2
    KEY = 0
    NCOMP = 1
    MAXEVAL = int(5e5)
    VERBOSE = 0
    EPSREL = 1e-3

    @cubify.Cubify
    def call_int(kx) : 
        return anaGkx(kx, omega, delta, m, v3)

    ################### starting integration #################
    start_time = time.perf_counter() if __debug__ else 0.0
    CUHREdict = pycuba.Cuhre(call_int, ndim=NDIM, key=KEY, maxeval=MAXEVAL, verbose=VERBOSE, epsrel=EPSREL) 
    # intarr = np.array([CUHREdict['results'][i]['integral'] for i in range(NCOMP)], dtype = np.float64)
    intarr = CUHREdict['results'][0]['integral'] 
    if __debug__: 
        elapsed = time.perf_counter() - start_time
        print(f"Finished integration for omega = {omega:.6f} in {elapsed} sec(s) with fail = {CUHREdict['fail']}, neval = {CUHREdict['neval']}.")

    return intarr


def process_r(r_index):
    r_values = np.arange(21,dtype=int)  # r = 0 is the LDOS, goes until r = 20
    r = r_values[r_index]
    # omegavals = np.array([0.0003,0.003,0.03,0.3])
    eps = 1e-5
    omegavals = np.sort(np.concatenate((np.logspace(np.log10(1e-6),np.log10(1e-2),300),np.linspace(1e-2+eps,5e-1,50))))
    omegavals = np.concatenate((-1.*omegavals[::-1], omegavals))

    PROCESSES = int(os.environ.get('SLURM_CPUS_PER_TASK','6'))
    start_time = time.perf_counter()
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
                'INFO' : 'A site of the 2x2 reduced BLG model , delta = 5e-2 * omega '
                }
    return savedict


if __name__ == "__main__":
    # Get the Slurm task ID
    task_id = int(sys.argv[1])  # Passed by Slurm as an argument
    if task_id != 0:
        raise(Exception('CURRENTLY ONLY LDOS SUPPORTED!!!! COMMAND LINE ARGS OTHER THAN 0 ARE NOT ACCEPTED'))
        exit(1)
    
    save_dict = process_r(task_id)
    savename = f"results_r_{task_id}"
    savefileoutput = savename + '.h5'
    dict2h5(save_dict,os.path.join(path_to_output,savefileoutput), verbose=True)


