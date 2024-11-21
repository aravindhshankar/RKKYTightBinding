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
    blg = BLG()	
    ret_H0 = blg.ret_H0
    ret_Ty = blg.ret_Ty
    dimH = blg.dimH
    omega = 3e-4
    r = 20
    omegavals = (omega,)
    dochecks = False
    delta = 5e-3 * omega # for BLG 
    # delta = 5e-2 * omega # for graphene 
    RECURSIONS = 30

    def call_int(kx) : 
        Gkx = MOMfastrecNLDOSfull(omega,r,kx,ret_H0(kx),ret_Ty(kx),RECURSIONS,delta)
        return np.array((Gkx[0,0], Gkx[1,1], Gkx[2,2], Gkx[3,3]))

    print('STARTED CUHRE')
    ##### initialize cubify ######
    cubify.set_limits(-np.pi,np.pi)
    Integrand = cubify.VECCubify(call_int)
    # Integrand = cubify.Cubify(call_int)
    NDIM = 2
    NCOMP = 4 
    KEY = 0
    MAXEVAL = int(1e5)
    verbose = 0
    start_time = time.perf_counter()
    CUHREdict = pycuba.Cuhre(Integrand, NDIM, ncomp = NCOMP, key=KEY, maxeval=MAXEVAL, verbose=verbose,epsrel=1e-4) 
    elapsed = time.perf_counter() - start_time
    print(f'Finished CUHRE in {elapsed} sec(s).')
    print('CUHRE dict = ', CUHREdict)
    intarr = np.array([CUHREdict['results'][i]['integral'] for i in range(NCOMP)], dtype = np.float64)
    print('integrated array = ', intarr) 



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









