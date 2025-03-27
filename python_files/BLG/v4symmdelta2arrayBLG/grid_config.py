# grid_config.py
import numpy as np
import sys, os
sys.path.insert(0,'..')
sys.path.insert(0,'../..')
from utils.decorators import cubify
from utils.models import BLG
import pycuba
import time 
from functools import partial
from FastRGF.solveRGF import MOMfastrecNLDOSfull

print = partial(print, flush=True)
# rvals = np.arange(0,10)
rvals = np.array([0,1,2,3,4,10,11,12,50,51,52,99,100,101])
# omegavals = np.logspace(np.log10(1e-5), np.log10(15.0), 840)
spc = 1e-5
omegavals = np.logspace(np.log10(1e-5), np.log10(1e-1), 1500)
omegavalslin = np.linspace(1e-1 + spc, 15., 1500) 
omegavals = np.concatenate((omegavals, omegavalslin))
omegavals = np.sort(np.concatenate((-1.*omegavals[::-1], omegavals))) #no need for sorting actually, but just for peace of mind since it's only done once
NUMGS = 18 #size of the array returned by helper_mp 
DELTEMPFLAG = False # set to true to delete the temporary files generated by each array: set this to false for the edge case that the last task is not the last to complete


def helper_mp(r, omega):
    dochecks = False
    omegastar = 2e-4
    omegaM = 2e-2
    # delta = 5e-2 * np.abs(omega) if np.abs(omega) > omegastar else 1e-5 #chosen such that delta(omegastar) = 1e-5
    if np.abs(omega) >= omegaM:
        delta = 1e-3
    elif np.abs(omega) >= omegastar:
        delta = 5e-2 * np.abs(omega)
    else:
        delta = 1e-5

    RECURSIONS = 30
    blg = BLG() #intialize model
    dimH = blg.dimH
    ret_H0 = blg.ret_H0
    ret_Ty = blg.ret_Ty

    ##### initialize cubify ######
    cubify.set_limits(-np.pi,np.pi)
    NDIM = 2
    KEY = 0
    NCOMP = int(NUMGS)
    MAXEVAL = int(5e5)
    VERBOSE = 0
    EPSREL = 1e-3

    @cubify.VECCubify
    def call_int(kx) : 
        # Gkx = MOMfastrecNLDOSfull(omega,r,kx,ret_H0(kx),ret_Ty(kx).conj().T,RECURSIONS,delta) #adding the right hopping matrix as needed for fast recursion in v2
        Gkx = MOMfastrecNLDOSfull(omega,r,kx,ret_H0(kx).T,ret_Ty(kx).T,RECURSIONS,delta) #adding the right hopping matrix as needed for fast recursion in v2
        return np.array((Gkx[0,0], Gkx[1,1], Gkx[2,2], Gkx[3,3], Gkx[0,4], Gkx[1,5], Gkx[2,6], Gkx[3,7], Gkx[0,1], Gkx[1,0], Gkx[0,5], Gkx[5,0], Gkx[2,3], Gkx[3,2], Gkx[2,6], Gkx[6,2], Gkx[2,7], Gkx[7,2]))

    ################### starting integration #################
    start_time = time.perf_counter() if __debug__ else 0.0
    CUHREdict = pycuba.Cuhre(call_int, NDIM, ncomp = NCOMP, key=KEY, maxeval=MAXEVAL, verbose=VERBOSE, epsrel=EPSREL) 
    intarr = np.array([CUHREdict['results'][i]['integral'] for i in range(NCOMP)], dtype = np.float64)
    if __debug__: 
        elapsed = time.perf_counter() - start_time
        print(f"Finished integration for omega = {omega:.8f} in {elapsed} sec(s) with fail = {CUHREdict['fail']}, neval = {CUHREdict['neval']}.")

    return intarr
