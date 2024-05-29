import sys
sys.path.insert(0,'..')
import os 
import numpy as np
import mpmath as mpm
import time
from FastRGF.RGF import MOMfastrecDOSfull as RGFnp
from FastRGF.mpmRGF import MOMfastrecDOSfull as RGFmpm
from scipy.linalg import norm
mpm.mp.dps = 9

# epsA1 = 0 
# deltaprime = 0.022
# epsA2 = deltaprime
# epsB1 = deltaprime
# epsB2 = 0
# gamma0 = 3.16
# gamma1 = 0.381
# gamma3 = 0.38
# gamma4 = 0.14

epsA1 = mpm.mpf('0') 
deltaprime = mpm.mpf('0.022')
epsA2 = deltaprime
epsB1 = deltaprime
epsB2 = mpm.mpf('0')
gamma0 = mpm.mpf('3.16')
gamma1 = mpm.mpf('0.381')
gamma3 = mpm.mpf('0.38')
gamma4 = mpm.mpf('0.14')




def retnp_H0(kx):
	Tx = np.zeros((8,8),dtype=np.cdouble)
	Tx[0,5] = -gamma3
	Tx[0,6] = -gamma0
	Tx[0,7] = gamma4
	Tx[2,5] = gamma4
	Tx[3,5] = -gamma0
	Tx = Tx * np.exp(-1j*kx)
	#P = Tx + Tx.conj().T

	#Horrible code ahead : please don't judge

	M = np.zeros((8,8),dtype=np.cdouble)
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

def retnp_Ty(kx):
	# non-hermitian : only couples along -ve y direction
	Ty = np.zeros((8,8),dtype = np.cdouble)
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

def retmpm_H0(kx):
	Tx = mpm.zeros(8)
	Tx[0,5] = -gamma3
	Tx[0,6] = -gamma0
	Tx[0,7] = gamma4
	Tx[2,5] = gamma4
	Tx[3,5] = -gamma0
	Tx = Tx * mpm.exp(-1j*kx)

	M = mpm.zeros(8)
	M[0,1] = -gamma3
	M[0,2] = -gamma0
	M[0,3] = gamma1
	M[1,2] = gamma4
	M[1,3] = -gamma0
	M[1,4] = -gamma3
	M[1,6] = gamma4
	M[1,7] = -gamma0
	M[2,3] = gamma1
	M[2,4] = -gamma0
	M[3,4] = gamma4
	M[4,5] = -gamma3
	M[4,6] = -gamma0
	M[4,7] = gamma4
	M[5,6] = gamma4
	M[5,7] = -gamma0
	M[6,7] = gamma1
	M = M + Tx
	M = M + M.transpose_conj()
	M[0,0] = epsB2
	M[1,1] = epsA1
	M[2,2] = epsA2
	M[3,3] = epsB1
	M[4,4] = epsB2
	M[5,5] = epsA1
	M[6,6] = epsA2
	M[7,7] = epsB1
	
	return M 

def retmpm_Ty(kx):
	# non-hermitian : only couples along -ve y direction
	Ty = mpm.zeros(8)
	Ty[0,2] = -gamma0
	Ty[0,3] = gamma4
	Ty[0,5] = -gamma3 * mpm.exp(-1j*kx)
	Ty[1,2] = gamma4
	Ty[1,3] = -gamma0
	Ty[1,4] = -gamma3
	Ty[6,4] = -gamma0
	Ty[6,5] = gamma4
	Ty[7,4] = gamma4
	Ty[7,5] = -gamma0

def rel_timing():
	RECURSIONS = 25
	delta = 1e-5
	kx = 1.3
	omega = 0.2
	start = time.perf_counter()
	# Gnp = RGFnp(omega, retnp_H0(kx), retnp_Ty(kx), RECURSIONS, delta)
	# stop = time.perf_counter()
	# print(f'Gnp = {Gnp} calculated in {(stop-start):.4f} seconds.')

	start = time.perf_counter()
	Gmpm = RGFmpm(mpm.mpf(omega), retmpm_H0(mpm.mpf(kx)), retmpm_Ty(mpm.mpf(kx)), RECURSIONS, mpm.mpf(delta))
	stop = time.perf_counter()
	print(f'Gmpm = {Gmpm} calculatd in {(stop-start):.4f} seconds.')

	# print(f'2 Norm  = {norm(Gmpm-Gnp)}')


def main():
	rel_timing()


if __name__ == '__main__':
	main()