import mpmath as mpm
import numpy as np
import time 
import sys 
sys.path.insert(0,'..')
from FastRGF.matops import getMatrixInverse, fixmul

mpm.mp.dps=32

A = np.random.rand(4).reshape((2,2))
# B = np.random.rand(4).reshape((2,2))
# C = np.random.rand(4).reshape((2,2))


mpmA = mpm.matrix(A)
# mpmB = mpm.matrix(B)
# mpmC = mpm.matrix(C)

# print(mpmA)
# print(mpmA, mpm.diag(mpm.diag(mpmA)))

# print(np.linalg.inv(A@B@C))
# print((mpmA*mpmB*mpmC)**-1) #inverse works


# print(mpmA + mpm.matrix(A))
# print(2*mpmA) # mpm matrix cannot be added to np arrays. convert explicitly using mpm.matrix()
# omega = 2.34323 + 1j*0.001
# # mpmomega = mpm.mpc(str(omega.real), str(omega.imag))
# mpmomega = mpm.mpf(str(omega.real)) + 1j*mpm.mpf(str(omega.imag))
# D = mpmomega * mpm.eye(2)
# # print(D)
# # print(D.transpose_conj()) #works
# print(D.apply(mpm.im))


# print(A.shape[0], mpmA.rows) #Use mpmA.rows to get the shape
# print(A.rows) # np array does not have attribute 'rows'

# checkcpy = mpmA
# print(mpmA is checkcpy) #A = B returns reference: no copies made 


# print(mpm.pi)

# E = mpm.matrix(2)
# E[0,0] = mpm.mpf('1e-6')
# E[0,1] = mpm.mpc('1.23' , '3.21')
# E[1,0] = mpm.exp(1j*mpm.mpf('2.21'))
# print(E) #works as expected

# A = mpm.linspace(2.,3.,5)
# B = mpm.linspace(2.5,4.5,5)
# C = mpm.matrix(np.concatenate([A,B]))
# print(C[0])
# print(type(C))


# A = mpm.mpf(2.47563874653)
# print(A.to_fixed(256))

A = np.random.rand(16).reshape((4,4))
mpmA = mpm.matrix(A)

# start = time.perf_counter()
# for i in range(20):
# 	mpmA = mpmA * mpmA
# stop = time.perf_counter()
# print(f'Direct multiplication finished in {stop-start} sec')

# start = time.perf_counter()
# for i in range(20):
# 	mpmA = fixmul(mpmA,mpmA,prec=64)
# stop = time.perf_counter()
# print(f'fixmul multiplication finished in {stop-start} sec')

start = time.perf_counter()
for i in range(50):
	mpmA = (mpmA)**-1
stop = time.perf_counter()
print(f'Direct Inverse finished in {stop-start} sec')

start = time.perf_counter()
for i in range(50):
	mpmA = getMatrixInverse(mpmA)
stop = time.perf_counter()
print(f'getMatrixInverse inverse finished in {stop-start} sec')
















