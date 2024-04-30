import mpmath as mpm
import numpy as np

mpm.mp.dps=32

A = np.random.rand(4).reshape((2,2))
B = np.random.rand(4).reshape((2,2))
C = np.random.rand(4).reshape((2,2))


mpmA = mpm.matrix(A)
mpmB = mpm.matrix(B)
mpmC = mpm.matrix(C)


# print(np.linalg.inv(A@B@C))
# print((mpmA*mpmB*mpmC)**-1) #inverse works


# print(mpmA + mpm.matrix(A))
# print(2*mpmA) # mpm matrix cannot be added to np arrays. convert explicitly using mpm.matrix()
omega = 2.34323 + 1j*0.001
# mpmomega = mpm.mpc(str(omega.real), str(omega.imag))
mpmomega = mpm.mpf(str(omega.real)) + 1j*mpm.mpf(str(omega.imag))
D = mpmomega * mpm.eye(2)
# print(D)
# print(D.transpose_conj()) #works
print(D.apply(mpm.im))


# print(A.shape[0], mpmA.rows) #Use mpmA.rows to get the shape
# print(A.rows) # np array does not have attribute 'rows'

# checkcpy = mpmA
# print(mpmA is checkcpy) #A = B returns reference: no copies made 


# print(mpm.pi)

E = mpm.matrix(2)
E[0,0] = mpm.mpf('2.12')
E[0,1] = mpm.mpc('1.23' , '3.21')
E[1,0] = mpm.exp(1j*mpm.mpf('2.21'))
print(E)