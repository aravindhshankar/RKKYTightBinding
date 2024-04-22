import numpy as np 
import matplotlib.pyplot as plt
from scipy.linalg import eigvals,eigvalsh
import multiprocessing as mp
from scipy.integrate import quad

def m1():
	ivals = np.array([1,2,3])
	jvals = np.array([10,100])
	A = np.array([i+j for i in ivals for j in jvals]).reshape((len(ivals),len(jvals)))
	print(A.shape)
	print(A)
	print(A[0], ivals[0]+jvals ) # A[0] contains A[i=0,jval]
	for Ai,Aval in enumerate(A):
		print(Ai,Aval)

def m2():
	ivals = np.array([1,2,3])
	jvals = np.array([10,100])
	dimH = 2
	Tmat = 2*np.eye(dimH)
	A = np.array([i*np.eye(dimH)+j*np.eye(dimH)
		 for i in ivals for j in jvals]).reshape((len(ivals),len(jvals), dimH, dimH))
	Ainv = np.linalg.inv(A)
	print(Ainv[2,0]@A[2,0]) #works
	#inv is broadcastable and if dims A = [M,N,d,d], then dims of Ainv are [M,N,d,d]
	# with the inverse taken as a vectorized operation on the last 2 dxd matrices

	# print(Tmat@A) #@ also acts only on the last dxd indices
	# print(A@Tmat)
	np.testing.assert_almost_equal(Tmat@A, A@Tmat)
	print('test passed!')

def m3():
	M = np.array([[1,2],[3,4]])
	print(np.diag(np.diag(M)))

def f(x):
	return x*x

def test_mp():
	print(f(2))
	with mp.Pool() as p:
		A = p.map(f, [1, 2, 3])
	print(A)


def test_quad_vec():
	integrand = lambda x: np.eye(2) * np.sin(x)
	vec_quad = np.vectorize(quad)
	intval = vec_quad(integrand, 0,np.pi)
	print(inval)



if __name__ == '__main__':
	# test_mp()
	test_quad_vec()

