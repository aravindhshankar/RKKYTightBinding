import numpy as np 
from scipy.linalg import inv as scipy_inv


def test_singlemat():
	A = np.array([[1,2],[3,4]])
	npinv = np.linalg.inv(A)
	spinv = scipy_inv(A)
	linv_A = np.linalg.solve(A.T.dot(A), A.T)

	print('Numpy inv ', npinv)
	print('Scipy inv', spinv)
	print('solver inv', linv_A)


def invmatmul():
	A = np.array([[1,2],[3,4]])
	y = np.array([[3,4],[5,6]])
	npinv = np.linalg.inv(A)
	# linv_A = np.linalg.solve(A.T.dot(A), A.T)
	naive = npinv@y
	print('naive:', naive)
	solved = np.linalg.solve(A,y)
	print('solved:', solved)


def scratch():
	x = np.array([np.inf, np.nan, 3.23 + 3.34j, np.inf + 2.3j, 2.1 - np.inf*1j, np.nan+1j, -1*np.inf+ (1j*np.nan)])
	print(x)
	print([np.isnan(xval) or np.isinf(xval) for xval in x])
	# x = np.array([[2,3],[5,6]], dtype = np.cdouble)
	# x0 = np.zeros_like(x,dtype=np.double)
	# print(x0, type(x0), type(x0[0,0]))
	A = np.array([-2,3,2,-1])
	print(A<0)


def main():
	scratch()

if __name__ == '__main__':
	main()