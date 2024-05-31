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


def main():
	invmatmul()

if __name__ == '__main__':
	main()