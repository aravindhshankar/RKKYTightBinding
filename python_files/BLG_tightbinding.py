##### Implement tight binding hamiltonian from McCann and Koshino Rep. Prog. Phys. 76 (2013) 056503 
##### Eq. 16

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigvals


a = 2.46 #Angstrom
acc = 1.42 #Angstrom


## energy quantities in units of eV
epsA1 = 0 
deltaprime = 0.022
epsA2 = deltaprime
epsB1 = deltaprime
epsB2 = 0
gamma0 = 3.16
gamma1 = 0.381
gamma3 = 0.38
gamma4 = 0.14



def f(k):
	t1 = np.exp(1j*k[1]/np.sqrt(3))
	t2 = 2 * np.exp(-1j * k[1]/(2*np.sqrt(3))) * np.cos(k[0]/2.)
	return t1 + t2

def Ham_BLG(k): 
	ham = [ [epsA1, -gamma0*f(k), gamma4*f(k), -gamma3*np.conj(f(k))],
			[-gamma0*np.conj(f(k)), epsB1, gamma1, gamma4*f(k)],
			[gamma4*np.conj(f(k)), gamma1, epsA2, -gamma0*f(k)],
			[-gamma3*f(k), gamma4*np.conj(f(k)), -gamma0*np.conj(f(k)), epsB2] ]
	return np.array(ham)


kxrange = np.linspace(-np.pi, np.pi, 200)
kyrange = np.linspace(-np.pi, np.pi, 200)





Elist = np.array([eigvals(Ham_BLG((kx,ky))) 
	for kx in kxrange for ky in kyrange]).real.reshape(len(kxrange),len(kyrange),4)







###### tests ##########

def contourplot():
	fig = plt.contour(kxrange,kyrange,Elist[:,:,2], levels = 200)
	plt.colorbar()
	plt.show()

def test_elist():
	print(Elist.shape)

def test_hermiticity():
	kvec = np.array([np.pi/2, np.pi/3])
	np.testing.assert_almost_equal(Ham_BLG(kvec) , np.conj(Ham_BLG(kvec).T))
	return True 


def main():
	test_hermiticity()
	contourplot()


if __name__ == '__main__':
	main()

