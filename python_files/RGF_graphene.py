import numpy as np
from scipy.linalg import eigvals, eigvalsh
import matplotlib.pyplot as plt

def main():
	epsB = 0.
	epsA = 0.
	t = 1. 
	a = 1. 
	kx = 2*np.pi/a * 0.1 
	kxvals = np.linspace(0,2*np.pi/a,10)
	kxvals = (kx,)

	kwargs = {  't':t, 
				'a':a,
				'epsA': epsA,
				'epsB': epsB 
				}

	def ret_H0(kx,t,a,epsA,epsB):
		P = np.array([[0,0,0,-np.conj(t)*np.exp(-1j*kx*a)],[0,0,0,0],[0,0,0,0],[-t*np.exp(1j*kx*a),0,0,0]]) 
		M = np.array([[epsB,-np.conj(t),0,0],[-t,epsA,-t,0],[0,-np.conj(t),epsB,-t],[0,0,-np.conj(t),epsA]])
		return M + P

	Q = np.array([[0,-t,0,0],[-np.conj(t),0,0,0],[0,0,0,-np.conj(t)],[0,0,-t,0]])
	Ty = np.array([[0,-t,0,0],[0,0,0,0],[0,0,0,0],[0,0,-t,0]]) #Right hopping matrix along Y
	# H0 = ret_H0(kx,**kwargs)
	np.testing.assert_almost_equal(Q,Q.conj().T)
	dimH = 4
	omega = 0.001
	omegavals = np.linspace(0,0.01,5)
	G0invarr = np.array([np.linalg.inv(omega - ret_H0(kx,**kwargs)) 
					for omega in omegavals for kx in kxvals]).reshape((len(omegavals),len(kxvals),dimH,dimH))
	#First index is omega index i.e. G0invarr[0] contains G0inv(omega=0, kx) for all kx
	Garr = np.linalg.inv(G0inv) #Initialize G to G0
	Tydag = Ty.conj().T
	RECURSIONS = 10
	for itern in range(RECURSIONS):
		Garr = np.linalg.inv(G0invarr - Ty@Garr@(Tydag))
	print(Garr)

if __name__ == '__main__': 
	main()