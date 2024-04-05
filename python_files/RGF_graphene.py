import numpy as np
from scipy.linalg import eigvals, eigvalsh
import matplotlib.pyplot as plt

def main():
	epsB = 0.
	epsA = 0.
	t = 1. 
	a = 1. 
	#kx = 2*np.pi/a * 0.1 
	kx = 0.5/a
	#kxvals = np.linspace(0,2*np.pi/a,10)
	kxvals = (kx,)
	#omega = 0.001
	delta = 0.001
	omegavals = np.linspace(-3.5,3.5,10000) 
	#omegavals = (omega,)

	kwargs = {  't':t, 
				'a':a,
				'epsA': epsA,
				'epsB': epsB 
				}

	def ret_H0(kx,t,a,epsA,epsB):
		P = np.array([[0,0,0,-np.conj(t)*np.exp(-1j*kx*a)],[0,0,0,0],[0,0,0,0],[-t*np.exp(1j*kx*a),0,0,0]]) 
		np.testing.assert_almost_equal(P,P.conj().T)
		M = np.array([[epsB,-np.conj(t),0,0],[-t,epsA,-t,0],[0,-np.conj(t),epsB,-t],[0,0,-np.conj(t),epsA]])
		return M + P

	Q = np.array([[0,-t,0,0],[-np.conj(t),0,0,0],[0,0,0,-np.conj(t)],[0,0,-t,0]])
	Ty = np.array([[0,-t,0,0],[0,0,0,0],[0,0,0,0],[0,0,-t,0]]) #Right hopping matrix along Y
	# H0 = ret_H0(kx,**kwargs)
	np.testing.assert_almost_equal(Q,Q.conj().T)
	dimH = 4
	G0invarr = np.array([(omega + 1j*delta)*np.eye(4) - ret_H0(kx,**kwargs) 
					for omega in omegavals for kx in kxvals]).reshape((len(omegavals),len(kxvals),dimH,dimH))
	#First index is omega index i.e. G0invarr[0] contains G0inv(omega=0, kx) for all kx
	Garr = np.linalg.inv(G0invarr) #Initialize G to G0
	# print(G0invarr.shape, Garr.shape)
	# print((G0invarr[0,0]@Garr[0,0]).real)
	Tydag = Ty.conj().T
	RECURSIONS = 500
	fig,ax = plt.subplots(1)
	for itern in range(RECURSIONS):
		Garr = np.linalg.inv(G0invarr - Ty@Garr@Tydag)
		# DOS = (-1./np.pi) * Garr[:,0,0,0].imag
		# ax.plot(omegavals,DOS,label=str(itern))
	DOS = (-1./np.pi) * Garr[:,0,0,0].imag
	#np.testing.assert_equal(len(omegavals),len(DOS))
	plt.plot(omegavals,DOS)
	#ax.legend()
	#ax.set_ylim(-1,1)
	plt.show()

if __name__ == '__main__': 
	main()