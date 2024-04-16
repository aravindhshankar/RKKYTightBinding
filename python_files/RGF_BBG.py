import numpy as np
from scipy.linalg import eigvals, eigvalsh
import matplotlib.pyplot as plt


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
	omegavals = np.linspace(-3.1,3.1,2000) 
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
	RECURSIONS = 20000 #So far, this is just the recursive method : not yet Fast recursion
	for itern in range(RECURSIONS):
		Garr = np.linalg.inv(G0invarr - Ty@Garr@Tydag)
	Gfull = np.linalg.inv(np.linalg.inv(Garr) - Ty@Garr@Tydag)
	DOSend0 = (-1./np.pi) * Garr[:,0,0,0].imag
	DOSfull0 = (-1./np.pi) * Gfull[:,0,0,0].imag
	DOSend1 = (-1./np.pi) * Garr[:,0,1,1].imag
	DOSfull1 = (-1./np.pi) * Gfull[:,0,1,1].imag
	np.testing.assert_equal(len(omegavals),len(DOSend0))
	np.testing.assert_equal(len(omegavals),len(DOSfull1))

	fig, ax = plt.subplots(2)
	fig.suptitle(f'Recursions = {RECURSIONS}')
	ax[0].plot(omegavals,DOSend0,label='index 0')
	ax[0].plot(omegavals,DOSend1,label='index 1')
	ax[0].set_ylim(0,2.2)
	ax[0].set_title('DOSend')
	ax[0].legend()

	ax[1].plot(omegavals,DOSfull0,label='index 0')
	ax[1].plot(omegavals,DOSfull1,label='index 1')
	ax[1].set_ylim(0,2.2)
	ax[1].set_title('DOSfull')
	ax[1].legend()


	plt.show()

if __name__ == '__main__': 
	main()










