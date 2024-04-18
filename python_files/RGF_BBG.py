import numpy as np
from scipy.linalg import eigvals, eigvalsh
import matplotlib.pyplot as plt

## Maybe you want to rewrite this section as data members of a class that you can inherit from
## Or even add these data quantities to a separate module which can be imported

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

kwargs = {  'epsA1': epsA1, 
			'epsA1': epsA1,
			'epsA2': epsA2,
			'epsB1': epsB1, 
			'epsB2': epsB2,
			'gamma0': gamma0, 
			'gamma1': gamma1, 
			'gamma3': gamma3, 
			'gamma4': gamma4
			}


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

def ret_H0(kx,**kwargs):
	Tx = np.zeros((8,8))
	Tx[0,5] = -kwargs['gamma3']
	Tx[0,6] = -kwargs['gamma0']
	Tx[0,7] = kwargs['gamma4']
	Tx[2,5] = kwargs['gamma4']
	Tx[3,5] = -kwargs['gamma0']
	Tx = Tx * np.exp(-1j*kx)
	#P = Tx + Tx.conj().T

	#Horrible code ahead : please don't judge

	M = np.zeros((8,8))
	M[0,0] = kwargs['epsB2']
	M[0,1] = -kwargs['gamma3']
	M[0,2] = -kwargs['gamma0']
	M[0,3] = kwargs['gamma1']
	M[1,1] = kwargs['epsA1']
	M[1,2] = kwargs['gamma4']
	M[1,3] = -kwargs['gamma0']
	M[1,4] = -kwargs['gamma3']
	M[1,6] = kwargs['gamma4']
	M[1,7] = -kwargs['gamma0']
	M[2,2] = kwargs['epsA2']
	M[2,3] = kwargs['gamma1']
	M[2,4] = -kwargs['gamma0']
	M[3,3] = kwargs['epsB1']
	M[3,4] = kwargs['gamma4']
	M[4,4] = kwargs['epsB2']
	M[4,5] = -kwargs['gamma3']
	M[4,6] = -kwargs['gamma0']
	M[4,7] = kwargs['gamma4']
	M[5,5] = kwargs['epsA1']
	M[5,6] = kwargs['gamma4']
	M[5,7] = -kwargs['gamma0']
	M[6,6] = kwargs['epsA2']
	M[6,7] = kwargs['gamma1']
	M[7,7] = kwargs['epsB1']
	M = M + Tx
	M = M + M.conj().T - np.diag(np.diag(M))

	return M 


def ret_Ty(kx,**kwargs):
	# non-hermitian : only couples along -ve y direction
	Ty = np.zeros((8,8),dtype = complex)
	Ty[0,2] = -kwargs['gamma0']
	Ty[0,3] = kwargs['gamma4']
	Ty[0,5] = -kwargs['gamma3'] * np.exp(-1j*kx)
	Ty[1,2] = kwargs['gamma4']
	Ty[1,3] = -kwargs['gamma0']
	Ty[1,4] = -kwargs['gamma3']
	Ty[6,4] = -kwargs['gamma0']
	Ty[6,5] = kwargs['gamma4']
	Ty[7,4] = kwargs['gamma4']
	Ty[7,5] = -kwargs['gamma0']

	return Ty



def recG(omega, kx, kwargs, RECURSIONS=20, delta = 0.001):
	''' 
	Takes a single omega and a single kx and returns the greens function at the end of the chain
	So far, only the naive simple recursive process is implemented
	'''
	dimH = 8
	G0inv = (omega+1j*delta)*np.eye(dimH) - ret_H0(kx,**kwargs)
	G = np.linalg.inv(G0inv) #Initialize G to G0

	Ty = ret_Ty(kx,**kwargs)
	Tydag = Ty.conj().T
	for itern in range(RECURSIONS):
		G = np.linalg.inv(G0inv - Ty@G@Tydag) #Notice how this adds one extra site at a time
	Gfull = np.linalg.inv(np.linalg.inv(G) - Ty@G@Tydag)
	DOSend = (-1./np.pi) * np.diag(G.imag)
	DOSfull = (-1./np.pi) * np.diag(Gfull.imag)
	return (DOSend, DOSfull)




def boilerplate():
	#kx = 2*np.pi/a * 0.1 
	kx = 0.5/a
	#kxvals = np.linspace(0,2*np.pi/a,10)
	kxvals = (kx,)
	#omega = 0.001
	delta = 0.001
	omegavals = np.linspace(-3.1,3.1,2000) 
	#omegavals = (omega,)

	np.testing.assert_almost_equal(Q,Q.conj().T)
	dimH = 8
	G0invarr = np.array([(omega + 1j*delta)*np.eye(dimH) - ret_H0(kx,**kwargs) 
					for omega in omegavals for kx in kxvals]).reshape((len(omegavals),len(kxvals),dimH,dimH))
	#First index is omega index i.e. G0invarr[0] contains G0inv(omega=0, kx) for all kx
	Garr = np.linalg.inv(G0invarr) #Initialize G to G0

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



def main(): 
	RECURSIONS = 2000
	omega = 0.001 #units of eV
	kxvals = np.linspace(0,0.1,1000)
	DOS = np.array([recG(omega,kx,kwargs,RECURSIONS=RECURSIONS) for kx in kxvals])
	DOSend = DOS[:,0]
	DOSfull = DOS[:,1]
	#print(np.diag(DOSfull))
	#exit(0)










	################## Plotting #############################

	fig, ax = plt.subplots(2)
	fig.suptitle(f'Recursions = {RECURSIONS}, $\\omega$ = {omega:.4}')
	ax[0].plot(kxvals,DOSend[:,0],label='index 0')
	ax[0].plot(kxvals,DOSend[:,1],label='index 1')
	#ax[0].set_ylim(0,2.2)
	ax[0].set_title('DOSend')
	ax[0].set_xlabel('kx')
	ax[0].legend()

	ax[1].plot(kxvals,DOSfull[:,0],label='index 0')
	ax[1].plot(kxvals,DOSfull[:,1],label='index 1')
	#ax[1].set_ylim(0,2.2)
	ax[1].set_title('DOSfull')
	ax[1].set_xlabel('kx')
	ax[1].legend()

	fig.tight_layout()
	plt.show()

if __name__ == '__main__': 
	main()










