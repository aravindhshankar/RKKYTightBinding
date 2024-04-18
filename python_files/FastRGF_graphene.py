import numpy as np
from scipy.linalg import eigvals, eigvalsh
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.linalg import norm
from scipy.integrate import simpson, quad
from functools import partial



epsB = 0.
epsA = 0.
t = 1. 
a = 1. 
kwargs = {  't':t, 
			'a':a,
			'epsA': 2,
			'epsB': 5 
			}




def ret_H0(kx, **kwargs): 
	P = np.array([[0,0,0,-np.conj(t)*np.exp(-1j*kx*a)],[0,0,0,0],[0,0,0,0],[-t*np.exp(1j*kx*a),0,0,0]]) 
	np.testing.assert_almost_equal(P,P.conj().T)
	M = np.array([[epsB,-np.conj(t),0,0],[-t,epsA,-t,0],[0,-np.conj(t),epsB,-t],[0,0,-np.conj(t),epsA]])
	return M + P


def ret_Ty(**kwargs):
	Ty = np.array([[0,-t,0,0],[0,0,0,0],[0,0,0,0],[0,0,-t,0]]) #Right hopping matrix along Y
	return Ty

def fastrecG(omega,kx,**kwargs):
	'''
	Takes a single omega and a single kx 
	'''
	dimH = 4
	RECURSIONS=20
	delta=0.001
	G0inv = (omega+1j*delta)*np.eye(dimH) - ret_H0(kx,**kwargs)
	G = np.linalg.inv(G0inv) #Initialize G to G0
	Ty = ret_Ty(**kwargs)
	Tydag = Ty.conj().T

	tnf = G@Tydag
	tnb = G@Ty
	T = tnf
	tprod = tnb
	for itern in range(RECURSIONS):
		tmp = np.linalg.inv(np.eye(dimH) - tnf@tnb - tnb@tnf)
		tnf = tmp@tnf@tnf
		tnb = tmp@tnb@tnb
		T = T + tprod@tnf
		tprod = tprod@tnb
	Gn = np.linalg.inv(G0inv - Ty@T)
	G = Gn
	# G = np.linalg.inv(np.linalg.inv(G) - Ty@G@Tydag)
	#G = np.linalg.inv(np.linalg.inv(G) - Tydag@G@Ty) #couple other way around

	return G




def main():

	kx = 0.5/a
	# kxvals = np.linspace(0,2*np.pi/a,1000)
	kxvals = np.linspace(-np.pi/a,np.pi/a,1000)
	# kxvals = (kx,)
	omega = 0.4
	delta = 0.0001
	# omegavals = np.linspace(-3.1,3.1,2000) 
	# omegavals = np.linspace(-1.5,1.5, 200)
	#omegavals = [0.001,0.005,0.01,0.05,0.1,0.5,1,1.5]
	omegavals = (omega,)
	omegavals = np.linspace(0.001,2,4)
	omegavals = (0.002,0.01,0.07,0.3,0.9,1.4,2.8)
	err = 1e-1
	dimH = 4
	#R = np.arange(0,500)
	R = (0,)
	G = np.array([fastrecG(omega,kx,**kwargs) 
					for omega in omegavals for kx in kxvals]).reshape((len(omegavals),len(kxvals),dimH,dimH))



	DOS0 = (-1./np.pi) * G[:,:,0,0].imag
	DOS1 = (-1./np.pi) * G[:,:,1,1].imag
	# np.testing.assert_equal(len(kxvals),len(DOSend0))
	# np.testing.assert_equal(len(kxvals),len(DOSfull1))

	peaks = [find_peaks(DOS0[i,:],prominence=0.1*np.max(DOS0[i,:]))[0] for i in range(len(omegavals))]
	print(len(peaks))
	print(peaks)
	# GR0 = np.array([(0.5/np.pi)*simpson(np.exp(-1j * kxvals * R[0])*DOSfull0[i,:], kxvals) for i in range(len(omegavals))])
	# GR0 = np.array([0.5/np.pi*simpson(np.exp(-1j*0)*DOS0, kxvals])

	# print(GR0.size)








	#################PLOTTING######################
	NUMPLOTS = len(omegavals)
	assert NUMPLOTS < 10 , "TOO MANY PLOTS"
	fig, ax = plt.subplots(NUMPLOTS)
	#fig.suptitle(f'Recursions = {RECURSIONS}, kx = {kx:.3}')
	for i,omega in enumerate(omegavals):
		ax[i].plot(kxvals,DOS0[i,:],label='index 0')
		ax[i].plot(kxvals,DOS1[i,:],label='index 1')
		#ax[i].set_ylim(0,2.2)
		ax[i].set_title(f'$\\omega$ = {omega:.3}')
		ax[i].set_xlabel('$k_x$')
		for peak in peaks[i]:
			ax[i].axvline(kxvals[peak],ls='--',c='k')
		ax[i].legend()


	# fig.tight_layout()
	
	# fig, ax = plt.subplots(1)
	# ax.plot(R,GR0,'.-')
	# ax.set_xlabel('R')

	# fig,ax = plt.subplots(1)
	# ax.plot(omegavals, GR0)
	# ax.set_xlabel('$\\omega$')

	plt.show()



def test_kwargs():
	H0 = ret_H0(0.1, **kwargs)
	print(H0)







if __name__ == '__main__': 
	main()










