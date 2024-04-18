import numpy as np
from scipy.linalg import eigvals, eigvalsh
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.linalg import norm
from scipy.integrate import simpson

######### Benchmarked semi inf chain with andrew's notebook ########
######### inf chain still has some finite size artifacts 
######### sites [3,3] and [4,4] of andrew's notebook correspond to our [0,0] index in semi inf chani
def main():
	epsB = 0.
	epsA = 0.
	t = 1. 
	a = 1. 
	#kx = 2*np.pi/a * 0.1 
	kx = 0.5/a
	kxvals = np.linspace(0,2*np.pi/a,1000)
	kxvals = (kx,)
	omega = 0.4
	delta = 0.0001
	omegavals = np.linspace(-3.1,3.1,2000) 
	# omegavals = np.linspace(-1.5,1.5, 200)
	#omegavals = [0.001,0.005,0.01,0.05,0.1,0.5,1,1.5]
	# omegavals = (omega,)
	err = 1e-1
	R = np.arange(0,50)
	# R = (0,)

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
	itern = 0
	diff = 1
	while itern < RECURSIONS and diff > err:
		Goldarr = Garr[:,0,0,0]
		Garr = np.linalg.inv(G0invarr - Ty@Garr@Tydag)
		# diff = norm(Goldarr[0,:,0,0].imag - Garr[0,:,0,0].imag)
		diff = norm(Goldarr.imag - Garr[:,0,0,0].imag)
		itern += 1 
	print(f'finished with itern = {itern}, diff = {diff:.3}')
	Gfull = np.linalg.inv(np.linalg.inv(Garr) - Ty@Garr@Tydag)
	# DOSend0 = (-1./np.pi) * Garr[0,:,0,0].imag
	# DOSfull0 = (-1./np.pi) * Gfull[0,:,0,0].imag
	# DOSend1 = (-1./np.pi) * Garr[0,:,1,1].imag
	# DOSfull1 = (-1./np.pi) * Gfull[0,:,1,1].imag
	# np.testing.assert_equal(len(omegavals),len(DOSend0))
	# np.testing.assert_equal(len(omegavals),len(DOSfull1))
	DOSend0 = (-1./np.pi) * Garr[:,0,0,0].imag
	DOSfull0 = (-1./np.pi) * Gfull[:,0,0,0].imag
	DOSend1 = (-1./np.pi) * Garr[:,0,1,1].imag
	DOSfull1 = (-1./np.pi) * Gfull[:,0,1,1].imag
	# np.testing.assert_equal(len(kxvals),len(DOSend0))
	# np.testing.assert_equal(len(kxvals),len(DOSfull1))

	# peaks = find_peaks(DOSfull0,prominence=0.1*np.max(DOSfull0))[0]
	# print(peaks)
	# print(kxvals[peaks])

	# GR0 = np.array([(0.5/np.pi)*simpson(np.exp(-1j * kxvals * R[0])*DOSfull0[i,:], kxvals) for i in range(len(omegavals))])
	# GR0 = np.array([(0.5/np.pi)*simpson(np.exp(-1j * kxvals * Rval)*DOSfull0, kxvals) for Rval in R])

	# print(GR0.size)








	################PLOTTING######################
	fig, ax = plt.subplots(2)
	fig.suptitle(f'Recursions = {RECURSIONS}, kx = {kx:.3}')
	ax[0].plot(omegavals,DOSend0,label='index 0')
	ax[0].plot(omegavals,DOSend1,label='index 1')
	ax[0].set_ylim(0,0.6)
	ax[0].set_title('DOSend')
	ax[0].set_xlabel('$\\omega$')
	ax[0].legend()

	ax[1].plot(omegavals,DOSfull0,label='index 0')
	ax[1].plot(omegavals,DOSfull1,label='index 1')
	ax[1].set_ylim(0,0.6)
	ax[1].set_title('DOSfull')
	ax[1].set_xlabel('$\\omega$')
	ax[1].legend()

	# fig.suptitle(f'Recursions = {RECURSIONS}, $\\omega = $ {omega:.3}')
	# ax[0].plot(kxvals,DOSend0,label='index 0')
	# ax[0].plot(kxvals,DOSend1,label='index 1')
	# #ax[0].set_ylim(0,2.2)
	# ax[0].set_title('DOSend')
	# ax[0].set_xlabel('kx')
	# ax[0].legend()

	# ax[1].plot(kxvals,DOSfull0,label='index 0')
	# ax[1].plot(kxvals,DOSfull1,label='index 1')
	# #ax[1].set_ylim(0,2.2)
	# ax[1].set_title('DOSfull')
	# ax[1].set_xlabel('$kx')
	# for peak in peaks:
	# 	ax[1].axvline(kxvals[peak],ls='--')
	# ax[1].legend()


	fig.tight_layout()
	
	# fig, ax = plt.subplots(1)
	# ax.plot(R,GR0,'.-')
	# ax.set_xlabel('R')

	# fig,ax = plt.subplots(1)
	# ax.plot(omegavals, GR0)
	# ax.set_xlabel('$\\omega$')

	plt.show()










if __name__ == '__main__': 
	main()










