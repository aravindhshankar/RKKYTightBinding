import numpy as np 
import mpmath as mpm

mpm.mp.dps = 32

def fastrecGfwd(omega,H0,Ty,RECURSIONS=20,delta=mpm.mpf('0.001')):
	'''
	Takes a single omega and a single kx 
	Ty is the matrix propagating in the forward Y direction
	IMPORTANT : pass H0,Ty as mpmath matrices
	IMPORTANTL : pass omega, delta also as mpm numbers created with strings
	'''
	dimH = H0.rows #Assumes that H is a square matrix; also acts as a check that H0 is an mpm
	G0inv = (omega+1j*delta)*mpm.eye(dimH) - H0
	G = G0inv**-1 #Initialize G to G0
	Tydag = Ty.transpose_conj() #also checks that Ty is an mpm matrix

	tnf = G*Tydag
	tnb = G*Ty
	T = tnf
	tprod = tnb
	for itern in range(RECURSIONS):
		tmp = (mpm.eye(dimH) - tnf*tnb - tnb*tnf)**-1
		tnf = tmp*tnf*tnf
		tnb = tmp*tnb*tnb
		T = T + tprod*tnf
		tprod = tprod*tnb
	Gn = (G0inv - Ty*T)**-1
	G = Gn
	# G = np.linalg.inv(np.linalg.inv(G) - Ty@G@Tydag)
	#G = np.linalg.inv(np.linalg.inv(G) - Tydag@G@Ty) #couple other way around

	return G

def fastrecGrev(omega,H0,Ty,RECURSIONS=20,delta=mpm.mpf('0.001')):
	'''
	Takes a single omega and a single kx 
	Ty is still the forward coupling matrix
	Notice that instead of using this function, one can just pass Ty.conj().T to the fastrecGfwd method
	'''
	dimH = H0.shape[0] #Assumes that H is a square matrix 
	G0inv = (omega+1j*delta)*np.eye(dimH,dtype=np.cdouble) - H0
	G = np.linalg.inv(G0inv) #Initialize G to G0
	Tydag = Ty
	Ty = Tydag.conj().T

	tnf = G@Tydag
	tnb = G@Ty
	T = tnf
	tprod = tnb
	for itern in range(RECURSIONS):
		tmp = np.linalg.inv(np.eye(dimH,dtype=np.cdouble) - tnf@tnb - tnb@tnf)
		tnf = tmp@tnf@tnf
		tnb = tmp@tnb@tnb
		T = T + tprod@tnf
		tprod = tprod@tnb
	Gn = np.linalg.inv(G0inv - Ty@T)
	G = Gn
	# G = np.linalg.inv(np.linalg.inv(G) - Ty@G@Tydag)
	#G = np.linalg.inv(np.linalg.inv(G) - Tydag@G@Ty) #couple other way around

	return G


def MOMfastrecGfull(omega,H0,Ty,RECURSIONS=20,delta=0.001):
	'''
	Once again pass H0,Ty as a mpm matrices
	'''
	Tydag = Ty.transpose_conj()
	Gfwd = fastrecGfwd(omega,H0,Ty,RECURSIONS,delta)
	Grev = fastrecGfwd(omega,H0,Tydag,RECURSIONS,delta)
	# Grev = fastrecGrev(omega,kx,**kwargs)
	G = ((Grev)**-1 - Ty*Gfwd*Tydag)**-1
	return G


def MOMfastrecDOSfull(omega,H0,Ty,RECURSIONS=20,delta=0.001):
	'''
	Returns the full -1/pi Im G(kx, omega) matrix in the bulk 
	'''
	Tydag = Ty.transpose_conj()
	Gfwd = fastrecGfwd(omega,H0,Ty,RECURSIONS,delta)
	Grev = fastrecGfwd(omega,H0,Tydag,RECURSIONS,delta)
	# Grev = fastrecGrev(omega,kx,**kwargs)
	# DOS = (-1./np.pi) * np.imag(np.linalg.inv(np.linalg.inv(Grev) - Ty@Gfwd@Tydag))
	DOS = (-1./mpm.pi) * np.imag(((Grev)**-1 - Ty@Gfwd@Tydag)**-1)
	# return (-1./np.pi) * G.imag
	return DOS







