import numpy as np 
from scipy.linalg import inv as scipy_inv


# custinv = np.linalg.inv
# custinv = scipy_inv
custinv = lambda A: np.linalg.solve(A.T.dot(A), A.T)

def fastrecGfwd(omega,H0,Ty,RECURSIONS=20,delta=0.001):
	'''
	Takes a single omega and a single kx 
	Ty is the matrix propagating in the forward Y direction
	'''
	dimH = H0.shape[0] #Assumes that H is a square matrix 
	G0inv = (omega+1j*delta)*np.eye(dimH,dtype=np.cdouble) - H0
	G =   custinv(G0inv) #Initialize G to G0
	Tydag = Ty.conj().T

	tnf = G@Tydag
	tnb = G@Ty
	T = tnf
	tprod = tnb
	for itern in range(RECURSIONS):
		tmp = custinv(np.eye(dimH,dtype=np.cdouble) - tnf@tnb - tnb@tnf)
		tnf = tmp@tnf@tnf
		tnb = tmp@tnb@tnb
		T = T + tprod@tnf
		tprod = tprod@tnb
	Gn = custinv(G0inv - Ty@T)
	G = Gn
	# G = custinv(custinv(G) - Ty@G@Tydag)
	#G = custinv(custinv(G) - Tydag@G@Ty) #couple other way around

	return G

def fastrecGrev(omega,H0,Ty,RECURSIONS=20,delta=0.001):
	'''
	Takes a single omega and a single kx 
	Ty is still the forward coupling matrix
	Notice that instead of using this function, one can just pass Ty.conj().T to the fastrecGfwd method
	'''
	dimH = H0.shape[0] #Assumes that H is a square matrix 
	G0inv = (omega+1j*delta)*np.eye(dimH,dtype=np.cdouble) - H0
	G = custinv(G0inv) #Initialize G to G0
	Tydag = Ty
	Ty = Tydag.conj().T

	tnf = G@Tydag
	tnb = G@Ty
	T = tnf
	tprod = tnb
	for itern in range(RECURSIONS):
		tmp = custinv(np.eye(dimH,dtype=np.cdouble) - tnf@tnb - tnb@tnf)
		tnf = tmp@tnf@tnf
		tnb = tmp@tnb@tnb
		T = T + tprod@tnf
		tprod = tprod@tnb
	Gn = custinv(G0inv - Ty@T)
	G = Gn
	# G = custinv(custinv(G) - Ty@G@Tydag)
	#G = custinv(custinv(G) - Tydag@G@Ty) #couple other way around

	return G


def MOMfastrecGfull(omega,H0,Ty,RECURSIONS=20,delta=0.001):
	Tydag = Ty.conj().T
	Gfwd = fastrecGfwd(omega,H0,Ty,RECURSIONS,delta)
	Grev = fastrecGfwd(omega,H0,Tydag,RECURSIONS,delta)
	# Grev = fastrecGrev(omega,kx,**kwargs)
	G = custinv(custinv(Grev) - Ty@Gfwd@Tydag)
	return G


def MOMfastrecDOSfull(omega,H0,Ty,RECURSIONS=20,delta=0.001):
	'''
	Returns the full -1/pi Im G(kx, omega) matrix in the bulk 
	'''
	Tydag = Ty.conj().T
	Gfwd = fastrecGfwd(omega,H0,Ty,RECURSIONS,delta)
	Grev = fastrecGfwd(omega,H0,Tydag,RECURSIONS,delta)
	# Grev = fastrecGrev(omega,kx,**kwargs)
	# DOS = (-1./np.pi) * np.imag(custinv(custinv(Grev) - Ty@Gfwd@Tydag))
	DOS = (-1./np.pi) * np.imag(custinv(custinv(Grev) - Ty@Gfwd@Tydag))
	# return (-1./np.pi) * G.imag
	return DOS







