import numpy as np 
from scipy.linalg import inv as scipy_inv
from scipy.linalg import solve as scsolve

#Use the Ainv@y = nplinalg.solve(A,y)

custinv = np.linalg.inv
# custinv = scipy_inv
# custinv = lambda A: np.linalg.solve(A.T.dot(A), A.T)

def fastrecGfwd(omega,H0,Ty,RECURSIONS=20,delta=0.001):
	'''
	Takes a single omega and a single kx 
	Ty is the matrix propagating in the forward Y direction
	'''
	flag = True
	dimH = H0.shape[0] #Assumes that H is a square matrix 
	G0inv = (omega+1j*delta)*np.eye(dimH,dtype=np.cdouble) - H0
	G = custinv(G0inv) #Initialize G to G0
	if np.isnan(G).any() or np.isinf(G).any():
		flag = False
		print(f"FOUND overflow in G FIRST")
		return np.zeros_like(G0inv), flag
    
	Tydag = Ty.conj().T
	tnf = np.linalg.solve(G0inv,Tydag)
	if np.isnan(tnf).any() or np.isinf(tnf).any():
		flag = False
		print(f"FOUND overflow in tnf FIRST")
		print(tnf)
		return np.zeros_like(G0inv), flag
	tnb = np.linalg.solve(G0inv,Ty)
	if np.isnan(tnb).any() or np.isinf(tnb).any():
		flag = False
		print(f"FOUND overflow in tnb FIRST")
		print(tnb)
		return np.zeros_like(G0inv), flag
	T = tnf
	tprod = tnb
	for itern in range(RECURSIONS):
		tmpinv = np.eye(dimH,dtype=np.cdouble) - tnf@tnb - tnb@tnf
		if np.isnan(tmpinv).any() or np.isinf(tmpinv).any():
			flag = False
			print(f"FOUND overflow in tmpinv")
			return np.zeros_like(G0inv), flag
		# tnf = np.linalg.solve(tmpinv,tnf@tnf)
		# tnb = np.linalg.solve(tmpinv,tnb@tnb)
		tnf = np.linalg.solve(tmpinv,tnf)@tnf
		if np.isnan(tnf).any() or np.isinf(tnf).any():
			flag = False
			print(f"FOUND overflow in tnf")
			print(tnf)
			return np.zeros_like(G0inv), flag
		tnb = np.linalg.solve(tmpinv,tnb)@tnb
		#if itern>1:
			#print(itern, 'tnb', np.amax(np.abs(tnb)), np.unravel_index(np.argmax(np.abs(tnb)), tnb.shape))  

        
		if np.isnan(tnb).any() or np.isinf(tnb).any():
			flag = False
			print(f"FOUND overflow in tnb")
			print(tnb)
			return np.zeros_like(G0inv), flag
		T = T + tprod@tnf
		if np.isnan(T).any() or np.isinf(T).any():
			flag = False
			print(f"FOUND overflow in T")
			return np.zeros_like(G0inv), flag
		tprod = tprod@tnb
		#new addition
		#if itern>1:
			#print(itern, np.amax(np.abs(tprod)), np.unravel_index(np.argmax(np.abs(tprod)), tprod.shape))  
		if np.isnan(tprod).any() or np.isinf(tprod).any():
			flag = False
			print(f"FOUND overflow in tprod", 'iter=', str(itern))
			print(tprod)
			return np.zeros_like(G0inv), flag
	Gn = custinv(G0inv - Ty@T)
	if np.isnan(Gn).any() or np.isinf(Gn).any():
		flag = False
		print(f"FOUND overflow in Gn")
		return np.zeros_like(G0inv), flag
	G = Gn
	# G = custinv(custinv(G) - Ty@G@Tydag)
	#G = custinv(custinv(G) - Tydag@G@Ty) #couple other way around

	return G, flag

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


def MOMfastrecDOSfull(omega,H0,Ty,RECURSIONS=20,delta=0.001, printing=False):
    '''
        Returns the full -1/pi Im G(kx, omega) matrix in the bulk 
        '''
    Tydag = Ty.conj().T
    Gfwd, fwdflag = fastrecGfwd(omega,H0,Ty,RECURSIONS,delta)
    Grev, revflag = fastrecGfwd(omega,H0,Tydag,RECURSIONS,delta)
    if printing:
        print(Gfwd,fwdflag)
        print(f'maxabsGfwd = {np.max(np.abs(Gfwd))}')
        print(f'maxabsGrev = {np.max(np.abs(Grev))}')
        print(Grev,revflag)
    # Grev = fastrecGrev(omega,kx,**kwargs)
    # DOS = (-1./np.pi) * np.imag(custinv(custinv(Grev) - Ty@Gfwd@Tydag))
    if fwdflag and revflag:
        itrmdt = custinv(Grev) - Ty@Gfwd@Tydag
        if np.isnan(itrmdt).any() or np.isinf(itrmdt).any():
            print(f"FOUND overflow in itrmdt")
            return np.zeros_like(H0,dtype=np.double)
        DOS = (-1./np.pi) * np.imag(custinv(itrmdt))
        if np.isnan(DOS).any() or np.isinf(DOS).any():
            print(f"FOUND overflow in DOS")
            return np.zeros_like(H0,dtype=np.double)
    else: 
        DOS = np.zeros_like(H0,dtype=np.double)
    # return (-1./np.pi) * G.imag
    return DOS







