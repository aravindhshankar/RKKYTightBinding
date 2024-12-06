import numpy as np 
from dataclasses import dataclass

class Graphene:
	dimH = 4
	def __init__(self, epsB=0., epsA=0., t=1., a=1.):
		self.epsB = epsB
		self.epsA = epsA
		self.t = t
		self.a = a 

	def ret_H0(self,kx): 
		P = np.array([[0,0,0,-np.conj(self.t)*np.exp(-1j*kx*self.a)],[0,0,0,0],[0,0,0,0],[-self.t*np.exp(1j*kx*self.a),0,0,0]]) 
		# np.testing.assert_almost_equal(P,P.conj().T)
		M = np.array([[self.epsB,-np.conj(self.t),0,0],[-self.t,self.epsA,-self.t,0],[0,-np.conj(self.t),self.epsB,-self.t],[0,0,-np.conj(self.t),self.epsA]])
		return M + P

	def ret_Ty(self,kx):
		return np.array([[0,-self.t,0,0],[0,0,0,0],[0,0,0,0],[0,0,-self.t,0]]) #Right hopping matrix along Y


class BLG:
	dimH = 8
	def __init__(self, epsA1 = 0, deltaprime = 0.022, epsA2 = None, epsB1 = None, epsB2 = 0, gamma0 = 3.16, gamma1 = 0.381, gamma3 = 0.38, gamma4 = 0.14):
		self.epsA1 = epsA1
		self.deltaprime = deltaprime

		if epsA2 == None:
			self.epsA2 = deltaprime
		else:
			self.epsA2 = epsA2
		if epsB1 == None:
			self.epsB1 = deltaprime
		else:
			self.epsB1 = epsB1

		self.epsB2 = epsB2
		self.gamma0 = gamma0
		self.gamma1 = gamma1
		self.gamma3 = gamma3
		self.gamma4 = gamma4
	
	def ret_H0(self,kx):
		Tx = np.zeros((8,8),dtype=np.cdouble)
		Tx[0,5] = -self.gamma3
		Tx[0,6] = -self.gamma0
		Tx[0,7] = self.gamma4
		Tx[2,5] = self.gamma4
		Tx[3,5] = -self.gamma0
		Tx = Tx * np.exp(-1j*kx)

		M = np.zeros((8,8),dtype=np.cdouble)
		M[0,0] = self.epsB2
		M[0,1] = -self.gamma3
		M[0,2] = -self.gamma0
		# M[0,3] = self.gamma1
		M[0,3] = self.gamma4 #changed! This was a typo that said gamma1 before
		M[1,1] = self.epsA1
		M[1,2] = self.gamma4
		M[1,3] = -self.gamma0
		M[1,4] = -self.gamma3
		M[1,6] = self.gamma4
		M[1,7] = -self.gamma0
		M[2,2] = self.epsA2
		M[2,3] = self.gamma1
		M[2,4] = -self.gamma0
		M[3,3] = self.epsB1
		M[3,4] = self.gamma4
		M[4,4] = self.epsB2
		M[4,5] = -self.gamma3
		M[4,6] = -self.gamma0
		M[4,7] = self.gamma4
		M[5,5] = self.epsA1
		M[5,6] = self.gamma4
		M[5,7] = -self.gamma0
		M[6,6] = self.epsA2
		M[6,7] = self.gamma1
		M[7,7] = self.epsB1
		M = M + Tx
		M = M + M.conj().T - np.diag(np.diag(M))

		return M 

	def ret_Ty(self,kx):
		# non-hermitian : only couples along -ve y direction
		Ty = np.zeros((8,8),dtype = np.cdouble)
		Ty[0,2] = -self.gamma0
		Ty[0,3] = self.gamma4
		Ty[0,5] = -self.gamma3 * np.exp(-1j*kx)
		Ty[1,2] = self.gamma4
		Ty[1,3] = -self.gamma0
		Ty[1,4] = -self.gamma3
		Ty[6,4] = -self.gamma0
		Ty[6,5] = self.gamma4
		Ty[7,4] = self.gamma4
		Ty[7,5] = -self.gamma0

		return Ty



if __name__ == '__main__':
	graphene = Graphene()
	print(graphene.epsB)
	print(graphene.ret_H0(0.1))
	blg = BLG()	
	print(blg.gamma4, blg.epsA2, blg.gamma4)
	print(blg.ret_H0(0))
