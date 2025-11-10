import numpy as np 
from FastRGF.solveRGF import MOMfastrecGfull
from utils.models import BLG
from scipy.linalg import norm

'''
Tests that NLDOS ImG(x, omega) = ImG(-x, omega)
All that's needed to see is that ReG(kx) = ReG(-kx)
'''
blg = BLG()
omega = 0
kx = np.random.rand(1) * np.pi * 2 - np.pi
H0 = blg.ret_H0(kx)
Ty = blg.ret_Ty(kx)
delta = 1e-6
Gkx = MOMfastrecGfull(omega,H0,Ty,delta=delta, RECURSIONS=30,dochecks=False)
kxminus = -kx
H0minus = blg.ret_H0(kxminus)
Tyminus = blg.ret_Ty(kxminus)
Gkxminus = MOMfastrecGfull(omega,H0minus,Tyminus,delta=delta,RECURSIONS=30,dochecks=False)

print('kx = ', kx)
print(Gkx[0,1].real, Gkxminus[0,1].real)
print(norm(Gkx.real - Gkxminus.real))


