import numpy as np 
import cmath

def Gkx(px, omega, eta, m, v3):
    '''
    In these units, m = [1/2m + v3/4sqrt3], v3 = v3, where each RHS is according to the mccann-koshino convention
    furthermore, v3 = sqrt(3) * gamma3 / 2 
    v = sqrt(3) * gamma0 / 2 
    m = gamma1/2v**2 = 2gamma1/3gamma**2
    upto omega = 5, it's more than sufficient, overkill even to just do the infinite integral from -pi to pi, as seen from the plot
    function returns -1/pi Im G(px, omega) 
    '''
    return np.imag(((m*cmath.sqrt(1/(2*m**2*px**2 + 6*m*px*v3 + v3**2 - cmath.sqrt(32*m**3*px**3*v3 + 12*m*px*v3**3 + v3**4 + 4*m**2*(9*px**2*v3**2 - (eta - 1j*omega)**2)))) - m*cmath.sqrt(1/(2*m**2*px**2 + 6*m*px*v3 + v3**2 + cmath.sqrt(32*m**3*px**3*v3 + 12*m*px*v3**3 + v3**4 + 4*m**2*(9*px**2*v3**2 - (eta - 1j*omega)**2)))))*(1j*eta + omega))/cmath.sqrt(16*m**3*px**3*v3 + 6*m*px*v3**3 + v3**4/2. + 2*m**2*(9*px**2*v3**2 - (eta - 1j*omega)**2))) 
