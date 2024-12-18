import numpy as np 

def Gkx(px, omega, eta, m, v3):
    '''
    In these units, m = [1/2m + v3/4sqrt3], v3 = v3, where each RHS is according to the mccann-koshino convention
    furthermore, v3 = sqrt(3) * gamma3 / 2 
    v = sqrt(3) * gamma0 / 2 
    m = gamma1/2v**2 = 2gamma1/3gamma**2
    '''
    return m*np.pi*omega*np.imag((-((-1)**np.floor((np.pi + np.angle(2*m**2*px**2 + 6*m*px*v3 + v3**2 - 2j*m*eta + np.sqrt(v3**4 + 4*m**2*(9*px**2*v3**2 + px*v3*(-4j*eta - 4*omega) - (eta - 1j*omega)**2) + 16*m**3*px**2*(2*px*v3 - 1j*eta - omega) + 4*m*v3**2*(3*px*v3 - 1j*eta - omega)) - 2*m*omega))/(2.*np.pi))/np.sqrt(2*m**2*px**2 + v3**2 + np.sqrt(v3**4 + 4*m**2*(9*px**2*v3**2 + px*v3*(-4j*eta - 4*omega) - (eta - 1j*omega)**2) + 16*m**3*px**2*(2*px*v3 - 1j*eta - omega) + 4*m*v3**2*(3*px*v3 - 1j*eta - omega)) + m*(6*px*v3 - 2j*eta - 2*omega))) + (1j*(-1)**np.floor(np.angle(-2*m**2*px**2 - v3**2 + np.sqrt(v3**4 + 4*m**2*(9*px**2*v3**2 + px*v3*(-4j*eta - 4*omega) - (eta - 1j*omega)**2) + 16*m**3*px**2*(2*px*v3 - 1j*eta - omega) + 4*m*v3**2*(3*px*v3 - 1j*eta - omega)) + 2*m*(-3*px*v3 + 1j*eta + omega))/(2.*np.pi)))/np.sqrt(-2*m**2*px**2 - v3**2 + np.sqrt(v3**4 + 4*m**2*(9*px**2*v3**2 + px*v3*(-4j*eta - 4*omega) - (eta - 1j*omega)**2) + 16*m**3*px**2*(2*px*v3 - 1j*eta - omega) + 4*m*v3**2*(3*px*v3 - 1j*eta - omega)) + 2*m*(-3*px*v3 + 1j*eta + omega)))/np.sqrt(v3**4/2. + 2*m**2*(9*px**2*v3**2 + px*v3*(-4j*eta - 4*omega) - (eta - 1j*omega)**2) + 8*m**3*px**2*(2*px*v3 - 1j*eta - omega) + 2*m*v3**2*(3*px*v3 - 1j*eta - omega)))
