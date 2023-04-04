import numpy as np
import scipy.integrate as integrate
import sys

sys.path.append('../../src')
from utils import print_msg

def get_test_prob(prob_num):
    """
    Generates the functions needed for a test_problem
    """
    
    if prob_num == 0:
        """
        Smooth-ish in space and angle
        """
                
        def kappa(x, y):
            return 1.0
        
        def sigma(x, y):
            return 0.1 * kappa(x, y)

        g = 0.8
        def Phi_HG(th, phi):
            return (1. - g**2) / (1. + g**2 - 2. * g * np.cos(th - phi))**(3./2.)
        
        [coeff, err] = integrate.quad(lambda phi: Phi_HG(2. * np.pi, phi),
                                      0., 2. * np.pi)
        
        def Phi(th, phi):
            cosTh = np.cos(th - phi)
            
            return (1. / coeff) * Phi_HG(th, phi)
        
        def bcs(x, y, th):
            ang_min = 3.0 * np.pi / 2.0 - 0.1
            ang_max = 3.0 * np.pi / 2.0 + 0.1
            if (y == 2.0) and (ang_min <= th) and (th <= ang_max):
                return 10.0
            else:
                return 0.0
        
        def f(x, y):
            return 0

        

    else:
        print_msg('Problem number {} unsupported.'.format(prob_num))
        quit()
        
    return [kappa, sigma, Phi, bcs, f]
