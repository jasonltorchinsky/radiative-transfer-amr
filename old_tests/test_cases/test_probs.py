import numpy as np
import scipy.integrate as integrate
import sys

sys.path.append("../../src")
from utils import print_msg

def get_test_prob(prob_num, mesh):
    """
    Generates the functions needed for a test_problem
    """
    
    if prob_num == 0:
        """
        HG scattering, step function in angle at top of domain.
        """
                
        def kappa(x, y):
            return 1.1
        
        def sigma(x, y):
            return 1.0

        g = 0.1
        def Phi_HG(th, phi):
            return (1. - g**2) / (1. + g**2 - 2. * g * np.cos(th - phi))**(3./2.)
        
        [coeff, err] = integrate.quad(lambda phi: Phi_HG(2. * np.pi, phi),
                                      0., 2. * np.pi)
        
        def Phi(th, phi):
            return (1. / coeff) * Phi_HG(th, phi)
        
        [x_bot, y_bot] = [0.0, 0.0]
        [x_top, y_top] = mesh.Ls
        
        def bcs(x, y, th):
            ang_min = 3.0 * np.pi / 2.0 - 1.
            ang_max = 3.0 * np.pi / 2.0 + 1.
            if (y == y_top) and (ang_min <= th) and (th <= ang_max):
                return 10.0
            else:
                return 0.0

        dirac = [None, y_top, None]
        
        def f(x, y):
            return 0

    elif prob_num == 1:
        """
        HG scattering, dirac-delta in angle at top of domain.
        """
                
        def kappa(x, y):
            return 1.1
        
        def sigma(x, y):
            return 1.0

        g = 0.8
        def Phi_HG(th, phi):
            return (1. - g**2) / (1. + g**2 - 2. * g * np.cos(th - phi))**(3./2.)
        
        [coeff, err] = integrate.quad(lambda phi: Phi_HG(2. * np.pi, phi),
                                      0., 2. * np.pi)
        
        def Phi(th, phi):
            return (1. / coeff) * Phi_HG(th, phi)
        
        [x_bot, y_bot] = [0.0, 0.0]
        [x_top, y_top] = mesh.Ls[:]
        th_star = 3. * np.pi / 2. + 0.1
        
        def bcs(x, y, th):
            if (y == y_top) and (th == th_star):
                return 1.0
            else:
                return 0.0

        dirac = [None, y_top, th_star]
        
        def f(x, y):
            return 0

    elif prob_num == 2:
        """
        HG scattering, dirac-delta in angle and x at top of domain.
        """
                
        def kappa(x, y):
            return 1.1
        
        def sigma(x, y):
            return 1.0

        g = 0.1
        def Phi_HG(th, phi):
            return (1. - g**2) / (1. + g**2 - 2. * g * np.cos(th - phi))**(3./2.)
        
        [coeff, err] = integrate.quad(lambda phi: Phi_HG(2. * np.pi, phi),
                                      0., 2. * np.pi)
        
        def Phi(th, phi):
            return (1. / coeff) * Phi_HG(th, phi)
        
        [x_bot, y_bot] = [0.0, 0.0]
        [x_top, y_top] = mesh.Ls[:]
        x_star = (3. * x_bot + x_top) / 4.
        th_star = 5. * np.pi / 3.
        
        def bcs(x, y, th):
            if (x == x_star) and (y == y_top) and (th == th_star):
                return 1.0
            else:
                return 0.0

        dirac = [x_star, y_top, th_star]
        
        def f(x, y):
            return 0

    elif prob_num == 3:
        """
        HG scattering, Gaussian in angle at top of domain.
        """
                
        def kappa(x, y):
            return 1.1
        
        def sigma(x, y):
            return 1.0

        g = 0.2
        def Phi_HG(th, phi):
            return (1. - g**2) / (1. + g**2 - 2. * g * np.cos(th - phi))**(3./2.)
        
        [coeff, err] = integrate.quad(lambda phi: Phi_HG(2. * np.pi, phi),
                                      0., 2. * np.pi)
        
        def Phi(th, phi):
            return (1. / coeff) * Phi_HG(th, phi)
        
        [x_bot, y_bot] = [0.0, 0.0]
        [x_top, y_top] = mesh.Ls
        th_star = 7. * np.pi / 4.
        
        def bcs(x, y, th):
            if (y == y_top):
                return np.exp(-(th - th_star)**2)
            else:
                return 0.0

        dirac = [None, y_top, None]
        
        def f(x, y):
            return 0

    else:
        print_msg("Problem number {} unsupported.".format(prob_num))
        quit()
        
    return [kappa, sigma, Phi, [bcs, dirac], f]
