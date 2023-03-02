import numpy as np
from scipy.special import ellipe
import sys

sys.path.append('../../src')
from utils import print_msg

def get_cons_soln(prob_name, sol_num):
    """
    Generates the functions needed for a contructed analytic solution for a
    test (sub-)problem.
    """
    
    if sol_num == 0:
        """
        Sinusoidal * Gaussian analytic solution.
        Highly oscillatory extinction coefficient.
        Highly oscillatory scattering coefficient (0.1 * kappa).
        Rayleigh scattering.
        """
        
        # Every sub-problem can have the same analytic solution,
        # extinction function, scattering coefficient, and phase function
        # The forcing term will change
        def anl_sol(x, y, th):
            return np.sin(th)**2 * np.exp(-(x**2 + y**2))
        
        def kappa(x, y):
            return (x + 1.)**6 * (np.sin(18. * np.pi * y))**2 + 1.
        
        def sigma(x, y):
            return 0.1 * kappa(x, y)
        
        def Phi(theta, phi):
            cosTh = np.cos(theta) * np.cos(phi) + np.sin(theta) * np.sin(phi)
            return (1.0 / (3.0 * np.pi)) * (1 + cosTh**2)
        
        if prob_name == 'mass':
            # kappa * u = f
            def f(x, y, th):
                return kappa(x, y) * anl_sol(x, y, th)
            
        elif prob_name == 'scat':
            # kappa * u - sigma * int_0^2pi Phi u dth' = f
            def f(x, y, th):
                return kappa(x, y) * anl_sol(x, y, th) \
                    + ((1. / 120.) * np.exp(-(x**2 + y**2)) \
                       * (np.cos(2. * th) - 6.) * kappa(x, y))
            
        elif prob_name == 'conv':
            # s.grad(u) = f
            def f(x, y, th):
                return -2. * anl_sol(x, y, th) * (x * np.cos(th) + y * np.sin(th))
            
        elif prob_name == 'comp':
            # s.grad(u) + kappa * u - sigma * int_0^2pi Phi u dth' = f
            def f(x, y, th):
                return -2. * anl_sol(x, y, th) * (x * np.cos(th) + y * np.sin(th)) \
                    + kappa(x, y) * anl_sol(x, y, th) \
                    + ((1. / 120.) * np.exp(-(x**2 + y**2)) \
                       * (np.cos(2. * th) - 6.) * kappa(x, y))
            
        else:
            msg = 'ERROR: Problem name {} is unsupported.'.format(prob_name)
            print_msg(msg)
            quit()

    elif sol_num == 1:
        """
        Sinusoidal * Gaussian analytic solution.
        Slightly oscillatory extinction coefficient.
        Slightly oscillatory scattering coefficient (0.1 * kappa).
        Rayleigh scattering.
        """
        
        # Every sub-problem can have the same analytic solution,
        # extinction function, scattering coefficient, and phase function
        # The forcing term will change
        def anl_sol(x, y, th):
            return np.sin(th)**2 * np.exp(-(x**2 + y**2))
        
        def kappa(x, y):
            return (x + 1.)**2 * (np.sin(2. * np.pi * y))**2 + 1.
        
        def sigma(x, y):
            return 0.1 * kappa(x, y)
        
        def Phi(theta, phi):
            cosTh = np.cos(theta) * np.cos(phi) + np.sin(theta) * np.sin(phi)
            return (1.0 / (3.0 * np.pi)) * (1 + cosTh**2)
        
        if prob_name == 'mass':
            # kappa * u = f
            def f(x, y, th):
                return kappa(x, y) * anl_sol(x, y, th)
            
        elif prob_name == 'scat':
            # kappa * u - sigma * int_0^2pi Phi u dth' = f
            def f(x, y, th):
                return kappa(x, y) * anl_sol(x, y, th) \
                    - ((1. / 120.) * np.exp(-(x**2 + y**2)) \
                       * (np.cos(2. * th) - 6.) * kappa(x, y))
            
        elif prob_name == 'conv':
            # s.grad(u) = f
            def f(x, y, th):
                return -2. * anl_sol(x, y, th) \
                    * (x * np.cos(th) + y * np.sin(th))
            
        elif prob_name == 'comp':
            # s.grad(u) + kappa * u - sigma * int_0^2pi Phi u dth' = f
            def f(x, y, th):
                return -2. * anl_sol(x, y, th) * (x * np.cos(th) + y * np.sin(th)) \
                    + kappa(x, y) * anl_sol(x, y, th) \
                    + ((1. / 120.) * np.exp(-(x**2 + y**2)) \
                       * (np.cos(2. * th) - 6.) * kappa(x, y))
            
        else:
            msg = 'ERROR: Problem name {} is unsupported.'.format(prob_name)
            print_msg(msg)
            quit()
            
    elif sol_num == 2: # NOT FUNCTIONING
        """
        Sinusoidal * Gaussian analytic solution.
        Highly oscillatory extinction coefficient.
        Highly oscillatory scattering coefficient (0.1 * kappa).
        Henyey-Greenstein scattering.
        """
        
        # Every sub-problem can have the same analytic solution,
        # extinction function, scattering coefficient, and phase function
        # The forcing term will change
        def anl_sol(x, y, th):
            return np.sin(th)**2 * np.exp(-(x**2 + y**2))
        
        def kappa(x, y):
            return (x + 1.)**6 * (np.sin(18. * np.pi * y))**2 + 1.
        
        def sigma(x, y):
            return 0.1 * kappa(x, y)
        
        def Phi(theta, phi):
            cosTh = np.cos(theta) * np.cos(phi) + np.sin(theta) * np.sin(phi)
            
            g = 0.8
            coeff = (1. - g) * ellipe((-4. * g) / (1. - g)**2) \
                + (1. + g) * ellipe((4. * g) / (1. + g)**2)
            numer = (1. - g**2)**2
            denom = 2. * (1. + g**2 - 2. * cosTh)**(3./2.)
            
            return (1.0 / coeff) * numer / denom
        
        if prob_name == 'mass':
            # kappa * u = f
            def f(x, y, th):
                return kappa(x, y) * anl_sol(x, y, th)
            
        elif prob_name == 'scat':
            # kappa * u - sigma * int_0^2pi Phi u dth' = f
            def f(x, y, th):
                return kappa(x, y) * anl_sol(x, y, th) \
                    - ()
            
        elif prob_name == 'conv':
            # s.grad(u) = f
            def f(x, y, th):
                return -2. * anl_sol(x, y, th) \
                    * (x * np.cos(th) + y * np.sin(th))
            
        elif prob_name == 'comp':
            # s.grad(u) + kappa * u - sigma * int_0^2pi Phi u dth' = f
            def f(x, y, th):
                return -2. * anl_sol(x, y, th) * (x * np.cos(th) + y * np.sin(th)) \
                    + kappa(x, y) * anl_sol(x, y, th) \
                    - ()

            (1. / 120.) * np.exp(-(x**2 + y**2)) \
                    * ((54. - 59. * np.cos(2. * th)) * kappa(x, y) \
                       - 60. * (1. - np.cos(2. * th)) \
                       - 240. * (np.sin(th))**2 * (x * np.cos(th)
                                                   + y * np.sin(th)
                                                   - 0.5))
            
        else:
            msg = 'ERROR: Problem name {} is unsupported.'.format(prob_name)
            print_msg(msg)
            quit()
            
    else:
        msg = 'ERROR: Solution number {} is unsupported.'.format(sol_num)
        print_msg(msg)
        quit()
        
    return [anl_sol, kappa, sigma, Phi, f]
