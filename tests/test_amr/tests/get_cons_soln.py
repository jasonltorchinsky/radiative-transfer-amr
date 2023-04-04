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
        Smooth-ish in space and angle
        """
        def anl_sol(x, y, th):
            return np.sin(th)**2 * np.exp(-((x - 1.5)**2 + (y - 1.)**2))
        
        def kappa(x, y):
            return (x + 1.)**6 * (np.sin(18. * np.pi * y))**2 + 1.
        
        def sigma(x, y):
            return 0.1 * kappa(x, y)
        
        def Phi(th, phi):
            return (1.0 / (3.0 * np.pi)) * (1 + (np.cos(th - phi))**2)
        
        def f_mass(x, y, th): # *JUST* kappa * u
            return kappa(x, y) * anl_sol(x, y, th)

        def f_scat(x, y, th): # *JUST* sigma * int_0^2pi Phi(th, th') * u(x, y, th') dth'
            return sigma(x, y) * np.exp(-((x - 1.5)**2 + (y - 1.)**2)) \
                * (1. / 12.) * (6. - np.cos(2. * th))

        def f_conv(x, y, th): # *JUST* s.grad(u)
            return -2. * anl_sol(x, y, th) \
                * ((x - 1.5) * np.cos(th) + (y - 1.) * np.sin(th))

        def anl_sol_intg_th(x, y):
            return np.pi * np.exp(-((x - 1.5)**2 + (y - 1.)**2))

    elif sol_num == 1:
        """
        Smooth-ish in angle, steep gradient in space.
        """
        
        def anl_sol(x, y, th):
            return (np.cos(th))**2 * np.exp(-((x - 1.5)**4 + (y - 1.)**4))
        
        def kappa(x, y):
            return np.exp(-((x - 1.5)**2 + (y - 1.)**2)) + 1.
        
        def sigma(x, y):
            return 0.1 * kappa(x, y)
        
        def Phi(th, phi):
            return (1.0 / (3.0 * np.pi)) * (1 + (np.cos(th - phi))**2)
        
        def f_mass(x, y, th): # *JUST* kappa * u
            return kappa(x, y) * anl_sol(x, y, th)

        def f_scat(x, y, th): # *JUST* sigma * int_0^2pi Phi(th, th') * u(x, y, th') dth'
            return sigma(x, y) * np.exp(-((x - 1.5)**4 + (y - 1.)**4)) \
                * (1. / 12.) * (6. + np.cos(2. * th))

        def f_conv(x, y, th): # *JUST* s.grad(u)
            return -4. * anl_sol(x, y, th) \
                * ((x - 1.5)**3 * np.cos(th) + (y - 1.)**3 * np.sin(th))

        def anl_sol_intg_th(x, y):
            return np.pi * np.exp(-((x - 1.5)**4 + (y - 1.)**4))

    elif sol_num == 2:
        """
        Smooth-ish in space, steep gradient in angle.
        """
        
        def anl_sol(x, y, th):
            return (np.cos(th/2 - (3. * np.pi / 5.)))**32 \
                * np.exp(-((x - 1.5)**2 + (y - 1.)**2))
        
        def kappa(x, y):
            return np.exp(-((x - 1.5)**2 + (y - 1.)**2)) + 1.
        
        def sigma(x, y):
            return 0.1 * kappa(x, y)
        
        def Phi(th, phi):
            return (1.0 / (3.0 * np.pi)) * (1 + (np.cos(th - phi))**2)
        
        def f_mass(x, y, th): # *JUST* kappa * u
            return kappa(x, y) * anl_sol(x, y, th)

        def f_scat(x, y, th): # *JUST* sigma * int_0^2pi Phi(th, th') * u(x, y, th') dth'
            return sigma(x, y) * np.exp(-((x - 1.5)**4 + (y - 1.)**4)) \
                * (1964315. / 2147483648.) * (153. + 40. * np.sin(2. * th + np.pi / 10.))

        def f_conv(x, y, th): # *JUST* s.grad(u)
            return -2. * anl_sol(x, y, th) \
                * ((x - 1.5) * np.cos(th) + (y - 1.) * np.sin(th))

        def anl_sol_intg_th(x, y):
            return (300540195. * np.pi / 1073741824.) \
                * np.exp(-((x - 1.5)**2 + (y - 1.)**2))
    
    elif sol_num == 3:
        """
        Flat in space, steep gradient in angle.
        """
        
        def anl_sol(x, y, th):
            return (np.cos(th/2 - (3. * np.pi / 5.)))**32
        
        def kappa(x, y):
            return np.exp(-((x - 1.5)**2 + (y - 1.)**2)) + 1.
        
        def sigma(x, y):
            return 0.1 * kappa(x, y)
        
        def Phi(th, phi):
            return (1.0 / (3.0 * np.pi)) * (1 + (np.cos(th - phi))**2)
        
        def f_mass(x, y, th): # *JUST* kappa * u
            return kappa(x, y) * anl_sol(x, y, th)

        def f_scat(x, y, th): # *JUST* sigma * int_0^2pi Phi(th, th') * u(x, y, th') dth'
            return sigma(x, y) * (1964315. / 2147483648.) \
                * (153. + 40. * np.sin(2. * th + np.pi / 10.))

        def f_conv(x, y, th): # *JUST* s.grad(u)
            return 0

        def anl_sol_intg_th(x, y):
            return (300540195. * np.pi / 1073741824.)

    elif sol_num == 4:
        """
        Flat in space, gentle in angle.
        """
        
        def anl_sol(x, y, th):
            return (1. / 2.) * (np.sin(th / 2.))**2
        
        def kappa(x, y):
            return np.exp(-((x - 1.5)**2 + (y - 1.)**2)) + 1.
        
        def sigma(x, y):
            return 0.1 * kappa(x, y)
        
        def Phi(th, phi):
            return (1.0 / (3.0 * np.pi)) * (1 + (np.cos(th - phi))**2)
        
        def f_mass(x, y, th): # *JUST* kappa * u
            return kappa(x, y) * anl_sol(x, y, th)

        def f_scat(x, y, th): # *JUST* sigma * int_0^2pi Phi(th, th') * u(x, y, th') dth'
            return sigma(x, y) * (1. / 4.)

        def f_conv(x, y, th): # *JUST* s.grad(u)
            return 0

        def anl_sol_intg_th(x, y):
            return np.pi / 2
        
    elif sol_num == 5: # NOT FUNCTIONING
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

    if prob_name == 'mass':
        def f(x, y, th):
            return f_mass(x, y, th)
        
    elif prob_name == 'scat':
        def f(x, y, th):
            return f_mass(x, y, th) - f_scat(x, y, th)
        
    elif prob_name == 'conv':
        def f(x, y, th):
            return f_conv(x, y, th)
        
    elif prob_name == 'comp':
        # s.grad(u) + kappa * u - sigma * int_0^2pi Phi u dth' = f
        def f(x, y, th):
            return f_conv(x, y, th) + f_mass(x, y, th) - f_scat(x, y, th)
        
    else:
        msg = 'ERROR: Problem name {} is unsupported.'.format(prob_name)
        print_msg(msg)
        quit()
        
    return [anl_sol, kappa, sigma, Phi, f, anl_sol_intg_th]
