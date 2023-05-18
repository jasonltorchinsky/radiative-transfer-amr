import numpy as np
from scipy.special import erf, erfi
import sys

sys.path.append('../../src')
from utils import print_msg

def get_cons_prob(prob_name, prob_num, mesh):
    """
    Generates the functions needed for a contructed analytic solution for a
    test (sub-)problem.
    """

    [Lx, Ly] = mesh.Ls[:]
    
    if prob_num == 0:
        """
        Smooth-ish in space and angle
        """
        def u(x, y, th):
            return np.exp(-((th - 3. * np.pi / 2.)**2)) \
                * np.exp(-((x - Lx/3.)**2 + (y - Ly/3.)**2))
        
        def kappa(x, y):
            return 1. + np.exp(-((x - Lx/2.)**2 + (y - Ly/2.)**2))
        
        def sigma(x, y):
            return 0.9 * kappa(x, y)
        
        def Phi(th, phi):
            return (1.0 / (3.0 * np.pi)) * (1 + (np.cos(th - phi))**2)
        
        def f_mass(x, y, th): # *JUST* kappa * u
            return kappa(x, y) * u(x, y, th)

        erf0 = erf(np.pi / 2)
        erf1 = erf(3. * np.pi / 2.)
        erfi0 = erfi(1. + 1.j * np.pi / 2.)
        erfi1 = erfi(1. - 1.j * 3. * np.pi / 2.)
        erfi2 = erfi(1. - 1.j * np.pi / 2.)
        erfi3 = erfi(1. + 1.j * 3. * np.pi / 2.)
        def f_scat(x, y, th): # *JUST* sigma * int_0^2pi Phi(th, th') * u(x, y, th') dth'
            return sigma(x, y) * np.exp(-((x - Lx/3.)**2 + (y - Ly/3.)**2)) \
                * np.real(
                    ( (1. / (24. * np.sqrt(np.pi)))
                      * ( 6. * (erf0 + erf1)
                          + 1.j * np.exp(-1 - 2.j * th) * (erfi0 - erfi1)
                          + np.exp(4.j * th) * (- erfi2 + erfi3) )
                     )
                    )

        def f_conv(x, y, th): # *JUST* s.grad(u)
            return -2. * u(x, y, th) \
                * ((x - Lx/3.) * np.cos(th) + (y - Ly/3.) * np.sin(th))

        def u_intg_th(x, y):
            return 0.5 * np.sqrt(np.pi) * (erf0 + erf1) \
                * np.exp(-((x - Lx/3.)**2 + (y - Ly/3.)**2))

        erfx0 = erf(Lx / 3.)
        erfx1 = erf(2. * Lx/3.)
        erfy0 = erf(Ly / 3.)
        erfy1 = erf(2. * Ly / 3.)
        def u_intg_xy(th):
            return 0.25 * np.pi * (erfx0 + erfx1) * (erfy0 + erfy1) \
                * np.exp(-((th - 3. * np.pi / 2.)**2))

    elif prob_num == 1:
        """
        Smooth-ish in angle, steep gradient in space.
        """
        
        def u(x, y, th):
            return (np.cos(th))**2 * np.exp(-(32. * (x - Lx/3.)**2
                                              + 16. * (y - Ly/2.)**2))
        
        def kappa(x, y):
            return np.exp(-((x - Lx/2.)**2 + (y - Ly/2.)**2)) + 1.
        
        def sigma(x, y):
            return 0.9 * kappa(x, y)
        
        def Phi(th, phi):
            return (1.0 / (3.0 * np.pi)) * (1 + (np.cos(th - phi))**2)
        
        def f_mass(x, y, th): # *JUST* kappa * u
            return kappa(x, y) * u(x, y, th)

        def f_scat(x, y, th): # *JUST* sigma * int_0^2pi Phi(th, th') * u(x, y, th') dth'
            return sigma(x, y) * (1. / 12.) * (6. + np.cos(2. * th)) \
                * np.exp(-(32. * (x - Lx/3.)**2 + 16. * (y - Ly/2.)**2))

        def f_conv(x, y, th): # *JUST* s.grad(u)
            return u(x, y, th) * (-64. * (x - Lx/3.) * np.cos(th)
                                        + 32. * (y - Ly/2.) * np.sin(th))

        def u_intg_th(x, y):
            return np.pi * np.exp(-(32. * (x - Lx/3.)**2
                                    + 16. * (y - Ly/2.)**2))
        
        erfx = erf(4. * np.sqrt(2) * Lx/3.) + erf(8. * np.sqrt(2) * Lx/3.)
        erfy = erf(2. * Ly)
        def u_intg_xy(th):
            return (np.pi / (32. * np.sqrt(2.))) * erfx * erfy * (np.cos(th))**2

    elif prob_num == 2:
        """
        Flat in space, steep gradient in angle.
        """

        def u(x, y, th):
            return np.exp(-30. * (th - 7. * np.pi / 5.)**2)
        
        def kappa(x, y):
            return 1.1
        
        def sigma(x, y):
            return 0.9 * kappa(x, y)
        
        def Phi(th, phi):
            return (1.0 / (3.0 * np.pi)) * (1 + (np.cos(th - phi))**2)
        
        def f_mass(x, y, th): # *JUST* kappa * u
            return kappa(x, y) * u(x, y, th)

        # Calculate the error function terms outside of the function evaluation
        erf0 = erf(3. * np.sqrt(6. / 5.) * np.pi)
        erf1 = erf(7. * np.sqrt(6. / 5.) * np.pi)
        erf2 = erf((1.j - 18. * np.pi) / np.sqrt(30.))
        erf3 = erf((1.j + 42. * np.pi) / np.sqrt(30.))
        erfi0 = erfi((1. - (18.j * np.pi)) / np.sqrt(30.))
        erfi1 = erfi((1. + (42.j * np.pi)) / np.sqrt(30.))
        
        def f_scat(x, y, th): # *JUST* sigma * int_0^2pi Phi(th, th') * u(x, y, th') dth'
            coeff0 = (1. / (24. * np.sqrt(30. * np.pi))) \
                * np.exp(-(1. / 30.) - 2.j * th)
            term0 = 6. * np.exp((1. / 30.) + 2.j * th) * (erf0 + erf1)
            term1 = np.exp((4. * np.pi / 5.) * 1.j) * (-erf2 + erf3)
            term2 = np.exp(((7. * np.pi / 10.) + 4. * th) * 1.j) * (-erfi0 + erfi1)
            return sigma(x, y) * np.real(coeff0 * (term0 + term1 + term2))

        def f_conv(x, y, th): # *JUST* s.grad(u)
            return 0.

        def u_intg_th(x, y):
            return 0.5 * np.sqrt(np.pi / 30.) * (erf0 + erf1)

        def u_intg_xy(th):
            return Lx * Ly * np.exp(-30. * (th - 7. * np.pi / 5.)**2)
    
    elif prob_num == 3:
        """
        Steep gradient in space and angle.
        """

        def u(x, y, th):
            return np.exp(-30. * (th - 7. * np.pi / 5.)**2) \
                * np.exp(-(32. * (x - Lx/3.)**2 + 16. * (y - Ly/2.)**2))
        
        def kappa(x, y):
            return np.exp(-((x - Lx/2.)**2 + (y - Ly/2.)**2)) + 1.
        
        def sigma(x, y):
            return 0.9 * kappa(x, y)
        
        def Phi(th, phi):
            return (1.0 / (3.0 * np.pi)) * (1 + (np.cos(th - phi))**2)
        
        def f_mass(x, y, th): # *JUST* kappa * u
            return kappa(x, y) * u(x, y, th)

        # Calculate the error function terms outside of the function evaluation
        erf0 = erf(3. * np.sqrt(6. / 5.) * np.pi)
        erf1 = erf(7. * np.sqrt(6. / 5.) * np.pi)
        erf2 = erf((1.j - 18. * np.pi) / np.sqrt(30.))
        erf3 = erf((1.j + 42. * np.pi) / np.sqrt(30.))
        erfi0 = erfi((1. - (18.j * np.pi)) / np.sqrt(30.))
        erfi1 = erfi((1. + (42.j * np.pi)) / np.sqrt(30.))
        
        def f_scat(x, y, th): # *JUST* sigma * int_0^2pi Phi(th, th') * u(x, y, th') dth'
            coeff0 = (1. / (24. * np.sqrt(30. * np.pi))) \
                * np.exp(-(1. / 30.) - 2.j * th)
            term0 = 6. * np.exp((1. / 30.) + 2.j * th) * (erf0 + erf1)
            term1 = np.exp((4. * np.pi / 5.) * 1.j) * (-erf2 + erf3)
            term2 = np.exp(((7. * np.pi / 10.) + 4. * th) * 1.j) * (-erfi0 + erfi1)
            return sigma(x, y) * np.real(coeff0 * (term0 + term1 + term2)) \
                * np.exp(-(32. * (x - Lx/3.)**2 + 16. * (y - Ly/2.)**2))

        def f_conv(x, y, th): # *JUST* s.grad(u)
            return -1. * u(x, y, th) \
                * (64. * (x - Lx/3.) * np.cos(th) + 32. * (y - Ly/2.) * np.sin(th))

        def u_intg_th(x, y):
            return 0.5 * np.sqrt(np.pi / 30.) * (erf0 + erf1) \
                * np.exp(-(32. * (x - Lx/3.)**2 + 16. * (y - Ly/2.)**2))

        erfx = erf(4. * np.sqrt(2) * Lx/3.) + erf(8. * np.sqrt(2) * Lx/3.)
        erfy = erf(2. * Ly)
        def u_intg_xy(th):
            return (np.pi / (32. * np.sqrt(2.))) * erfx * erfy \
                * np.exp(-30. * (th - 7. * np.pi / 5.)**2)
        

    elif prob_num == 4:
        """
        Flat in space, gentle in angle.
        """
        
        def u(x, y, th):
            return (1. / 2.) * (np.sin(th / 2.))**2
        
        def kappa(x, y):
            return np.exp(-((x - 1.5)**2 + (y - 1.)**2)) + 1.
        
        def sigma(x, y):
            return 0.1 * kappa(x, y)
        
        def Phi(th, phi):
            return (1.0 / (3.0 * np.pi)) * (1 + (np.cos(th - phi))**2)
        
        def f_mass(x, y, th): # *JUST* kappa * u
            return kappa(x, y) * u(x, y, th)

        def f_scat(x, y, th): # *JUST* sigma * int_0^2pi Phi(th, th') * u(x, y, th') dth'
            return sigma(x, y) * (1. / 4.)

        def f_conv(x, y, th): # *JUST* s.grad(u)
            return 0

        def u_intg_th(x, y):
            return np.pi / 2
        
        def u_intg_xy(th):
            return (1. / 2.) * (np.sin(th / 2.))**2
            
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
        
    return [u, kappa, sigma, Phi, f, u_intg_th, u_intg_xy]
