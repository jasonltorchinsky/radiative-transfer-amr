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
    [x_num, y_num, th_num] = prob_num[:]
    # 0 - Constant
    # 1 - Linear
    # 2 - Shallow gradient
    # 3 - Steep gradient

    # Get x-part
    if x_num == 0:
        def X(x):
            return 1.
        
        def dXdx(x):
            return 0.
        
        X_intg = Lx
        
        def kappa_x(x):
            return 1.
        
    elif x_num == 1:
        def X(x):
            return (1. / (10. * Lx)) * x
        
        def dXdx(x):
            return (1. / (10. * Lx))
        
        X_intg = Lx / 20.
        
        def kappa_x(x):
            return (x / Lx**2) * (5.6 * Lx - 5.5 * x) + 0.1

    elif x_num == 2:
        def X(x):
            return np.exp(-((1. / Lx) * (x - (Lx / 3.)))**2)
        
        def dXdx(x):
            return -(2. / Lx**2) * (x - (Lx / 3.)) * X(x)
        
        erf0 = erf(1. / 3.)
        erf1 = erf(2. / 3.)

        X_intg = (1. / 2.) * Lx * np.sqrt(np.pi) * np.real(erf0 + erf1) 
        
        def kappa_x(x):
            return 0.4 * np.exp((4. / Lx**2) * x * (x - Lx)) + 0.6

    elif x_num == 3:
        def X(x):
            return np.exp(-((16. / Lx) * (x - (Lx / 3.)))**2)
        
        def dXdx(x):
            return -(512. / Lx**2) * (x - (Lx / 3.)) * X(x)
        
        erf0 = erf(16. / 3.)
        erf1 = erf(32. / 3.)
        
        X_intg = (1. / 32.) * Lx * np.sqrt(np.pi) * np.real(erf0 + erf1) 
        
        def kappa_x(x):
            return np.exp(-((8. / Lx) * (x - (5. * Lx / 7.)))**2) + 1.0

    # Get y-part
    if y_num == 0:
        def Y(y):
            return 1.
        
        def dYdy(y):
            return 0.
        
        Y_intg = Ly
        
        def kappa_y(y):
            return 1.
        
    elif y_num == 1:
        def Y(y):
            return (1. / (6. * Ly)) * y
        
        def dYdy(y):
            return (1. / (6. * Ly))
        
        Y_intg = Ly / 12.
        
        def kappa_y(y):
            return (y / Ly**2) * (5.4 * Ly - 5.5 * y) + 0.2

    elif y_num == 2:
        def Y(y):
            return np.exp(-((1. / Ly) * (y - (2. * Ly / 3.)))**2)
        
        def dYdy(y):
            return -(2. / Ly**2) * (y - (2. * Ly / 3.)) * Y(y)
        
        erf0 = erf(1. / 3.)
        erf1 = erf(2. / 3.)
        Y_intg = (1. / 2.) * Ly * np.sqrt(np.pi) * np.real(erf0 + erf1) 
        
        def kappa_y(y):
            return 0.3 * np.exp((1. / Ly**2) * y * (8.75 * y - 8.25 * Ly)) + 0.5

    elif y_num == 3:
        def Y(y):
            return np.exp(-((16. / Ly) * (y - (2. * Ly / 3.)))**2)
        
        def dYdy(y):
            return -(512. / Ly**2) * (y - (2. * Ly / 3.)) * Y(y)
        
        erf0 = erf(32. / 3.)
        erf1 = erf(64. / 3.)
        
        Y_intg = (1. / 64.) * Ly * np.sqrt(np.pi) * np.real(erf0 + erf1) 
        
        def kappa_y(y):
            return np.exp(-((64. / Ly) * (y - (Ly / 5.)))**2) + 1.0

    # Get th-part
    if th_num == 0:
        def TH(th):
            return 1.
        
        TH_intg = 2. * np.pi

        def TH_scat(th):
            return 1
        
    elif th_num == 1:
        def TH(th):
            return 1. - np.abs((th - np.pi) / (4. * np.pi))

        TH_intg = 7. * np.pi / 4.

        def TH_scat(th):
            return 7. / 8.

    elif th_num == 2:
        def TH(th):
            return (np.sin((th / 2.) - (7. * np.pi / 5.)))**2
        
        TH_intg = np.pi

        def TH_scat(th):
            return 1. / 2.

    elif th_num == 3:
        def TH(th):
            return np.exp(-((12. / np.pi) * (th - (7. * np.pi / 5.)))**2)
        
        erf0 = erf(36. / 5.)
        erf1 = erf(84. / 5.)
        erf01 = erf0 + erf1
        
        TH_intg = (1. / 24.) * np.pi**(3. / 2.) * np.real(erf01) 
        
        erf2 = erf((84. / 5.) - (1.j * np.pi) / 12.)
        erf3 = erf((36. / 5.) + (1.j * np.pi) / 12.)
        erf23 = erf2 + erf3
        
        erf4 = erf((36. / 5.) - (1.j * np.pi) / 12.)
        erf5 = erf((84. / 5.) + (1.j * np.pi) / 12.)
        erf45 = erf4 + erf5

        def TH_scat(th):
            val = (1. / 288.) * np.sqrt(np.pi) \
                * (6. * erf01
                   + ( (-1.)**(1./5.) * np.exp(-(np.pi**2 / 144.) - 2.j * th)
                       * (-np.exp(4.j * th) * erf23 + (-1.)**(3./5.) * erf45)
                       )
                   )
            return np.real(val)
        
    def u(x, y, th):
        return X(x) * Y(y) * TH(th)
        
    def kappa(x, y):
        return kappa_x(x) * kappa_y(y)

    def sigma(x, y):
        return 0.9 * kappa(x, y)

    def Phi(th, phi):
        return (1.0 / (3.0 * np.pi)) * (1 + (np.cos(th - phi))**2)

    def f_mass(x, y, th):
        return kappa(x, y) * u(x, y, th)

    def f_scat(x, y, th):
        return sigma(x, y) * X(x) * Y(y) * TH_scat(th)

    def f_conv(x, y, th):
        return (dXdx(x) * Y(y) * np.cos(th) + X(x) * dYdy(y) * np.sin(th)) * TH(th)

    def u_intg_th(x, y):
        return X(x) * Y(y) * TH_intg

    def u_intg_xy(th):
        return X_intg * Y_intg * TH(th)

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
