import numpy as np
from scipy.integrate import quad
from scipy.special import erf, erfi
import sys

sys.path.append("../../src")
from utils import print_msg


def get_cons_prob(prob_name, prob_num, mesh, **kwargs):
    """
    Generates the functions needed for a contructed analytic solution for a
    test (sub-)problem.
    """
    
    default_kwargs = {"sx"  : 16.,
                      "sy"  : 24.,
                      "sth" : 32.,
                      "g"   : 0.8}
    kwargs = {**default_kwargs, **kwargs}
    
    [x_num, y_num, th_num, scat_num] = prob_num[:]
    
    [X, dXdx, X_intg, kappa_x] = get_X_part(mesh,  x_num,  **kwargs)
    [Y, dYdy, Y_intg, kappa_y] = get_Y_part(mesh,  y_num,  **kwargs)
    [Th,      Th_intg        ] = get_Th_part(mesh, th_num, **kwargs)
    
    [Phi, Th_scat] = get_scat_part(scat_num, Th)
    
    def u(x, y, th):
        return X(x) * Y(y) * Th(th)
        
    def kappa(x, y):
        return kappa_x(x) * kappa_y(y)
    
    def sigma(x, y):
        return 0.1 * kappa(x, y)
    
    def f_mass(x, y, th):
        return kappa(x, y) * u(x, y, th)
    
    def f_scat(x, y, th):
        return sigma(x, y) * X(x) * Y(y) * Th_scat(th)
    
    def f_conv(x, y, th):
        return (dXdx(x) * Y(y) * np.cos(th) + X(x) * dYdy(y) * np.sin(th)) * Th(th)
    
    def u_intg_th(x, y, th0, th1):
        return X(x) * Y(y) * Th_intg(th0, th1)
    
    def u_intg_xy(x0, x1, y0, y1, th):
        return X_intg(x0, x1) * Y_intg(y0, y1) * Th(th)
    
    if prob_name == "mass":
        def f(x, y, th):
            return f_mass(x, y, th)
        
    elif prob_name == "scat":
        def f(x, y, th):
            return f_mass(x, y, th) - f_scat(x, y, th)
        
    elif prob_name == "conv":
        def f(x, y, th):
            return f_conv(x, y, th)
        
    elif prob_name == "comp":
        # s.grad(u) + kappa * u - sigma * int_0^2pi Phi u dth" = f
        def f(x, y, th):
            return f_conv(x, y, th) + f_mass(x, y, th) - f_scat(x, y, th)
        
    else:
        msg = "ERROR: Problem name {} is unsupported.".format(prob_name)
        print_msg(msg)
        quit()
        
    return [u, kappa, sigma, Phi, f, u_intg_th, u_intg_xy]

def get_X_part(mesh, x_num, **kwargs):
    
    default_kwargs = {"sx" : 16.}
    kwargs = {**default_kwargs, **kwargs}
    
    [Lx, _] = mesh.Ls[:]
    sx = kwargs["sx"]

    if x_num == 0: # Constant
        def X(x):
            return np.ones_like(x)
        def dXdx(x):
            return np.zeros_like(x)
        def X_intg(x0, x1):
            return x1 - x0
        def kappa_x(x):
            return np.ones_like(x)
    elif x_num == 1: # Linear
        def X(x):
            return (1. / (10. * Lx)) * x
        def dXdx(x):
            return (1. / (10. * Lx)) * np.ones_like(x)
        def X_intg(x0, x1):
            return (x1**2 - x0**2) / (20. * Lx)
        def kappa_x(x):
            return (x / Lx**2) * (5.6 * Lx - 5.5 * x) + 0.1
    elif x_num == 2: # Shallow gradient
        def X(x):
            return np.exp(-((1. / Lx) * (x - (Lx / 3.)))**2)
        def dXdx(x):
            return -(2. / Lx**2) * (x - (Lx / 3.)) * X(x)
        def X_intg(x0, x1):
            erf0 = erf((1. / 3.) - (x0 / Lx))
            erf1 = erf((1. / 3.) - (x1 / Lx))
            return (1. / 2.) * Lx * np.sqrt(np.pi) * np.real(erf0 - erf1) 
        def kappa_x(x):
            return 0.4 * np.exp((4. / Lx**2) * x * (x - Lx)) + 0.6
    elif x_num == 3: # Steep gradient
        def X(x):
            return np.exp(-((sx / Lx) * (x - (Lx / 3.)))**2)
        def dXdx(x):
            return -((2. * sx**2) / Lx**2) * (x - Lx / 3.) * X(x)
        def X_intg(x0, x1):
            erf0 = erf(sx * (Lx - 3. * x0) / (3. * Lx))
            erf1 = erf(sx * (Lx - 3. * x1) / (3. * Lx))
            return (1. / (2. * sx)) * Lx * np.sqrt(np.pi) * np.real(erf0 - erf1)
        def kappa_x(x):
            return np.exp(-((sx / (2. * Lx)) * (x - (5. * Lx / 7.)))**2) + 1.0
    
    return [X, dXdx, X_intg, kappa_x]

def get_Y_part(mesh, y_num, **kwargs):
    
    default_kwargs = {"sy" : 24.}
    kwargs = {**default_kwargs, **kwargs}
    
    [Ly, _] = mesh.Ls[:]
    sy = kwargs["sy"]
    
    if y_num == 0: # Constant
        def Y(y):
            return np.ones_like(y)
        def dYdy(y):
            return np.zeros_like(y)
        def Y_intg(y0, y1):
            return y1 - y0
        def kappa_y(y):
            return np.ones_like(y)
    elif y_num == 1: # Linear
        def Y(y):
            return (1. / (6. * Ly)) * y
        def dYdy(y):
            return (1. / (6. * Ly)) * np.ones_like(y)
        def Y_intg(y0, y1):
            return (y1**2 - y0**2) / (12. * Ly)
        def kappa_y(y):
            return (y / Ly**2) * (5.4 * Ly - 5.5 * y) + 0.2
    elif y_num == 2: # Shallow gradient
        def Y(y):
            return np.exp(-((1. / Ly) * (y - (2. * Ly / 3.)))**2)
        def dYdy(y):
            return -(2. / Ly**2) * (y - (2. * Ly / 3.)) * Y(y)
        def Y_intg(y0, y1):
            erf0 = erf((2. / 3.) - (y0 / Ly))
            erf1 = erf((2. / 3.) - (y1 / Ly))
            return (1. / 2.) * Ly * np.sqrt(np.pi) * np.real(erf0 - erf1)
        def kappa_y(y):
            return 0.3 * np.exp((1. / Ly**2) * y * (8.75 * y - 8.25 * Ly)) + 0.5
    elif y_num == 3: # Steep gradient
        def Y(y):
            return np.exp(-((sy / Ly) * (y - (2. * Ly / 3.)))**2)
        def dYdy(y):
            return -((2. * sy**2) / Ly**2) * (y - (2. * Ly / 3.)) * Y(y)
        def Y_intg(y0, y1):
            erf0 = erf((sy / Ly) * ((2. * Ly / 3.) - y0))
            erf1 = erf((sy / Ly) * ((2. * Ly / 3.) - y1))
            return (1. / (2. * sy)) * Ly * np.sqrt(np.pi) * np.real(erf0 - erf1)
        def kappa_y(y):
            return np.exp(-((2. * sy / Ly) * (y - (Ly / 5.)))**2) + 1.0
        
    return [Y, dYdy, Y_intg, kappa_y]

def get_Th_part(mesh, th_num, **kwargs):
    
    default_kwargs = {"sth" : 32.}
    kwargs = {**default_kwargs, **kwargs}
    
    sth = kwargs["sth"]
    
    if th_num == 0: # Constant
        def Th(th):
            return np.ones_like(th)
        def Th_intg(th0, th1):
            return th1 - th0
    elif th_num == 1: # Linear
        def Th(th):
            return 1. - np.abs((th - np.pi) / (4. * np.pi))
        def Th_intg(th0, th1):
            if th0 > th1:
                [th0, th1] = [th1, th0]
            if (th0 <= np.pi) and (th1 <= np.pi):
                return (th1**2 - th0**2) / (8. * np.pi) + (th1 - th0) * (3. / 4)
            elif (th0 > np.pi) and (th1 > np.pi):
                return (th1**2 - th0**2) / (8. * np.pi) + (th1 - th0) * (5. / 4.)
            else:
                return -(th1**2 + th0**2) / (8. * np.pi) + (5. * th1 - 3. * th0 - np.pi) / 4.
    elif th_num == 2: # Shallow gradient
        def Th(th):
            return (np.sin((th / 2.) - (7. * np.pi / 5.)))**2
        def Th_intg(th0, th1):
            sin0 = np.sin((np.pi / 5.) + th0)
            sin1 = np.sin((np.pi / 5.) + th1)
            return 0.5 * ((th1 + sin1) - (th0 + sin0))
    elif th_num == 3: # Steep gradient
        def Th(th):
            return np.exp(-((sth / (2. * np.pi)) * (th - (7. * np.pi / 5.)))**2)
        def Th_intg(th0, th1):
            erf0 = erf((sth / (10. * np.pi)) * (7. * np.pi - 5. * th0))
            erf1 = erf((sth / (10. * np.pi)) * (7. * np.pi - 5. * th1))
            return (1. / sth) * np.pi**(3. / 2.) * np.real(erf0 - erf1) 
    return [Th, Th_intg]

def get_scat_part(scat_num, Th, **kwargs):

    default_kwargs = {"g" : 0.8}
    kwargs = {**default_kwargs, **kwargs}
    
    g = kwargs["g"]
    if scat_num == 0: # Isotropic scattering
        def Phi(th, phi):
            return 1.0 / (2.0 * np.pi) * np.ones_like(th) * np.ones_like(phi)
    elif scat_num == 1: # Rayleigh scattering
        def Phi(th, phi):
            return (1.0 / (3.0 * np.pi)) * (1 + (np.cos(th - phi))**2)
    elif scat_num == 2: # Henyey-Greenstein scattering
        def Phi_HG(Th):
            return (1. - g**2) / (1 + g**2 - 2. * g * np.cos(Th))**(3./2.)
        [norm, abserr] = quad(lambda Th : Phi_HG(Th), 0., 2. * np.pi)

        def Phi(th, phi):
            val = (1. - g**2) / (1 + g**2 - 2. * g * np.cos(th - phi))**(3./2.)
            return val / norm

    def Th_scat(th):
        [val, abserr] = quad(lambda phi: Phi(th, phi) * Th(phi), 0., 2. * np.pi)
        return val

    return [Phi, Th_scat]
    

def get_cons_prob_old(prob_name, prob_num, mesh):
    """
    Generates the functions needed for a contructed analytic solution for a
    test (sub-)problem.
    """

    [Lx, Ly] = mesh.Ls[:]
    [x_num, y_num, th_num] = prob_num[:]
    [sx, sy, sth] = [32., 48., 64.]
    # 0 - Constant
    # 1 - Linear
    # 2 - Shallow gradient
    # 3 - Steep gradient

    # Get x-part
    if x_num == 0:
        def X(x):
            return np.ones_like(x)
        
        def dXdx(x):
            return np.zeros_like(x)
        
        def X_intg(x0, x1):
            return x1 - x0
        
        def kappa_x(x):
            return np.ones_like(x)
        
    elif x_num == 1:
        def X(x):
            return (1. / (10. * Lx)) * x
        
        def dXdx(x):
            return (1. / (10. * Lx)) * np.ones_like(x)
        
        def X_intg(x0, x1):
            return (x1**2 - x0**2) / (20. * Lx)
        
        def kappa_x(x):
            return (x / Lx**2) * (5.6 * Lx - 5.5 * x) + 0.1

    elif x_num == 2:
        def X(x):
            return np.exp(-((1. / Lx) * (x - (Lx / 3.)))**2)
        
        def dXdx(x):
            return -(2. / Lx**2) * (x - (Lx / 3.)) * X(x)
        
        def X_intg(x0, x1):
            erf0 = erf((1. / 3.) - (x0 / Lx))
            erf1 = erf((1. / 3.) - (x1 / Lx))
            return (1. / 2.) * Lx * np.sqrt(np.pi) * np.real(erf0 - erf1) 
        
        def kappa_x(x):
            return 0.4 * np.exp((4. / Lx**2) * x * (x - Lx)) + 0.6

    elif x_num == 3:
        def X(x):
            return np.exp(-((sx / Lx) * (x - (Lx / 3.)))**2)
        
        def dXdx(x):
            return -((2. * sx**2) / Lx**2) * (x - Lx / 3.) * X(x)
        
        def X_intg(x0, x1):
            erf0 = erf(sx * (Lx - 3. * x0) / (3. * Lx))
            erf1 = erf(sx * (Lx - 3. * x1) / (3. * Lx))
            return (1. / (2. * sx)) * Lx * np.sqrt(np.pi) * np.real(erf0 - erf1) 
        
        def kappa_x(x):
            return np.exp(-((sx / (2. * Lx)) * (x - (5. * Lx / 7.)))**2) + 1.0

    # Get y-part
    if y_num == 0:
        def Y(y):
            return np.ones_like(y)
        
        def dYdy(y):
            return np.zeros_like(y)
        
        def Y_intg(y0, y1):
            return y1 - y0
        
        def kappa_y(y):
            return np.ones_like(y)
        
    elif y_num == 1:
        def Y(y):
            return (1. / (6. * Ly)) * y
        
        def dYdy(y):
            return (1. / (6. * Ly)) * np.ones_like(y)
        
        def Y_intg(y0, y1):
            return (y1**2 - y0**2) / (12. * Ly)
        
        def kappa_y(y):
            return (y / Ly**2) * (5.4 * Ly - 5.5 * y) + 0.2

    elif y_num == 2:
        def Y(y):
            return np.exp(-((1. / Ly) * (y - (2. * Ly / 3.)))**2)
        
        def dYdy(y):
            return -(2. / Ly**2) * (y - (2. * Ly / 3.)) * Y(y)
        
        def Y_intg(y0, y1):
            erf0 = erf((2. / 3.) - (y0 / Ly))
            erf1 = erf((2. / 3.) - (y1 / Ly))
            return (1. / 2.) * Ly * np.sqrt(np.pi) * np.real(erf0 - erf1) 
        
        def kappa_y(y):
            return 0.3 * np.exp((1. / Ly**2) * y * (8.75 * y - 8.25 * Ly)) + 0.5

    elif y_num == 3:
        def Y(y):
            return np.exp(-((sy / Ly) * (y - (2. * Ly / 3.)))**2)
        
        def dYdy(y):
            return -((2. * sy**2) / Ly**2) * (y - (2. * Ly / 3.)) * Y(y)
        
        def Y_intg(y0, y1):
            erf0 = erf((sy / Ly) * ((2. * Ly / 3.) - y0))
            erf1 = erf((sy / Ly) * ((2. * Ly / 3.) - y1))
            return (1. / (2. * sy)) * Ly * np.sqrt(np.pi) * np.real(erf0 - erf1) 
        
        def kappa_y(y):
            return np.exp(-((2. * sy / Ly) * (y - (Ly / 5.)))**2) + 1.0

    # Get th-part
    if th_num == 0:
        def TH(th):
            return np.ones_like(th)
        
        def TH_intg(th0, th1):
            return th1 - th0

        def TH_scat(th):
            return np.ones_like(th)
        
    elif th_num == 1:
        def TH(th):
            return 1. - np.abs((th - np.pi) / (4. * np.pi))

        def TH_intg(th0, th1):
            if th0 > th1:
                [th0, th1] = [th1, th0]

            if (th0 <= np.pi) and (th1 <= np.pi):
                return (th1**2 - th0**2) / (8. * np.pi) + (th1 - th0) * (3. / 4)
            elif (th0 > np.pi) and (th1 > np.pi):
                return (th1**2 - th0**2) / (8. * np.pi) + (th1 - th0) * (5. / 4.)
            else:
                return -(th1**2 + th0**2) / (8. * np.pi) + (5. * th1 - 3. * th0 - np.pi) / 4.

        def TH_scat(th):
            return 7. / 8. * np.ones_like(th)

    elif th_num == 2:
        def TH(th):
            return (np.sin((th / 2.) - (7. * np.pi / 5.)))**2
        
        def TH_intg(th0, th1):
            sin0 = np.sin((np.pi / 5.) + th0)
            sin1 = np.sin((np.pi / 5.) + th1)
            return 0.5 * ((th1 + sin1) - (th0 + sin0))
        
        def TH_scat(th):
            return 1. / 2. * np.ones_like(th)
        
    elif th_num == 3:
        def TH(th):
            return np.exp(-((sth / (2. * np.pi)) * (th - (7. * np.pi / 5.)))**2)
        
        def TH_intg(th0, th1):
            erf0 = erf((sth / (10. * np.pi)) * (7. * np.pi - 5. * th0))
            erf1 = erf((sth / (10. * np.pi)) * (7. * np.pi - 5. * th1))
            return (1. / sth) * np.pi**(3. / 2.) * np.real(erf0 - erf1) 
        
        erf0 = erf(3. * sth / 10.)
        erf1 = erf(7. * sth / 10.)
        
        erfi2 = erfi((2. * np.pi) / sth + (3.j * sth) / 10.)
        erfi3 = erfi((2. * np.pi) / sth - (7.j * sth) / 10.)
        
        erfi4 = erfi((2. * np.pi) / sth - (3.j * sth) / 10.)
        erfi5 = erfi((2. * np.pi) / sth + (7.j * sth) / 10.)

        def TH_scat(th):
            val = (1. / (12. * sth)) * np.sqrt(np.pi) \
                * (6. * (erf0 + erf1) \
                   + (-1.)**(3./10.) \
                     * np.exp(-((4. * np.pi**2)/(sth**2)) - 2.j * th) \
                     * (erfi2 - erfi3 + (-1.)**(2./5.) * np.exp(4.j * th) \
                        * (-erfi4 + erfi5)
                        )
                   )
            return np.real(val)
        
    def u(x, y, th):
        return X(x) * Y(y) * TH(th)
        
    def kappa(x, y):
        return kappa_x(x) * kappa_y(y)

    def sigma(x, y):
        return 0.7 * kappa(x, y)

    def Phi(th, phi):
        return (1.0 / (3.0 * np.pi)) * (1 + (np.cos(th - phi))**2)

    def f_mass(x, y, th):
        return kappa(x, y) * u(x, y, th)

    def f_scat(x, y, th):
        return sigma(x, y) * X(x) * Y(y) * TH_scat(th)

    def f_conv(x, y, th):
        return (dXdx(x) * Y(y) * np.cos(th) + X(x) * dYdy(y) * np.sin(th)) * TH(th)

    def u_intg_th(x, y, th0, th1):
        return X(x) * Y(y) * TH_intg(th0, th1)

    def u_intg_xy(x0, x1, y0, y1, th):
        return X_intg(x0, x1) * Y_intg(y0, y1) * TH(th)

    if prob_name == "mass":
        def f(x, y, th):
            return f_mass(x, y, th)
        
    elif prob_name == "scat":
        def f(x, y, th):
            return f_mass(x, y, th) - f_scat(x, y, th)
        
    elif prob_name == "conv":
        def f(x, y, th):
            return f_conv(x, y, th)
        
    elif prob_name == "comp":
        # s.grad(u) + kappa * u - sigma * int_0^2pi Phi u dth" = f
        def f(x, y, th):
            return f_conv(x, y, th) + f_mass(x, y, th) - f_scat(x, y, th)
        
    else:
        msg = "ERROR: Problem name {} is unsupported.".format(prob_name)
        print_msg(msg)
        quit()
        
    return [u, kappa, sigma, Phi, f, u_intg_th, u_intg_xy]
