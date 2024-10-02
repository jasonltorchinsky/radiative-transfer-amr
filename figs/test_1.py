"""
Test 1: Manufactured Solution.
"""

# Standard Library Imports

# Third-Party Library Imports
import numpy as np
from   scipy.integrate import quad, dblquad

# Local Library Imports


# End-Combo Parameters
max_ndof   = 2**20#2**19    # Max number of DOFs
max_ntrial = 256      # Max number of trials
min_err    = 1.e-6    # Min error before cutoff
max_mem    = 95       # Max memory usage (percentage of 100)

# Mesh parameters common to each combination
[Lx, Ly] = [3., 2.]
pbcs     = [False, False]
has_th   = True
        
# Uniform Angular h-Refinement
h_uni_ang  = {"full_name"  : "Uniform Angular h-Refinement",
              "short_name" : "h-uni-ang",
              "ref_kind"   : "ang",
              "ndofs"      : [5, 5, 3],
              "nref_ang"   : 3,
              "nref_spt"   : 3,
              "Ls"         : [Lx, Ly],
              "pbcs"       : pbcs,
              "has_th"     : has_th,
              "spt_res_offset" : 0,
              "ang_res_offset" : 2
}

# Uniform Angular p-Refinement
p_uni_ang  = {"full_name"  : "Uniform Angular p-Refinement",
              "short_name" : "p-uni-ang",
              "ref_kind"   : "ang",
              "ndofs"      : [5, 5, 3],
              "nref_ang"   : 3,
              "nref_spt"   : 3,
              "Ls"         : [Lx, Ly],
              "pbcs"       : pbcs,
              "has_th"     : has_th,
              "spt_res_offset" : 0,
              "ang_res_offset" : 2}
        
# Adaptive Angular h-Refinement
h_amr_ang  = {"full_name"  : "Adaptive Angular h-Refinement",
              "short_name" : "h-amr-ang",
              "ref_kind"   : "ang",
              "ndofs"      : [5, 5, 3],
              "nref_ang"   : 3,
              "nref_spt"   : 3,
              "Ls"         : [Lx, Ly],
              "pbcs"       : pbcs,
              "has_th"     : has_th,
              "spt_res_offset" : 0,
              "ang_res_offset" : 2,
              "kwargs_ang_nneg" : {"ref_col"       : False,
                                   "col_ref_form"  : None,
                                   "col_ref_kind"  : None,
                                   "col_ref_tol"   : None,
                                   "ref_cell"      : True,
                                   "cell_ref_form" : "h",
                                   "cell_ref_kind" : "ang",
                                   "cell_ref_tol"  : -10.**10},
              "kwargs_ang_jmp"  : {"ref_col"       : False,
                                   "col_ref_form"  : None,
                                   "col_ref_kind"  : None,
                                   "col_ref_tol"   : None,
                                   "ref_cell"      : True,
                                   "cell_ref_form" : "h",
                                   "cell_ref_kind" : "ang",
                                   "cell_ref_tol"  : 0.8}
              }
        
# Adaptive Angular hp-Refinement
hp_amr_ang = {"full_name"  : "Adaptive Angular hp-Refinement",
              "short_name" : "hp-amr-ang",
              "ref_kind"   : "ang",
              "ndofs"      : [5, 5, 3],
              "nref_ang"   : 3,
              "nref_spt"   : 3,
              "Ls"         : [Lx, Ly],
              "pbcs"       : pbcs,
              "has_th"     : has_th,
              "spt_res_offset" : 0,
              "ang_res_offset" : 2,
              "kwargs_ang_nneg" : {"ref_col"       : False,
                                   "col_ref_form"  : None,
                                   "col_ref_kind"  : None,
                                   "col_ref_tol"   : None,
                                   "ref_cell"      : True,
                                   "cell_ref_form" : "h",
                                   "cell_ref_kind" : "ang",
                                   "cell_ref_tol"  : -10.**10},
              "kwargs_ang_jmp"  : {"ref_col"       : False,
                                   "col_ref_form"  : None,
                                   "col_ref_kind"  : None,
                                   "col_ref_tol"   : None,
                                   "ref_cell"      : True,
                                   "cell_ref_form" : "hp",
                                   "cell_ref_kind" : "ang",
                                   "cell_ref_tol"  : 0.8}
              }
        
combos = [
    h_uni_ang,
    p_uni_ang,
    h_amr_ang,
    hp_amr_ang
]

# Manufactured solution
def X(x):
    return np.exp(-((1. / Lx) * (x - (Lx / 3.)))**2)
def dXdx(x):
    return -(2. / Lx**2) * (x - (Lx / 3.)) * X(x)
def Y(y):
    return np.exp(-4. * (Ly - y) / Ly)
def dYdy(y):
    return (4. / Ly) * Y(y)
def XY(x, y):
    return X(x) * Y(y)
sth = 96.
def Theta(th):
    return np.exp(-((sth / (2. * np.pi)) * (th - (7. * np.pi / 5.)))**2)
def u(x, y, th):
    return XY(x, y) * Theta(th)
        
def kappa_x(x):
    return np.exp(-((1. / Lx) * (x - (Lx / 2.)))**2)
def kappa_y(y):
    return np.exp(-y / Ly)
def kappa(x, y):
    return 10. * kappa_x(x) * kappa_y(y)

def sigma(x, y):
    return 0.1 * kappa(x, y)

def Phi(th, phi):
    val = (1. / (3. * np.pi)) * (1. + (np.cos(th - phi))**2)
    return val

def f(x, y, th):
    # Propagation part
    prop = (np.cos(th) * dXdx(x) * Y(y) + np.sin(th) * X(x) * dYdy(y)) * Theta(th)
    # Extinction part
    extn = kappa(x, y) * u(x, y, th)
    # Scattering part
    [Theta_scat, _] = quad(lambda phi: Phi(th, phi) * Theta(phi), 0., 2. * np.pi,
                           epsabs = 1.e-9, epsrel = 1.e-9, limit = 100, maxp1 = 100)
    scat =  sigma(x, y) * XY(x, y) * Theta_scat
    return prop + extn - scat
        
def bcs(x, y, th):
    return u(x, y, th)
dirac = [None, None, None]
bcs_dirac = [bcs, dirac]
        
def u_intg_th(x, y, th0, th1):
    [Theta_intg, _] = quad(lambda th: Theta(th), th0, th1,
                           epsabs = 1.e-9, epsrel = 1.e-9,
                           limit = 100, maxp1 = 100)
    return XY(x, y) * Theta_intg
        
def u_intg_xy(x0, x1, y0, y1, th):
    [XY_intg, _] = dblquad(lambda x, y: XY(x, y), x0, x1, y0, y1,
                           epsabs = 1.e-9, epsrel = 1.e-9,
                           limit = 100, maxp1 = 100)
    return XY_intg * Theta(th)