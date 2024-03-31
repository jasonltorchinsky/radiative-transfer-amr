"""
Test 2: Clear sky with negligible scattering.
"""

# Standard Library Imports

# Third-Party Library Imports
import numpy as np
from   scipy.integrate import quad, dblquad

# Local Library Imports


# End-Combo Parameters
max_ndof   = int(2.8e5)  # Max number of DOFs
max_ntrial = 1024        # Max number of trials
min_err    = 1.e-6       # Min error before cutoff
max_mem    = 95          # Max memory usage (percentage of 100)

# Mesh parameters common to each combination
[Lx, Ly] = [3., 2.]
pbcs     = [False, False]
has_th   = True
        
# Uniform Angular h-Refinement
h_uni_ang  = {'full_name'  : 'Uniform Angular h-Refinement',
              'short_name' : 'h-uni-ang',
              'ref_kind'   : 'ang',
              'ndofs'      : [3, 3, 4],
              'nref_ang'   : 3,
              'nref_spt'   : 3,
              'Ls'         : [Lx, Ly],
              'pbcs'       : pbcs,
              'has_th'     : has_th,
              'spt_res_offset' : 0,
              'ang_res_offset' : 3
}

# Uniform Angular p-Refinement
p_uni_ang  = {'full_name'  : 'Uniform Angular p-Refinement',
              'short_name' : 'p-uni-ang',
              'ref_kind'   : 'ang',
              'ndofs'      : [3, 3, 4],
              'nref_ang'   : 3,
              'nref_spt'   : 3,
              'Ls'         : [Lx, Ly],
              'pbcs'       : pbcs,
              'has_th'     : has_th,
              'spt_res_offset' : 0,
              'ang_res_offset' : 3}
        
# Adaptive Angular h-Refinement
h_amr_ang  = {'full_name'  : 'Adaptive Angular h-Refinement',
              'short_name' : 'h-amr-ang',
              'ref_kind'   : 'ang',
              'ndofs'      : [3, 3, 4],
              'nref_ang'   : 3,
              'nref_spt'   : 3,
              'Ls'         : [Lx, Ly],
              'pbcs'       : pbcs,
              'has_th'     : has_th,
              'spt_res_offset' : 0,
              'ang_res_offset' : 3,
              'kwargs_ang_nneg' : {'ref_col'       : False,
                                   'col_ref_form'  : None,
                                   'col_ref_kind'  : None,
                                   'col_ref_tol'   : None,
                                   'ref_cell'      : True,
                                   'cell_ref_form' : 'h',
                                   'cell_ref_kind' : 'ang',
                                   'cell_ref_tol'  : -10.**10},
              'kwargs_ang_jmp'  : {'ref_col'       : False,
                                   'col_ref_form'  : None,
                                   'col_ref_kind'  : None,
                                   'col_ref_tol'   : None,
                                   'ref_cell'      : True,
                                   'cell_ref_form' : 'h',
                                   'cell_ref_kind' : 'ang',
                                   'cell_ref_tol'  : 0.8}
              }
        
# Adaptive Angular hp-Refinement
hp_amr_ang = {'full_name'  : 'Adaptive Angular hp-Refinement',
              'short_name' : 'hp-amr-ang',
              'ref_kind'   : 'ang',
              'ndofs'      : [3, 3, 4],
              'nref_ang'   : 3,
              'nref_spt'   : 3,
              'Ls'         : [Lx, Ly],
              'pbcs'       : pbcs,
              'has_th'     : has_th,
              'spt_res_offset' : 0,
              'ang_res_offset' : 3,
              'kwargs_ang_nneg' : {'ref_col'       : False,
                                   'col_ref_form'  : None,
                                   'col_ref_kind'  : None,
                                   'col_ref_tol'   : None,
                                   'ref_cell'      : True,
                                   'cell_ref_form' : 'h',
                                   'cell_ref_kind' : 'ang',
                                   'cell_ref_tol'  : -10.**10},
              'kwargs_ang_jmp'  : {'ref_col'       : False,
                                   'col_ref_form'  : None,
                                   'col_ref_kind'  : None,
                                   'col_ref_tol'   : None,
                                   'ref_cell'      : True,
                                   'cell_ref_form' : 'hp',
                                   'cell_ref_kind' : 'ang',
                                   'cell_ref_tol'  : 0.8}
              }
        
combos = [
    h_uni_ang,
    p_uni_ang,
    h_amr_ang,
    hp_amr_ang
]

# Manufactured solution
u = None
        
def kappa_x(x):
    return np.ones_like(x)
def kappa_y(y):
    return np.ones_like(y)
def kappa(x, y):
    r = (Ly / 3.) - np.sqrt((x - (3. * Lx / 5.))**2 + (y - (2. * Ly / 5.))**2)
    return 1.1 / (1. + np.exp(-15 * r))

def sigma(x, y):
    return 0.9 * kappa(x, y)

g = 0.9
def Phi_HG(Th):
    return (1. - g**2) / (1 + g**2 - 2. * g * np.cos(Th))**(3./2.)
[Phi_norm, abserr] = quad(lambda Th : Phi_HG(Th), 0., 2. * np.pi,
                          epsabs = 1.e-9, epsrel = 1.e-9,
                          limit = 100, maxp1 = 100)
def Phi(th, phi):
    val = (1. - g**2) / (1 + g**2 - 2. * g * np.cos(th - phi))**(3./2.)
    return val / Phi_norm

def f(x, y, th):
    return 0
        
x_right = 0.
y_top = Ly
def bcs(x, y, th):
    sth = 96. * 4.
    if (y == y_top) or (x == x_right):
        return np.exp(-((sth / (2. * np.pi)) * (th - (7. * np.pi / 4.)))**2)
    #if (y == y_top) and (th == (8. * np.pi / 5.)):
    #    return 1
    else:
        return 0
dirac = [None, None, None]
bcs_dirac = [bcs, dirac]