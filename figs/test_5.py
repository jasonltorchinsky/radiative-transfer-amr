"""
Test 5: Many small clouds.
"""

# Standard Library Imports

# Third-Party Library Imports
import numpy as np
from   scipy.integrate import quad, dblquad

# Local Library Imports


# End-Combo Parameters
max_ndof   = int(3.2e5) # Max number of DOFs
max_ntrial = 1024       # Max number of trials
min_err    = 1.e-6      # Min error before cutoff
max_mem    = 95         # Max memory usage (percentage of 100)

# Mesh parameters common to each combination
[Lx, Ly] = [3., 2.]
pbcs     = [False, False]
has_th   = True
        
# Adaptive hp-Spatial, Uniform p-Angular Refinement
hp_amr_spt = {'full_name'  : 'Adaptive Spatial hp-Refinement',
              'short_name' : 'hp-amr-spt',
              'ref_kind'   : 'all',
              'ndofs'      : [3, 3, 4],
              'nref_ang'   : 3,
              'nref_spt'   : 2,
              'Ls'         : [Lx, Ly],
              'pbcs'       : pbcs,
              'has_th'     : has_th,
              'spt_res_offset' : 2,
              'ang_res_offset' : 2,
              'kwargs_spt_nneg' : {'ref_col'       : True,
                                   'col_ref_form'  : 'h',
                                   'col_ref_kind'  : 'spt',
                                   'col_ref_tol'   : -10.**10,
                                   'ref_cell'      : False,
                                   'cell_ref_form' : None,
                                   'cell_ref_kind' : None,
                                   'cell_ref_tol'  : None},
              'kwargs_spt_jmp'  : {'ref_col'       : True,
                                   'col_ref_form'  : 'hp',
                                   'col_ref_kind'  : 'spt',
                                   'col_ref_tol'   : 0.8,
                                   'ref_cell'      : False,
                                   'cell_ref_form' : None,
                                   'cell_ref_kind' : None,
                                   'cell_ref_tol'  : None}
              }

# Adaptive hp-Angular, Uniform p-Spatial Refinement
hp_amr_ang = {'full_name'  : 'Adaptive Angular hp-Refinement',
              'short_name' : 'hp-amr-ang',
              'ref_kind'   : 'all',
              'ndofs'      : [3, 3, 4],
              'nref_ang'   : 3,
              'nref_spt'   : 2,
              'Ls'         : [Lx, Ly],
              'pbcs'       : pbcs,
              'has_th'     : has_th,
              'spt_res_offset' : 2,
              'ang_res_offset' : 2,
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
        
# Adaptive hp-Angular, Adaptive hp-Spatial Refinement
hp_amr_all = {'full_name'  : 'Adaptive Spatio-Angular hp-Refinement',
              'short_name' : 'hp-amr-all',
              'ref_kind'   : 'all',
              'ndofs'      : [3, 3, 4],
              'nref_ang'   : 3,
              'nref_spt'   : 2,
              'Ls'         : [Lx, Ly],
              'pbcs'       : pbcs,
              'has_th'     : has_th,
              'spt_res_offset' : 2,
              'ang_res_offset' : 2,
              'kwargs_ang_nneg' : {'ref_col'       : False,
                                   'col_ref_form'  : None,
                                   'col_ref_kind'  : None,
                                   'col_ref_tol'   : None,
                                   'ref_cell'      : True,
                                   'cell_ref_form' : 'h',
                                   'cell_ref_kind' : 'ang',
                                   'cell_ref_tol'  : -10.**10},
              'kwargs_spt_nneg' : {'ref_col'       : True,
                                   'col_ref_form'  : 'h',
                                   'col_ref_kind'  : 'spt',
                                   'col_ref_tol'   : -10.**10,
                                   'ref_cell'      : False,
                                   'cell_ref_form' : None,
                                   'cell_ref_kind' : None,
                                   'cell_ref_tol'  : None},
              'kwargs_ang_jmp'  : {'ref_col'       : False,
                                   'col_ref_form'  : None,
                                   'col_ref_kind'  : None,
                                   'col_ref_tol'   : None,
                                   'ref_cell'      : True,
                                   'cell_ref_form' : 'hp',
                                   'cell_ref_kind' : 'ang',
                                   'cell_ref_tol'  : 0.8},
              'kwargs_spt_jmp'  : {'ref_col'       : True,
                                   'col_ref_form'  : 'hp',
                                   'col_ref_kind'  : 'spt',
                                   'col_ref_tol'   : 0.8,
                                   'ref_cell'      : False,
                                   'cell_ref_form' : None,
                                   'cell_ref_kind' : None,
                                   'cell_ref_tol'  : None}
              }
        
combos = [
    #hp_amr_spt,
    hp_amr_ang,
    hp_amr_all
]

# Manufactured solution
u = None
        
Ax = 0.5
fx = 1. / Lx
deltax = 0.05
def kappa_x(x):
    return (2. * Ax / np.pi) * np.arctan(np.sin(2. * np.pi * fx * (x - Lx / 3.)) / deltax) + 0.5
Ay = 0.5
fy = 1. / Ly
deltay = 0.05
def kappa_y(y):
    return (2. * Ay / np.pi) * np.arctan(np.sin(2. * np.pi * fy * (y - Ly / 3.)) / deltay) + 0.5
def kappa(x, y):
    r1 = (Ly / 9.) - np.sqrt((x - (7. * Lx / 16.))**2 + (y - (9. * Ly / 16.))**2)
    kappa1 = (10.) / (1. + np.exp(-2. * 15. * r1))

    r2 = (Ly / 9.) - np.sqrt((x - (3. * Lx / 4.))**2 + (y - (1. * Ly / 2.))**2)
    kappa2 = (10.) / (1. + np.exp(-2. * 15. * r2))

    r3 = (Ly / 9.) - np.sqrt((x - (7. * Lx / 8.))**2 + (y - (3. * Ly / 16.))**2)
    kappa3 = (10.) / (1. + np.exp(-2. * 15. * r3))

    r4 = (Ly / 9.) - np.sqrt((x - (9. * Lx / 16.))**2 + (y - (1. * Ly / 4.))**2)
    kappa4 = (10.) / (1. + np.exp(-2. * 15. * r4))

    return kappa1 + kappa2 + kappa3 + kappa4

def sigma(x, y):
    return 0.9 * kappa(x, y)

g = 0.8
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
        
x_left = 0.
y_top = Ly
def bcs(x, y, th):
    sth = (48.) / 2.
    if (y == y_top) or (x == x_left):
        return 5. * np.exp(-((sth / (2. * np.pi)) * (th - (7. * np.pi / 4.)))**2)
    else:
        return 0
dirac = [None, None, None]
bcs_dirac = [bcs, dirac]