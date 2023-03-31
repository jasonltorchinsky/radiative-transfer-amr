import sys

from .calc_mass_matrix import calc_mass_matrix
from .calc_scat_matrix import calc_scat_matrix
from .calc_intr_conv_matrix import calc_intr_conv_matrix
from .calc_bdry_conv_matrix import calc_bdry_conv_matrix
from .calc_forcing_vec import calc_forcing_vec
from .calc_bcs_vec import calc_bcs_vec

from utils import print_msg

sys.path.append('../src')
from dg.matrix import get_intr_mask, split_matrix

def rtdg_amr(mesh, kappa, sigma, Phi, f = None,
             bcs = [None, None, None, None]):
    """
    Solve the RT problem.
    """
    
    M_mass      = calc_mass_matrix(mesh, kappa)
    M_scat      = calc_scat_matrix(mesh, sigma, Phi)
    M_intr_conv = calc_intr_conv_matrix(mesh)
    M_bdry_conv = calc_bdry_conv_matrix(mesh)
    
    M = (M_bdry_conv - M_intr_conv) + M_mass - M_scat
    
    if f is None:
        def forcing(x, y, th):
            return 0
    else:
        def forcing(x, y, th):
            return f(x, y)
    
    f_vec = calc_forcing_vec(mesh, f)
    bcs_vec = calc_bcs_vec(mesh, bcs)
    
    intr_mask = get_intr_mask(mesh)
    bdry_mask = np.invert(intr_mask)

    print_msg('Hello world!')
    
    
