from time import perf_counter

from .calc_mass_matrix import calc_mass_matrix
from .calc_scat_matrix import calc_scat_matrix
from .calc_intr_conv_matrix import calc_intr_conv_matrix
from .calc_bdry_conv_matrix import calc_bdry_conv_matrix

from utils import print_msg

import matplotlib.pyplot as plt

def rtdg_amr(mesh, kappa, sigma, Phi):
    """
    Solve the RT problem.
    mesh - Mesh to contruct the discretized equation.
    uh_init - Initial guess at teh soluition.
    kappa - Extinction coefficient.
    sigma - Scattering coefficient.
    Phi   - Phase function.
    """

    """
    print_msg('Constructing the mass matrix...')
    
    M_mass = calc_mass_matrix(mesh, kappa)

    fig = plt.figure()
    im_mass = plt.spy(M_mass, marker = '.', markersize = 0.1)
    plt.savefig('M_mass.png', dpi = 500)
    plt.close(fig)
    """

    """
    print_msg('Created the mass matrix! Constructing the scattering matrix...')

    M_scat = calc_scat_matrix(mesh, sigma, Phi)

    fig = plt.figure()
    im_scat = plt.spy(M_scat, marker = '.', markersize = 0.1)
    plt.savefig('M_scat.png', dpi = 500)
    plt.close(fig)
    """

    """
    print_msg(('Created the scattering matrix!' +
               ' Constructing the interior convection matrix...'))
    
    M_intr_conv = calc_intr_conv_matrix(mesh)

    fig = plt.figure()
    im_intr_conv = plt.spy(M_intr_conv, marker = '.', markersize = 0.1)
    plt.savefig('M_intr_conv.png', dpi = 500)
    plt.close(fig)
    """

    """
    print_msg(('Created the interior convection matrix!' +
               ' Constructing the boundary convection matrix...'))
    
    M_bdry_conv = calc_bdry_conv_matrix(mesh)

    fig = plt.figure()
    im_mass = plt.spy(M_bdry_conv, marker = '.', markersize = 0.1)
    plt.savefig('M_bdry_conv.png', dpi = 1000.0)
    plt.close(fig)
    """
