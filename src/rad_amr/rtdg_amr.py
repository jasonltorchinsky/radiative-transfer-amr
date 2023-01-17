import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, block_diag, bmat
from scipy.sparse.linalg import gmres
from scipy.linalg import eig
import sys
from time import perf_counter

from .Projection import Projection_2D

import matplotlib.pyplot as plt

import dg.quadrature as qd
from dg.mesh import ji_mesh, tools

import matplotlib.pyplot as plt

def rtdg_amr(mesh, uh_init, kappa, sigma, phi, **kwargs):
    '''
    Solve the radiative transfer problem.
    '''

    uh = uh_init

    # Construct the mass matrix
    #if kwargs['verbose']:
    #    print('Constructing the mass matrix...')
    #    t_start = perf_counter()
    
    M_mass = calc_mass_matrix(mesh, kappa)

    fig = plt.figure()
    im_mass = plt.spy(M_mass, marker = '.', markersize = 0.1)
    plt.savefig('M_mass.png', dpi = 500)
    plt.close(fig)

            
    M_scat = calc_scat_matrix(mesh, sigma, phi)

    fig = plt.figure()
    im_mass = plt.spy(M_scat, marker = '.', markersize = 0.1)
    plt.savefig('M_scat.png', dpi = 500)
    plt.close(fig)

    sys.exit(2)
