from mesh import ji_mesh
from mesh import tools as mesh_tools
import quadrature as qd
import rad_amr as rd
from rad_amr import tools as rd_tools
import reg_est as re

import sys, getopt
import numpy as np
import matplotlib.pyplot as plt

def main(argv):

    # Simulation parameters
    Lx = 16
    Ly = 16
    dof_x = 2
    dof_y = 2
    dof_a = 2

    # Convergence test parameters
    max_nref = 4
    max_dof = 1e6
    max_iters = int(1e5)
    err_thresh = 1e-10

    # Functions for calculations
    def zz(x, y, a):
        return 0

    def rr(x, y, cx, cy):
        return np.sqrt((x - cx)**2 + (y - cy)**2)

    def sigma(x, y):
        return np.exp(-0.5 * (rr(x, y, Lx/2., Ly/2.) / (Ly/10.))**2)
    
    def kappa(x, y):
        return 10**(-3) + 1.01 * sigma(x, y)

    def varphi(a):
        return np.sin((1 - 2 * a / np.pi) * np.pi / 2) \
               * (a >= 0) * (a <= np.pi / 2) \
            + (2 / np.pi * (a - 3 * np.pi / 2))**3 \
              * (a >= 3 * np.pi / 2) * (a <= 2 * np.pi)

    ccc = 10
    mid = 2.2
    def U(x, y):
        return (Ly + y * np.tanh(ccc * (x - Lx / mid))) * x * (Lx - x)

    def u(x, y, a):
        return U(x, y) * varphi(a)

    def term_conv(x, y, a):
        return varphi(a) * np.cos(a) \
            * ( (Lx - 2 * x) * (Ly + y * np.tanh(ccc * (x - Lx / mid))) \
                + (Lx * x - x**2) \
                  * (ccc * y * (1 - np.tanh(ccc * (x - Lx / mid))**2)) ) \
            + varphi(a) * np.sin(a) \
              * ( (Lx * x - x**2) * np.tanh(ccc * (x - Lx / mid)) )

    def term_scat(x, y, a): # Calculated integrals analytically using Mathematica
        val = (1. / (144 * np.pi**4)) \
            * ( 9 * np.pi**3 * (8 + np.pi) \
                + 4 * (-36 + 9 * np.pi**2 + 2 * np.pi**3) * np.cos(2 * a) \
                + 4 * np.pi * (18 + np.pi**2) * np.sin(2 * a) )
        return -sigma(x, y) * U(x, y) * val

    def source(x, y, a):
        return term_conv(x, y, a) + kappa(x, y) * u(x, y, a) + term_scat(x, y, a)

    def bdry_top(x, y, a):
        return u(x, y, a)

    def bdry_bot(x, y, a):
        return u(x, y, a)

    # Set up the mesh
    mesh = ji_mesh.Mesh(Ls = [Lx, Ly],
                        pbcs = [True, False],
                        is_flat = False)
    col = mesh.cols[0]
    cell = col.cells[0]
    cell.ndof = [dof_x, dof_y, dof_a]

    for ii in range(0, 2):
        col.ref_col()
    
    for ii in range(0, 2):
        mesh.ref_mesh()

    '''mesh_tools.plot_mesh(mesh, file_name = 'mesh_0_2d.png',
                         label_cells = True, plot_dim = 2)
    mesh_tools.plot_mesh(mesh, file_name = 'mesh_0_3d.png',
                         label_cells = False, plot_dim = 3)

    proj = rd.Projection(mesh = mesh, u = u, has_a = True)
    rd_tools.plot_projection(mesh = mesh, uh = proj,
                             file_name = 'uh.png',
                             show_mesh = True,
                             colormap = 'gray')'''
    

    for ref in range(0, max_nref):
        # Plot the current mesh
        mesh_tools.plot_mesh(mesh = mesh, ax = None,
                             file_name = 'mesh_' + str(ref) + '.png',
                             label_cells = False,
                             plot_dim = 3)

        
        # Calculate the solution on the current mesh.
        uh_init = rd.Projection(mesh = mesh, has_a = True)
        uh = rd.rtdg_amr(mesh, uh_init, kappa, sigma, source,
                         bdry_top, bdry_bot,
                         max_iters = max_iters,
                         err_thresh = err_thresh,
                         scat_type = None,
                         verbose = True)
    '''
        rd_tools.plot_projection(mesh, uh,
                           file_name = 'uh_' + str(ref) + '.png',
                           angles = [0, np.pi / 2, np.pi, 3 * np.pi / 2],
                           show_mesh = True,
                           label_cells = False,
                           shading = 'flat',
                           colormap = 'gray')


        
        # Calculate the error of the numerical solution
        error = ar.err_ind(mesh, uh)
        [regth, regmm] = re.reg_deg(mesh, uh)
        mesh = ar.ref_by_ind(mesh, error, 0.2, regth, regmm)
        
        print(None)
    '''


if __name__ == '__main__':

    main(sys.argv[1:])
