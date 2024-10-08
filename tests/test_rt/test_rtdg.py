# Standard Library Imports
import os

# Third-Party Library Imports
import numpy as np
import pytest
from scipy.integrate import quad

# Local Library Imports
import consts
from dg.mesh import Mesh
from rt import rtdg
from tools.dg.mesh import plot_mesh
from tools.dg.projection import plot_th, plot_xy, plot_xth, plot_yth, plot_xyth

# Relative Imports

@pytest.mark.mpi(minsize = 1)
def test_rtdg(tmp_path):
    # Create a mesh
    Ls: list = [1., 2.]
    pbcs: list = [False, False]
    ndofs: list = [3, 3, 3]
    has_th: bool = True

    mesh: Mesh = Mesh(Ls, pbcs, ndofs, has_th)

    for _ in range(0, 4):
        mesh.ref_mesh(kind = "all", form = "hp")

    ## Define a simple test problem
    [Lx, Ly] = Ls
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
        return np.exp(-((sth / (2. * consts.PI)) * (th - (7. * consts.PI / 5.)))**2)
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
        val = (1. / (3. * consts.PI)) * (1. + (np.cos(th - phi))**2)
        return val

    def f(x, y, th):
        # Propagation part
        prop = (np.cos(th) * dXdx(x) * Y(y) + np.sin(th) * X(x) * dYdy(y)) * Theta(th)
        # Extinction part
        extn = kappa(x, y) * u(x, y, th)
        # Scattering part
        [Theta_scat, _] = quad(lambda phi: Phi(th, phi) * Theta(phi), 0., 2. * consts.PI,
                               epsabs = 1.e-9, epsrel = 1.e-9, limit = 100, maxp1 = 100)
        scat =  sigma(x, y) * XY(x, y) * Theta_scat
        return prop + extn - scat
        
    def bcs(x, y, th):
        return u(x, y, th)
    dirac = [None, None, None]
    bcs_dirac = [bcs, dirac]

    ## Solve the problem once
    [uh, _, _] = rtdg(mesh, kappa, sigma, Phi, bcs_dirac, f = f)

    ## Write the projection to file and read from it
    proj_file_name: str = "uh.npy"
    proj_file_path: str = os.path.join(tmp_path, proj_file_name)

    mesh_file_name: str = "mesh.json"
    mesh_file_path: str = os.path.join(tmp_path, mesh_file_name)

    uh.to_file(proj_file_path, write_mesh = True, mesh_file_path = mesh_file_path)

    ## Plot the mesh
    file_name: str = "mesh.png"
    file_path: str = os.path.join(tmp_path, file_name)
    plot_mesh(mesh, file_path = file_path)

    ## Plot the projection
    file_name: str = "proj_th.png"
    file_path: str = os.path.join(tmp_path, file_name)
    plot_th(uh, file_path = file_path)

    file_name: str = "proj_xth.png"
    file_path: str = os.path.join(tmp_path, file_name)
    plot_xth(uh, file_path = file_path, cmap = "hot", scale = "normal")

    file_name: str = "proj_yth.png"
    file_path: str = os.path.join(tmp_path, file_name)
    plot_yth(uh, file_path = file_path, cmap = "hot", scale = "normal")

    file_name: str = "proj_xy.png"
    file_path: str = os.path.join(tmp_path, file_name)
    plot_xy(uh, file_path = file_path, cmap = "hot", scale = "normal")

    file_name: str = "proj_xyth.png"
    file_path: str = os.path.join(tmp_path, file_name)
    plot_xyth(uh, file_path = file_path, cmap = "hot", scale = "normal")