import os, sys
src_dir: str = os.path.normpath(os.path.join(os.path.dirname(__file__),
                                             os.pardir, os.pardir, "src"))

if src_dir not in sys.path:
    sys.path.append(src_dir)

# Standard Library Imports
import argparse
import json
from typing import Callable

# Third-Party Library Imports
import matplotlib.pyplot as plt
import numpy as np
import petsc4py
from mpi4py import MPI
from petsc4py import PETSc

# Local Library Imports
import consts
from dg.mesh import Mesh
from dg.projection import Projection
from tools.dg.projection import plot_th, plot_xy, plot_xth, plot_yth, plot_xyth

# Relative Imports
from problem import kappa, sigma, Phi, u

def main():
    ## Initialize parallel communicators
    mpi_comm: MPI.Intracomm = MPI.COMM_WORLD
    
    if not PETSc.Sys.isInitialized():
        petsc4py.init(comm = mpi_comm)

    ## Read command-line input
    parser_desc: str = ( "Generates visualizations of the problem." )
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description = parser_desc)
    
    parser.add_argument("--o",
                        action = "store",
                        nargs = 1, 
                        type = str,
                        required = False,
                        default = "out",
                        help = "Output directory path.")
    
    args = parser.parse_args()

    if (args.o != "out"):
        out_dir_path: str = os.path.normpath(args.o[0])
    else:
        out_dir_path: str = args.o

    ## Read input - hardcoded file names
    with open("input.json", "r") as input_file:
        input_dict: dict = json.load(input_file)

    ## Create a high-resolution mesh for the plots
    Ls: list = input_dict["mesh_params"]["Ls"]
    pbcs: list = [False, False]
    ndofs: list = [5, 5, 5]
    has_th: bool = True

    mesh: Mesh = Mesh(Ls, pbcs, ndofs, has_th)
    for _ in range(0, 4):
        mesh.ref_mesh(kind = "all", form = "h")

    ## Plot the extinction coefficient kappa - rescale because integrates in theta
    kappa_p = lambda x, y, th: kappa(x, y) / (2. * consts.PI)
    kappa_h: Projection = Projection(mesh, kappa_p)

    kappa_h_plot_file_name: str = "kappa.png"
    kappa_h_plot_file_path: str = os.path.join(out_dir_path,
                                               kappa_h_plot_file_name)
    
    if not os.path.isfile(kappa_h_plot_file_path):
        plot_xy(kappa_h, kappa_h_plot_file_path, show_mesh = False, cmap = "gray")

    ## Plot the scattering coefficient sigma - rescale because integrates in theta
    sigma_p = lambda x, y, th: sigma(x, y) / (2. * consts.PI)
    sigma_h: Projection = Projection(mesh, sigma_p)

    sigma_h_plot_file_name: str = "sigma.png"
    sigma_h_plot_file_path: str = os.path.join(out_dir_path,
                                               sigma_h_plot_file_name)
    
    if not os.path.isfile(sigma_h_plot_file_path):
        plot_xy(sigma_h, sigma_h_plot_file_path, show_mesh = False, cmap = "gray")

    ## Plot the scattering phase function
    Phi_plot_file_name: str = "Phi.png"
    Phi_plot_file_path: str = os.path.join(out_dir_path,
                                           Phi_plot_file_name)
    
    if not os.path.isfile(Phi_plot_file_path):
        plot_Phi(Phi, Phi_plot_file_path)

    ## Plot the analytic solution
    u_proj: Projection = Projection(mesh, u)

    # u - th
    u_th_file_name: str = "u_th.png"
    u_th_file_path: str = os.path.join(out_dir_path, u_th_file_name)
    if not os.path.isfile(u_th_file_path):
        plot_th(u_proj, file_path = u_th_file_path, show_mesh = False)
    # u - xy
    u_xy_file_name: str = "u_xy.png"
    u_xy_file_path: str = os.path.join(out_dir_path, u_xy_file_name)
    if not os.path.isfile(u_xy_file_path):
        plot_xy(u_proj, file_path = u_xy_file_path, show_mesh = False)
    # u - xth
    u_xth_file_name: str = "u_xth.png"
    u_xth_file_path: str = os.path.join(out_dir_path, u_xth_file_name)
    if not os.path.isfile(u_xth_file_path):
        plot_xth(u_proj, file_path = u_xth_file_path, show_mesh = False)
    # u - yth
    u_yth_file_name: str = "u_yth.png"
    u_yth_file_path: str = os.path.join(out_dir_path, u_yth_file_name)
    if not os.path.isfile(u_yth_file_path):
        plot_yth(u_proj, file_path = u_yth_file_path, show_mesh = False)
    # u - xyth
    u_xyth_file_name: str = "u_xyth.png"
    u_xyth_file_path: str = os.path.join(out_dir_path, u_xyth_file_name)
    if not os.path.isfile(u_xyth_file_path):
        plot_xyth(u_proj, file_path = u_xyth_file_path, show_mesh = False)

def plot_Phi(Phi: Callable[[np.ndarray, np.ndarray], np.ndarray],
             file_path: str) -> None:
    
    fig, ax = plt.subplots(subplot_kw = {"projection": "polar"})

    ## Generate data to plot
    th: np.ndarray = np.linspace(0, 2. * consts.PI, num = 720)
    rr: np.ndarray = Phi(0, th)
    
    ## Generate ticks and tick labels
    max_r: float = np.max(rr)

    ax.set_rlim([0, max_r])

    ntick: int = 2
    r_ticks: np.ndarray = np.linspace(max_r / ntick, max_r, ntick)
    r_tick_labels: list = ["{:3.2f}".format(r_tick) for r_tick in r_ticks]
    ax.set_rticks(r_ticks, r_tick_labels)

    th_ticks: np.ndarray = np.linspace(0, 2. * consts.PI, num = 8, endpoint = False)
    th_tick_labels: list = [r"${:3.2f} \pi$".format(th_tick / consts.PI)
                            for th_tick in th_ticks]
    ax.set_xticks(th_ticks, th_tick_labels)
    
    ax.plot(th, rr, color = "black")
    
    ax.set_title(r"$\Phi\left( \theta - \theta' \right)$")
    
    plt.tight_layout()
    plt.savefig(file_path, dpi = 300)
    
    plt.close(fig)

if __name__ == "__main__":
    main()