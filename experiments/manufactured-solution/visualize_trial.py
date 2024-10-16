import os, sys
src_dir: str = os.path.normpath(os.path.join(os.path.dirname(__file__),
                                             os.pardir, os.pardir, "src"))

if src_dir not in sys.path:
    sys.path.append(src_dir)

# Standard Library Imports
import argparse

# Third-Party Library Imports
import petsc4py
from mpi4py import MPI
from petsc4py import PETSc

# Local Library Imports
from amr.error_indicator import Error_Indicator
from amr.error_indicator import from_file as error_indicator_from_file
from dg.projection import Projection
from dg.projection import from_file as projection_from_file
from tools.dg.mesh import plot_mesh
from tools.dg.projection import plot_th, plot_xy, plot_xth, plot_yth, plot_xyth
from tools.amr import plot_error_indicator

# Relative Imports

def main():
    ## Initialize parallel communicators
    mpi_comm: MPI.Intracomm = MPI.COMM_WORLD
    
    if not PETSc.Sys.isInitialized():
        petsc4py.init(comm = mpi_comm)

    ## Read command-line input
    parser_desc: str = ( "Generates all visualizations for each trial of the " +
                         "analytic solution experiment." )
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description = parser_desc)
    
    parser.add_argument("--o",
                        action = "store",
                        nargs = 1, 
                        type = str,
                        required = False,
                        default = "out",
                        help = "Trial directory path.")
    
    args = parser.parse_args()

    if (args.o != "out"):
        trial_dir_path: str = os.path.normpath(args.o[0])
    else:
        trial_dir_path: str = args.o
    
    ## Read the projection from file and visualize it
    mesh_file_name: str = "mesh.json"
    mesh_file_path: str = os.path.join(trial_dir_path, mesh_file_name)
    
    uh_file_name: str = "uh.npy"
    uh_file_path: str = os.path.join(trial_dir_path, uh_file_name)
    uh: Projection = projection_from_file(mesh_file_path, uh_file_path)

    # Mesh
    mesh_plot_file_name: str = "mesh.png"
    mesh_plot_file_path: str = os.path.join(trial_dir_path, mesh_plot_file_name)
    if not os.path.isfile(mesh_plot_file_path):
        plot_mesh(uh.mesh, mesh_plot_file_path)

    # uh - th
    uh_th_file_name: str = "uh_th.png"
    uh_th_file_path: str = os.path.join(trial_dir_path, uh_th_file_name)
    if not os.path.isfile(uh_th_file_path):
        plot_th(uh, file_path = uh_th_file_path)

    # uh - xy
    uh_xy_file_name: str = "uh_xy.png"
    uh_xy_file_path: str = os.path.join(trial_dir_path, uh_xy_file_name)
    if not os.path.isfile(uh_xy_file_path):
        plot_xy(uh, file_path = uh_xy_file_path)

    # uh - xth
    uh_xth_file_name: str = "uh_xth.png"
    uh_xth_file_path: str = os.path.join(trial_dir_path, uh_xth_file_name)
    if not os.path.isfile(uh_xth_file_path):
        plot_xth(uh, file_path = uh_xth_file_path)

    # uh - yth
    uh_yth_file_name: str = "uh_yth.png"
    uh_yth_file_path: str = os.path.join(trial_dir_path, uh_yth_file_name)
    if not os.path.isfile(uh_yth_file_path):
        plot_yth(uh, file_path = uh_yth_file_path)

    # uh - xyth
    uh_xyth_file_name: str = "uh_xyth.png"
    uh_xyth_file_path: str = os.path.join(trial_dir_path, uh_xyth_file_name)
    if not os.path.isfile(uh_xyth_file_path):
        plot_xyth(uh, file_path = uh_xyth_file_path)

    ## Read the error indicators from file and plot them
    # Analytic error indicator
    err_ind_anl_file_name: str = "err_ind_anl.json"
    err_ind_anl_file_path: str = os.path.join(trial_dir_path,
                                              err_ind_anl_file_name)
    err_ind_anl: Error_Indicator = error_indicator_from_file(mesh_file_path,
                                                             uh_file_path,
                                                             err_ind_anl_file_path)
    
    err_ind_anl_plot_file_name: str = "err_ind_anl.png"
    err_ind_anl_plot_file_path: str = os.path.join(trial_dir_path,
                                                   err_ind_anl_plot_file_name)
    if not os.path.isfile(err_ind_anl_plot_file_path):
        plot_error_indicator(err_ind_anl, err_ind_anl_plot_file_path)

    # High-resolution error indicator
    err_ind_hr_file_name: str = "err_ind_hr.json"
    err_ind_hr_file_path: str = os.path.join(trial_dir_path,
                                             err_ind_hr_file_name)
    err_ind_hr: Error_Indicator = error_indicator_from_file(mesh_file_path,
                                                            uh_file_path,
                                                            err_ind_hr_file_path)
    
    err_ind_hr_plot_file_name: str = "err_ind_hr.png"
    err_ind_hr_plot_file_path: str = os.path.join(trial_dir_path,
                                                  err_ind_hr_plot_file_name)
    if not os.path.isfile(err_ind_hr_plot_file_path):
        plot_error_indicator(err_ind_hr, err_ind_hr_plot_file_path)

    # Angular-Jump error indicator
    err_ind_jmp_file_name: str = "err_ind_jmp.json"
    err_ind_jmp_file_path: str = os.path.join(trial_dir_path,
                                              err_ind_jmp_file_name)
    err_ind_jmp: Error_Indicator = error_indicator_from_file(mesh_file_path,
                                                             uh_file_path,
                                                             err_ind_jmp_file_path)
    
    err_ind_jmp_plot_file_name: str = "err_ind_jmp.png"
    err_ind_jmp_plot_file_path: str = os.path.join(trial_dir_path,
                                                   err_ind_jmp_plot_file_name)
    if not os.path.isfile(err_ind_jmp_plot_file_path):
        plot_error_indicator(err_ind_jmp, err_ind_jmp_plot_file_path)

    ## Read the high-resolution projection from file and visualize it
    mesh_hr_file_name: str = "mesh_hr.json"
    mesh_hr_file_path: str = os.path.join(trial_dir_path, mesh_hr_file_name)
    
    uh_hr_file_name: str = "uh_hr.npy"
    uh_hr_file_path: str = os.path.join(trial_dir_path, uh_hr_file_name)
    uh_hr: Projection = projection_from_file(mesh_hr_file_path, uh_hr_file_path)

    # Mesh
    mesh_hr_plot_file_name: str = "mesh_hr.png"
    mesh_hr_plot_file_path: str = os.path.join(trial_dir_path, mesh_hr_plot_file_name)
    if not os.path.isfile(mesh_hr_plot_file_path):
        plot_mesh(uh_hr.mesh, mesh_hr_plot_file_path)

    # uh_hr - th
    uh_hr_th_file_name: str = "uh_hr_th.png"
    uh_hr_th_file_path: str = os.path.join(trial_dir_path, uh_hr_th_file_name)
    if not os.path.isfile(uh_hr_th_file_path):
        plot_th(uh_hr, file_path = uh_hr_th_file_path)

    # uh_hr - xy
    uh_hr_xy_file_name: str = "uh_hr_xy.png"
    uh_hr_xy_file_path: str = os.path.join(trial_dir_path, uh_hr_xy_file_name)
    if not os.path.isfile(uh_hr_xy_file_path):
        plot_xy(uh_hr, file_path = uh_hr_xy_file_path)

    # uh_hr - xth
    uh_hr_xth_file_name: str = "uh_hr_xth.png"
    uh_hr_xth_file_path: str = os.path.join(trial_dir_path, uh_hr_xth_file_name)
    if not os.path.isfile(uh_hr_xth_file_path):
        plot_xth(uh_hr, file_path = uh_hr_xth_file_path)

    # uh_hr - yth
    uh_hr_yth_file_name: str = "uh_hr_yth.png"
    uh_hr_yth_file_path: str = os.path.join(trial_dir_path, uh_hr_yth_file_name)
    if not os.path.isfile(uh_hr_yth_file_path):
        plot_yth(uh_hr, file_path = uh_hr_yth_file_path)

    # uh - xyth
    uh_hr_xyth_file_name: str = "uh_hr_xyth.png"
    uh_hr_xyth_file_path: str = os.path.join(trial_dir_path, uh_hr_xyth_file_name)
    if not os.path.isfile(uh_hr_xyth_file_path):
        plot_xyth(uh_hr, file_path = uh_hr_xyth_file_path)

if __name__ == "__main__":
    main()