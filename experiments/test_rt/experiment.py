import os, sys
src_dir: str = os.path.normpath(os.path.join(os.path.dirname(__file__),
                                             os.pardir, os.pardir, "src"))

if src_dir not in sys.path:
    sys.path.append(src_dir)

# Standard Library Imports
import argparse

# Third-Party Library Imports
import numpy as np
import petsc4py
import scipy.sparse as sp
from mpi4py   import MPI
from petsc4py import PETSc

# Local Library Imports
from dg.mesh.column import Column
from dg.mesh.column.cell import Cell
from tools.dg.mesh import plot_mesh
from tools.dg.projection import plot_th, plot_xth, plot_yth, plot_xy, plot_xyth
from tools.rt import plot_matrix

# Relative Imports
from mesh import mesh
from problem import problem

def main():
    ## Initialize parallel communicators
    mpi_comm: MPI.Intracomm = MPI.COMM_WORLD
    
    if not PETSc.Sys.isInitialized():
        petsc4py.init(comm = mpi_comm)
    petsc_comm: PETSc.Comm = PETSc.COMM_WORLD
    comm_rank: int = petsc_comm.getRank()

    ## Read command-line input
    parser_desc: str = ( "Runs the numerical experiment for the hp-adaptive DG" +
                         " method for radiative transfer." )
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description = parser_desc)
    
    parser.add_argument("--o",
                        action = "store",
                        nargs = 1, 
                        type = str,
                        required = False,
                        default = "figs",
                        help = "Output directory path.")
    
    args = parser.parse_args()

    if (args.o != "figs"):
        out_dir_path: str = os.path.normpath(args.o[0])
    else:
        out_dir_path: str = args.o
    
    ## Create the figs output directory
    if comm_rank == 0:
        figs_dir_name: str = "figs"
        figs_dir_path: str = os.path.join(out_dir_path, figs_dir_name)
        os.makedirs(figs_dir_path, exist_ok = True)

    ## Output the initial mesh
    if comm_rank == 0:
        mesh_file_name: str = "mesh.json"
        mesh_file_path: str = os.path.join(out_dir_path, mesh_file_name)
        mesh.to_file(mesh_file_path)

        mesh_plot_file_name: str = "mesh.png"
        mesh_plot_file_path: str = os.path.join(figs_dir_path,
                                                mesh_plot_file_name)
        plot_mesh(mesh, mesh_plot_file_path)

    ## Obtain and output the mass matrix
    ma_mat: sp._coo.coo_matrix = problem.mass_matrix(mesh)

    if comm_rank == 0:
        ## Plot the intra-process mass matrix on process 0
        ownership_range: tuple = ma_mat.getOwnershipRange()
        ma_mat_global: np.ndarray = ma_mat[ownership_range[0]:ownership_range[1],
                                           ownership_range[0]:ownership_range[1]]

        ma_mat_global_plot_file_name: str = "mass_global.png"
        ma_mat_global_plot_file_path: str = \
            os.path.join(figs_dir_path, ma_mat_global_plot_file_name)
        plot_matrix(ma_mat_global, ma_mat_global_plot_file_path,
                    title = "Global Mass Matrix", marker_size = 0.5)
        
        ## Plot the intra-column mass matrix for the column with minimal key
        col_key: int = min(mesh.cols.keys())
        col: Column = mesh.cols[col_key]
        col_ndof: int = 0
        for _, cell in col.cells.items():
            col_ndof += cell.ndofs[0]
        col_ndof *= col.ndofs[0] * col.ndofs[1]
        
        ma_mat_column: np.ndarray = ma_mat[0:col_ndof, 0:col_ndof]
        ma_mat_column_plot_file_name: str = "mass_column.png"
        ma_mat_column_plot_file_path: str = \
            os.path.join(figs_dir_path, ma_mat_column_plot_file_name)
        plot_matrix(ma_mat_column, ma_mat_column_plot_file_path,
                    title = "Intra-Column Mass Matrix", marker_size = 1.)

        ## Plot the intra-cell mass matrix for the cell with minimal key
        cell_key: int = min(mesh.cols[col_key].cells.keys())
        cell: Cell = mesh.cols[col_key].cells[cell_key]
        cell_ndof: int = col.ndofs[0] * col.ndofs[1] * cell.ndofs[0]

        ma_mat_cell: np.ndarray = ma_mat[0:cell_ndof, 0:cell_ndof]
        ma_mat_cell_plot_file_name: str = "mass_cell.png"
        ma_mat_cell_plot_file_path: str = \
            os.path.join(figs_dir_path, ma_mat_cell_plot_file_name)
        plot_matrix(ma_mat_cell, ma_mat_cell_plot_file_path,
                    title = "Intra-Cell Mass Matrix", marker_size = 2.)
    
    ## Obtain and output scattering matrix
    sc_mat: sp._coo.coo_matrix = problem.scattering_matrix(mesh)

    if comm_rank == 0:
        ## Plot the intra-process scattering matrix on process 0
        ownership_range: tuple = sc_mat.getOwnershipRange()
        sc_mat_global: np.ndarray = sc_mat[ownership_range[0]:ownership_range[1],
                                           ownership_range[0]:ownership_range[1]]

        sc_mat_global_plot_file_name: str = "scattering_global.png"
        sc_mat_global_plot_file_path: str = \
            os.path.join(figs_dir_path, sc_mat_global_plot_file_name)
        plot_matrix(sc_mat_global, sc_mat_global_plot_file_path,
                    title = "Global Scattering Matrix", marker_size = 0.5)
        
        ## Plot the intra-column scattering matrix for the column with minimal key
        col_key: int = min(mesh.cols.keys())
        col: Column = mesh.cols[col_key]
        col_ndof: int = 0
        for _, cell in col.cells.items():
            col_ndof += cell.ndofs[0]
        col_ndof *= col.ndofs[0] * col.ndofs[1]
        
        sc_mat_column: np.ndarray = sc_mat[0:col_ndof, 0:col_ndof]
        sc_mat_column_plot_file_name: str = "scattering_column.png"
        sc_mat_column_plot_file_path: str = \
            os.path.join(figs_dir_path, sc_mat_column_plot_file_name)
        plot_matrix(sc_mat_column, sc_mat_column_plot_file_path,
                    title = "Intra-Column Scattering Matrix", marker_size = 1.0)

        ## Plot the intra-cell scattering matrix for the cell with minimal key
        cell_key: int = min(mesh.cols[col_key].cells.keys())
        cell: Cell = mesh.cols[col_key].cells[cell_key]
        cell_ndof: int = col.ndofs[0] * col.ndofs[1] * cell.ndofs[0]

        sc_mat_cell: np.ndarray = sc_mat[0:cell_ndof, 0:cell_ndof]
        sc_mat_cell_plot_file_name: str = "scattering_cell.png"
        sc_mat_cell_plot_file_path: str = \
            os.path.join(figs_dir_path, sc_mat_cell_plot_file_name)
        plot_matrix(sc_mat_cell, sc_mat_cell_plot_file_path,
                    title = "Intra-Cell Scattering Matrix", marker_size = 2.0)

    ## Obtain and output the interior convection matrix
    ic_mat: sp._coo.coo_matrix = problem.interior_convection_matrix(mesh)

    if comm_rank == 0:
        ## Plot the intra-process interior_convection matrix on process 0
        ownership_range: tuple = ic_mat.getOwnershipRange()
        ic_mat_global: np.ndarray = ic_mat[ownership_range[0]:ownership_range[1],
                                           ownership_range[0]:ownership_range[1]]

        ic_mat_global_plot_file_name: str = "interior_convection_global.png"
        ic_mat_global_plot_file_path: str = \
            os.path.join(figs_dir_path, ic_mat_global_plot_file_name)
        plot_matrix(ic_mat_global, ic_mat_global_plot_file_path,
                    title = "Global Interior Convection Matrix", marker_size = 0.5)
        
        ## Plot the intra-column interior_convection matrix for the column with minimal key
        col_key: int = min(mesh.cols.keys())
        col: Column = mesh.cols[col_key]
        col_ndof: int = 0
        for _, cell in col.cells.items():
            col_ndof += cell.ndofs[0]
        col_ndof *= col.ndofs[0] * col.ndofs[1]
        
        ic_mat_column: np.ndarray = ic_mat[0:col_ndof, 0:col_ndof]
        ic_mat_column_plot_file_name: str = "interior_convection_column.png"
        ic_mat_column_plot_file_path: str = \
            os.path.join(figs_dir_path, ic_mat_column_plot_file_name)
        plot_matrix(ic_mat_column, ic_mat_column_plot_file_path,
                    title = "Intra-Column Interior Convection Matrix", marker_size = 1.0)

        ## Plot the intra-cell interior_convection matrix for the cell with minimal key
        cell_key: int = min(mesh.cols[col_key].cells.keys())
        cell: Cell = mesh.cols[col_key].cells[cell_key]
        cell_ndof: int = col.ndofs[0] * col.ndofs[1] * cell.ndofs[0]

        ic_mat_cell: np.ndarray = ic_mat[0:cell_ndof, 0:cell_ndof]
        ic_mat_cell_plot_file_name: str = "interior_convection_cell.png"
        ic_mat_cell_plot_file_path: str = \
            os.path.join(figs_dir_path, ic_mat_cell_plot_file_name)
        plot_matrix(ic_mat_cell, ic_mat_cell_plot_file_path,
                    title = "Intra-Cell Interior Convection Matrix", marker_size = 2.0)

    ## Obtain and output the boundary convection matrix
    bc_mat: sp._coo.coo_matrix = problem.boundary_convection_matrix(mesh)

    if comm_rank == 0:
        ## Plot the intra-process interior_convection matrix on process 0
        ownership_range: tuple = bc_mat.getOwnershipRange()
        bc_mat_global: np.ndarray = bc_mat[ownership_range[0]:ownership_range[1],
                                           ownership_range[0]:ownership_range[1]]

        bc_mat_global_plot_file_name: str = "boundary_convection_global.png"
        bc_mat_global_plot_file_path: str = \
            os.path.join(figs_dir_path, bc_mat_global_plot_file_name)
        plot_matrix(bc_mat_global, bc_mat_global_plot_file_path,
                    title = "Global Boundary Convection Matrix", marker_size = 0.5)
        
        ## Plot the intra-column interior_convection matrix for the column with minimal key
        col_key: int = min(mesh.cols.keys())
        col: Column = mesh.cols[col_key]
        col_ndof: int = 0
        for _, cell in col.cells.items():
            col_ndof += cell.ndofs[0]
        col_ndof *= col.ndofs[0] * col.ndofs[1]
        
        bc_mat_column: np.ndarray = bc_mat[0:col_ndof, 0:col_ndof]
        bc_mat_column_plot_file_name: str = "boundary_convection_column.png"
        bc_mat_column_plot_file_path: str = \
            os.path.join(figs_dir_path, bc_mat_column_plot_file_name)
        plot_matrix(bc_mat_column, bc_mat_column_plot_file_path,
                    title = "Intra-Column Boundary Convection Matrix", marker_size = 1.0)

        ## Plot the intra-cell interior_convection matrix for the cell with minimal key
        cell_key: int = min(mesh.cols[col_key].cells.keys())
        cell: Cell = mesh.cols[col_key].cells[cell_key]
        cell_ndof: int = col.ndofs[0] * col.ndofs[1] * cell.ndofs[0]

        bc_mat_cell: np.ndarray = bc_mat[0:cell_ndof, 0:cell_ndof]
        bc_mat_cell_plot_file_name: str = "boundary_convection_cell.png"
        bc_mat_cell_plot_file_path: str = \
            os.path.join(figs_dir_path, bc_mat_cell_plot_file_name)
        plot_matrix(bc_mat_cell, bc_mat_cell_plot_file_path,
                    title = "Intra-Cell Boundary Convection Matrix", marker_size = 2.0)
        
    ## Obtain and output the system matrix
    sys_mat: sp._coo.coo_matrix = ma_mat - sc_mat + (bc_mat - ic_mat)

    if comm_rank == 0:
        ## Plot the intra-process interior_convection matrix on process 0
        ownership_range: tuple = sys_mat.getOwnershipRange()
        sys_mat_global: np.ndarray = sys_mat[ownership_range[0]:ownership_range[1],
                                           ownership_range[0]:ownership_range[1]]

        sys_mat_global_plot_file_name: str = "system_global.png"
        sys_mat_global_plot_file_path: str = \
            os.path.join(figs_dir_path, sys_mat_global_plot_file_name)
        plot_matrix(sys_mat_global, sys_mat_global_plot_file_path,
                    title = "Global System Matrix", marker_size = 0.5)
        
        ## Plot the intra-column interior_convection matrix for the column with minimal key
        col_key: int = min(mesh.cols.keys())
        col: Column = mesh.cols[col_key]
        col_ndof: int = 0
        for _, cell in col.cells.items():
            col_ndof += cell.ndofs[0]
        col_ndof *= col.ndofs[0] * col.ndofs[1]
        
        sys_mat_column: np.ndarray = sys_mat[0:col_ndof, 0:col_ndof]
        sys_mat_column_plot_file_name: str = "system_column.png"
        sys_mat_column_plot_file_path: str = \
            os.path.join(figs_dir_path, sys_mat_column_plot_file_name)
        plot_matrix(sys_mat_column, sys_mat_column_plot_file_path,
                    title = "Intra-Column System Matrix", marker_size = 1.0)

        ## Plot the intra-cell interior_convection matrix for the cell with minimal key
        cell_key: int = min(mesh.cols[col_key].cells.keys())
        cell: Cell = mesh.cols[col_key].cells[cell_key]
        cell_ndof: int = col.ndofs[0] * col.ndofs[1] * cell.ndofs[0]

        sys_mat_cell: np.ndarray = sys_mat[0:cell_ndof, 0:cell_ndof]
        sys_mat_cell_plot_file_name: str = "system_cell.png"
        sys_mat_cell_plot_file_path: str = \
            os.path.join(figs_dir_path, sys_mat_cell_plot_file_name)
        plot_matrix(sys_mat_cell, sys_mat_cell_plot_file_path,
                    title = "Intra-Cell System Matrix", marker_size = 2.0)
    
    ## Calculate, output, and plot the solution
    [uh, _, _] = problem.solve(mesh)

    if comm_rank == 0:
        file_name: str = "uh.npy"
        file_path: str = os.path.join(out_dir_path, file_name)
        uh.to_file(file_path, write_mesh = False)

        file_name: str = "uh_th.png"
        file_path: str = os.path.join(figs_dir_path, file_name)
        plot_th(uh, file_path = file_path)

        file_name: str = "uh_xth.png"
        file_path: str = os.path.join(figs_dir_path, file_name)
        plot_xth(uh, file_path = file_path, cmap = "hot", scale = "normal")

        file_name: str = "uh_yth.png"
        file_path: str = os.path.join(figs_dir_path, file_name)
        plot_yth(uh, file_path = file_path, cmap = "hot", scale = "normal")

        file_name: str = "uh_xy.png"
        file_path: str = os.path.join(figs_dir_path, file_name)
        plot_xy(uh, file_path = file_path, cmap = "hot", scale = "normal")

        file_name: str = "uh_xyth.png"
        file_path: str = os.path.join(figs_dir_path, file_name)
        plot_xyth(uh, file_path = file_path, cmap = "hot", scale = "normal")
        
if __name__ == "__main__":
    main()