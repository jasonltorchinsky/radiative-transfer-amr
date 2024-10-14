# Standard Library Imports
import copy

# Third-Party Library Imports
import numpy as np
import petsc4py
from   mpi4py import MPI
from   petsc4py import PETSc

# Local Library Imports
import consts
from dg.mesh import Mesh
from dg.mesh.column import Column
from dg.mesh.column.cell import Cell
from dg.projection import Projection
from dg.quadrature import lag_eval, quad_xyth
from rt import Problem

# Relative Imports
from ..error_indicator_column import Error_Indicator_Column
from ..error_indicator_column.error_indicator_cell import Error_Indicator_Cell

phi_projs: dict = {}
psi_projs: dict = {}
xsi_projs: dict = {}

def error_high_resolution(self, problem: Problem, uh_hr: Projection = None,
                          **kwargs) -> list:
    default_kwargs = {"ang_ref_offset" : 0, # # of times to refine mesh angularly 
                      "spt_ref_offset" : 0, # # of times to refine mesh spatially
                      "verbose"  : False, # Print info while executing
                      "blocking" : True # Synchronize ranks before exiting
                      }
    kwargs = {**default_kwargs, **kwargs}
    
    ## Initialize parallel communicators
    mpi_comm: MPI.Intracomm = MPI.COMM_WORLD
    
    if not PETSc.Sys.isInitialized():
        petsc4py.init(comm = mpi_comm)
    petsc_comm: PETSc.Comm = PETSc.COMM_WORLD
    comm_rank: int = petsc_comm.getRank()
    
    if uh_hr is None:
        ang_ref_offset: int = kwargs["ang_ref_offset"]
        spt_ref_offset: int = kwargs["spt_ref_offset"]

        if comm_rank == consts.COMM_ROOT:
            # Get high-resolution mesh
            mesh_hr: Mesh = copy.deepcopy(self.proj.mesh)
            for _ in range(0, ang_ref_offset):
                mesh_hr.ref_mesh(kind = "ang", form = "p")
            for _ in range(0, spt_ref_offset):
                mesh_hr.ref_mesh(kind = "spt", form = "p")
            mesh_hr: Mesh = mpi_comm.bcast(mesh_hr, root = consts.COMM_ROOT)
        else:
            mesh_hr: Mesh = None
            mesh_hr: Mesh = mpi_comm.bcast(mesh_hr, root = consts.COMM_ROOT)

        [uh_hr, info, mat_info] = problem.solve(mesh_hr)
        PETSc.garbage_cleanup(petsc_comm)
    else:
        info = None
        mat_info = None
    
    if comm_rank == consts.COMM_ROOT:
        # Integrate high-resolution solution
        intg_uh_hr2: float= 0.
        col_items_hr: list = sorted(uh_hr.mesh.cols.items())
        for col_key_hr, col_hr in col_items_hr:
            assert(col_hr.is_lf)
            
            [nx_hr, ny_hr] = col_hr.ndofs[:]
            [x0, y0, x1, y1] = col_hr.pos[:]
            [dx, dy]  = [x1 - x0, y1 - y0]
            [_, wx_hr, _, wy_hr, _, _] = quad_xyth(nnodes_x = nx_hr,
                                                   nnodes_y = ny_hr)
            wx_hr: np.ndarray = wx_hr.reshape([nx_hr, 1, 1])
            wy_hr: np.ndarray = wy_hr.reshape([1, ny_hr, 1])
            
            dcoeff: float = (dx * dy) / 4.
            
            cell_items_hr: list = sorted(col_hr.cells.items())
            for cell_key_hr, cell_hr in cell_items_hr:
                assert(cell_hr.is_lf)
                
                [nth_hr] = cell_hr.ndofs[:]
                [th0, th1] = cell_hr.pos[:]
                dth: float = th1 - th0
                
                [_, _, _, _, _, wth_hr] = quad_xyth(nnodes_th = nth_hr)
                
                wth_hr: np.ndarray = wth_hr.reshape([1, 1, nth_hr])
                
                uh_hr_cell: np.ndarray = uh_hr.cols[col_key_hr].cells[cell_key_hr].vals[:, :, :]
                
                intg_uh_hr2 += dcoeff * (dth / 2.) \
                    * np.sum(wx_hr * wy_hr * wth_hr * (uh_hr_cell)**2)
            
        # Now calculate the error. We project low-resolution solution to the 
        # high-resolution mesh. This may get into some weird quadrature rule
        # stuff, but that's the bullet we'll bite.
        # The meshs have the same column and cell keys.

        # Store maximum errors to calculate hp-steering only where needed
        col_max_err: float  = -consts.INF
        cell_max_err: float = -consts.INF

        # Store the info needed for error_indicator
        cols: dict = {}
        mesh_err: float = 0.

        col_keys: list = sorted(self.proj.mesh.cols.keys())
        for col_key in col_keys:
            col: Column = self.proj.mesh.cols[col_key]
            col_hr: Column = uh_hr.mesh.cols[col_key]
            assert(col.is_lf and col_hr.is_lf)
            
            ## Column information for quadrature
            [x0, y0, xf, yf] = col.pos[:]
            [dx, dy] = [xf - x0, yf - y0]
            [nx, ny] = col.ndofs[:]
            [nx_hr, ny_hr] = col_hr.ndofs[:]

            ## Store spatial projection matrices for later reuse
            global phi_projs
            if (nx, nx_hr) in phi_projs.keys():
                phi_proj: np.ndarray = phi_projs[(nx, nx_hr)][:]
            else:
                [xxb,    _, _, _, _, _] = quad_xyth(nnodes_x = nx)

                [xxb_hr, _, _, _, _, _] = quad_xyth(nnodes_x = nx_hr)

                phi_proj: np.ndarray = np.zeros([nx, nx_hr])
                for ii in range(0, nx):
                    for pp in range(0, nx_hr):
                        phi_proj[ii, pp] = lag_eval(xxb, ii, xxb_hr[pp])
                phi_projs[(nx, nx_hr)] = phi_proj[:]

            global psi_projs
            if (ny, ny_hr) in psi_projs.keys():
                psi_proj: np.ndarray = psi_projs[(ny, ny_hr)][:]
            else:
                [_, _, yyb   , _, _, _] = quad_xyth(nnodes_y = ny)

                [_, _, yyb_hr, _, _, _] = quad_xyth(nnodes_y = ny_hr)

                psi_proj: np.ndarray = np.zeros([ny, ny_hr])
                for jj in range(0, ny):
                    for qq in range(0, ny_hr):
                        psi_proj[jj, qq] = lag_eval(yyb, jj, yyb_hr[qq])
                psi_projs[(ny, ny_hr)] = psi_proj[:]

            [xxb, wx, yyb, wy, _, _] = quad_xyth(nnodes_x = nx_hr, nnodes_y = ny_hr)
            wx: np.ndarray = wx.reshape([nx_hr, 1, 1])
            wy: np.ndarray = wy.reshape([1, ny_hr, 1])

            # Store the info needed for error_indicator_columns
            col_err: float = 0.
            cells: dict = {}

            # Loop through cells to calculate error
            cell_keys: list = sorted(col.cells.keys())
            for cell_key in cell_keys:
                cell: Cell = col.cells[cell_key]
                cell_hr: Cell = col_hr.cells[cell_key]
                assert(cell.is_lf and cell_hr.is_lf)

                # Cell information for quadrature
                [th0, th1] = cell.pos[:]
                dth: float = th1 - th0
                [nth] = cell.ndofs[:]
                [nth_hr] = cell_hr.ndofs[:]

                # Store angular projection matrices for later reuse
                global xsi_projs
                if (nth, nth_hr) in xsi_projs.keys():
                    xsi_proj: np.ndarray = xsi_projs[(nth, nth_hr)][:]
                else:
                    [_, _, _, _, thb,    _] = quad_xyth(nnodes_th = nth)

                    [_, _, _, _, thb_hr, _] = quad_xyth(nnodes_th = nth_hr)

                    xsi_proj: np.ndarray = np.zeros([nth, nth_hr])
                    for aa in range(0, nth):
                        for rr in range(0, nth_hr):
                            xsi_proj[aa, rr] = lag_eval(thb, aa, thb_hr[rr])
                    xsi_projs[(nth, nth_hr)] = xsi_proj[:]

                [_, _, _, _, _, wth] = quad_xyth(nnodes_th = nth_hr)
                wth: np.ndarray = wth.reshape([1, 1, nth_hr])

                # Calculate error
                uh_cell: np.ndarray = self.proj.cols[col_key].cells[cell_key].vals[:,:,:]
                uh_cell_hr: np.ndarray = np.zeros([nx_hr, ny_hr, nth_hr])
                for pp in range(0, nx_hr):
                    for qq in range(0, ny_hr):
                        for rr in range(0, nth_hr):
                            for ii in range(0, nx):
                                phi_ip: float = phi_proj[ii, pp]
                                for jj in range(0, ny):
                                    psi_jq: float = psi_proj[jj, qq]
                                    for aa in range(0, nth):
                                        xsi_ar: float = xsi_proj[aa, rr]

                                        uh_ija: float = uh_cell[ii, jj, aa]

                                        uh_cell_hr[pp, qq, rr] += uh_ija * phi_ip * psi_jq * xsi_ar

                uh_hr_cell: np.ndarray = uh_hr.cols[col_key].cells[cell_key].vals[:,:,:]
                cell_err: float = \
                    (dx * dy * dth / 8.) * np.sum(wx * wy * wth * (uh_hr_cell - uh_cell_hr)**2)

                col_err += cell_err
                mesh_err += cell_err

                cells[cell_key] = Error_Indicator_Cell(np.sqrt(cell_err / intg_uh_hr2)) # sqrt at the end to avoid sqrt then square
                cell_max_err: float = max(cell_max_err, np.sqrt(cell_err / intg_uh_hr2))

            cols[col_key] = Error_Indicator_Column(np.sqrt(col_err / intg_uh_hr2), cells)
            col_max_err: float = max(col_max_err, np.sqrt(col_err / intg_uh_hr2))
    
        self.cols: dict = cols
        self.col_max_error: float = col_max_err
        self.cell_max_error: float = cell_max_err
        self.error: float = np.sqrt(mesh_err / intg_uh_hr2)

        ## Calculate if cols/cells need to be refined, and calculate hp-steering
        col_items: list = sorted(self.proj.mesh.cols.items())
        ang_ref_thrsh: float = self.ang_ref_tol * self.cell_max_error
        spt_ref_thrsh: float = self.spt_ref_tol * self.col_max_error
        for col_key, col in col_items:
            assert(col.is_lf)

            if self.ref_kind in ["spt", "all"]:
                if self.cols[col_key].error >= spt_ref_thrsh: # Does this one need to be refined?
                    self.cols[col_key].do_ref = True
                    if self.ref_form == "hp": # Does the form of refinement need to be chosen?
                        self.cols[col_key].ref_form = self.col_hp_steer(col_key)
                    else:
                        self.cols[col_key].ref_form = self.ref_form
                else: # Needn't be refined
                    self.cols[col_key].do_ref = False

            if self.ref_kind in ["ang", "all"]:
                cell_items: list = sorted(col.cells.items())
                for cell_key, cell in cell_items:
                    assert(cell.is_lf)

                    if self.cols[col_key].cells[cell_key].error >= ang_ref_thrsh: # Does this one need to be refined?
                        self.cols[col_key].cells[cell_key].do_ref = True
                        if self.ref_form == "hp": # Does the form of refinement need to be chosen?
                            self.cols[col_key].cells[cell_key].ref_form = \
                                self.cell_hp_steer(col_key, cell_key)
                        else:
                            self.cols[col_key].cells[cell_key].ref_form = self.ref_form
                    else: # Needn't be refined
                        self.cols[col_key].cells[cell_key].do_ref = False

        self = mpi_comm.bcast(self, root = consts.COMM_ROOT)
    else:
        self = mpi_comm.bcast(self, root = consts.COMM_ROOT)

    mpi_comm.barrier()
    return [uh_hr, info, mat_info]