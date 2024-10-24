# Standard Library Imports
from typing import Callable

# Third-Party Library Imports
import numpy as np
from scipy.integrate import nquad

# Local Library Imports
import consts
from dg.quadrature import lag_eval, quad_xyth
from dg.projection import push_forward

# Relative Imports
from ..error_indicator_column import Error_Indicator_Column
from ..error_indicator_column.error_indicator_cell import Error_Indicator_Cell

intg_u2: float = None
phi_projs: dict = {}
psi_projs: dict = {}
xsi_projs: dict = {}

def error_analytic(self, anl_sol: Callable[[np.ndarray, np.ndarray, np.ndarray],
                                            np.ndarray]) -> None:
    # L2 error by cell and column (Sqrt[Sum[cell_err**2]] / Sqrt[Intg[u**2]])
    # Project u_hp to higher-resolution mesh b/c we need to integrat (u - u_hp)**2
    
    # Integrate square of analytic solution to weight the relative error
    global intg_u2
    if intg_u2 is None:
        [Lx, Ly] = self.proj.mesh.Ls[:]
        [intg_u2, _] = nquad(lambda x, y, th: (anl_sol(x, y, th))**2,
                             [[0, Lx], [0, Ly], [0, 2. * consts.PI]])
        
    # Store maximum errors to calculate hp-steering only where needed
    col_max_err: float  = -consts.INF
    cell_max_err: float = -consts.INF
    
    # Store the info needed for error_indicator
    cols: dict = {}
    mesh_err: float = 0.

    # Calculate the errors
    col_items: list = sorted(self.proj.mesh.cols.items())
    for col_key, col in col_items:
        assert(col.is_lf)
        
        ## Column information for quadrature
        [x0, y0, xf, yf] = col.pos[:]
        [dx, dy] = [xf - x0, yf - y0]
        [nx, ny] = col.ndofs[:]
        [nx_hr, ny_hr] = [nx + 3, ny + 3]

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
        
        xxf: np.ndarray = push_forward(x0, xf, xxb).reshape([nx_hr, 1, 1])
        wx: np.ndarray  = wx.reshape([nx_hr, 1, 1])
        yyf: np.ndarray = push_forward(y0, yf, yyb).reshape([1, ny_hr, 1])
        wy: np.ndarray  = wy.reshape([1, ny_hr, 1])
        
        # Store the info needed for error_indicator_columns
        col_err: float = 0.
        cells: dict = {}

        # Loop through cells to calculate error
        cell_items: list = sorted(col.cells.items())
        for cell_key, cell in cell_items:
            assert(cell.is_lf)

            # Cell information for quadrature
            [th0, th1] = cell.pos[:]
            dth: float = th1 - th0
            [nth]  = cell.ndofs[:]
            
            nth_hr: int = nth + 3
            
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
                
            [_, _, _, _, thb, wth] = quad_xyth(nnodes_th = nth_hr)
            thf: np.ndarray = push_forward(th0, th1, thb).reshape([1, 1, nth_hr])
            wth: np.ndarray = wth.reshape([1, 1, nth_hr])
            
            # Calculate error
            u_cell: np.ndarray = anl_sol(xxf, yyf, thf)
            uh_cell: np.ndarray = self.proj.cols[col_key].cells[cell_key].vals[:,:,:]
            uh_cell_hr = np.zeros([nx_hr, ny_hr, nth_hr])
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
            
            cell_err: float = \
                (dx * dy * dth / 8.) * np.sum(wx * wy * wth * (u_cell - uh_cell_hr)**2)

            col_err += cell_err
            mesh_err += cell_err
            
            cells[cell_key] = Error_Indicator_Cell(np.sqrt(cell_err / intg_u2)) # sqrt at the end to avoid sqrt then square
            cell_max_err: float = max(cell_max_err, np.sqrt(cell_err / intg_u2))
                
        cols[col_key] = Error_Indicator_Column(np.sqrt(col_err / intg_u2), cells)
        col_max_err: float = max(col_max_err, np.sqrt(col_err / intg_u2))
    
    self.cols: dict = cols
    self.col_max_error: float = col_max_err
    self.cell_max_error: float = cell_max_err
    self.error: float = np.sqrt(mesh_err / intg_u2)
    self.error_to_resolve: float = 0.

    ## Calculate if cols/cells need to be refined, and calculate hp-steering
    ang_ref_thrsh: float = self.ang_ref_tol * self.cell_max_error
    spt_ref_thrsh: float = self.spt_ref_tol * self.col_max_error
    for col_key, col in col_items:
        assert(col.is_lf)

        if self.ref_kind in ["spt", "all"]:
            if self.cols[col_key].error >= spt_ref_thrsh: # Does this one need to be refined?
                self.cols[col_key].do_ref = True
                self.error_to_resolve += self.cols[col_key].error
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
                    self.error_to_resolve += self.cols[col_key].cells[cell_key].error
                    if self.ref_form == "hp": # Does the form of refinement need to be chosen?
                        self.cols[col_key].cells[cell_key].ref_form = \
                            self.cell_hp_steer(col_key, cell_key)
                    else:
                        self.cols[col_key].cells[cell_key].ref_form = self.ref_form
                else: # Needn't be refined
                    self.cols[col_key].cells[cell_key].do_ref = False