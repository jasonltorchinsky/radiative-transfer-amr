# Standard Library Imports

# Third-Party Library Imports
import numpy as np

# Local Library Imports
import consts
from dg.projection.projection_column import Projection_Column
from dg.projection.projection_column.projection_cell import Projection_Cell
from dg.projection import push_forward, pull_back
from dg.quadrature import lag_eval, quad_xyth

# Relative Imports
from ..error_indicator_column import Error_Indicator_Column
from ..error_indicator_column.error_indicator_cell import Error_Indicator_Cell

def error_spatial_jump(self) -> None:
    # Spatial jump error

    # Store maximum errors to calculate hp-steering only where needed
    col_max_err: float  = -consts.INF
    cell_max_err: float = -consts.INF
    
    # Store the info needed for error_indicator
    cols: dict = {}
    mesh_err: float = 0.
    
    # We get the jumps for each pair of neighboring cells
    # _0 refers to self, _1 refers to neighbor
    # _A => smaller spatial domain, _B => larger number of spatial DoFs
    # _G => smaller angular domain, _H => larger number of angular DoFs
    col_items: list = sorted(self.proj.cols.items())
    for col_key_0, col_0 in col_items:
        assert(col_0.is_lf)
            
        # Get self-column info
        [x0_0, y0_0, x1_0, y1_0] = col_0.pos[:]
        [dx_0, dy_0]  = [x1_0 - x0_0, y1_0 - y0_0]
        perim: float  = 2. * dx_0 + 2. * dy_0
        [nx_0, ny_0]  = col_0.ndofs[:]
        
        [xxb_0, _, yyb_0, _, _, _] = quad_xyth(nnodes_x = nx_0, nnodes_y = ny_0)
        
        # Store the info needed for error_indicator_columns
        col_err: float = 0.
        cells: dict = {}
        
        # Since column-jump error only has values for columns, we're
        # going to fill in dummy values for the error indicator cells
        cell_items_0: list = sorted(col_0.cells.items())
        for cell_key, cell in cell_items_0:
            assert(cell.is_lf)

            cells[cell_key] = Error_Indicator_Cell(-consts.INF)
        col_err: float = -consts.INF

        # Loop through faces to calculate error
        col_jump: float = 0.
        for col_face in range(0, 4):
            nhbr_keys: list = list(set(col_0.nhbr_keys[col_face]))
            
            for col_key_1 in nhbr_keys:
                if (col_key_1 is not None):
                    
                    col_1: Projection_Column = self.proj.cols[col_key_1]
                    assert(col_1.is_lf)

                    # Get nhbr-column info
                    [x0_1, y0_1, x1_1, y1_1] = col_1.pos[:]
                    [dx_1, dy_1] = [x1_1 - x0_1, y1_1 - y0_1]
                    [nx_1, ny_1] = col_1.ndofs[:]

                    [xxb_1, _, yyb_1, _, _, _] = quad_xyth(nnodes_x = nx_1,
                                                           nnodes_y = ny_1)

                    # Get integration interval, quadrature info
                    if (col_face%2 == 0):
                        if dy_0 <= dy_1:
                            [y0_A, y1_A] = [y0_0, y1_0]
                            [dy_A] = [dy_0]
                        else:
                            [y0_A, y1_A] = [y0_1, y1_1]
                            [dy_A] = [dy_1]

                        if ny_0 >= ny_1:
                            [ny_B] = [ny_0]
                        else:
                            [ny_B] = [ny_1]

                        [_, _, yyb_B, wy_B, _, _] = quad_xyth(nnodes_y = ny_B)
                        wy_B: np.ndarray = wy_B.reshape([ny_B, 1])

                        yyf_A: np.ndarray   = push_forward(y0_A, y1_A, yyb_B)
                        yyb_A_0: np.ndarray = pull_back(y0_0, y1_0, yyf_A)
                        yyb_A_1: np.ndarray = pull_back(y0_1, y1_1, yyf_A)

                        psi_0_mtx: np.ndarray = np.zeros([ny_0, ny_B])
                        for jj in range(0, ny_0):
                            for jj_p in range(0, ny_B):
                                psi_0_mtx[jj, jj_p] = lag_eval(yyb_0, jj, yyb_A_0[jj_p])

                        psi_1_mtx: np.ndarray = np.zeros([ny_1, ny_B])
                        for qq in range(0, ny_1):
                            for jj_p in range(0, ny_B):
                                psi_1_mtx[qq, jj_p] = lag_eval(yyb_1, qq, yyb_A_1[jj_p])

                    else: # (F%2 == 1)
                        if dx_0 <= dx_1:
                            [x0_A, x1_A] = [x0_0, x1_0]
                            [dx_A]       = [dx_0]
                        else:
                            [x0_A, x1_A] = [x0_1, x1_1]
                            [dx_A]       = [dx_1]

                        if nx_0 >= nx_1:
                            [nx_B] = [nx_0]
                        else:
                            [nx_B] = [nx_1]

                        [xxb_B, wx_B, _, _, _, _] = quad_xyth(nnodes_x = nx_B)
                        wx_B: np.ndarray    = wx_B.reshape([nx_B, 1])

                        xxf_A: np.ndarray   = push_forward(x0_A, x1_A, xxb_B)
                        xxb_A_0: np.ndarray = pull_back(x0_0, x1_0, xxf_A)
                        xxb_A_1: np.ndarray = pull_back(x0_1, x1_1, xxf_A)

                        phi_0_mtx: np.ndarray = np.zeros([nx_0, nx_B])
                        for ii in range(0, nx_0):
                            for ii_p in range(0, nx_B):
                                phi_0_mtx[ii, ii_p] = lag_eval(xxb_0, ii, xxb_A_0[ii_p])

                        phi_1_mtx: np.ndarray = np.zeros([nx_1, nx_B])
                        for pp in range(0, nx_1):
                            for ii_p in range(0, nx_B):
                                phi_1_mtx[pp, ii_p] = lag_eval(xxb_1, pp, xxb_A_1[ii_p])

                    # Loop through self-cells
                    for cell_key_0, cell_0 in cell_items_0:
                        assert(cell_0.is_lf)

                        # Get self-cell info
                        [th0_0, th1_0] = cell_0.pos[:]
                        [dth_0] = [th1_0 - th0_0]
                        [nth_0] = cell_0.ndofs[:]

                        [_, _, _, _, thb_0, _] = quad_xyth(nnodes_th = nth_0)

                        # Get solution values in cell 0
                        if col_face == 0:
                            uh_0: np.ndarray = cell_0.vals[-1,:,:]
                        elif col_face == 1:
                            uh_0: np.ndarray = cell_0.vals[:,-1,:]
                        elif col_face == 2:
                            uh_0: np.ndarray = cell_0.vals[0,:,:]
                        else: # F == 3
                            uh_0: np.ndarray = cell_0.vals[:,0,:]

                        # Get nhbr_cell info
                        nhbr_keys: list = self.proj.mesh.nhbr_cells_in_nhbr_col(col_key_0,
                                                                                cell_key_0,
                                                                                col_key_1)
                        for cell_key_1 in nhbr_keys:
                            if (cell_key_1 is not None):

                                cell_1: Projection_Cell = col_1.cells[cell_key_1]
                                assert(cell_1.is_lf)

                                # Get nhbr-cell info
                                [th0_1, th1_1] = cell_1.pos[:]
                                [dth_1] = [th1_1 - th0_1]
                                [nth_1] = cell_1.ndofs[:]

                                [_, _, _, _, thb_1, _] = quad_xyth(nnodes_th = nth_1)

                                # Get solution values in cell 1
                                # F refers to the face of _0, so
                                # it"s opposite for _1
                                if col_face == 0:
                                    uh_1: np.ndarray = cell_1.vals[0,:,:]
                                elif col_face == 1:
                                    uh_1: np.ndarray = cell_1.vals[:,0,:]
                                elif col_face == 2:
                                    uh_1: np.ndarray = cell_1.vals[-1,:,:]
                                else: # F == 3
                                    uh_1: np.ndarray = cell_1.vals[:,-1,:]

                                # Get integration interval, quadrature info
                                if dth_0 <= dth_1:
                                    [th0_G, th1_G] = [th0_0, th1_0]
                                    [dth_G] = [dth_0]
                                else:
                                    [th0_G, th1_G] = [th0_1, th1_1]
                                    [dth_G] = [dth_1]
                                if nth_0 >= nth_1:
                                    [nth_H] = [nth_0]
                                else:
                                    [nth_H] = [nth_1]

                                [_, _, _, _, thb_H, wth_H] = quad_xyth(nnodes_th = nth_H)
                                wth_H: np.ndarray = wth_H.reshape([1, nth_H])

                                thf_G: np.ndarray = push_forward(th0_G, th1_G, thb_H)
                                thb_G_0: np.ndarray = pull_back(th0_0, th1_0, thf_G)
                                thb_G_1: np.ndarray = pull_back(th0_1, th1_1, thf_G)

                                xsi_0_mtx: np.ndarray = np.zeros([nth_0, nth_H])
                                for aa in range(0, nth_0):
                                    for aa_p in range(0, nth_H):
                                        xsi_0_mtx[aa, aa_p] = lag_eval(thb_0, aa, thb_G_0[aa_p])

                                xsi_1_mtx: np.ndarray = np.zeros([nth_1, nth_H])
                                for rr in range(0, nth_1):
                                    for aa_p in range(0, nth_H):
                                        xsi_1_mtx[rr, aa_p] = lag_eval(thb_1, rr, thb_G_1[aa_p])

                                if (col_face%2 == 0):
                                    # Project uh_0, uh_1 to the same
                                    # quadrature and integrate jump
                                    uh_0_proj: np.ndarray = np.zeros([ny_B, nth_H])
                                    for jj in range(0, ny_0):
                                        for aa in range(0, nth_0):
                                            for jj_p in range(0, ny_B):
                                                for aa_p in range(0, nth_H):
                                                    uh_0_proj[jj_p, aa_p] += uh_0[jj, aa] * psi_0_mtx[jj, jj_p] * xsi_0_mtx[aa, aa_p]
                                    uh_1_proj: np.ndarray = np.zeros([ny_B, nth_H])
                                    for qq in range(0, ny_1):
                                        for rr in range(0, nth_1):
                                            for jj_p in range(0, ny_B):
                                                for aa_p in range(0, nth_H):
                                                    uh_1_proj[jj_p, aa_p] += uh_1[qq, rr] * psi_1_mtx[qq, jj_p] * xsi_1_mtx[rr, aa_p]
                                    col_jump += (dy_A * dth_G / 4.) * np.sum(wy_B * wth_H * (uh_0_proj - uh_1_proj)**2)
                                else: # (F%2 == 1)
                                    uh_0_proj: np.ndarray = np.zeros([nx_B, nth_H])
                                    for ii in range(0, nx_0):
                                        for aa in range(0, nth_0):
                                            for ii_p in range(0, nx_B):
                                                for aa_p in range(0, nth_H):
                                                    uh_0_proj[ii_p, aa_p] += uh_0[ii, aa] * phi_0_mtx[ii, ii_p] * xsi_0_mtx[aa, aa_p]
                                    uh_1_proj: np.ndarray = np.zeros([nx_B, nth_H])
                                    for pp in range(0, nx_1):
                                        for rr in range(0, nth_1):
                                            for ii_p in range(0, nx_B):
                                                for aa_p in range(0, nth_H):
                                                    uh_1_proj[ii_p, aa_p] += uh_1[pp, rr] * phi_1_mtx[pp, ii_p] * xsi_1_mtx[rr, aa_p]
                                    col_jump += (dx_A * dth_G / 4.) * np.sum(wx_B * wth_H * (uh_0_proj - uh_1_proj)**2)
                                                
        col_err: float = (1. / (perim * 2. * consts.PI)) * col_jump
        mesh_err += col_err

        cols[col_key_0] = Error_Indicator_Column(np.sqrt(col_err), cells)
        col_max_err: float = max(col_max_err, np.sqrt(col_err))

    self.cols: dict = cols
    self.col_max_error: float = col_max_err
    self.cell_max_error: float = cell_max_err
    self.error: float = np.sqrt(mesh_err)
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