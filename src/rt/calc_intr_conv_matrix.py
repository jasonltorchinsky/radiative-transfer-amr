import numpy        as np
import petsc4py
import scipy.sparse as sp
from   mpi4py       import MPI
from   petsc4py     import PETSc
from   time         import perf_counter

import dg.matrix     as mat
import dg.projection as proj
import dg.quadrature as qd
import utils

ddy_psis = {}
ddx_phis = {}

def calc_intr_conv_matrix(mesh, **kwargs):

    default_kwargs = {'verbose'  : False, # Print info while executing
                      'blocking' : True   # Synchronize ranks before exiting
                      }
    kwargs = {**default_kwargs, **kwargs}
    
    # Initialize parallel communicators
    MPI_comm = MPI.COMM_WORLD
    
    petsc4py.init()
    comm      = PETSc.COMM_WORLD
    comm_rank = comm.getRank()
    comm_size = comm.getSize()
    
    if kwargs['verbose']:
        t0 = perf_counter()
        msg = (
            'Constructing Interior Propagation Matrix...\n'
            )
        utils.print_msg(msg)
        
    if comm_rank == 0:
        # Create column indexing for constructing global interior convection  matrix
        [ncols, col_idxs] = proj.get_col_idxs(mesh)
        col_mtxs = [None] * ncols # Global interior convection matrix is block-diagonal,
        # and so there are only ncol non-zero column mass matrices
        
        for col_key, col in sorted(mesh.cols.items()):
            if col.is_lf:
                # Get column information, quadrature weights
                col_idx = col_idxs[col_key]
                [x0, y0, x1, y1] = col.pos
                [dx, dy] = [x1 - x0, y1 - y0]
                [ndof_x, ndof_y] = col.ndofs
                
                [xxb, w_x, yyb, w_y, _, _] = qd.quad_xyth(nnodes_x = ndof_x,
                                                          nnodes_y = ndof_y)
                
                w_x = w_x.reshape([ndof_x, 1])
                w_y = w_y.reshape([1, ndof_y])
                
                wx_wy = w_x * w_y
                
                # Create cell indexing for constructing column mass matrix
                [ncells, cell_idxs] = proj.get_cell_idxs(mesh, col_key)
                cell_mtxs = [None] * ncells # Column interior convection  matrix is
                # block-diagonal, and so there are only 
                # ncell non-zero cell interior convection
                # matrices
                
                for cell_key, cell in sorted(col.cells.items()):
                    if cell.is_lf:
                        # Get cell information, quadrature weights
                        cell_idx   = cell_idxs[cell_key]
                        [th0, th1] = cell.pos
                        dth        = th1 - th0
                        [ndof_th]  = cell.ndofs
                        
                        [_, _, _, _, thb, w_th] = qd.quad_xyth(nnodes_th = ndof_th)
                        
                        thf = proj.push_forward(th0, th1, thb)
                        
                        wth_sinth = w_th * np.sin(thf)
                        wth_costh = w_th * np.cos(thf)
                        
                        # Indexing from i, j, a to beta
                        # Same formula for p, q, r to alpha, but we define alpha
                        # anyway for clarity
                        alpha = mat.get_idx_map(ndof_x, ndof_y, ndof_th)
                        beta  = mat.get_idx_map(ndof_x, ndof_y, ndof_th)
                        
                        # Set up arrays for delta_ip * delta_ar term
                        cell_ndof      = ndof_x * ndof_y * ndof_th
                        cell_ndof_ipar = ndof_x * ndof_y**2 * ndof_th
                        alphalist_ipar = np.zeros([cell_ndof_ipar], dtype = np.int32)
                        betalist_ipar  = np.zeros([cell_ndof_ipar], dtype = np.int32)
                        vlist_ipar     = np.zeros([cell_ndof_ipar])
                        
                        # Construct delta_ip * delta_ar term
                        # i = p, a = r
                        # When we take the derivative of psi, we end up with
                        # 2/dy * psi_bar in normalized coordinates, so dcoeff is
                        dcoeff = dx * dth / 4.
                        
                        # Store the ddy_psi matrix if we haven't yet calculated it
                        if ndof_y in ddy_psis.keys():
                            ddy_psi = ddy_psis[ndof_y]
                        else:
                            ddy_psi = np.zeros([ndof_y, ndof_y])
                            for qq in range(0, ndof_y):
                                for jj in range(0, ndof_y):
                                    ddy_psi[qq, jj] = qd.lag_ddx_eval(yyb, qq, yyb[jj])
                            ddy_psis[ndof_y] = ddy_psi
                            
                        idx = 0
                        for ii in range(0, ndof_x):
                            for jj in range(0, ndof_y):
                                wxi_wyj = wx_wy[ii, jj]
                                for aa in range(0, ndof_th):
                                    wth_sinth_a = wth_sinth[aa]
                                    for qq in range(0, ndof_y):
                                        ddy_psi_qj = ddy_psi[qq, jj]
                                        
                                        alphalist_ipar[idx] = alpha(ii, qq, aa)
                                        betalist_ipar[idx]  = beta( ii, jj, aa)
                                        
                                        vlist_ipar[idx] = dcoeff * wxi_wyj \
                                            * wth_sinth_a * ddy_psi_qj
                                        
                                        idx += 1
                                        
                        delta_ipar = sp.coo_matrix((vlist_ipar,
                                                    (alphalist_ipar, betalist_ipar)),
                                                   shape = (cell_ndof, cell_ndof))
                        
                        # Set up arrays for  delta_jq * delta_ar term
                        cell_ndof_jqar = ndof_x**2 * ndof_y * ndof_th
                        alphalist_jqar = np.zeros([cell_ndof_jqar], dtype = np.int32)
                        betalist_jqar  = np.zeros([cell_ndof_jqar], dtype = np.int32)
                        vlist_jqar     = np.zeros([cell_ndof_jqar])
                        
                        # Construct delta_jq * delta_ar term
                        # j = q, a = r
                        # When we take the derivative of phi, we end up with
                        # 2/dx * phi_bar in normalized coordinates, so dcoeff is
                        dcoeff = dy * dth / 4.
                        
                        # Store the ddy_psi matrix if we haven't yet calculated it
                        if ndof_x in ddx_phis.keys():
                            ddx_phi = ddx_phis[ndof_x]
                        else:
                            ddx_phi = np.zeros([ndof_x, ndof_x])
                            for pp in range(0, ndof_x):
                                for ii in range(0, ndof_x):
                                    ddx_phi[pp, ii] = qd.lag_ddx_eval(xxb, pp, xxb[ii])
                            ddx_phis[ndof_x] = ddx_phi
                            
                        idx = 0
                        for ii in range(0, ndof_x):
                            for jj in range(0, ndof_y):
                                wxi_wyj = wx_wy[ii, jj]
                                for aa in range(0, ndof_th):
                                    wth_costh_a = wth_costh[aa]
                                    for pp in range(0, ndof_x):
                                        ddx_phi_pi = ddx_phi[pp, ii]
                                        
                                        alphalist_jqar[idx] = alpha(pp, jj, aa)
                                        betalist_jqar[idx]  = beta( ii, jj, aa)
                                        
                                        vlist_jqar[idx] = dcoeff * wxi_wyj * wth_costh_a \
                                            * ddx_phi_pi
                                        
                                        idx += 1
                                        
                        delta_jqar = sp.coo_matrix((vlist_jqar,
                                                    (alphalist_jqar, betalist_jqar)),
                                                   shape = (cell_ndof, cell_ndof))
                        
                        cell_mtxs[cell_idx] = delta_ipar + delta_jqar
                        
                col_mtxs[col_idx] = sp.block_diag(cell_mtxs, format = 'coo')
                
        # Global interior convection matrix is block-diagonal
        # with the column matrices as the blocks
        intr_conv_mtx = sp.block_diag(col_mtxs, format = 'csr')
        
    else:
        intr_conv_mtx = 0
        
    if kwargs['verbose']:
        tf = perf_counter()
        msg = (
            'Constructed Interior Propagation Matrix\n' +
            12 * ' '  + 'Time Elapsed: {:8.4f} [s]\n'.format(tf - t0)
        )
        utils.print_msg(msg)
        
    if kwargs['blocking']:
        MPI_comm.Barrier()
        
    return intr_conv_mtx
    
