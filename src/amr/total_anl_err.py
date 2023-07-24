import numpy           as np
import petsc4py
from   mpi4py          import MPI
from   petsc4py        import PETSc
import scipy.integrate as integrate

import dg.projection as proj
import dg.quadrature as qd

intg_u2  = {}
phi_mtxs = {}
psi_mtxs = {}
xsi_mtxs = {}

def total_anl_err(mesh, num_sol, anl_sol, **kwargs):
    if kwargs['res_coeff'] == 1:
        return total_anl_err_lr(mesh, num_sol, anl_sol, **kwargs)
    else:
        return total_anl_err_hr(mesh, num_sol, anl_sol, **kwargs)

def total_anl_err_lr(mesh, num_sol, anl_sol, **kwargs):
    """
    Calculate the L2-error by cell (and column), weighted to be the relative error.
    """
    
    default_kwargs = {'res_coeff' : 1,   # Factor to multiply DoFs
                      'key'       : ' ', # Key to hold intg_u2
                      'blocking'  : True # Synchronize ranks before exiting
                      }
    kwargs = {**default_kwargs, **kwargs}
    
    # Initialize parallel communicators
    MPI_comm = MPI.COMM_WORLD
    
    petsc4py.init()
    comm      = PETSc.COMM_WORLD
    comm_rank = comm.getRank()
    comm_size = comm.getSize()
    
    if comm_rank == 0:
        col_items = sorted(mesh.cols.items())
        
        # Integrate th analytic solution and the square of the analytic solution here
        global intg_u2
        if kwargs['key'] not in intg_u2.keys():
            [Lx, Ly] = mesh.Ls[:]
            [intg_u2[kwargs['key']], _] = integrate.nquad(lambda x, y, th: (anl_sol(x, y, th))**2,
                                           [[0, Lx], [0, Ly], [0, 2. * np.pi]])
            
        err = 0.
        # Calculate the errors
        for col_key, col in col_items:
            if col.is_lf:
                # Column information for quadrature
                [x0, y0, x1, y1] = col.pos[:]
                [dx, dy]         = [x1 - x0, y1 - y0]
                [ndof_x, ndof_y] = col.ndofs[:]
                
                [xxb, wx, yyb, wy, _, _] = qd.quad_xyth(nnodes_x = ndof_x,
                                                        nnodes_y = ndof_y)
                
                xxf = proj.push_forward(x0, x1, xxb).reshape([ndof_x, 1, 1])
                yyf = proj.push_forward(y0, y1, yyb).reshape([1, ndof_y, 1])
                
                wx = wx.reshape([ndof_x, 1, 1])
                wy = wy.reshape([1, ndof_y, 1])
                
                # Loop through cells to calculate error
                cell_items = sorted(col.cells.items())
                for cell_key, cell in cell_items:
                    if cell.is_lf:
                        # Cell information for quadrature
                        [th0, th1] = cell.pos[:]
                        dth = th1 - th0
                        [ndof_th]  = cell.ndofs[:]
                        
                        [_, _, _, _, thb, wth] = qd.quad_xyth(nnodes_th = ndof_th)
                        
                        thf = proj.push_forward(th0, th1, thb).reshape([1, 1, ndof_th])
                        wth = wth.reshape([1, 1, ndof_th])
                        
                        # Calculate error
                        u_cell  = anl_sol(xxf, yyf, thf)
                        uh_cell = num_sol.cols[col_key].cells[cell_key].vals[:, :, :]
                        
                        err += (dx * dy * dth / 8.) * np.sum(wx * wy * wth * (u_cell - uh_cell)**2)
                    
        err = np.sqrt(err/intg_u2[kwargs['key']])
    else:
        err = 0

    err = MPI_comm.bcast(err, root = 0)

    if kwargs['blocking']:
        MPI_comm.Barrier()

    return err

def total_anl_err_hr(mesh, num_sol, anl_sol, **kwargs):
    """
    Calculate the L2-error by cell (and column), weighted to be the relative error.
    """
    
    default_kwargs = {'res_coeff' : 3,   # Factor to multiply DoFs
                      'blocking'  : True # Synchronize ranks before exiting
                      }
    kwargs = {**default_kwargs, **kwargs}
    
    # Initialize parallel communicators
    MPI_comm = MPI.COMM_WORLD
    
    petsc4py.init()
    comm      = PETSc.COMM_WORLD
    comm_rank = comm.getRank()
    comm_size = comm.getSize()

    res_coeff = kwargs['res_coeff']
    
    if comm_rank == 0:
        col_items = sorted(mesh.cols.items())
        
        # Integrate th analytic solution and the square of the analytic solution here
        global intg_u2
        if kwargs['key'] not in intg_u2.keys():
            [Lx, Ly] = mesh.Ls[:]
            [intg_u2[kwargs['key']], _] = integrate.nquad(lambda x, y, th: (anl_sol(x, y, th))**2,
                                           [[0, Lx], [0, Ly], [0, 2. * np.pi]])
            
        err = 0.
        # Calculate the errors
        for col_key, col in col_items:
            if col.is_lf:
                # Column information for quadrature
                [x0, y0, x1, y1] = col.pos[:]
                [dx, dy]         = [x1 - x0, y1 - y0]
                [ndof_x, ndof_y] = col.ndofs[:]
                
                [xxb, _, yyb, _, _, _] = qd.quad_xyth(nnodes_x = ndof_x,
                                                      nnodes_y = ndof_y)
                
                [ndof_x_hr, ndof_y_hr] = [int(res_coeff * ndof_x), int(res_coeff * ndof_y)]
                
                [xxb_hr, wx_hr, yyb_hr, wy_hr, _, _] = qd.quad_xyth(nnodes_x = ndof_x_hr,
                                                                    nnodes_y = ndof_y_hr)
                
                xxf_hr = proj.push_forward(x0, x1, xxb_hr).reshape([ndof_x_hr, 1, 1])
                yyf_hr = proj.push_forward(y0, y1, yyb_hr).reshape([1, ndof_y_hr, 1])
                
                wx_hr = wx_hr.reshape([ndof_x_hr, 1, 1])
                wy_hr = wy_hr.reshape([1, ndof_y_hr, 1])
                
                if (ndof_x, ndof_x_hr) in phi_mtxs.keys():
                    phi_mtx = phi_mtxs[(ndof_x, ndof_x_hr)]
                else:
                    phi_mtx = np.zeros([ndof_x, ndof_x_hr])
                    for ii in range(0, ndof_x):
                        for pp in range(0, ndof_x_hr):
                            phi_mtx[ii, pp] = qd.lag_eval(xxb, ii, xxb_hr[pp])
                    phi_mtxs[(ndof_x, ndof_x_hr)] = phi_mtx
                    
                if (ndof_y, ndof_y_hr) in psi_mtxs.keys():
                    psi_mtx = psi_mtxs[(ndof_y, ndof_y_hr)]
                else:
                    psi_mtx = np.zeros([ndof_y, ndof_y_hr])
                    for jj in range(0, ndof_y):
                        for qq in range(0, ndof_y_hr):
                            psi_mtx[jj, qq] = qd.lag_eval(yyb, jj, yyb_hr[qq])
                    psi_mtxs[(ndof_y, ndof_y_hr)] = psi_mtx
                    
                # Loop through cells to calculate error
                cell_items = sorted(col.cells.items())
                for cell_key, cell in cell_items:
                    if cell.is_lf:
                        # Cell information for quadrature
                        [th0, th1] = cell.pos[:]
                        dth = th1 - th0
                        [ndof_th]  = cell.ndofs[:]
                        
                        [_, _, _, _, thb, _] = qd.quad_xyth(nnodes_th = ndof_th)
                        
                        [ndof_th_hr]  = [int(res_coeff * ndof_th)]
                        
                        [_, _, _, _, thb_hr, wth_hr] = qd.quad_xyth(nnodes_th = ndof_th_hr)
                        
                        thf_hr = proj.push_forward(th0, th1, thb_hr).reshape([1, 1, ndof_th_hr])
                        wth_hr = wth_hr.reshape([1, 1, ndof_th_hr])
                        
                        if (ndof_th, ndof_th_hr) in xsi_mtxs.keys():
                            xsi_mtx = xsi_mtxs[(ndof_th, ndof_th_hr)]
                        else:
                            xsi_mtx = np.zeros([ndof_th, ndof_th_hr])
                            for aa in range(0, ndof_th):
                                for rr in range(0, ndof_th_hr):
                                    xsi_mtx[aa, rr] = qd.lag_eval(thb, aa, thb_hr[rr])
                            xsi_mtxs[(ndof_th, ndof_th_hr)] = xsi_mtx
                            
                        # Calculate error
                        u_cell  = anl_sol(xxf_hr, yyf_hr, thf_hr)
                        uh_cell = num_sol.cols[col_key].cells[cell_key].vals
                        uh_hr_cell = np.zeros([ndof_x_hr, ndof_y_hr, ndof_th_hr])
                        for pp in range(0, ndof_x_hr):
                            for qq in range(0, ndof_y_hr):
                                for rr in range(0, ndof_th_hr):
                                    for ii in range(0, ndof_x):
                                        phi_ip = phi_mtx[ii, pp]
                                        for jj in range(0, ndof_y):
                                            psi_jq = psi_mtx[jj, qq]
                                            for aa in range(0, ndof_th):
                                                xsi_ar = xsi_mtx[aa, rr]
                                                
                                                uh_hr_cell[pp, qq, rr] += uh_cell[ii, jj, aa] * phi_ip * psi_jq * xsi_ar
                    
                        err += (dx * dy * dth / 8.) * np.sum(wx_hr * wy_hr * wth_hr * (u_cell - uh_hr_cell)**2)
                    
        err = np.sqrt(err/intg_u2[kwargs['key']])
    else:
        err = 0

    err = MPI_comm.bcast(err, root = 0)

    if kwargs['blocking']:
        MPI_comm.Barrier()

    return err
