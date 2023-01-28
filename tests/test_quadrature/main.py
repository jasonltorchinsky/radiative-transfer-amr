import argparse
from datetime import datetime
import numpy as np
from time import perf_counter
import os

from tests import test_0, test_1, test_2, test_3, test_4, test_5

def main():

    parser_desc = 'Determine which tests to run and where to put output.'
    parser = argparse.ArgumentParser(description = parser_desc)
    parser.add_argument('--dir', nargs = 1, default = 'test_quad',
                        required = False, help = 'Subdirectory to store output')
    parser.add_argument('--test_all', nargs = 1, default = [0],
                        type = int, choices = [0, 1], required = False,
                        help = 'Do not run (0) or run (1) all tests (overrides other flags)')
    parser.add_argument('--test_0', nargs = 1, default = [0],
                        type = int, choices = [0, 1], required = False,
                        help = 'Do not run (0) or run (1) Test 0 - LG/LGL Node Placement')
    parser.add_argument('--test_1', nargs = 1, default = [0],
                        type = int, choices = [0, 1], required = False,
                        help = 'Do not run (0) or run (1) Test 1 - LG/LGL 1D Function Projection Between Bases')
    parser.add_argument('--test_2', nargs = 1, default = [0],
                        type = int, choices = [0, 1], required = False,
                        help = 'Do not run (0) or run (1) Test 2 - LG/LGL 1D Function Projection Comparison')
    parser.add_argument('--test_3', nargs = 1, default = [0],
                        type = int, choices = [0, 1], required = False,
                        help = 'Do not run (0) or run (1) Test 3 - LG/LGL 1D Function Projection Accuracy')
    parser.add_argument('--test_4', nargs = 1, default = [0],
                        type = int, choices = [0, 1], required = False,
                        help = 'Do not run (0) or run (1) Test 4 - LG/LGL 1D Derivative Function Projection Comparison')
    parser.add_argument('--test_5', nargs = 1, default = [0],
                        type = int, choices = [0, 1], required = False,
                        help = 'Do not run (0) or run (1) Test 5 - LG/LGL 1D Function Derivative Projection Accuracy')

    args = parser.parse_args()
    ntests = 6
    if args.test_all[0]:
        run_tests = [True] * ntests
    else:
        run_tests = [args.test_0[0], args.test_1[0],
                     args.test_2[0], args.test_3[0],
                     args.test_4[0], args.test_5[0]]

    dir_name = args.dir
    os.makedirs(dir_name, exist_ok = True)

    if run_tests[0]:
        perf_0 = perf_counter()
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        msg = '[{}]: Starting Test 0...'.format(current_time)
        print(msg)

        test_0(nnodes = 5, quad_type = 'lg', dir_name = dir_name)
        test_0(nnodes = 5, quad_type = 'lgl', dir_name = dir_name)

        perf_f = perf_counter()
        perf_diff = perf_f - perf_0
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        msg = ('[{}]: Completed Test 0! ' +
               'Time Elapsed: {:06.3f} [s]').format(current_time, perf_diff)
        print(msg)
        
    if run_tests[1]:
        perf_0 = perf_counter()
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        msg = '[{}]: Starting Test 1...'.format(current_time)
        print(msg)
        
        test_1(func = f, src_nnodes = 41, trgt_nnodes = 10,
               quad_type = 'lg', dir_name = dir_name)
        test_1(func = f, src_nnodes = 10, trgt_nnodes = 41,
               quad_type = 'lg', dir_name = dir_name)

        test_1(func = f, src_nnodes = 41, trgt_nnodes = 10,
               quad_type = 'lgl', dir_name = dir_name)
        test_1(func = f, src_nnodes = 10, trgt_nnodes = 41,
               quad_type = 'lgl', dir_name = dir_name)

        perf_f = perf_counter()
        perf_diff = perf_f - perf_0
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        msg = ('[{}]: Completed Test 1! ' +
               'Time Elapsed: {:06.3f} [s]').format(current_time, perf_diff)
        print(msg)

    if run_tests[2]:
        perf_0 = perf_counter()
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        msg = '[{}]: Starting Test 2...'.format(current_time)
        print(msg)
        
        test_2(func = f, quad_type = 'lg', dir_name = dir_name)
        test_2(func = f, quad_type = 'lgl', dir_name = dir_name)

        perf_f = perf_counter()
        perf_diff = perf_f - perf_0
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        msg = ('[{}]: Completed Test 2! ' +
               'Time Elapsed: {:06.3f} [s]').format(current_time, perf_diff)
        print(msg)

    if run_tests[3]:
        perf_0 = perf_counter()
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        msg = '[{}]: Starting Test 3...'.format(current_time)
        print(msg)
        
        test_3(func = f, quad_type = 'lg', dir_name = dir_name)
        test_3(func = f, quad_type = 'lgl', dir_name = dir_name)

        perf_f = perf_counter()
        perf_diff = perf_f - perf_0
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        msg = ('[{}]: Completed Test 3! ' +
               'Time Elapsed: {:06.3f} [s]').format(current_time, perf_diff)
        print(msg)

    if run_tests[4]:
        perf_0 = perf_counter()
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        msg = '[{}]: Starting Test 4...'.format(current_time)
        print(msg)
        
        test_4(func = f, func_ddx = dfdx, quad_type = 'lg', dir_name = dir_name)
        #test_4(func = f, func_ddx = dfdx, quad_type = 'lgl', dir_name = dir_name)

        perf_f = perf_counter()
        perf_diff = perf_f - perf_0
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        msg = ('[{}]: Completed Test 4! ' +
               'Time Elapsed: {:06.3f} [s]').format(current_time, perf_diff)
        print(msg)

    if run_tests[5]:
        perf_0 = perf_counter()
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        msg = '[{}]: Starting Test 5...'.format(current_time)
        print(msg)
        
        test_5(func = f, func_ddx = dfdx, quad_type = 'lg', dir_name = dir_name)
        #test_5(func = f, func_ddx = dfdx, quad_type = 'lgl', dir_name = dir_name)

        perf_f = perf_counter()
        perf_diff = perf_f - perf_0
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        msg = ('[{}]: Completed Test 5! ' +
               'Time Elapsed: {:06.3f} [s]').format(current_time, perf_diff)
        print(msg)
"""

    # Test rate of convergence w.r.t. order of approximation
    #test_3(norders = 5, nx = 1024, dir_name = dir_name)
    
    # Test accuracy of differentiation
    #test_4(norders = 5, nx = 1024, dir_name = dir_name)
    
    # Test accuracy of integration
    #test_5(norders = 5, intv = [-np.pi / 3, np.pi], dir_name = dir_name)

    # Test 2D function projection.
    #test_6(order_x = 6,  nx = 41,
    #       order_y = 11, ny = 81,
    #       dir_name = dir_name)
    #test_6(order_x = 12, nx = 71,
    #       order_y = 17, ny = 121,
    #       dir_name = dir_name)

    #[tab_f2f, tablist_upd] = qd.face_2_face([], 1, 2)
    #[tab_f2df, tab_df2f, tablist_upd] = qd.face_2_dface([], 1, 2)
"""
    
def f(x):
    """
    Test function for approximation.
    """

    return (x + 0.25)**2 * np.sin(2 * np.pi * x)

    #return (x + 0.25)**6 * np.sin(18 * np.pi * x)

def dfdx(x):
    '''
    Test function for differentiation.
    '''

    return 2 * (x + 0.25) * (np.sin(2 * np.pi * x)
                             + np.pi * (x + 0.25) * np.cos(2 * np.pi * x))

    #return 6 * (x + 0.25)**5 * (np.sin(18 * np.pi * x)
    #                            + 3 * np.pi * (x + 0.25) * np.cos(18 * np.pi * x))

def dg(x):
    '''
    Analytic derivative of test function g.
    '''

    return np.pi * np.cos(np.pi * x)

def G(x):
    '''
    Analytic antiderivatiuve of test function g
    '''

    return -(1.0 / np.pi) * np.cos(np.pi * x)

def f_2D(x, y):
    '''
    Test function for 2D approximation.
    '''

    res = (3 * x + 0.25 * (y - 0.25))**4 * np.sin(np.pi * x) * np.sin(np.pi * y)
    
    return res


"""
def test_3(norders, nx, dir_name):
    '''
    Plot the order of convergence w.r.t. order.
    '''

    errors_1 = np.zeros([norders])
    errors_2 = np.zeros([norders])
    errors_max = np.zeros([norders])
    
    orders = 2 ** np.arange(1, norders + 1)
    
    for order_idx in range(0, norders):
        order = orders[order_idx]
        
        [nodes, weights, vand] = qd.lgl_quad(order)
        
        # Calculate true function values
        xx = np.linspace(-1, 1, nx)
        f_true = f(xx)
        
        # Reconstruct f in GL basis
        f_aprx = np.zeros([nx])
        d = f(xx)
        [A, Adag] = qd.gl_proj(xx, nodes, True)
        k = Adag @ d
        for i in range(0, order + 1):
            for x_idx in range(0, nx):
                f_aprx[x_idx] += k[i] * qd.gl_eval(nodes, i, xx[x_idx])   

        for x_idx in range(0, nx):
            errors_1[order_idx] += np.abs(f_true[x_idx] - f_aprx[x_idx])
            errors_2[order_idx] += (f_true[x_idx] - f_aprx[x_idx])**2

        errors_2[order_idx] = np.sqrt(errors_2[order_idx])
        errors_max[order_idx] = np.amax(np.abs(f_true - f_aprx))

    # Get best-fit lines
    [m_1, b_1] = np.polyfit(orders, np.log(errors_1), 1)
    [m_2, b_2] = np.polyfit(orders, np.log(errors_2), 1)
    [m_max, b_max] = np.polyfit(orders, np.log(errors_max), 1)
            
    # Plot errors
    fig, ax = plt.subplots()
    ax.plot(orders, errors_1,
            label = '1-Norm: Order {:4.2f}'.format(np.abs(m_1)),
            linestyle = 'None',
            color = 'r',
            marker = 'o')
    ax.plot(orders, np.exp(m_1 * orders + b_1), label = None,
            linestyle = '--',
            color = 'r',
            marker = 'None')
    
    ax.plot(orders, errors_2,
            label = '2-Norm: Order {:4.2f}'.format(np.abs(m_2)),
            linestyle = 'None',
            color = 'b',
            marker = '^')
    ax.plot(orders, np.exp(m_2 * orders + b_2), label = None,
            linestyle = '--',
            color = 'b',
            marker = 'None')
    
    ax.plot(orders, errors_max,
            label = 'Max-Norm: Order {:4.2f}'.format(np.abs(m_max)),
            linestyle = 'None',
            color = 'g',
            marker = 's')
    ax.plot(orders, np.exp(m_max * orders + b_max), label = None,
            linestyle = '--',
            color = 'g',
            marker = 'None')
    
    ax.legend()

    ax.set_yscale('Log')

    ax.set_xlabel('Order of Approximation')
    ax.set_ylabel('Error')

    ax.set_xticks(orders)

    file_name = 'conv_test_{}.png'.format(nx)
    fig.set_size_inches(6.5, 6.5)
    plt.savefig(os.path.join(dir_name, file_name), dpi = 300)
    plt.close(fig)

def test_4(norders, nx, dir_name):
    '''
    Plot the order of convergence w.r.t. order for derivative
    '''

    errors_1 = np.zeros([norders])
    errors_2 = np.zeros([norders])
    errors_max = np.zeros([norders])
    
    orders = 2 ** np.arange(1, norders + 1)
    
    for order_idx in range(0, norders):
        order = orders[order_idx]
        
        [nodes, weights, vand] = qd.lgl_quad(order)
        
        # Calculate true function values
        g_true = g(nodes)
        dg_true = dg(nodes)
        
        # Calculate derivate of g
        ddx = qd.gl_ddx(nodes)
        dg_aprx =  ddx @ g_true
        for i in range(0, order + 1):
            errors_1[order_idx] += np.abs(dg_true[i] - dg_aprx[i])
            errors_2[order_idx] += (dg_true[i] - dg_aprx[i])**2

        errors_2[order_idx] = np.sqrt(errors_2[order_idx])
        errors_max[order_idx] = np.amax(np.abs(dg_true - dg_aprx))

    # Get best-fit lines
    [m_1, b_1] = np.polyfit(orders, np.log(errors_1), 1)
    [m_2, b_2] = np.polyfit(orders, np.log(errors_2), 1)
    [m_max, b_max] = np.polyfit(orders, np.log(errors_max), 1)
            
    # Plot errors
    fig, ax = plt.subplots()
    ax.plot(orders, errors_1,
            label = '1-Norm: Order {:4.2f}'.format(np.abs(m_1)),
            linestyle = 'None',
            color = 'r',
            marker = 'o')
    ax.plot(orders, np.exp(m_1 * orders + b_1), label = None,
            linestyle = '--',
            color = 'r',
            marker = 'None')
    
    ax.plot(orders, errors_2,
            label = '2-Norm: Order {:4.2f}'.format(np.abs(m_2)),
            linestyle = 'None',
            color = 'b',
            marker = '^')
    ax.plot(orders, np.exp(m_2 * orders + b_2), label = None,
            linestyle = '--',
            color = 'b',
            marker = 'None')
    
    ax.plot(orders, errors_max,
            label = 'Max-Norm: Order {:4.2f}'.format(np.abs(m_max)),
            linestyle = 'None',
            color = 'g',
            marker = 's')
    ax.plot(orders, np.exp(m_max * orders + b_max), label = None,
            linestyle = '--',
            color = 'g',
            marker = 'None')
    
    ax.legend()

    ax.set_yscale('Log')

    ax.set_xlabel('Order of Approximation')
    ax.set_ylabel('Error')

    ax.set_xticks(orders)

    file_name = 'ddx_conv_test_{}.png'.format(nx)
    fig.set_size_inches(6.5, 6.5)
    plt.savefig(os.path.join(dir_name, file_name), dpi = 300)
    plt.close(fig)

def test_5(norders, intv, dir_name):
    '''
    Plot the order of convergence w.r.t. order integral
    '''

    # Calculate true integral values
    G_true = G(intv[1]) - G(intv[0])
    
    errors = np.zeros([norders])
    
    orders = 2 ** np.arange(1, norders + 1)
    
    for order_idx in range(0, norders):
        order = orders[order_idx]
        
        [nodes, weights] = qd.lg_quad(order, intv)

        # Calculate integral of g numerically
        g_true = g(nodes)
        G_aprx = g_true @ weights
        errors[order_idx] = np.abs(G_true - G_aprx)


    # Get best-fit lines
    [m, b] = np.polyfit(orders, np.log(errors), 1)
            
    # Plot errors
    fig, ax = plt.subplots()
    ax.plot(orders, errors,
            label = 'Order {:4.2f}'.format(np.abs(m)),
            linestyle = 'None',
            color = 'r',
            marker = 'o')
    ax.plot(orders, np.exp(m * orders + b), label = None,
            linestyle = '--',
            color = 'r',
            marker = 'None')
    
    ax.legend()

    ax.set_yscale('Log')

    ax.set_xlabel('Order of Approximation')
    ax.set_ylabel('Error')

    ax.set_xticks(orders)

    file_name = 'int_conv_test.png'
    fig.set_size_inches(6.5, 6.5)
    plt.savefig(os.path.join(dir_name, file_name), dpi = 300)
    plt.close(fig)

def test_6(order_x, nx, order_y, ny, dir_name):
    '''
    Test the projection of an analytic function onto a basis.
    '''

    [nodes_x, _, nodes_y, _, _, _] = qd.quad_xya(order_x, order_y, 1)

    # Calculate true function values
    xx = np.linspace(-1, 1, nx)
    yy = np.linspace(-1, 1, ny)
    f_true = np.zeros([nx, ny])
    for i in range(0, nx):
        for j in range(0, ny):
            f_true[i, j] = f_2D(xx[i], yy[j])

    # Reconstruct f in GL basis
    f_aprx = np.zeros([nx, ny])
    d = np.zeros([nx * ny])
    for i in range(0, nx*ny):
        x_idx = int(np.mod(i, nx))
        y_idx = int(np.floor(i / nx))
        d[i] = f_2D(xx[x_idx], yy[y_idx])
        
    [A, Adag] = qd.gl_proj_2D(xx, nodes_x, yy, nodes_y, True)
    k = Adag @ d
    for i in range(0, nx*ny):
        x_idx = int(np.mod(i, nx))
        y_idx = int(np.floor(i / nx))
        for j in range(0, order_x * order_y):
            ox_idx = int(np.mod(j, order_x))
            oy_idx = int(np.floor(j / order_x))
            f_aprx[x_idx, y_idx] += k[j] \
                * qd.gl_eval(nodes_x, ox_idx, xx[x_idx]) \
                * qd.gl_eval(nodes_y, oy_idx, yy[y_idx])
            
    # Plot approximation and truth
    fig, axs = plt.subplots(1, 2)
    axs[0].contourf(xx, yy, f_true.transpose())
    im = axs[1].contourf(xx, yy, f_aprx.transpose())
    fig.colorbar(im, ax = axs)

    axs[0].set_title('Truth: {} X {}'.format(nx, ny))
    axs[1].set_title('Truth: {} X {}'.format(order_x, order_y))
    
    for ax in axs.flat:
        ax.label_outer()

    file_name = 'proj_test_2d_{}_{}.png'.format(order_x, order_y)
    fig.set_size_inches(14, 6.5)
    plt.savefig(os.path.join(dir_name, file_name), dpi = 300)
    plt.close(fig)
"""
    
if __name__ == '__main__':

    main()
