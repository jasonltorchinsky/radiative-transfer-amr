import dg.quadrature as qd

import numpy as np
import matplotlib.pyplot as plt
import sys, getopt
import os

def main(argv):

    dir_name = 'test_quadrature'
    os.makedirs(dir_name, exist_ok = True)

    # Test LGL node placement, GL basis functions
    test_1(order = 5, dir_name = dir_name)

    # Test function projection
    #test_2(order = 5,  nx = 41, dir_name = dir_name)
    #test_2(order = 11, nx = 41, dir_name = dir_name)

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
    
    print('\n')
    
    
def f(x):
    '''
    Test function for approximation.
    '''

    res = (x + 0.25)**2 * np.sin(2 * np.pi * x)
    
    return res

def g(x):
    '''
    Test function for differentiation.
    '''
    return np.sin(np.pi * x)

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

def test_1(order, dir_name):
    '''
    Test LGL node placement, GL basis function evaluation.
    '''

    # Test LGL nodes, evaluating GL
    #[nodes, weights, _] = qd.lgl_quad(order)
    [nodes, weights] = qd.lg_quad(order + 1, [-1, 1])

    nx = 500
    xx = np.linspace(-1, 1, nx)
    gl_funcs = np.zeros([order + 1, nx])
    for i in range(0, order + 1):
        for x_idx in range(0, nx):
            gl_funcs[i, x_idx] = qd.gl_eval(nodes, i, xx[x_idx])

    fig, ax = plt.subplots()
    for i in range(0, order + 1):
        lbl = 'Basis Function {}'.format(i)
        ax.plot(xx, gl_funcs[i, :], label = lbl,
            linestyle = '-')
    ax.legend()

    file_name = 'lg_basis.png'
    fig.set_size_inches(6.5, 6.5)
    plt.savefig(os.path.join(dir_name, file_name), dpi = 300)
    plt.close(fig)


def test_2(order, nx, dir_name):
    '''
    Test the projection of an analytic function onto a basis.
    '''

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
        
            
    # Plot approximation and truth
    fig, ax = plt.subplots()
    ax.plot(xx, f_true, label = 'Truth',
            linestyle = '-',
            color = 'k',
            marker = '.')
    # When we plot the approximation, we only want markers at the LGL nodes
    ax.plot(xx, f_aprx, label = 'Approx',
            linestyle = '--',
            color = 'r',
            marker = 'None')
    ax.plot(nodes, k, label = None,
            linestyle = 'None',
            color = 'r',
            marker = '.')
    ax.legend()

    file_name = 'proj_test_{}.png'.format(order)
    fig.set_size_inches(6.5, 6.5)
    plt.savefig(os.path.join(dir_name, file_name), dpi = 300)
    plt.close(fig)

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
    
if __name__ == '__main__':

    main(sys.argv[1:])
