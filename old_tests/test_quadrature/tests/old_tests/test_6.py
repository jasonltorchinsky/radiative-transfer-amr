import numpy as np
import matplotlib.pyplot as plt
import os, sys

sys.path.append("../../src")
import dg.quadrature as qd

def test_6(func, Func,
           quad_type = "lg",
           dir_name = "test_quad"):
    """
    Tests the integration of an analytic function f onto the Legendre-Gauss/
    Legendre-Gauss-Lobatto basis.
    """

    dir_name = os.path.join(dir_name, "test_6")
    os.makedirs(dir_name, exist_ok = True)

    nnodes = 2**np.arange(2, 7, 1)
    quad_nnode = 2 * np.amax(nnodes)

    if quad_type == "lg":
        [quad_nodes, quad_weights] = qd.lg_quad(quad_nnode)
        quad_type_str = "Legendre-Gauss"

    elif quad_type == "lgl":
        [quad_nodes, quad_weights] = qd.lgl_quad(quad_nnode)
        quad_type_str = "Legendre-Gauss-Lobatto"
    else:
        print("ERROR: Test 6 recieved invalid quad_type. Please use "lg" or "lgl".")
        quit()
        
    for nnode in nnodes:
        if quad_type == "lg":
            [nodes, weights] = qd.lg_quad(nnode)
        elif quad_type == "lgl":
            [nodes, weights] = qd.lgl_quad(nnode)
        else:
            print("ERROR: Test 6 recieved invalid quad_type. Please use "lg" or "lgl".")
            quit()        

        # Define 
        def Func_apprx(x):
            if x == -1.:
                return 0
            else:
                dx = (x + 1.) / 2.
                val = 0
                for ii in range(0, nnode):
                    zz = nodes[ii]
                    val += weights[ii] * func(dx * (zz + 1.) - 1.)
                val /= dx
                
                return val
    
        # Calculate F throughout the interval
        nx = 1000
        xx = np.linspace(-1, 1, nx)
        F_anl = Func(xx) - Func(-1)
        F_apr = np.zeros(nx)
        for ii in range(0, nx):
            F_apr[ii] = Func_apprx(xx[ii])
        
        
        # Plot all functions for comparison
        fig, ax = plt.subplots()
        ax.plot(xx, F_anl,  label = "Analytic",
                color = "k", linestyle = "-")
        ax.plot(xx, F_apr, label = "Approximation",
                color = "b", linestyle = "-")

        ax.legend()
        title_str = ("1-D Function Anti-differentiation on\n"
                     + "{} Nodal Bases\n").format(quad_type_str)
        ax.set_title(title_str)
        
        file_name = "{}_integration_{:03d}.png".format(quad_type, nnode)
        fig.set_size_inches(6.5, 6.5)
        plt.savefig(os.path.join(dir_name, file_name), dpi = 300)
        plt.close(fig)
