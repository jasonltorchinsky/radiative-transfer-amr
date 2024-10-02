import numpy as np
import matplotlib.pyplot as plt
import os, sys

sys.path.append("../../tests")
from test_cases import get_cons_funcs

sys.path.append("../../src")
import dg.quadrature as qd

def test_3(quad_type = "lg", dir_name = "test_quad", **kwargs):
    """
    Tests convergence of the quadrature rule when integrating on two
    half-intervals.
    """

    default_kwargs = {}
    kwargs = {**default_kwargs, **kwargs}
    
    test_dir = os.path.join(dir_name, "test_3")
    os.makedirs(test_dir, exist_ok = True)

    if quad_type == "lg":
        quad_type_str = "Legendre-Gauss"
    elif quad_type == "lgr":
        quad_type_str = "Legendre-Gauss-Radau"
    elif quad_type == "lgl":
        quad_type_str = "Legendre-Gauss-Lobatto"
    elif quad_type == "uni":
        quad_type_str = "Uniform"

    max_ntrial = 8
    for func_num in range(0, 4):
        [F, f, _] = get_cons_funcs(func_num = func_num)
        
        nnodes = [0] * max_ntrial
        errs_0  = [0] * max_ntrial
        errs_l  = [0] * max_ntrial
        errs_r  = [0] * max_ntrial
        errs_lr = [0] * max_ntrial
        
        for trial in range(0, max_ntrial):
            nnode = 2**(trial + 1)
            nnodes[trial] = nnode
            
            if quad_type == "lg":
                [nodes, weights] = qd.lg_quad(nnode)
            elif quad_type == "lgr":
                [nodes, weights] = qd.lgr_quad(nnode)
            elif quad_type == "lgl":
                [nodes, weights] = qd.lgl_quad(nnode)
            elif quad_type == "uni":
                [nodes, weights] = qd.uni_quad(nnode)
            else:
                msg = (
                    "ERROR: Test 2 recieved invalid quad_type. " +
                    "Please use "lg", "lgr", "lgl", "uni"."
                )
                print(msg)
                quit()


            # Push nodes forward onto interval [-1, 3], then integrate
            # numerically on [-1, 1]
            nodes_0 = nodes # Full interval
            nodes_1l = 0.5 * (nodes - 1.) # Left half-interval
            nodes_1r = 0.5 * (nodes + 1.) # Right half-interval
            
            f_0 = f(nodes_0)
            nintg_0 = 0.
            nintg_l = 0.
            nintg_r = 0.
            
            for nn in range(0, nnode):
                nintg_0 += weights[nn] * f_0[nn]
                for nn_p in range(0, nnode):
                    nintg_l += 0.5 * weights[nn] * f_0[nn_p] * qd.lag_eval(nodes_0, nn_p, nodes_1l[nn])
                    nintg_r += 0.5 * weights[nn] * f_0[nn_p] * qd.lag_eval(nodes_0, nn_p, nodes_1r[nn])
                
            aintg_0 = F(1) - F(-1)
            aintg_l = F(0) - F(-1)
            aintg_r = F(1) - F(0)

            errs_0[trial]  = np.abs(aintg_0 - nintg_0)
            errs_l[trial]  = np.abs(aintg_l - nintg_l)
            errs_r[trial]  = np.abs(aintg_r - nintg_r)
            errs_lr[trial] = np.abs(aintg_0 - (nintg_l + nintg_r))
            
        # Plot errors
        fig, ax = plt.subplots()
        ax.plot(nnodes, errs_0,
                label = "0",
                color = "k",
                linestyle = "-")
        ax.plot(nnodes, errs_l,
                label = "l",
                color = "b",
                linestyle = "--")
        ax.plot(nnodes, errs_r,
                label = "r",
                color = "r",
                linestyle = "--")
        ax.plot(nnodes, errs_l,
                label = "lr",
                color = "k",
                linestyle = "--")
        
        ax.set_xscale("log", base = 2)
        ax.set_yscale("log", base = 10)
        
        ax.set_xlabel("Number of Nodes")
        ax.set_ylabel("Error")

        ax.legend()
        
        title_str = ("1-D Function Integration via\n"
                     + "{} Quadrature").format(quad_type_str)
        ax.set_title(title_str)
        
        file_name = "{}_intg_acc_{}.png".format(quad_type, func_num)
        fig.set_size_inches(6.5, 6.5)
        plt.savefig(os.path.join(test_dir, file_name), dpi = 300)
        plt.close(fig)
