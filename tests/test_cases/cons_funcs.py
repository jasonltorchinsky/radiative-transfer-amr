import numpy as np

def get_cons_funcs(func_num):

    if func_num == 0:
        # Runge Function
        
        def F(x):
            return (1. / 5.) * (np.arctan(5.) + np.arctan(5. * x))
        
        def f(x):
            return 1. / (1. + 25. * x**2)

        def dfdx(x):
            return -50. / (1. + 25. * x**2)**2

    return [F, f, dfdx]
