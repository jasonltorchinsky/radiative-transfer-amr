import numpy as np
from scipy.special import erf

def get_cons_funcs(func_num):

    # 0) Constant
    # 1) Linear
    # 2) Shallow gradient
    # 3) Steep gradient
    # 4*) Runge function

    if func_num == 0:
        def F(x):
            return x
        
        def f(x):
            return np.ones_like(x)

        def dfdx(x):
            return np.zeros_like(x)

    elif func_num == 1:
        def F(x):
            return (1. / 6.) * x**2 + 0.3 * x
        
        def f(x):
            return (1. / 3.) * x + 0.3

        def dfdx(x):
            return (1. / 3.)

    elif func_num == 2:
        def F(x):
            return np.sqrt(np.pi) * erf((1. / 2.) * (x - (2. / 3.)))
        
        def f(x):
            return np.exp(-((1. / 2.) * (x - (2. / 3.)))**2)

        def dfdx(x):
            return -(1. / 2.) * (x - (2. / 3.)) * f(x)

    elif func_num == 3:
        def F(x):
            return (1. / 16.) * np.sqrt(np.pi) * erf(8. * (x - (1. / 3.)))
        
        def f(x):
            return np.exp(-(8. * (x - (1. / 3.)))**2)

        def dfdx(x):
            return -128. * (x - (1. / 3.)) * f(x)

    elif func_num == 4:
        def F(x):
            return (1. / 5.) * (np.arctan(5.) + np.arctan(5. * x))
        
        def f(x):
            return 1. / (1. + 25. * x**2)

        def dfdx(x):
            return -50. / (1. + 25. * x**2)**2

    elif func_num == 5:
        def F(x):
            return (1. / (12. * np.pi**2)) * (6. * np.pi * x + np.sin(2. * np.pi * x))

        def f(x):
            return (1. / (3. * np.pi)) * (1. + (np.cos(np.pi * (x + 1.)))**2)

        def dfdx(x):
            return -(2. / 3.) * np.cos(np.pi * (x + 1.)) * np.sin(np.pi * (x + 1.))

    return [F, f, dfdx]
