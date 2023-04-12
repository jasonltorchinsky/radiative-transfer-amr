import numpy as np
from scipy.special import erf

def get_test_funcs(func_num, mesh):

    [Lx, Ly] = mesh.Ls[:]
    erfLx = erf(Lx/2.)
    erfLy = erf(Ly/2.)
    erfLxLy2 = erf(Lx * Ly**2 / 2.)
    erfLx2Ly = erf(Lx**2 * Ly / 2.)
    
    if func_num == 0:
        # Smooth-ish in space and angle
        
        def func_1d_x(x):
            return np.sqrt(np.pi) * erfLy * np.exp(-(x - Lx/2.)**2)

        def func_1d_y(y):
            return np.sqrt(np.pi) * erfLx * np.exp(-(y - Ly/2.)**2)
        
        def func_2d(x, y):
            return np.exp(-((x - Lx/2.)**2 + (y - Ly/2.)**2))

        def func_3d(x, y, th):
            return func_2d(x, y) * (np.cos(th))**2 / np.pi

    elif func_num == 1:
        # Steep in space, smooth-ish in angle

        def func_1d_x(x):
            return np.sqrt(np.pi) * erfLxLy2 / (Lx * Ly) \
                * np.exp(-(Lx * Ly)**2 * (x - Lx/2.)**2)

        def func_1d_y(y):
            return np.sqrt(np.pi) * erfLx2Ly / (Lx * Ly) \
                * np.exp(-(Lx * Ly)**2 * (y - Ly/2.)**2)
        
        def func_2d(x, y):
            return np.exp(-(Lx * Ly)**2 * ((x - Lx/2.)**2 + (y - Ly/2.)**2))
        
        def func_3d(x, y, th):
            return func_2d(x, y) * (np.cos(th))**2 / np.pi

    elif func_num == 2:
        # Smooth-ish in space, steep in angle

        def func_1d_x(x):
            return np.sqrt(np.pi) * erfLy * np.exp(-(x - Lx/2.)**2)

        def func_1d_y(y):
            return np.sqrt(np.pi) * erfLx * np.exp(-(y - Ly/2.)**2)
        
        def func_2d(x, y):
            return np.exp(-((x - Lx/2.)**2 + (y - Ly/2.)**2))

        def func_3d(x, y, th):
            return func_2d(x, y) * (np.sin(8. * th))**4 * (4. / (3. * np.pi))

    return [func_1d_x, func_1d_y, func_2d, func_3d]
