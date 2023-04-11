import numpy as np

def get_test_funcs(func_num = 0):

    if func_num == 0:
        # Smooth-ish in space and angle
        
        def func_2d(x, y):
            return np.exp(np.sin(np.pi * x) + np.cos(np.pi * y))

        def func_3d(x, y, th):
            return func_2d(x, y) * (np.cos(th))**2 / np.pi

    elif func_num == 1:
        # Steep in space, smooth-ish in angle
        
        def func_2d(x, y):
            return np.exp(np.sin(16. * np.pi * x) + np.cos(13. * np.pi * y))
        
        def func_3d(x, y, th):
            return func_2d(x, y) * (np.cos(th))**2 / np.pi

    elif func_num == 2:
        # Smooth-ish in space, steep in angle
        
        def func_2d(x, y):
            return np.exp(np.sin(np.pi * x) + np.cos(np.pi * y))

        def func_3d(x, y, th):
            return func_2d(x, y) * (np.sin(8. * th))**4 * (4. / (3. * np.pi))

    return [func_2d, func_3d]
