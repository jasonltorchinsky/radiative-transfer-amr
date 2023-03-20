import numpy as np

def func_2D_0(x, y):

    return np.exp(-(x**2 + y**2))

def func_2D_1(x, y):

    return np.exp(np.sin(np.pi * x) + np.cos(np.pi * y))

def func_3D_0(x, y, th):

    return func_2D_0(x, y) * np.abs(np.cos(th))

def func_3D_1(x, y, th):

    return func_2D_1(x, y) * np.abs(np.cos(th))
