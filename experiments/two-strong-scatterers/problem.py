# Standard Library Imports
import json

# Third-Party Library Imports
import numpy as np
from scipy.integrate import quad

# Local Library Imports
import consts
from rt import Problem

# Relative Imports

## Read input files - hardcoded file names
input_file = open("input.json")
input_dict: dict = json.load(input_file)
input_file.close()

mesh_params: dict = input_dict["mesh_params"]

## Setup the Problem
[Lx, Ly] = mesh_params["Ls"]

def kappa(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    r1: np.ndarray = (Ly / 5.) - np.sqrt((x - (9. * Lx / 20.))**2 + (y - (2. * Ly / 5.))**2)
    kappa1: np.ndarray = (100.) / (1. + np.exp(-30. * r1))

    r2: np.ndarray = (Ly / 7.) - np.sqrt((x - (4. * Lx / 5.))**2 + (y - (Ly / 4.))**2)
    kappa2: np.ndarray = (100.) / (1. + np.exp(-30. * r2))

    return kappa1 + kappa2

def sigma(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return 0.7 * kappa(x, y)

g: float = 0.8
def Phi_HG(Th: np.ndarray) -> np.ndarray:
    return (1. - g**2) / (1 + g**2 - 2. * g * np.cos(Th))**(3./2.)
[Phi_norm, abserr] = quad(lambda Th : Phi_HG(Th), 0., 2. * consts.PI,
                          epsabs = 1.e-9, epsrel = 1.e-9,
                          limit = 100, maxp1 = 100)
def Phi(th: np.ndarray, phi: np.ndarray) -> np.ndarray:
    return ((1. - g**2) / Phi_norm) / (1 + g**2 - 2. * g * np.cos(th - phi))**(3./2.)

def f(x: np.ndarray, y: np.ndarray, th: np.ndarray) -> np.ndarray:
    return 0.
        
x_left: float = 0.
y_top: float = Ly
def bcs(x: np.ndarray, y: np.ndarray, th: np.ndarray) -> np.ndarray:
    sth: float = 48.
    if (y == y_top) or (x == x_left):
        return np.exp(-((sth / (2. * consts.PI)) * (th - (7. * consts.PI / 4.)))**2)
    else:
        return 0
dirac: list = [None, None, None]
bcs_dirac: list = [bcs, dirac]

problem: Problem = Problem(kappa, sigma, Phi, f, bcs_dirac)