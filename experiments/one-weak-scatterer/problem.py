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

def r(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (Ly / 6.) - np.sqrt((x - (5. * Lx / 8.))**2 + (y - (3. * Ly / 8.))**2)
def kappa(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return 5.5 / (1. + np.exp(-15. * r(x, y)))
def sigma(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return 0.7 * kappa(x, y)

g: float = 0.7
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
    sth: float = 96. * 2.
    if (y == y_top) or (x == x_left):
        return np.exp(-((sth / (2. * consts.PI)) * (th - (7. * consts.PI / 4.)))**2)
    else:
        return 0
dirac: list = [None, None, None]
bcs_dirac: list = [bcs, dirac]

problem: Problem = Problem(kappa, sigma, Phi, f, bcs_dirac)