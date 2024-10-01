# Standard Library Imports
import json, time

# Third-Party Library Imports
import numpy as np
from   scipy.integrate import quad, dblquad

# Local Library Imports

## Read input files - hardcoded file names
input_file = open("input.json")
input_dict: dict = json.load(input_file)
input_file.close()

if (input_dict["seed"] == 0):
    seed: int = int(1.e8 * (time.time() % 1))
else:
    seed: int = input_dict["seed"]
stopping_conditions: dict = input_dict["stopping_conditions"]
mesh_params: dict = input_dict["mesh_params"]
amr_params: dict = input_dict["amr_params"]
output_options: dict = input_dict["output_options"]

## Set the kwargs for each combination
# h-Uniform Angular
h_uni_ang_file = open("h-uni-ang.json")
h_uni_ang: dict = json.load(h_uni_ang_file)
h_uni_ang_file.close()

h_uni_ang: dict = {**h_uni_ang, **mesh_params}

# p-Uniform Angular
p_uni_ang_file = open("p-uni-ang.json")
p_uni_ang: dict = json.load(p_uni_ang_file)
p_uni_ang_file.close()

p_uni_ang: dict = {**p_uni_ang, **mesh_params}

# h-Adaptive Angular
h_amr_ang_file = open("h-amr-ang.json")
h_amr_ang: dict = json.load(h_amr_ang_file)
h_amr_ang_file.close()

h_amr_ang: dict = {**h_amr_ang, **mesh_params, **amr_params}
h_amr_ang["kwargs_ang_nneg"]["cell_ref_form"] = "h"
h_amr_ang["kwargs_ang_jmp"]["cell_ref_form"] = "h"

# hp-Adaptive Angular
hp_amr_ang_file = open("hp-amr-ang.json")
hp_amr_ang: dict = json.load(hp_amr_ang_file)
hp_amr_ang_file.close()

hp_amr_ang: dict = {**hp_amr_ang, **mesh_params, **amr_params}
hp_amr_ang["kwargs_ang_nneg"]["cell_ref_form"] = "hp"
hp_amr_ang["kwargs_ang_jmp"]["cell_ref_form"] = "hp"
        
ref_strats: list = [h_uni_ang, p_uni_ang, h_amr_ang, hp_amr_ang]

## Set the manufactured solution
[Lx, Ly] = mesh_params["Ls"]

def X(x):
    return np.exp(-((1. / Lx) * (x - (Lx / 3.)))**2)
def dXdx(x):
    return -(2. / Lx**2) * (x - (Lx / 3.)) * X(x)
def Y(y):
    return np.exp(-4. * (Ly - y) / Ly)
def dYdy(y):
    return (4. / Ly) * Y(y)
def XY(x, y):
    return X(x) * Y(y)
sth = 96.
def Theta(th):
    return np.exp(-((sth / (2. * np.pi)) * (th - (7. * np.pi / 5.)))**2)
def u(x, y, th):
    return XY(x, y) * Theta(th)
        
def kappa_x(x):
    return np.exp(-((1. / Lx) * (x - (Lx / 2.)))**2)
def kappa_y(y):
    return np.exp(-y / Ly)
def kappa(x, y):
    return 10. * kappa_x(x) * kappa_y(y)

def sigma(x, y):
    return 0.1 * kappa(x, y)

def Phi(th, phi):
    val = (1. / (3. * np.pi)) * (1. + (np.cos(th - phi))**2)
    return val

def f(x, y, th):
    # Propagation part
    prop = (np.cos(th) * dXdx(x) * Y(y) + np.sin(th) * X(x) * dYdy(y)) * Theta(th)
    # Extinction part
    extn = kappa(x, y) * u(x, y, th)
    # Scattering part
    [Theta_scat, _] = quad(lambda phi: Phi(th, phi) * Theta(phi), 0., 2. * np.pi,
                           epsabs = 1.e-9, epsrel = 1.e-9, limit = 100, maxp1 = 100)
    scat =  sigma(x, y) * XY(x, y) * Theta_scat
    return prop + extn - scat
        
def bcs(x, y, th):
    return u(x, y, th)
dirac = [None, None, None]
bcs_dirac = [bcs, dirac]
        
def u_intg_th(x, y, th0, th1):
    [Theta_intg, _] = quad(lambda th: Theta(th), th0, th1,
                           epsabs = 1.e-9, epsrel = 1.e-9,
                           limit = 100, maxp1 = 100)
    return XY(x, y) * Theta_intg
        
def u_intg_xy(x0, x1, y0, y1, th):
    [XY_intg, _] = dblquad(lambda x, y: XY(x, y), x0, x1, y0, y1,
                           epsabs = 1.e-9, epsrel = 1.e-9)
    return XY_intg * Theta(th)