"""
    Generates figures of the final mesh for each combination in each test.
"""

# Standard Library Imports
import argparse
import gc
import json
import os
import sys

# Third-Party Library Imports
import matplotlib        as mpl
import matplotlib.pyplot as plt
import numpy             as np
import petsc4py
import psutil

# Local Library Imports
sys.path.append('../src')
import dg.mesh       as ji_mesh
import dg.mesh.utils
import utils

def main():
    figs_dir = "figs"
    test_num = 3
    test_dir = "test_" + str(test_num) + "_figs"
    test_path = os.path.join(figs_dir, test_dir)

    strat_name = "hp-amr-spt"
    trial_num  = 25

    mesh_file_path = os.path.join(test_path, strat_name, 
                             "trial_" + str(trial_num), "mesh.json")
    mesh = ji_mesh.read_mesh(mesh_file_path)

    [Lx, Ly] = mesh.Ls[:]
    plot_file_path = os.path.join(test_path, strat_name + "_mesh.png")
    ji_mesh.utils.plot_mesh_new(mesh,
                                [Lx * (2. / 8.), Lx * (4. / 8.)],
                                [Ly * (0. / 8.), Ly * (2. / 8.)],
                                plot_file_path)


if __name__ == '__main__':
    main()
