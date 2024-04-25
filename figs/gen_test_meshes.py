"""
    Generates figures of the final mesh for each combination in each test.
"""

# Standard Library Imports
import os
import sys

# Third-Party Library Imports


# Local Library Imports
sys.path.append('../src')
import dg.mesh       as ji_mesh
import dg.mesh.utils

def main():
    figs_dir = "figs"
    test_num = 2
    test_dir = "test_" + str(test_num) + "_figs"
    test_path = os.path.join(figs_dir, test_dir)

    strat_name = "h-amr-ang"
    trial_num  = 80

    mesh_file_path = os.path.join(test_path, strat_name, 
                             "trial_" + str(trial_num), "mesh.json")
    mesh = ji_mesh.read_mesh(mesh_file_path)

    [Lx, Ly] = mesh.Ls[:]
    plot_file_path = os.path.join(test_path, strat_name + "_mesh_full.png")
    ji_mesh.utils.plot_mesh_new(mesh,
                                [Lx * (0. / 8.), Lx * (8. / 8.)],
                                [Ly * (0. / 8.), Ly * (8. / 8.)],
                                plot_file_path)


if __name__ == '__main__':
    main()
