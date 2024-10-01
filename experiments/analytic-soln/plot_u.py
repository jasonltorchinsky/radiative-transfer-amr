import os, sys
src_dir: str = os.path.normpath(os.path.join(os.path.dirname(__file__),
                                             os.pardir, os.pardir, "src"))

if src_dir not in sys.path:
    sys.path.append(src_dir)

# Standard Library Imports
import argparse

# Third-Party Library Imports

# Local Library Imports
import dg.mesh       as ji_mesh
import dg.projection as proj
import dg.projection.utils as proj_utils

import params

def main(argv):
    ## Read command-line input
    parser_desc = ( "Runs the numerical experiment for the hp-adaptive DG" +
                    " method for radiative transfer." )
    parser = argparse.ArgumentParser(description = parser_desc)
    
    parser.add_argument("--o",
                        action = "store",
                        nargs = 1, 
                        type = str,
                        required = False,
                        default = "figs",
                        help = "Output directory path.")
    
    args    = parser.parse_args()

    if (args.o != "figs"):
        out_dir_path: str = os.path.normpath(args.o[0])
    else:
        out_dir_path: str = args.o
    
    figs_dir_name: str = "figs"
    figs_dir: str = os.path.join(out_dir_path, figs_dir_name)
    os.makedirs(figs_dir, exist_ok = True)

    # Mesh parameters
    [Lx, Ly] = params.mesh_params["Ls"]

    # Analtyic solution
    u = params.u

    # Plot analytic solution
    gen_u_plot([Lx, Ly], u, figs_dir)
    
def gen_u_plot(Ls, u, figs_dir):
    mesh = ji_mesh.Mesh(Ls     = Ls[:],
                        pbcs   = [False, False],
                        ndofs  = [8, 8, 8],
                        has_th = True)
    for _ in range(0, 4):
        mesh.ref_mesh(kind = 'ang', form = 'h')
    for _ in range(0, 4):
        mesh.ref_mesh(kind = 'spt', form = 'h')
        
    file_names: list = ['u_th.png', 'u_xy.png', 'u_xth.png', 'u_yth.png', 'u_xyth.png']
    file_paths: list = []
    is_file_paths = []
    for file_name in file_names:
        file_path      = os.path.join(figs_dir, file_name)
        file_paths    += [file_path]
        is_file_paths += [os.path.isfile(file_path)]
        
    if not all(is_file_paths):
        u_proj = proj.Projection(mesh, u)
        
        if not os.path.isfile(file_paths[0]):
            proj_utils.plot_th(mesh, u_proj, file_name = file_paths[0])
            
        if not os.path.isfile(file_paths[1]):
            proj_utils.plot_xy(mesh, u_proj, file_name = file_paths[1])
            
        if not os.path.isfile(file_paths[2]):
            proj_utils.plot_xth(mesh, u_proj, file_name = file_paths[2])
            
        if not os.path.isfile(file_paths[3]):
            proj_utils.plot_yth(mesh, u_proj, file_name = file_paths[3])
            
        if not os.path.isfile(file_paths[4]):
            proj_utils.plot_xyth(mesh, u_proj, file_name = file_paths[4])

if __name__ == "__main__":
    main(sys.argv[1:])