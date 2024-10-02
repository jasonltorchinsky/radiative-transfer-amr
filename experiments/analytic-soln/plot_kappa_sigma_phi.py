import os, sys
src_dir: str = os.path.normpath(os.path.join(os.path.dirname(__file__),
                                             os.pardir, os.pardir, "src"))

if src_dir not in sys.path:
    sys.path.append(src_dir)

# Standard Library Imports
import argparse

# Third-Party Library Imports
import matplotlib        as mpl
import matplotlib.pyplot as plt
import numpy             as np

# Local Library Imports

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
    
    args = parser.parse_args()

    if (args.o != "figs"):
        out_dir_path: str = os.path.normpath(args.o[0])
    else:
        out_dir_path: str = args.o
    
    figs_dir_name: str = "figs"
    figs_dir: str = os.path.join(out_dir_path, figs_dir_name)
    os.makedirs(figs_dir, exist_ok = True)

    # Mesh parameters
    [Lx, Ly] = params.mesh_params["Ls"]

    # Coefficients, forcing, and boundary conditions for this experiment
    kappa     = params.kappa
    sigma     = params.sigma
    Phi       = params.Phi

    # Plot extinction coefficient, scattering coefficient, and scattering phase function
    kappa_file_name: str = "kappa.png"
    sigma_file_name: str = "sigma.png"
    Phi_file_name: str   = "Phi.png"
    gen_kappa_sigma_plots([Lx, Ly], kappa, sigma, figs_dir,
                          [kappa_file_name, sigma_file_name])
    gen_Phi_plot(Phi, figs_dir, Phi_file_name)

def gen_kappa_sigma_plots(Ls, kappa, sigma, figs_dir, file_names):
    [Lx, Ly] = Ls[:]
    
    xx = np.linspace(0, Lx, num = 1000).reshape([1, 1000])
    yy = np.linspace(0, Ly, num = 1000).reshape([1000, 1])
    [XX, YY] = np.meshgrid(xx, yy)
    
    kappa_c = kappa(xx, yy)
    sigma_c = sigma(xx, yy)
    [vmin, vmax] = [0., max(np.amax(kappa_c), np.amax(sigma_c))]
    
    cmap = mpl.cm.gray
    norm = mpl.colors.Normalize(vmin = vmin, vmax = vmax)
    
    # kappa Plot
    file_path = os.path.join(figs_dir, file_names[0])
    if not os.path.isfile(file_path):
        fig, ax = plt.subplots()
        
        kappa_plot = ax.contourf(XX, YY, kappa_c, levels = 16,
                                 cmap = cmap, norm = norm)
        
        ax.set_xlim([0, Lx])
        ax.set_ylim([0, Ly])
        
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(r"$\kappa\left( x,\ y \right)$")
        
        fig.colorbar(mpl.cm.ScalarMappable(norm = norm, cmap = cmap), ax = ax)
        
        file_path = os.path.join(figs_dir, file_names[0])
        plt.tight_layout()
        plt.savefig(file_path, dpi = 300)
        
        plt.close(fig)

    # sigma Plot
    file_path = os.path.join(figs_dir, file_names[1])
    if not os.path.isfile(file_path):
        fig, ax = plt.subplots()
        
        kappa_plot = ax.contourf(XX, YY, sigma_c, levels = 16,
                                 cmap = cmap, norm = norm)
        
        ax.set_xlim([0, Lx])
        ax.set_ylim([0, Ly])
        
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(r"$\sigma\left( x,\ y \right)$")
        
        fig.colorbar(mpl.cm.ScalarMappable(norm = norm, cmap = cmap), ax = ax)
        
        plt.tight_layout()
        plt.savefig(file_path, dpi = 300)
        
        plt.close(fig)
    
def gen_Phi_plot(Phi, figs_dir, file_name):
    
    file_path = os.path.join(figs_dir, file_name)
    if not os.path.isfile(file_path):
        th = np.linspace(0, 2. * np.pi, num = 720)
        rr = Phi(0, th)
        
        max_r = np.amax(rr)
        ntick = 2
        r_ticks = np.linspace(max_r / ntick, max_r, ntick)
        r_tick_labels = ["{:3.2f}".format(r_tick) for r_tick in r_ticks]
        th_ticks = np.linspace(0, 2. * np.pi, num = 8, endpoint = False)
        th_tick_labels = [r"${:3.2f} \pi$".format(th_tick/np.pi)
                          for th_tick in th_ticks]
        
        fig, ax = plt.subplots(subplot_kw = {"projection": "polar"})
        
        Phi_plot = ax.plot(th, rr, color = "black")
        
        ax.set_rlim([0, max_r])
        ax.set_rticks(r_ticks, r_tick_labels)
        ax.set_xlabel(r"$\theta - \theta"$")
        ax.set_xticks(th_ticks, th_tick_labels)
        ax.set_title(r"$\Phi\left( \theta - \theta" \right)$")
        
        plt.tight_layout()
        plt.savefig(file_path, dpi = 300)
        
        plt.close(fig)

if __name__ == "__main__":
    main(sys.argv[1:])