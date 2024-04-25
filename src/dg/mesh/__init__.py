# Standard Library Imports
import json

# Third-Party Library Imports
import numpy as np

# Local Library Imports

# Relative Imports
from .Column import Column
from .Cell import Cell

from .get_nhbr import get_cell_nhbr_in_col
from .calc_key import calc_col_key, calc_cell_key

class Mesh:
    def __init__(self, Ls, pbcs, ndofs = [2, 2, 2], has_th = False):
        self.Ls     = Ls     # Lengths of spatial domain
        self.pbcs   = pbcs   # Periodicity of spatial domain
        self.has_th = has_th # Include angular domain in mesh?

        # Create first cell
        if has_th:
            cell = Cell(pos       = [0, 2 * np.pi],
                        idx       = 0,
                        lv        = 0,
                        is_lf     = True,
                        ndofs     = [ndofs[2]],
                        quad      = None,
                        nhbr_keys = [0, 0])
        else:
            cell = Cell(pos       = [0, 0],
                        idx       = 0,
                        lv        = 0,
                        is_lf     = True,
                        ndofs     = [1],
                        quad      = None,
                        nhbr_keys = [None, None])

        # Create first column.
        # Determine neighbor keys flags for column
        # F  => Right face, proceed counterclockwise
        nhbr_keys = [
            [None, None], # F = 0
            [None, None], # F = 1
            [None, None], # F = 2
            [None, None]  # F = 3
        ]
        
        # Which faces have a neighbor?
        if pbcs[0]: # Periodic in x, is own neighbor in x.
            nhbr_keys[0] = [0, 0]
            nhbr_keys[2] = [0, 0]

        if pbcs[1]: # Periodic in y, is own neighbor in y.
            nhbr_keys[1] = [0, 0]
            nhbr_keys[3] = [0, 0]
            
        col = Column(pos       = [0, 0, Ls[0], Ls[1]],
                     idx       = [0, 0],
                     lv        = 0,
                     is_lf     = True,
                     ndofs     = ndofs[0:2],
                     cells     = {0 : cell},
                     nhbr_keys = nhbr_keys)
        self.cols = {0 :  col} # Columns in mesh
        
    def __str__(self):
        msg = ( 'Ls     :  {}\n'.format(self.Ls) +
                'pbcs   :  {}\n'.format(self.pbcs) +
                'has_th :  {}\n'.format(self.has_th) +
                'cols   :  {}\n'.format(sorted(self.cols.keys()))
               )
        
        return msg

    from .add_col import add_col
    from .del_col import del_col
    from .ref_col import ref_col
    from .ref_col_spt import ref_col_spt
    from .ref_col_ang import ref_col_ang
    from .ref_cell import ref_cell
    from .ref_mesh import ref_mesh
    from .get_ndof import get_ndof
    
def get_hasnt_th(mesh):

    Ls     = mesh.Ls [:]
    pbcs   = mesh.pbcs[:]
    has_th = False

    mesh_hasnt_th = Mesh(Ls, pbcs, has_th = has_th)
    mesh_hasnt_th.cols = {}
    col_items = sorted(mesh.cols.items())
    for col_key, col in col_items:
        if col.is_lf:
            pos = col.pos[:]
            idx = col.idx[:]
            lv = col.lv
            key = col.key
            is_lf = col.is_lf
            ndofs = col.ndofs[:]
            nhbr_keys = col.nhbr_keys[:]
            
            cell = Cell(pos       = [0, 0],
                        idx       = 0,
                        lv        = 0,
                        is_lf     = True,
                        ndofs     = [1],
                        quad      = None,
                        nhbr_keys = [None, None])

            col_copy = Column(pos       = pos,
                              idx       = idx,
                              lv        = lv,
                              is_lf     = is_lf,
                              ndofs     = ndofs,
                              cells     = {0 : cell},
                              nhbr_keys = nhbr_keys)
            
            mesh_hasnt_th.cols[col_key] = col_copy
            
    return mesh_hasnt_th

def write_mesh(mesh, out_fname : str = "mesh.json"):
    """
    Converts a mesh to a dict and writes it our to a .json file.
    """

    mesh_dict = {}

    # Mesh properties
    mesh_dict["Ls"]     = mesh.Ls
    mesh_dict["pbcs"]   = mesh.pbcs
    mesh_dict["has_th"] = mesh.has_th
    mesh_dict["cols"]   = {}

    # Copy each column
    for col_key, col in mesh.cols.items():
        col_dict = {}

        col_dict["pos"]       = col.pos
        col_dict["idx"]       = col.idx
        col_dict["lv"]        = col.lv
        col_dict["key"]       = col.key
        col_dict["is_lf"]     = col.is_lf
        col_dict["ndofs"]     = col.ndofs
        col_dict["nhbr_keys"] = col.nhbr_keys
        col_dict["cells"]     = {}

        # Copy each cell
        for cell_key, cell in col.cells.items():
            cell_dict = {}

            cell_dict["pos"]       = cell.pos
            cell_dict["idx"]       = cell.idx
            cell_dict["lv"]        = cell.lv
            cell_dict["key"]       = cell.key
            cell_dict["is_lf"]     = cell.is_lf
            cell_dict["ndofs"]     = cell.ndofs
            cell_dict["quad"]      = cell.quad
            cell_dict["nhbr_keys"] = cell.nhbr_keys

            col_dict["cells"][cell_key] = cell_dict

        mesh_dict["cols"][col_key] = col_dict

    with open(out_fname, "w") as out_file:
        json.dump(mesh_dict, out_file)

def read_mesh(in_fname : str = "mesh.json"):
    """
    Reads a .json file for a mesh.
    """

    # Load in mesh dictionary
    with open(in_fname, "r") as in_file:
        mesh_dict = json.load(in_file)

    # Make the mesh object, empty out its columns
    Ls     = mesh_dict["Ls"]
    pbcs   = mesh_dict["pbcs"]
    has_th = mesh_dict["has_th"]
    mesh   = Mesh(Ls = Ls, pbcs = pbcs, has_th = has_th)
    mesh.del_col(0)

    # Copy each Column to the mesh
    for col_key in mesh_dict["cols"].keys():
        col_dict = mesh_dict["cols"][col_key]
        col_pos = col_dict["pos"]
        col_idx = col_dict["idx"]
        col_lv  = col_dict["lv"]
        col_is_lf     = col_dict["is_lf"]
        col_ndofs     = col_dict["ndofs"]
        col_nhbr_keys = col_dict["nhbr_keys"]

        col_cells = {}

        # Copy each Cell to the mesh
        for cell_key in col_dict["cells"].keys():
            cell_dict = col_dict["cells"][cell_key]

            cell_pos       = cell_dict["pos"]
            cell_idx       = cell_dict["idx"]
            cell_lv        = cell_dict["lv"]
            cell_is_lf     = cell_dict["is_lf"]
            cell_ndofs     = cell_dict["ndofs"]
            cell_quad      = cell_dict["quad"]
            cell_nhbr_keys = cell_dict["nhbr_keys"]

            col_cells[cell_key] = Cell(pos = cell_pos, idx = cell_idx,
                                       lv = cell_lv, is_lf = cell_is_lf, 
                                       ndofs = cell_ndofs, quad = cell_quad,
                                       nhbr_keys = cell_nhbr_keys)

        col = Column(pos = col_pos, idx = col_idx, lv = col_lv,
                     is_lf = col_is_lf, ndofs = col_ndofs, cells = col_cells,
                     nhbr_keys = col_nhbr_keys)
        
        mesh.add_col(col)

    return mesh