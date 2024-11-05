# Standard Library Imports

# Third-Party Library Imports
import json

# Local Library Imports
import consts

# Relative Imports
from .column import Column
from .column.cell import Cell
from .class_Mesh import Mesh

def remove_th(mesh: Mesh) -> Mesh:
    Ls: list     = mesh.Ls [:]
    pbcs: list   = mesh.pbcs[:]
    has_th: bool = False

    mesh_hasnt_th: Mesh = Mesh(Ls, pbcs, has_th = has_th)
    mesh_hasnt_th.cols = {}
    col_items = sorted(mesh.cols.items())
    for col_key, col in col_items:
        if col.is_lf:
            pos: list = col.pos[:]
            idx: list = col.idx[:]
            lv: int = col.lv
            is_lf: bool = col.is_lf
            ndofs: list = col.ndofs[:]
            nhbr_keys: list = col.nhbr_keys[:]
            
            cell: Cell = Cell(pos       = [0, 0],
                              idx       = 0,
                              lv        = 0,
                              is_lf     = True,
                              ndofs     = [1],
                              quad      = None,
                              nhbr_keys = [None, None])

            col_copy: Column = Column(pos       = pos,
                                      idx       = idx,
                                      lv        = lv,
                                      is_lf     = is_lf,
                                      ndofs     = ndofs,
                                      cells     = {0 : cell},
                                      nhbr_keys = nhbr_keys)
            
            mesh_hasnt_th.cols[col_key] = col_copy
            
    return mesh_hasnt_th