# Standard Library Imports

# Third-Party Library Imports

# Local Library Imports

# Relative Imports
from .cell import Cell

def add_cell(self, cell: Cell) -> None:
    self.cells[cell.key] = cell