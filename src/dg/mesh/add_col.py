# Standard Library Imports

# Third-Party Library Imports

# Local Library Imports

# Relative Imports
from .column import Column

def add_col(self, col: Column) -> None:
    self.cols[col.key] = col