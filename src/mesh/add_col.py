# Standard Library Imports

# Third-Party Library Imports

# Local Library Imports
from .column import Column

# Relative Imports

def add_col(self, col: Column) -> None:
    self.cols[col.key] = col