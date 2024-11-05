import os, sys
src_dir: str = os.path.normpath(os.path.join(os.path.dirname(__file__),
                                             os.pardir, os.pardir))

if src_dir not in sys.path:
    sys.path.append(src_dir)
    
# Standard Library Imports

# Third-Party Library Imports

# Local Library Imports

# Relative Imports
from . import column
from .class_Mesh import Mesh
from .from_file import from_file
from .remove_th import remove_th