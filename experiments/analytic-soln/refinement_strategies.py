# Standard Library Imports
import json

# Third-Party Library Imports
import numpy as np
from scipy.integrate import quad, dblquad

# Local Library Imports
import consts
from rt import Problem

# Relative Imports

## Read input files - hardcoded file names
input_file = open("input.json")
input_dict: dict = json.load(input_file)
input_file.close()

refinement_strategies: dict = {
    "h-uni-ang"  : input_dict["h-uni-ang"],
    "p-uni-ang"  : input_dict["p-uni-ang"],
    "h-amr-ang"  : input_dict["h-amr-ang"],
    "hp-amr-ang" : input_dict["hp-amr-ang"]
}