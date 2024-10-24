# Standard Library Imports
import json

# Third-Party Library Imports

# Local Library Imports

# Relative Imports

## Read input files - hardcoded file names
with open("input.json", "r") as input_file:
    input_dict: dict = json.load(input_file)

refinement_strategies: dict = {
    "h-uni-ang"  : input_dict["h-uni-ang"],
    "p-uni-ang"  : input_dict["p-uni-ang"],
    "h-amr-ang"  : input_dict["h-amr-ang"],
    "hp-amr-ang" : input_dict["hp-amr-ang"]
}