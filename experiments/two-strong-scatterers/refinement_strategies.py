# Standard Library Imports
import json

# Third-Party Library Imports

# Local Library Imports

# Relative Imports

## Read input files - hardcoded file names
with open("input.json", "r") as input_file:
    input_dict: dict = json.load(input_file)

refinement_strategies: dict = {
    "hp-amr-spt-p-uni-ang"  : input_dict["hp-amr-spt-p-uni-ang"],
    "hp-amr-ang-p-uni-spt"  : input_dict["hp-amr-ang-p-uni-spt"],
    "hp-amr-ang-hp-amr-spt" : input_dict["hp-amr-ang-hp-amr-spt"]
}