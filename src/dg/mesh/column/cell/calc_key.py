# Standard Library Imports

# Third-Party Library Imports

# Local Library Imports

# Relative Imports

def calc_key(idx: int, lv: int) -> int:

    key: int = 0
    for ll in range(0, lv):
        key += 2**ll

    key += idx

    return key
