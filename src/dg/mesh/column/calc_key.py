# Standard Library Imports

# Third-Party Library Imports

# Local Library Imports

# Relative Imports

def calc_key(idx: int, lv: int) -> int:
    
    key: int = 0
    for ll in range(0, lv):
        key += 4 ** ll
        
    key += 2**lv * idx[1] + idx[0]
        
    return key

