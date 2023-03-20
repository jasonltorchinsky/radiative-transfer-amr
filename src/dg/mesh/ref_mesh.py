# Refine each coumn in the mesh spatially ('spt'), angularly ('ang'),
# or both ('all').

def ref_mesh(self, kind = 'all'):
    col_keys = sorted(self.cols.keys())
    if (kind == 'ang') or (kind == 'all'):
        for col_key in col_keys:
            self.ref_col(col_key, kind = 'ang')
    
    if (kind == 'spt') or (kind == 'all'):
        for col_key in col_keys:
            self.ref_col(col_key, kind = 'spt')
