import sys

def ref_col(self, col_key, kind = 'all'):
    col = self.cols[col_key]
    if col.is_lf:     
        if (kind == 'ang') or (kind == 'all'):
            cell_keys = sorted(col.cells.keys())
            for cell_key in cell_keys:
                self.ref_cell(col_key, cell_key)
                
        if (kind == 'spt') or (kind == 'all'):
            self.ref_col_spt(col_key)
            
        if (kind != 'spt') and (kind != 'ang') and (kind != 'all'):
            msg = ( 'ERROR IN REFINING COLUMN, ' +
                    'UNSUPPORTED REFINEMENT TYPE - {}').format(kind)
            print(msg)
            sys.exit(0)
