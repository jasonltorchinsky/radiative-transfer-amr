`matrix`
================================================================================

This subdirectory contains the source code to handle matrices across multiple MPI processes.

Source Files
--------------------------------------------------------------------------------

The source files contained in the `matrix` subdirectory are:

- `get_idxs` : Contains indexing maps for, e.g., unrolling tensors into vectors, constructing global vectors/matrices, etc.
- `get_masks` : Constains functions for obtaining masks for vectors/matrices to obtain only the entries corresponding to interior/boundary values.
- `merge_vectors` : Merge vectors split across multiple MPI processes into a single vector on the `ROOT` process.
- `split_matrix` : Split a matrix on a single MPI process across all MPI processes in the MPI communicator.
- `split_vector` : Split a vector on a single MPI process across all MPI processes in the MPI communicator.