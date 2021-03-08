# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

# Defines helper utilities for working with sparse data
import numpy as np
from scipy.sparse import csr_matrix


def tile_csr_matrix(matrix, nsamples):
    shape = matrix.shape
    nnz = matrix.nnz
    data_rows, data_cols = shape
    rows = data_rows * nsamples
    shape = rows, data_cols
    if nnz == 0:
        tiled_data = csr_matrix(shape, dtype=matrix.dtype)
    else:
        data = matrix.data
        indices = matrix.indices
        indptr = matrix.indptr
        last_indptr_idx = indptr[len(indptr) - 1]
        indptr_wo_last = indptr[:-1]
        new_indptrs = []
        for i in range(0, nsamples - 1):
            new_indptrs.append(indptr_wo_last + (i * last_indptr_idx))
        new_indptrs.append(indptr + ((nsamples - 1) * last_indptr_idx))
        new_indptr = np.concatenate(new_indptrs)
        new_data = np.tile(data, nsamples)
        new_indices = np.tile(indices, nsamples)
        tiled_data = csr_matrix((new_data, new_indices, new_indptr), shape=shape)
    return tiled_data
