# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator


class IdentityTransformer(BaseEstimator):
    def __init__(self):
        pass

    def fit(self, x, y):
        return self

    def transform(self, x):
        return x


class FuncTransformer(BaseEstimator):
    def __init__(self, func):
        self.func = func

    def fit(self, x, y):
        return self

    def transform(self, x):
        return self.func(x)


class SparseTransformer(BaseEstimator):
    def __init__(self):
        pass

    def fit(self, x, y):
        return self

    def transform(self, x):
        return csr_matrix(x)


def _get_feature_map_from_indices_list(indices_list, num_raw_cols, num_generated_cols):
    # indices_list is list of lists containing indices of generated that map to a raw feature
    feature_map = np.zeros((num_raw_cols, num_generated_cols))
    for i, index in enumerate(indices_list):
        feature_map[i, index] = 1

    return feature_map
