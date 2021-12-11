# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

# Common utilities for transformations used by tests

import numpy as np
import pandas as pd
from raw_explain.utils import IdentityTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


def get_transformations_one_to_many_smaller(feature_names):
    # results in number of features smaller than original features
    transformations = []
    # Take out last feature after taking a copy
    feature_names = list(feature_names)
    feature_names.pop()

    index = 0
    for f in feature_names:
        transformations.append(("{}".format(index), "passthrough", [f]))
        index += 1

    return ColumnTransformer(transformations)


def get_transformations_one_to_many_greater(feature_names):
    # results in number of features greater than original features
    # copy all features except last one. For last one, replicate columns to create 3 more features
    transformations = []
    feature_names = list(feature_names)
    index = 0
    for f in feature_names[:-1]:
        transformations.append(("{}".format(index), "passthrough", [f]))
        index += 1

    def copy_func(x):
        return np.tile(x, (1, 3))

    copy_transformer = FunctionTransformer(copy_func)

    transformations.append(("copy_transformer", copy_transformer, [feature_names[-1]]))

    return ColumnTransformer(transformations)


def get_transformations_many_to_many(feature_names):
    # Instantiate data mapper with many to many transformer support and test whether the feature map is generated

    # IdentityTransformer is our custom transformer, so not recognized as one to many
    transformations = [
        ("column_0_1_2_3", Pipeline([
            ("scaler", StandardScaler()),
            ("identity", IdentityTransformer())]), [f for f in feature_names[:-2]]),
        ("column_4_5", StandardScaler(), [f for f in feature_names[-2:]])
    ]

    # add transformations with pandas index types
    transformations.append(("pandas_index_columns", "passthrough",
                            pd.Index([feature_names[0], feature_names[1]])))

    column_transformer = ColumnTransformer(transformations)

    return column_transformer


def get_transformations_from_col_transformer(col_transformer):
    transformers = []
    for _, tr, column_name, in col_transformer.transformers_:
        if tr == "passthrough":
            tr = None
        if tr != "drop":
            transformers.append((column_name, tr))

    return transformers
