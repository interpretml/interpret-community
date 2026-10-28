# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Tests for interpret-community NumPy compatibility."""

import numpy as np
import pandas as pd
from interpret_community import TabularExplainer
from interpret_community._internal.raw_explain.raw_explain_utils import \
    get_datamapper_and_transformed_data
from interpret_community.common.serialization_utils import _serialize_json_safe
from interpret_community.mimic.models.linear_model import \
    LinearExplainableModel
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def test_public_imports_and_serialization_support_numpy_1_and_2():
    """Verify public imports and serialization with NumPy 1.x and 2.x."""
    assert int(np.__version__.split('.')[0]) in (1, 2)
    assert TabularExplainer is not None

    values = np.array([1.0, np.nan, np.inf])
    assert _serialize_json_safe(values) == [1.0, 0, 0]


def test_latest_scikit_learn_compatibility():
    """Verify custom transformers and multiclass linear models."""
    data = pd.DataFrame({
        'category': ['a', 'b', 'a', 'c'],
        'category_2': ['x', 'y', 'x', 'z'],
        'numeric': [1.0, 2.0, np.nan, 4.0]
    })
    transformations = ColumnTransformer([
        ('category', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), ['category', 'category_2']),
        ('numeric', SimpleImputer(strategy='mean'), ['numeric'])
    ])
    transformations.fit(data)

    _, transformed_data = get_datamapper_and_transformed_data(
        examples=data,
        transformations=transformations
    )
    assert transformed_data.shape[0] == len(data)

    model = LinearExplainableModel(multiclass=True)
    model.fit(np.eye(6), np.array([0, 1, 2, 0, 1, 2]))
    assert model.predict(np.eye(6)).shape == (6,)
