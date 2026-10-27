# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Tests for interpret-community NumPy compatibility."""

import numpy as np
from interpret_community import TabularExplainer
from interpret_community.common.serialization_utils import _serialize_json_safe


def test_public_imports_and_serialization_support_numpy_1_and_2():
    """Verify public imports and serialization with NumPy 1.x and 2.x."""
    assert int(np.__version__.split('.')[0]) in (1, 2)
    assert TabularExplainer is not None

    values = np.array([1.0, np.nan, np.inf])
    assert _serialize_json_safe(values) == [1.0, 0, 0]
