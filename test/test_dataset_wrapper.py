# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Tests for DatasetWrapper class"""

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from interpret_community.dataset.dataset_wrapper import DatasetWrapper


class TestDatasetWrapper:
    def test_supported_types(self):
        test_dataframe = pd.DataFrame(data=[[1, 2, 3]], columns=['c1,', 'c2', 'c3'])
        DatasetWrapper(dataset=test_dataframe)

        test_array = test_dataframe.values
        DatasetWrapper(dataset=test_array)

        test_series = test_dataframe.squeeze()
        DatasetWrapper(dataset=test_series)

        sparse_matrix = csr_matrix((3, 4),
                                   dtype=np.int8)
        DatasetWrapper(dataset=sparse_matrix)

        test_list = test_array.tolist()
        DatasetWrapper(test_list)
        with pytest.raises(TypeError):
            DatasetWrapper(test_list)
