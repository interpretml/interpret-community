# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import numpy as np
import pytest
from constants import owner_email_tools_and_ux
from interpret_community._internal.raw_explain import DataMapper
from raw_explain.utils import (FuncTransformer, IdentityTransformer,
                               SparseTransformer)
from scipy.sparse import csr_matrix, issparse
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


@pytest.mark.owner(email=owner_email_tools_and_ux)
@pytest.mark.usefixtures('_clean_dir')
class TestDataMapper:
    def setup_class(self):
        self._identity_mapper_list = DataMapper([([0], IdentityTransformer()), ([1], IdentityTransformer())])
        column_transformer = ColumnTransformer(
            [("column1", IdentityTransformer(), [0]), ("column2", IdentityTransformer(), [1])]
        )
        x = np.ones((10, 5))
        column_transformer.fit(x)
        self._identity_mapper_column_transformer = DataMapper(column_transformer)

    def _transform_numpy(self, dmapper):
        x = np.ones((10, 5))
        result = dmapper.transform(x)
        assert result.shape == (10, 2)

    def test_mixed_dtypes(self):
        x = np.ones((10, 2))
        data_mapper = DataMapper([([0], IdentityTransformer()), ([1], SparseTransformer())])
        result = data_mapper.transform(x)
        assert issparse(result)

    def test_transform_sparse(self):
        x = csr_matrix(np.zeros((10, 2)))
        result = self._identity_mapper_list.transform(x)
        assert result.shape == x.shape
        assert issparse(result)

    def test_column_with_brackets(self):
        x = np.ones((2, 3))
        x[0, 0] = 0
        encoder = OneHotEncoder()
        encoder.fit(x[0].reshape(-1, 1))
        data_mapper = DataMapper([([0], encoder)])
        result = data_mapper.transform(x)
        assert result.shape == (2, 2)

    def test_large_sparse_transformation(self):
        x = np.linspace((1,) * 8000, (10,) * 8000, 10)
        encoder = OneHotEncoder()
        encoder.fit(x)
        data_mapper = DataMapper([(np.arange(8000), encoder)])
        result = data_mapper.transform(x)
        assert result.shape == (10, 80000)
        for i in range(10):
            for j in range(i, 80000, 10):
                assert result[i, j] == 1
                result[i, j] = 0
        result.eliminate_zeros()
        assert result.count_nonzero() == 0

    def test_transform_numpy_list(self):
        self._transform_numpy(self._identity_mapper_list)

    def test_transform_numpy_column_transformer(self):
        self._transform_numpy(self._identity_mapper_column_transformer)

    def test_column_without_brackets(self):
        data_mapper = DataMapper([(0, FuncTransformer(lambda x: x.reshape(-1, 1)))])
        result = data_mapper.transform(np.ones((2, 3)))

        assert np.all(result == [1, 1])

    def test_column_with_none_transformer(self):
        x = np.ones((2, 3))
        data_mapper = DataMapper([(0, None)])
        result = data_mapper.transform(x)
        assert np.all(result == np.array([[1, 1]]))

    def test_column_passthrough_column_transformer(self):
        x = np.ones((2, 3))
        column_transformer = ColumnTransformer([
            ("column0", "passthrough", [0])
        ])
        column_transformer.fit(x)
        data_mapper = DataMapper(column_transformer)
        result = data_mapper.transform(x)
        assert np.all(result == np.array([[1, 1]]))

    def test_pipeline_transform_list(self):
        pipeline = Pipeline([("imputer", SimpleImputer()), ("onehotencoder", OneHotEncoder())])
        x = np.ones((3, 2))
        pipeline.fit(x)
        data_mapper = DataMapper([([0, 1], pipeline)])
        result = data_mapper.transform(x)
        assert result.shape == (3, 2)

    def test_pipeline_transform_column_transformer(self):
        pipeline = Pipeline([("imputer", SimpleImputer()), ("onehotencoder", OneHotEncoder())])
        x = np.ones((3, 2))
        column_transformer = ColumnTransformer([
            ("column", pipeline, [0, 1])
        ])
        column_transformer.fit(x)
        data_mapper = DataMapper(column_transformer)
        result = data_mapper.transform(x)
        assert result.shape == (3, 2)

    def test_many_to_many_exception_list(self):
        # A transformer that takes input many columns. Since we do not recognize this transformer and it uses
        # many input columns - it is treated as many to many/one map.
        with pytest.raises(
                ValueError,
                match="Many to many or many to one transformers not supported in raw explanations when "
                      "explainer instantiated with allow_all_transformations is set to False. Change this "
                      "parameter to True in order to get explanations."):
            DataMapper([([0, 1], IdentityTransformer())])

    def test_many_to_many_exception_column_transformer(self):
        # A transformer that takes input many columns. Since we do not recognize this transformer and it uses
        # many input columns - it is treated as many to many/one map.
        column_transformer = ColumnTransformer([
            ("column_0_1", IdentityTransformer(), [0, 1])
        ])
        x = np.ones((2, 2))
        column_transformer.fit(x)
        with pytest.raises(
                ValueError,
                match="Many to many or many to one transformers not supported in raw explanations when "
                      "explainer instantiated with allow_all_transformations is set to False. Change this "
                      "parameter to True in order to get explanations."):
            DataMapper(column_transformer)

    def test_many_to_many_support_transformations(self):
        # Instantiate data mapper with many to many transformer support and test whether the feature map is generated
        column_transformer = ColumnTransformer([
            ("column_0_1_2_3", IdentityTransformer(), [0, 1, 2, 3]),
            ("column_4_5", OneHotEncoder(), [4, 5])
        ])
        x = np.ones((10, 6))
        # so that one hot encoder doesn't complain of only one category
        x[0, 4] = 0
        x[0, 5] = 0
        column_transformer.fit(x)
        data_mapper = DataMapper(column_transformer, allow_all_transformations=True)
        data_mapper.transform(x)
        # check feature mapper contents
        feature_map_indices = [
            [0, 1, 2, 3],
            [0, 1, 2, 3],
            [0, 1, 2, 3],
            [0, 1, 2, 3],
            [4, 5],
            [6, 7]
        ]
        x_out = column_transformer.transform(x)
        feature_map = np.zeros((x.shape[1], x_out.shape[1]))
        num_rows = 0
        for i, row in enumerate(feature_map_indices[:4]):
            feature_map[i, row] = 0.25
            num_rows += 1
        for i, row in enumerate(feature_map_indices[4:], start=num_rows):
            feature_map[i, row] = 1.0
        assert (data_mapper.feature_map == feature_map).all()
