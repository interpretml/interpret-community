# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import numpy as np
import pytest
from constants import owner_email_tools_and_ux
from interpret_community._internal.raw_explain.feature_mappers import (
    IdentityMapper, ManytoManyMapper, PassThroughMapper,
    get_feature_mapper_for_pipeline)
from raw_explain.utils import FuncTransformer, IdentityTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


@pytest.mark.owner(email=owner_email_tools_and_ux)
@pytest.mark.usefixtures('_clean_dir')
class TestFeatureMappers:
    def _get_nested_pipelines_and_data(self, last_transformer=None):
        # returns a pipeline that can be used to test nested pipelines. When last_transformer is not None, it is
        # added as the last transformer in pipeline_1
        steps = [("a", SimpleImputer()), ("b", OneHotEncoder())]
        if last_transformer:
            steps.append(("c", last_transformer))
        pipeline_1 = Pipeline(steps)
        pipeline = Pipeline([("a", SimpleImputer()), ("b", pipeline_1)])
        x = np.zeros((5, 2))
        x[0, 0] = 1
        x[0, 1] = 1
        x[1, 0] = 2

        return pipeline.fit(x), x

    def test_get_feature_mapper_tuple_for_pipeline(self):
        pipeline = Pipeline([("a", SimpleImputer()), ("b", SimpleImputer()), ("c", OneHotEncoder())])
        x = np.zeros((5, 2))
        x[0, 0] = 1
        x[0, 1] = 1
        pipeline.fit(x)
        feature_mapper = get_feature_mapper_for_pipeline(pipeline)
        feature_mapper.transform(x)

        feature_map = np.zeros((2, 4))
        feature_map[0, :2] = 1
        feature_map[1, 2:] = 1
        assert np.all(feature_mapper.feature_map == feature_map)

    def test_identity_mapper(self):
        x = np.zeros((5, 2))
        imputer = SimpleImputer()
        imputer.fit(x)
        mapper = IdentityMapper(imputer)
        mapper.transform(x)

        feature_map = np.eye(2)
        assert np.all(mapper.feature_map == feature_map)

    def test_pass_through_mapper(self):
        x = np.array([0, 1, 2, 5])
        x = x.reshape(-1, 1)
        encoder = OneHotEncoder()
        encoder.fit(x)
        mapper = PassThroughMapper(encoder)
        mapper.transform(x)

        feature_map = np.ones((1, 4))
        assert np.all(mapper.feature_map == feature_map)

    def test_nested_pipelines(self):
        pipeline, x = self._get_nested_pipelines_and_data()
        feature_mapper = get_feature_mapper_for_pipeline(pipeline)
        feature_mapper.transform(x)

        feature_map = np.zeros((2, 5))
        feature_map[0, [0, 1, 2]] = 1
        feature_map[1, [3, 4]] = 1
        assert np.all(feature_mapper.feature_map == feature_map)

    def test_many_to_many_mapper(self):
        x = np.ones((10, 5))
        mapper = ManytoManyMapper(FuncTransformer(lambda x: np.hstack((x, x))))
        mapper.transform(x)
        assert np.all(mapper.feature_map == np.ones((5, 10)) * 1.0 / 5)

    def test_many_to_many_mapper_nested_pipelines(self):
        pipeline, x = self._get_nested_pipelines_and_data(IdentityTransformer())
        feature_mapper = get_feature_mapper_for_pipeline(pipeline)
        feature_mapper.transform(x)

        feature_map = np.zeros((2, 5))
        feature_map[0, :] = 0.6
        feature_map[1, :] = 0.4

        assert np.all(feature_mapper.feature_map == pytest.approx(feature_map))
