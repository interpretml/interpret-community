# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import pytest

import numpy as np

from common_utils import create_sklearn_svm_classifier, create_sklearn_random_forest_regressor
from constants import DatasetConstants

from constants import owner_email_tools_and_ux


@pytest.mark.owner(email=owner_email_tools_and_ux)
@pytest.mark.usefixtures('clean_dir')
class TestRawExplanations:
    def test_get_global_raw_explanations_classification(self, iris, tabular_explainer):
        model = create_sklearn_svm_classifier(iris[DatasetConstants.X_TRAIN], iris[DatasetConstants.Y_TRAIN])

        exp = tabular_explainer(model, iris[DatasetConstants.X_TRAIN], features=iris[DatasetConstants.FEATURES],
                                classes=iris[DatasetConstants.CLASSES])

        global_explanation = exp.explain_global(iris[DatasetConstants.X_TEST])
        num_engineered_feats = len(iris[DatasetConstants.FEATURES])

        feature_map = np.eye(num_engineered_feats - 1, num_engineered_feats)
        feature_names = [str(i) for i in range(feature_map.shape[0])]

        global_raw_explanation = global_explanation.get_raw_explanation(
            [feature_map], raw_feature_names=feature_names[:feature_map.shape[0]])

        per_class_values = global_raw_explanation.get_ranked_per_class_values()
        assert len(per_class_values) == len(iris[DatasetConstants.CLASSES])
        assert len(per_class_values[0]) == feature_map.shape[0]
        assert len(global_raw_explanation.get_ranked_per_class_names()[0]) == feature_map.shape[0]
        feat_imps_global_local = np.array(global_raw_explanation.local_importance_values)
        assert feat_imps_global_local.shape[-1] == feature_map.shape[0]

        assert global_raw_explanation.is_raw
        assert len(global_raw_explanation.get_ranked_global_values()) == feature_map.shape[0]
        assert len(global_raw_explanation.get_ranked_global_names()) == feature_map.shape[0]
        assert (global_raw_explanation.classes == iris[DatasetConstants.CLASSES]).all()

        assert global_raw_explanation.features == feature_names

        feat_imps_global = np.array(global_raw_explanation.global_importance_values)

        assert feat_imps_global.shape[-1] == feature_map.shape[0]

    def test_get_global_raw_explanations_regression(self, boston, tabular_explainer):
        model = create_sklearn_random_forest_regressor(boston[DatasetConstants.X_TRAIN],
                                                       boston[DatasetConstants.Y_TRAIN])

        exp = tabular_explainer(model, boston[DatasetConstants.X_TRAIN], features=boston[DatasetConstants.FEATURES])

        global_explanation = exp.explain_global(boston[DatasetConstants.X_TEST])
        num_engineered_feats = len(boston[DatasetConstants.FEATURES])
        feature_map = np.eye(num_engineered_feats - 1, num_engineered_feats)

        global_raw_explanation = global_explanation.get_raw_explanation([feature_map])

        assert np.array(global_raw_explanation.local_importance_values).shape[-1] == feature_map.shape[0]

        assert global_raw_explanation.is_raw
        assert np.array(global_raw_explanation.global_importance_values).shape[-1] == feature_map.shape[0]

    def test_get_local_raw_explanations_classification(self, iris, tabular_explainer):
        model = create_sklearn_svm_classifier(iris[DatasetConstants.X_TRAIN], iris[DatasetConstants.Y_TRAIN])

        exp = tabular_explainer(model, iris[DatasetConstants.X_TRAIN], features=iris[DatasetConstants.FEATURES],
                                classes=iris[DatasetConstants.CLASSES])

        local_explanation = exp.explain_local(iris[DatasetConstants.X_TEST][0])

        num_engineered_feats = len(iris[DatasetConstants.FEATURES])
        feature_map = np.eye(num_engineered_feats - 1, num_engineered_feats)

        local_raw_explanation = local_explanation.get_raw_explanation([feature_map])

        assert len(local_raw_explanation.local_importance_values) == len(iris[DatasetConstants.CLASSES])
        assert len(local_raw_explanation.local_importance_values[0]) == feature_map.shape[0]

        local_rank = local_raw_explanation.get_local_importance_rank()
        assert len(local_rank) == len(iris[DatasetConstants.CLASSES])
        assert len(local_rank[0]) == feature_map.shape[0]

        ranked_names = local_raw_explanation.get_ranked_local_names()
        assert len(ranked_names) == len(iris[DatasetConstants.CLASSES])
        assert len(ranked_names[0]) == feature_map.shape[0]

        ranked_values = local_raw_explanation.get_ranked_local_values()
        assert len(ranked_values) == len(iris[DatasetConstants.CLASSES])
        assert len(ranked_values[0]) == feature_map.shape[0]

    def test_get_local_raw_explanations_regression(self, boston, tabular_explainer):
        model = create_sklearn_random_forest_regressor(boston[DatasetConstants.X_TRAIN],
                                                       boston[DatasetConstants.Y_TRAIN])

        exp = tabular_explainer(model, boston[DatasetConstants.X_TRAIN], features=boston[DatasetConstants.FEATURES])

        num_engineered_feats = len(boston[DatasetConstants.FEATURES])
        feature_map = np.eye(num_engineered_feats - 1, num_engineered_feats)

        local_explanation = exp.explain_local(boston[DatasetConstants.X_TEST][0])

        local_raw_explanation = local_explanation.get_raw_explanation([feature_map])

        assert len(local_raw_explanation.local_importance_values) == feature_map.shape[0]

        local_rank = local_raw_explanation.get_local_importance_rank()
        assert len(local_rank) == feature_map.shape[0]

        ranked_names = local_raw_explanation.get_ranked_local_names()
        assert len(ranked_names) == feature_map.shape[0]

        ranked_values = local_raw_explanation.get_ranked_local_values()
        assert len(ranked_values) == feature_map.shape[0]
