# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import pytest

import numpy as np

from common_utils import create_sklearn_svm_classifier, create_sklearn_random_forest_regressor, \
    create_sklearn_linear_regressor, create_multiclass_sparse_newsgroups_data, \
    create_sklearn_logistic_regressor
from constants import DatasetConstants, owner_email_tools_and_ux
from datasets import retrieve_dataset
from sklearn.model_selection import train_test_split
from interpret_community.mimic.models.linear_model import LinearExplainableModel

LINEAR_METHOD = 'mimic.linear'


@pytest.mark.owner(email=owner_email_tools_and_ux)
@pytest.mark.usefixtures('clean_dir')
class TestRawExplanations:
    def test_get_global_raw_explanations_classification(self, iris, tabular_explainer):
        model = create_sklearn_svm_classifier(iris[DatasetConstants.X_TRAIN], iris[DatasetConstants.Y_TRAIN])

        exp = tabular_explainer(model, iris[DatasetConstants.X_TRAIN], features=iris[DatasetConstants.FEATURES],
                                classes=iris[DatasetConstants.CLASSES])

        global_explanation = exp.explain_global(iris[DatasetConstants.X_TEST])
        assert not global_explanation.is_raw
        assert not global_explanation.is_engineered
        num_engineered_feats = len(iris[DatasetConstants.FEATURES])

        feature_map = np.eye(num_engineered_feats - 1, num_engineered_feats)
        feature_names = [str(i) for i in range(feature_map.shape[0])]

        global_raw_explanation = global_explanation.get_raw_explanation(
            [feature_map], raw_feature_names=feature_names[:feature_map.shape[0]])

        self.validate_global_raw_explanation_classification(global_explanation, global_raw_explanation, feature_map,
                                                            iris[DatasetConstants.CLASSES], feature_names)

    def test_get_global_raw_explanations_regression(self, boston, tabular_explainer):
        model = create_sklearn_random_forest_regressor(boston[DatasetConstants.X_TRAIN],
                                                       boston[DatasetConstants.Y_TRAIN])

        exp = tabular_explainer(model, boston[DatasetConstants.X_TRAIN], features=boston[DatasetConstants.FEATURES])

        global_explanation = exp.explain_global(boston[DatasetConstants.X_TEST])
        assert not global_explanation.is_raw
        assert not global_explanation.is_engineered
        num_engineered_feats = len(boston[DatasetConstants.FEATURES])
        feature_map = np.eye(num_engineered_feats - 1, num_engineered_feats)

        global_raw_explanation = global_explanation.get_raw_explanation([feature_map])
        self.validate_global_raw_explanation_regression(global_explanation, global_raw_explanation, feature_map)

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

    def test_get_local_raw_explanations_sparse_regression(self, mimic_explainer):
        X, y = retrieve_dataset('a1a.svmlight')
        x_train, x_test, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=7)
        # Fit a linear regression model
        model = create_sklearn_linear_regressor(x_train, y_train)

        explainer = mimic_explainer(model, x_train, LinearExplainableModel,
                                    explainable_model_args={'sparse_data': True})
        global_explanation = explainer.explain_global(x_test)
        assert global_explanation.method == LINEAR_METHOD

        num_engineered_feats = x_train.shape[1]
        feature_map = np.eye(5, num_engineered_feats)

        global_raw_explanation = global_explanation.get_raw_explanation([feature_map])
        self.validate_global_raw_explanation_regression(global_explanation, global_raw_explanation, feature_map)

    def test_get_local_raw_explanations_sparse_classification(self, mimic_explainer):
        x_train, x_test, y_train, _, classes, _ = create_multiclass_sparse_newsgroups_data()
        # Fit a linear regression model
        model = create_sklearn_logistic_regressor(x_train, y_train)

        explainer = mimic_explainer(model, x_train, LinearExplainableModel,
                                    explainable_model_args={'sparse_data': True}, classes=classes)
        global_explanation = explainer.explain_global(x_test)
        assert global_explanation.method == LINEAR_METHOD

        num_engineered_feats = x_train.shape[1]
        feature_map = np.eye(5, num_engineered_feats)
        feature_names = [str(i) for i in range(feature_map.shape[0])]
        raw_names = feature_names[:feature_map.shape[0]]
        global_raw_explanation = global_explanation.get_raw_explanation([feature_map], raw_feature_names=raw_names)
        self.validate_global_raw_explanation_classification(global_explanation, global_raw_explanation, feature_map,
                                                            classes, feature_names, is_sparse=True)

    def validate_global_raw_explanation_regression(self, global_explanation, global_raw_explanation, feature_map):
        assert not global_explanation.is_raw
        assert hasattr(global_explanation, 'eval_data')
        assert global_explanation.is_engineered

        assert np.array(global_raw_explanation.local_importance_values).shape[-1] == feature_map.shape[0]

        assert global_raw_explanation.is_raw
        assert not hasattr(global_raw_explanation, 'eval_data')
        assert not global_raw_explanation.is_engineered
        assert np.array(global_raw_explanation.global_importance_values).shape[-1] == feature_map.shape[0]

    def validate_global_raw_explanation_classification(self, global_explanation, global_raw_explanation,
                                                       feature_map, classes, feature_names, is_sparse=False):
        assert not global_explanation.is_raw
        assert global_explanation.is_engineered

        assert global_raw_explanation.expected_values == global_explanation.expected_values

        per_class_values = global_raw_explanation.get_ranked_per_class_values()
        assert len(per_class_values) == len(classes)
        assert len(per_class_values[0]) == feature_map.shape[0]
        assert len(global_raw_explanation.get_ranked_per_class_names()[0]) == feature_map.shape[0]
        feat_imps_global_local = np.array(global_raw_explanation.local_importance_values)
        assert feat_imps_global_local.shape[-1] == feature_map.shape[0]

        assert global_raw_explanation.is_raw
        assert not hasattr(global_raw_explanation, 'eval_data')
        assert not global_raw_explanation.is_engineered
        assert len(global_raw_explanation.get_ranked_global_values()) == feature_map.shape[0]
        assert len(global_raw_explanation.get_ranked_global_names()) == feature_map.shape[0]
        if isinstance(classes, list):
            assert global_raw_explanation.classes == classes
        else:
            assert (global_raw_explanation.classes == classes).all()

        assert global_raw_explanation.features == feature_names

        feat_imps_global = np.array(global_raw_explanation.global_importance_values)

        assert feat_imps_global.shape[-1] == feature_map.shape[0]
