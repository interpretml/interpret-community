# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import numpy as np
import pandas as pd
import pytest
from common_utils import (LIGHTGBM_METHOD, LINEAR_METHOD,
                          create_binary_sparse_newsgroups_data,
                          create_multiclass_classification_dataset,
                          create_multiclass_sparse_newsgroups_data,
                          create_sklearn_linear_regressor,
                          create_sklearn_logistic_regressor,
                          create_sklearn_random_forest_regressor,
                          create_sklearn_svm_classifier)
from constants import DatasetConstants, owner_email_tools_and_ux
from datasets import retrieve_dataset
from interpret_community.common.constants import ExplainParams, ExplainType
from interpret_community.explanation.explanation import (
    _create_local_explanation, _DatasetsMixin)
from interpret_community.mimic.models.lightgbm_model import \
    LGBMExplainableModel
from interpret_community.mimic.models.linear_model import \
    LinearExplainableModel
from sklearn.model_selection import train_test_split
from transformation_utils import (get_transformations_from_col_transformer,
                                  get_transformations_one_to_many_smaller)


@pytest.mark.owner(email=owner_email_tools_and_ux)
@pytest.mark.usefixtures('_clean_dir')
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

        self.validate_global_explanation_classification(global_explanation, global_raw_explanation, feature_map,
                                                        iris[DatasetConstants.CLASSES], feature_names)

    def test_get_global_raw_explanations_classification_complex_mapping(self, mimic_explainer):
        x_train, y_train, x_test, y_test, classes = create_multiclass_classification_dataset(num_features=21,
                                                                                             num_informative=10)
        model = create_sklearn_svm_classifier(x_train, y_train)

        exp = mimic_explainer(model, x_train, LGBMExplainableModel, classes=classes)

        global_explanation = exp.explain_global(x_test)
        assert not global_explanation.is_raw
        assert not global_explanation.is_engineered

        feature_map = [[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]]
        feature_map = np.array(feature_map)
        feature_names = [str(i) for i in range(feature_map.shape[0])]

        global_raw_explanation = global_explanation.get_raw_explanation(
            [feature_map], raw_feature_names=feature_names[:feature_map.shape[0]])

        self.validate_global_explanation_classification(global_explanation, global_raw_explanation, feature_map,
                                                        classes, feature_names)

    def test_get_global_raw_explanations_regression(self, housing, tabular_explainer):
        model = create_sklearn_random_forest_regressor(housing[DatasetConstants.X_TRAIN],
                                                       housing[DatasetConstants.Y_TRAIN])

        exp = tabular_explainer(model, housing[DatasetConstants.X_TRAIN], features=housing[DatasetConstants.FEATURES])

        global_explanation = exp.explain_global(housing[DatasetConstants.X_TEST])
        assert not global_explanation.is_raw
        assert not global_explanation.is_engineered
        num_engineered_feats = len(housing[DatasetConstants.FEATURES])
        feature_map = np.eye(num_engineered_feats - 1, num_engineered_feats)

        global_raw_explanation = global_explanation.get_raw_explanation([feature_map])
        self.validate_global_explanation_regression(global_explanation, global_raw_explanation, feature_map)

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

    def test_get_local_raw_explanations_regression(self, housing, tabular_explainer):
        model = create_sklearn_random_forest_regressor(housing[DatasetConstants.X_TRAIN],
                                                       housing[DatasetConstants.Y_TRAIN])

        exp = tabular_explainer(model, housing[DatasetConstants.X_TRAIN], features=housing[DatasetConstants.FEATURES])

        num_engineered_feats = len(housing[DatasetConstants.FEATURES])
        feature_map = np.eye(num_engineered_feats - 1, num_engineered_feats)

        local_explanation = exp.explain_local(housing[DatasetConstants.X_TEST][0])

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
        self.validate_global_explanation_regression(global_explanation, global_raw_explanation, feature_map)

    def test_get_global_raw_explanations_classification_pandas(self, iris, mimic_explainer):
        x_train = pd.DataFrame(iris[DatasetConstants.X_TRAIN])
        x_test = pd.DataFrame(iris[DatasetConstants.X_TEST])
        model = create_sklearn_svm_classifier(x_train, iris[DatasetConstants.Y_TRAIN])

        exp = mimic_explainer(model, x_train, LinearExplainableModel, features=iris[DatasetConstants.FEATURES],
                              classes=iris[DatasetConstants.CLASSES])

        global_explanation = exp.explain_global(x_test)
        assert not global_explanation.is_raw
        assert not global_explanation.is_engineered
        num_engineered_feats = len(iris[DatasetConstants.FEATURES])

        # Note in this case we are adding a feature in engineered features from raw,
        # so the raw explanation will have one fewer column than engineered explanation
        feature_map = np.eye(num_engineered_feats - 1, num_engineered_feats)
        feature_names = [str(i) for i in range(feature_map.shape[0])]

        has_raw_eval_data_options = [True, False]
        for has_raw_eval_data_option in has_raw_eval_data_options:
            if has_raw_eval_data_option:
                global_raw_explanation = global_explanation.get_raw_explanation(
                    [feature_map], raw_feature_names=feature_names[:feature_map.shape[0]], eval_data=x_test)
            else:
                global_raw_explanation = global_explanation.get_raw_explanation(
                    [feature_map], raw_feature_names=feature_names[:feature_map.shape[0]])

            self.validate_global_raw_explanation_classification(global_raw_explanation, feature_map,
                                                                iris[DatasetConstants.CLASSES], feature_names,
                                                                has_raw_eval_data=has_raw_eval_data_option)

    def test_get_global_raw_explanations_classification_pandas_transformations(self, iris, mimic_explainer):
        feature_names = iris[DatasetConstants.FEATURES]
        x_train = pd.DataFrame(iris[DatasetConstants.X_TRAIN], columns=feature_names)
        x_test = pd.DataFrame(iris[DatasetConstants.X_TEST], columns=feature_names)
        # Note in this case the transformations drop a feature, so raw explanation
        # will have one more column than engineered explanation
        col_transformer = get_transformations_one_to_many_smaller(feature_names)
        x_train_transformed = col_transformer.fit_transform(x_train)
        transformations = get_transformations_from_col_transformer(col_transformer)

        model = create_sklearn_svm_classifier(x_train_transformed, iris[DatasetConstants.Y_TRAIN])

        exp = mimic_explainer(model, x_train, LinearExplainableModel, features=feature_names,
                              classes=iris[DatasetConstants.CLASSES], transformations=transformations)

        # Create global explanation without local importance values
        global_raw_explanation = exp.explain_global(x_test, include_local=False)
        num_raw_feats = len(iris[DatasetConstants.FEATURES])
        num_engineered_feats = num_raw_feats - 1
        feature_map = np.eye(num_raw_feats, num_engineered_feats)
        self.validate_global_raw_explanation_classification(global_raw_explanation, feature_map,
                                                            iris[DatasetConstants.CLASSES], feature_names,
                                                            has_raw_eval_data=True)

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
        self.validate_global_explanation_classification(global_explanation, global_raw_explanation, feature_map,
                                                        classes, feature_names)

    def test_get_local_raw_explanations_sparse_binary_classification(self, mimic_explainer):
        x_train, x_test, y_train, _, classes, _ = create_binary_sparse_newsgroups_data()
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
        self.validate_global_explanation_classification(global_explanation, global_raw_explanation, feature_map,
                                                        classes, feature_names)

    def test_get_global_raw_explanations_classification_eval_data(self, iris, tabular_explainer):
        model = create_sklearn_svm_classifier(iris[DatasetConstants.X_TRAIN], iris[DatasetConstants.Y_TRAIN])

        exp = tabular_explainer(model, iris[DatasetConstants.X_TRAIN], features=iris[DatasetConstants.FEATURES],
                                classes=iris[DatasetConstants.CLASSES])

        global_explanation = exp.explain_global(iris[DatasetConstants.X_TEST])
        assert not global_explanation.is_raw
        assert not global_explanation.is_engineered
        num_engineered_feats = len(iris[DatasetConstants.FEATURES])

        feature_map = np.eye(num_engineered_feats - 1, num_engineered_feats)
        feature_names = [str(i) for i in range(feature_map.shape[0])]

        raw_eval_data = np.ones_like(iris[DatasetConstants.X_TRAIN])
        global_raw_explanation = global_explanation.get_raw_explanation(
            [feature_map],
            raw_feature_names=feature_names[:feature_map.shape[0]],
            eval_data=raw_eval_data)

        assert np.array_equal(raw_eval_data, global_raw_explanation.eval_data)

        self.validate_global_explanation_classification(global_explanation, global_raw_explanation, feature_map,
                                                        iris[DatasetConstants.CLASSES], feature_names,
                                                        has_raw_eval_data=True)

    def test_get_global_raw_explanations_regression_eval_data(self, housing, tabular_explainer):
        model = create_sklearn_random_forest_regressor(housing[DatasetConstants.X_TRAIN],
                                                       housing[DatasetConstants.Y_TRAIN])

        exp = tabular_explainer(model, housing[DatasetConstants.X_TRAIN], features=housing[DatasetConstants.FEATURES])

        global_explanation = exp.explain_global(housing[DatasetConstants.X_TEST])
        assert not global_explanation.is_raw
        assert not global_explanation.is_engineered
        num_engineered_feats = len(housing[DatasetConstants.FEATURES])
        feature_map = np.eye(num_engineered_feats - 1, num_engineered_feats)

        raw_eval_data = np.ones_like(housing[DatasetConstants.X_TRAIN])
        global_raw_explanation = global_explanation.get_raw_explanation(
            [feature_map],
            eval_data=raw_eval_data)

        assert np.array_equal(raw_eval_data, global_raw_explanation.eval_data)

        self.validate_global_explanation_regression(global_explanation, global_raw_explanation, feature_map,
                                                    has_raw_eval_data=True)

    def test_get_raw_explanation_no_datasets_mixin(self, housing, mimic_explainer):
        model = create_sklearn_random_forest_regressor(housing[DatasetConstants.X_TRAIN],
                                                       housing[DatasetConstants.Y_TRAIN])

        explainer = mimic_explainer(model, housing[DatasetConstants.X_TRAIN], LGBMExplainableModel)
        global_explanation = explainer.explain_global(housing[DatasetConstants.X_TEST])
        assert global_explanation.method == LIGHTGBM_METHOD

        kwargs = {ExplainParams.METHOD: global_explanation.method}
        kwargs[ExplainParams.FEATURES] = global_explanation.features
        kwargs[ExplainParams.MODEL_TASK] = ExplainType.REGRESSION
        kwargs[ExplainParams.LOCAL_IMPORTANCE_VALUES] = global_explanation._local_importance_values
        kwargs[ExplainParams.EXPECTED_VALUES] = 0
        kwargs[ExplainParams.CLASSIFICATION] = False
        kwargs[ExplainParams.IS_ENG] = True
        synthetic_explanation = _create_local_explanation(**kwargs)

        num_engineered_feats = housing[DatasetConstants.X_TRAIN].shape[1]
        feature_map = np.eye(5, num_engineered_feats)
        feature_names = [str(i) for i in range(feature_map.shape[0])]
        raw_names = feature_names[:feature_map.shape[0]]
        assert not _DatasetsMixin._does_quack(synthetic_explanation)
        global_raw_explanation = synthetic_explanation.get_raw_explanation([feature_map],
                                                                           raw_feature_names=raw_names)
        self.validate_local_explanation_regression(synthetic_explanation,
                                                   global_raw_explanation,
                                                   feature_map,
                                                   has_eng_eval_data=False,
                                                   has_raw_eval_data=False,
                                                   has_dataset_data=False)

    def validate_global_explanation_regression(self, eng_explanation, raw_explanation, feature_map,
                                               has_eng_eval_data=True, has_raw_eval_data=False):
        self.validate_local_explanation_regression(eng_explanation,
                                                   raw_explanation,
                                                   feature_map,
                                                   has_eng_eval_data,
                                                   has_raw_eval_data)
        assert np.array(raw_explanation.global_importance_values).shape[-1] == feature_map.shape[0]

    def validate_local_explanation_regression(self, eng_explanation, raw_explanation, feature_map,
                                              has_eng_eval_data=True, has_raw_eval_data=False,
                                              has_dataset_data=True):
        assert not eng_explanation.is_raw
        assert hasattr(eng_explanation, 'eval_data') == has_eng_eval_data
        assert eng_explanation.is_engineered

        assert np.array(raw_explanation.local_importance_values).shape[-1] == feature_map.shape[0]

        assert raw_explanation.is_raw
        assert not raw_explanation.is_engineered

        if has_dataset_data:
            # Test the y_pred and y_pred_proba on the raw explanations
            assert raw_explanation.eval_y_predicted is not None
            assert raw_explanation.eval_y_predicted_proba is None

            # Test the raw data on the raw explanations
            assert hasattr(raw_explanation, 'eval_data')
            assert (raw_explanation.eval_data is not None) == has_raw_eval_data

    def validate_global_explanation_classification(self, eng_explanation, raw_explanation,
                                                   feature_map, classes, feature_names,
                                                   has_eng_eval_data=True, has_raw_eval_data=False):
        assert not eng_explanation.is_raw
        assert hasattr(eng_explanation, 'eval_data') == has_eng_eval_data
        assert eng_explanation.is_engineered

        assert raw_explanation.expected_values == eng_explanation.expected_values

        feat_imps_global_local = np.array(raw_explanation.local_importance_values)
        assert feat_imps_global_local.shape[-1] == feature_map.shape[0]

        # Validate feature importances on the raw explanation are consistent
        # for global and local case when taking abs mean
        local_imp_values = raw_explanation.local_importance_values
        global_imp_values = np.mean(np.mean(np.absolute(local_imp_values), axis=1), axis=0)
        assert np.array_equal(raw_explanation.global_importance_values, global_imp_values)

        self.validate_global_raw_explanation_classification(
            raw_explanation, feature_map, classes, feature_names,
            has_raw_eval_data=has_raw_eval_data)

    def validate_global_raw_explanation_classification(self, raw_explanation,
                                                       feature_map, classes, feature_names,
                                                       has_raw_eval_data=False):
        per_class_values = raw_explanation.get_ranked_per_class_values()
        assert len(per_class_values) == len(classes)
        assert len(per_class_values[0]) == feature_map.shape[0]
        assert len(raw_explanation.get_ranked_per_class_names()[0]) == feature_map.shape[0]
        assert raw_explanation.is_raw
        assert not raw_explanation.is_engineered
        assert len(raw_explanation.get_ranked_global_values()) == feature_map.shape[0]
        assert len(raw_explanation.get_ranked_global_names()) == feature_map.shape[0]

        assert raw_explanation.features == feature_names

        if isinstance(classes, list):
            assert raw_explanation.classes == classes
        else:
            assert (raw_explanation.classes == classes).all()

        feat_imps_global = np.array(raw_explanation.global_importance_values)

        assert feat_imps_global.shape[-1] == feature_map.shape[0]

        # Test the y_pred and y_pred_proba on the raw explanations
        assert raw_explanation.eval_y_predicted is not None
        assert raw_explanation.eval_y_predicted_proba is not None

        # Test the raw data on the raw explanations
        assert hasattr(raw_explanation, 'eval_data')
        assert (raw_explanation.eval_data is not None) == has_raw_eval_data
