# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import logging

# Tests for model explainability SDK
import numpy as np
import pandas as pd
import pytest
import shap
from common_utils import (create_keras_classifier, create_keras_regressor,
                          create_lightgbm_classifier, create_msx_data,
                          create_pytorch_classifier, create_pytorch_regressor,
                          create_pytorch_single_output_classifier,
                          create_scikit_keras_classifier,
                          create_scikit_keras_regressor,
                          create_sklearn_linear_regressor,
                          create_sklearn_random_forest_classifier,
                          create_sklearn_random_forest_regressor,
                          create_sklearn_svm_classifier, create_tf_model,
                          create_xgboost_classifier,
                          wrap_classifier_without_proba)
from constants import DatasetConstants, owner_email_tools_and_ux
from datasets import retrieve_dataset
from interpret_community.common.constants import InterpretData, ModelTask
from interpret_community.common.policy import SamplingPolicy
from interpret_community.shap import (DeepExplainer, GPUKernelExplainer,
                                      KernelExplainer, LinearExplainer,
                                      TreeExplainer)
from interpret_community.tabular_explainer import _get_uninitialized_explainers
from lightgbm import LGBMClassifier
from ml_wrappers import DatasetWrapper, wrap_model
from raw_explain.utils import _get_feature_map_from_indices_list
from scipy.sparse import csr_matrix
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (FunctionTransformer, OneHotEncoder,
                                   StandardScaler)

try:
    import tensorflow as tf
except ImportError:
    pass


test_logger = logging.getLogger(__name__)
test_logger.setLevel(logging.DEBUG)


DATA_SLICE = slice(10)


@pytest.mark.owner(email=owner_email_tools_and_ux)
@pytest.mark.usefixtures('_clean_dir')
class TestTabularExplainer(object):
    def test_working(self):
        assert True

    def test_pandas_with_feature_names(self, iris, tabular_explainer, verify_tabular):
        # create pandas dataframes
        features = iris[DatasetConstants.FEATURES]
        x_train = pd.DataFrame(data=iris[DatasetConstants.X_TRAIN], columns=features)
        x_test = pd.DataFrame(data=iris[DatasetConstants.X_TEST], columns=features)
        # Fit an SVM model
        model = create_sklearn_svm_classifier(x_train, iris[DatasetConstants.Y_TRAIN])

        exp = tabular_explainer(model,
                                x_train,
                                features=features,
                                classes=iris[DatasetConstants.CLASSES])
        test_logger.info("Running explain global for test_pandas_with_feature_names")
        explanation = exp.explain_global(x_test)
        ranked_global_values = explanation.get_ranked_global_values()
        ranked_per_class_values = explanation.get_ranked_per_class_values()
        ranked_global_names = explanation.get_ranked_global_names()
        ranked_per_class_names = explanation.get_ranked_per_class_names()

        self.verify_iris_overall_features(ranked_global_names, ranked_global_values, verify_tabular)
        self.verify_iris_per_class_features(ranked_per_class_names, ranked_per_class_values)

        global_data = explanation.data()
        self.verify_global_mli_data(global_data, explanation, features, is_full=False)
        full_data = explanation.data(key=-1)
        self.verify_global_mli_data(full_data, explanation, features, is_full=True)

    def test_pandas_no_feature_names(self, iris, tabular_explainer, verify_tabular):
        # create pandas dataframes
        x_train = pd.DataFrame(data=iris[DatasetConstants.X_TRAIN], columns=iris[DatasetConstants.FEATURES])
        x_test = pd.DataFrame(data=iris[DatasetConstants.X_TEST], columns=iris[DatasetConstants.FEATURES])
        # Fit an SVM model
        model = create_sklearn_svm_classifier(x_train, iris[DatasetConstants.Y_TRAIN])

        exp = tabular_explainer(model, x_train, classes=iris[DatasetConstants.CLASSES])
        test_logger.info("Running explain global for test_pandas_no_feature_names")
        explanation = exp.explain_global(x_test)
        ranked_global_values = explanation.get_ranked_global_values()
        ranked_per_class_values = explanation.get_ranked_per_class_values()
        ranked_global_names = explanation.get_ranked_global_names()
        ranked_per_class_names = explanation.get_ranked_per_class_names()

        self.verify_iris_overall_features(ranked_global_names, ranked_global_values, verify_tabular)
        self.verify_iris_per_class_features(ranked_per_class_names, ranked_per_class_values)

    def test_explain_model_local(self, verify_tabular):
        iris_overall_expected_features = verify_tabular.iris_overall_expected_features
        iris_per_class_expected_features = self.iris_per_class_expected_features
        verify_tabular.verify_explain_model_local(iris_overall_expected_features,
                                                  iris_per_class_expected_features)

    @pytest.mark.skip(reason="failing in deep explainer after shap upgrade")
    def test_explain_model_local_dnn(self, verify_tabular):
        verify_tabular.verify_explain_model_local_dnn()

    def test_explain_model_local_without_include_local(self, verify_tabular):
        iris_overall_expected_features = verify_tabular.iris_overall_expected_features
        iris_per_class_expected_features = self.iris_per_class_expected_features
        verify_tabular.verify_explain_model_local(iris_overall_expected_features,
                                                  iris_per_class_expected_features,
                                                  include_local=False)

    def test_explain_model_local_regression_without_include_local(self, verify_tabular):
        verify_tabular.verify_explain_model_local_regression(include_local=False)

    @pytest.mark.skip(reason="latest tensorflow version no longer works with shap deep explainer")
    def test_explain_model_local_regression_dnn(self, verify_tabular):
        verify_tabular.verify_explain_model_local_regression_dnn()

    def test_explanation_get_feature_importance_dict(self, iris, tabular_explainer):
        x_train = pd.DataFrame(data=iris[DatasetConstants.X_TRAIN], columns=iris[DatasetConstants.FEATURES])
        x_test = pd.DataFrame(data=iris[DatasetConstants.X_TEST], columns=iris[DatasetConstants.FEATURES])
        # Fit an SVM model
        model = create_sklearn_svm_classifier(x_train, iris[DatasetConstants.Y_TRAIN])

        exp = tabular_explainer(model, x_train, classes=iris[DatasetConstants.CLASSES])
        test_logger.info("Running explain global for test_explanation_get_feature_importance_dict")
        explanation = exp.explain_global(x_test)
        ranked_names = explanation.get_ranked_global_names()
        ranked_values = explanation.get_ranked_global_values()
        ranked_dict = explanation.get_feature_importance_dict()
        assert len(ranked_dict) == len(ranked_values)
        # Order isn't guaranteed for a python dictionary, but this has seemed to hold empirically
        assert ranked_names == list(ranked_dict.keys())

    def test_explain_single_local_instance_classification(self, iris, tabular_explainer):
        # Fit an SVM model
        model = create_sklearn_svm_classifier(iris[DatasetConstants.X_TRAIN], iris[DatasetConstants.Y_TRAIN])
        features = iris[DatasetConstants.FEATURES]
        classes = iris[DatasetConstants.CLASSES]
        exp = tabular_explainer(model, iris[DatasetConstants.X_TRAIN], features=features,
                                classes=classes)

        local_explanation = exp.explain_local(iris[DatasetConstants.X_TEST][0])

        assert len(local_explanation.local_importance_values) == len(classes)
        assert local_explanation.num_classes == len(classes)
        assert len(local_explanation.local_importance_values[0]) == len(features)
        assert local_explanation.num_features == len(features)

        local_rank = local_explanation.get_local_importance_rank()
        assert len(local_rank) == len(classes)
        assert len(local_rank[0]) == len(features)

        ranked_names = local_explanation.get_ranked_local_names()
        assert len(ranked_names) == len(classes)
        assert len(ranked_names[0]) == len(features)

        ranked_values = local_explanation.get_ranked_local_values()
        assert len(ranked_values) == len(classes)
        assert len(ranked_values[0]) == len(features)

        data = local_explanation.data()
        self.verify_local_mli_data(data)

    def test_explain_multi_local_instance_classification(self, iris, tabular_explainer):
        # Fit an SVM model
        model = create_sklearn_svm_classifier(iris[DatasetConstants.X_TRAIN], iris[DatasetConstants.Y_TRAIN])
        features = iris[DatasetConstants.FEATURES]
        classes = iris[DatasetConstants.CLASSES]
        exp = tabular_explainer(model, iris[DatasetConstants.X_TRAIN], features=features,
                                classes=classes)

        local_explanation = exp.explain_local(iris[DatasetConstants.X_TEST])

        assert len(local_explanation.local_importance_values) == len(classes)
        assert local_explanation.num_classes == len(classes)
        assert len(local_explanation.local_importance_values[0]) == len(iris[DatasetConstants.X_TEST])
        assert local_explanation.num_examples == len(iris[DatasetConstants.X_TEST])
        assert len(local_explanation.local_importance_values[0][0]) == len(features)
        assert local_explanation.num_features == len(features)

        local_rank = local_explanation.get_local_importance_rank()
        assert len(local_rank) == len(classes)
        assert len(local_rank[0]) == len(iris[DatasetConstants.X_TEST])
        assert len(local_rank[0][0]) == len(features)

        ranked_names = local_explanation.get_ranked_local_names()
        assert len(ranked_names) == len(classes)
        assert len(ranked_names[0]) == len(iris[DatasetConstants.X_TEST])
        assert len(ranked_names[0][0]) == len(features)

        ranked_values = local_explanation.get_ranked_local_values()
        assert len(ranked_values) == len(classes)
        assert len(ranked_values[0]) == len(iris[DatasetConstants.X_TEST])
        assert len(ranked_values[0][0]) == len(features)

        data = local_explanation.data()
        self.verify_local_mli_data(data)

    def test_explain_single_local_instance_regression(self, housing, tabular_explainer):
        # Fit an SVM model
        model = create_sklearn_random_forest_regressor(housing[DatasetConstants.X_TRAIN],
                                                       housing[DatasetConstants.Y_TRAIN])

        exp = tabular_explainer(model, housing[DatasetConstants.X_TRAIN], features=housing[DatasetConstants.FEATURES])

        local_explanation = exp.explain_local(housing[DatasetConstants.X_TEST][0])

        assert len(local_explanation.local_importance_values) == len(housing[DatasetConstants.FEATURES])
        assert local_explanation.num_features == len(housing[DatasetConstants.FEATURES])

        local_rank = local_explanation.get_local_importance_rank()
        assert len(local_rank) == len(housing[DatasetConstants.FEATURES])

        ranked_names = local_explanation.get_ranked_local_names()
        assert len(ranked_names) == len(housing[DatasetConstants.FEATURES])

        ranked_values = local_explanation.get_ranked_local_values()
        assert len(ranked_values) == len(housing[DatasetConstants.FEATURES])

    def test_explain_model_pandas_input(self, verify_tabular):
        verify_tabular.verify_explain_model_pandas_input()

    # TODO change these to actual local tests
    def test_explain_model_local_pandas(self, iris, tabular_explainer, verify_tabular):
        # create pandas dataframes
        x_train = pd.DataFrame(data=iris[DatasetConstants.X_TRAIN], columns=iris[DatasetConstants.FEATURES])
        x_test = pd.DataFrame(data=iris[DatasetConstants.X_TEST], columns=iris[DatasetConstants.FEATURES])
        # Fit an SVM model
        model = create_sklearn_svm_classifier(x_train, iris[DatasetConstants.Y_TRAIN])

        # Create tabular explainer
        exp = tabular_explainer(model,
                                x_train,
                                features=iris[DatasetConstants.FEATURES],
                                classes=iris[DatasetConstants.CLASSES])
        test_logger.info("Running explain global for test_explain_model_local_pandas")
        explanation = exp.explain_global(x_test)
        ranked_global_values = explanation.get_ranked_global_values()
        ranked_global_names = explanation.get_ranked_global_names()
        ranked_per_class_values = explanation.get_ranked_per_class_values()
        ranked_per_class_names = explanation.get_ranked_per_class_names()
        self.verify_iris_overall_features(ranked_global_names, ranked_global_values, verify_tabular)
        self.verify_iris_per_class_features(ranked_per_class_names, ranked_per_class_values)

    # TODO change these to actual local tests
    def test_explain_model_local_pandas_no_feature_names(self, iris, tabular_explainer, verify_tabular):
        # create pandas dataframes
        x_train = pd.DataFrame(data=iris[DatasetConstants.X_TRAIN], columns=iris[DatasetConstants.FEATURES])
        x_test = pd.DataFrame(data=iris[DatasetConstants.X_TEST], columns=iris[DatasetConstants.FEATURES])
        # Fit an SVM model
        model = create_sklearn_svm_classifier(x_train, iris[DatasetConstants.Y_TRAIN])

        # Create tabular explainer
        exp = tabular_explainer(model, x_train, classes=iris[DatasetConstants.CLASSES])
        test_logger.info("Running explain global for test_explain_model_local_pandas_no_feature_names")
        explanation = exp.explain_global(x_test)
        ranked_global_values = explanation.get_ranked_global_values()
        ranked_global_names = explanation.get_ranked_global_names()
        ranked_per_class_values = explanation.get_ranked_per_class_values()
        ranked_per_class_names = explanation.get_ranked_per_class_names()
        self.verify_iris_overall_features(ranked_global_names, ranked_global_values, verify_tabular)
        self.verify_iris_per_class_features(ranked_per_class_names, ranked_per_class_values)

    def test_explain_model_local_no_feature_names(self, iris, tabular_explainer, verify_tabular):
        # Fit an SVM model
        model = create_sklearn_svm_classifier(iris[DatasetConstants.X_TRAIN], iris[DatasetConstants.Y_TRAIN])

        # Create tabular explainer
        exp = tabular_explainer(model, iris[DatasetConstants.X_TRAIN], classes=iris[DatasetConstants.CLASSES])
        test_logger.info("Running explain global for test_explain_model_local_no_feature_names")
        explanation = exp.explain_global(iris[DatasetConstants.X_TEST])
        ranked_global_values = explanation.get_ranked_global_values()
        ranked_global_names = explanation.get_ranked_global_names()
        ranked_per_class_values = explanation.get_ranked_per_class_values()
        ranked_per_class_names = explanation.get_ranked_per_class_names()
        self.verify_iris_overall_features_no_names(ranked_global_names, ranked_global_values)
        self.verify_iris_per_class_features_no_names(ranked_per_class_names, ranked_per_class_values)

    def test_explain_model_npz_linear(self, verify_tabular):
        verify_tabular.verify_explain_model_npz_linear()

    def test_explain_model_npz_tree(self, tabular_explainer):
        # run explain global on a real sparse dataset from the field
        x_train, x_test, y_train, _ = create_msx_data(0.1)
        x_train = x_train[DATA_SLICE]
        x_test = x_test[DATA_SLICE]
        y_train = y_train[DATA_SLICE]
        # Fit a random forest regression model
        model = create_sklearn_random_forest_regressor(x_train, y_train.toarray().flatten())
        # Create tabular explainer
        exp = tabular_explainer(model, x_train)
        test_logger.info('Running explain global for test_explain_model_npz_tree')
        exp.explain_global(x_test)

    def test_explain_model_sparse(self, verify_tabular):
        verify_tabular.verify_explain_model_sparse()

    def test_explain_model_sparse_tree(self, tabular_explainer):
        X, y = retrieve_dataset('a1a.svmlight')
        x_train, x_test, y_train, _ = train_test_split(X, y, test_size=0.002, random_state=7)
        # Fit a random forest regression model
        model = create_sklearn_random_forest_regressor(x_train, y_train)
        _, cols = x_train.shape
        shape = 1, cols
        background = csr_matrix(shape, dtype=x_train.dtype)

        # Create tabular explainer
        exp = tabular_explainer(model, background)
        test_logger.info('Running explain global for test_explain_model_sparse_tree')
        policy = SamplingPolicy(allow_eval_sampling=True)
        exp.explain_global(x_test, sampling_policy=policy)

    def test_explain_model_hashing(self, verify_tabular):
        verify_tabular.verify_explain_model_hashing()

    def test_explain_model_with_summarize_data(self, verify_tabular):
        iris_overall_expected_features = verify_tabular.iris_overall_expected_features
        iris_per_class_expected_features = self.iris_per_class_expected_features
        verify_tabular.verify_explain_model_with_summarize_data(iris_overall_expected_features,
                                                                iris_per_class_expected_features)

    def test_explain_model_random_forest_classification(self, tabular_explainer):
        X, y = shap.datasets.adult()
        x_train, x_test, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=7)
        # Fit a tree model
        model = create_sklearn_random_forest_classifier(x_train, y_train)

        # Create tabular explainer
        exp = tabular_explainer(model, x_train, features=X.columns.values)
        test_logger.info('Running explain global for test_explain_model_random_forest_classification')
        explanation = exp.explain_global(x_test)
        self.verify_adult_overall_features(explanation.get_ranked_global_names(),
                                           explanation.get_ranked_global_values())
        self.verify_adult_per_class_features(explanation.get_ranked_per_class_names(),
                                             explanation.get_ranked_per_class_values())
        self.verify_top_rows_local_features_with_and_without_top_k(explanation,
                                                                   self.adult_local_features_first_three_rf,
                                                                   is_classification=True, top_rows=3)

    def test_explain_model_lightgbm_multiclass(self, tabular_explainer, iris):
        # Fit a lightgbm model
        model = create_lightgbm_classifier(iris[DatasetConstants.X_TRAIN], iris[DatasetConstants.Y_TRAIN])

        # Create tabular explainer
        exp = tabular_explainer(model, iris[DatasetConstants.X_TRAIN], features=iris[DatasetConstants.FEATURES],
                                classes=iris[DatasetConstants.CLASSES])
        test_logger.info('Running explain global for test_explain_model_lightgbm_multiclass')
        explanation = exp.explain_global(iris[DatasetConstants.X_TEST])
        assert len(explanation.local_importance_values[0]) == len(iris[DatasetConstants.X_TEST])
        assert explanation.num_examples == len(iris[DatasetConstants.X_TEST])
        assert len(explanation.local_importance_values) == len(iris[DatasetConstants.CLASSES])
        assert explanation.num_classes == len(iris[DatasetConstants.CLASSES])

    def test_explain_model_xgboost_multiclass(self, tabular_explainer, iris):
        # Fit an xgboost model
        model = create_xgboost_classifier(iris[DatasetConstants.X_TRAIN], iris[DatasetConstants.Y_TRAIN])

        # Create tabular explainer
        exp = tabular_explainer(model, iris[DatasetConstants.X_TRAIN], features=iris[DatasetConstants.FEATURES],
                                classes=iris[DatasetConstants.CLASSES])
        test_logger.info('Running explain global for test_explain_model_xgboost_multiclass')
        explanation = exp.explain_global(iris[DatasetConstants.X_TEST])
        assert len(explanation.local_importance_values[0]) == len(iris[DatasetConstants.X_TEST])
        assert explanation.num_examples == len(iris[DatasetConstants.X_TEST])
        assert len(explanation.local_importance_values) == len(iris[DatasetConstants.CLASSES])
        assert explanation.num_classes == len(iris[DatasetConstants.CLASSES])

    def _validate_binary_explain_model(self, tabular_explainer, create_model, test_name):
        X, y = shap.datasets.adult()
        x_train, x_test, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=7)
        model = create_model(x_train, y_train)
        classes = ["<50k", ">50k"]
        # Create tabular explainer
        exp = tabular_explainer(model, x_train, features=X.columns.values,
                                classes=classes)
        exp_glob_info_message = 'Running explain global for {}'.format(test_name)
        test_logger.info(exp_glob_info_message)
        explanation = exp.explain_global(x_test)
        assert len(explanation.local_importance_values[0]) == len(x_test)
        assert len(explanation.local_importance_values) == len(classes)

    def test_explain_model_xgboost_binary(self, tabular_explainer):
        # Fit an xgboost tree model
        def create_model(x_train, y_train):
            return create_xgboost_classifier(x_train, y_train)
        self._validate_binary_explain_model(tabular_explainer, create_model, 'test_explain_model_xgboost_binary')

    def test_explain_model_lightgbm_binary(self, tabular_explainer):
        # Fit a lightgbm tree model
        def create_model(x_train, y_train):
            return create_lightgbm_classifier(x_train, y_train)
        self._validate_binary_explain_model(tabular_explainer, create_model, 'test_explain_model_lightgbm_binary')

    def _explain_model_dnn_common(self, tabular_explainer, model, x_train, x_test, y_train, features):
        # Create tabular explainer
        exp = tabular_explainer(model, x_train.values, features=features, model_task=ModelTask.Classification)
        policy = SamplingPolicy(allow_eval_sampling=True)
        exp.explain_global(x_test.values, sampling_policy=policy)

    @pytest.mark.skip(reason="latest tensorflow version no longer works with shap deep explainer")
    def test_explain_model_keras(self, tabular_explainer):
        X, y = shap.datasets.adult()
        x_train, x_test, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=7)
        # Fit a keras DNN model
        model = create_keras_classifier(x_train.values, y_train)
        test_logger.info('Running explain global for test_explain_model_keras')
        self._explain_model_dnn_common(tabular_explainer, model, x_train, x_test, y_train, X.columns.values)

    def test_explain_model_pytorch(self, tabular_explainer):
        X, y = shap.datasets.adult()
        x_train, x_test, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=7)
        # Fit a pytorch DNN model
        model = create_pytorch_classifier(x_train.values, y_train)
        test_logger.info('Running explain global for test_explain_model_pytorch')
        self._explain_model_dnn_common(tabular_explainer, model, x_train, x_test, y_train, X.columns.values)

    def test_explain_model_pytorch_binary_single_output(self, tabular_explainer):
        X, y = shap.datasets.adult()
        x_train, x_test, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=7)
        # Fit a pytorch DNN model
        model = create_pytorch_single_output_classifier(x_train.values, y_train)
        test_logger.info('Running explain global for test_explain_model_pytorch_binary_single_output')
        self._explain_model_dnn_common(tabular_explainer, model, x_train, x_test, y_train, X.columns.values)

    def test_explain_model_random_forest_regression(self, housing, tabular_explainer):
        # Fit a random forest regression model
        model = create_sklearn_random_forest_regressor(housing[DatasetConstants.X_TRAIN],
                                                       housing[DatasetConstants.Y_TRAIN])

        # Create tabular explainer
        exp = tabular_explainer(model, housing[DatasetConstants.X_TRAIN], features=housing[DatasetConstants.FEATURES])
        test_logger.info('Running explain global for test_explain_model_random_forest_regression')
        explanation = exp.explain_global(housing[DatasetConstants.X_TEST])
        self.verify_housing_overall_features_rf(explanation.get_ranked_global_names(),
                                                explanation.get_ranked_global_values())

    def test_explain_model_local_tree_regression(self, housing, tabular_explainer):
        # Fit a random forest regression model
        model = create_sklearn_random_forest_regressor(housing[DatasetConstants.X_TRAIN],
                                                       housing[DatasetConstants.Y_TRAIN])

        # Create tabular explainer
        exp = tabular_explainer(model, housing[DatasetConstants.X_TRAIN], features=housing[DatasetConstants.FEATURES])
        test_logger.info('Running explain local for test_explain_model_local_tree_regression')
        explanation = exp.explain_local(housing[DatasetConstants.X_TEST])
        assert explanation.local_importance_values is not None
        assert len(explanation.local_importance_values) == len(housing[DatasetConstants.X_TEST])
        assert explanation.num_examples == len(housing[DatasetConstants.X_TEST])
        assert len(explanation.local_importance_values[0]) == len(housing[DatasetConstants.FEATURES])
        assert explanation.num_features == len(housing[DatasetConstants.FEATURES])
        self.verify_top_rows_local_features_with_and_without_top_k(explanation,
                                                                   self.housing_local_features_first_five_rf)

    def _explain_model_local_dnn_classification_common(self, tabular_explainer, model, x_train,
                                                       x_test, y_train, features):
        # Create tabular explainer
        exp = tabular_explainer(model, x_train.values, features=features, model_task=ModelTask.Classification)
        explanation = exp.explain_local(x_test.values)
        assert explanation.local_importance_values is not None
        assert len(explanation.local_importance_values[0]) == len(x_test.values)
        assert explanation.num_examples == len(x_test.values)
        assert len(explanation.local_importance_values[0][0]) == len(features)
        assert explanation.num_features == len(features)

    @pytest.mark.skip(reason="latest tensorflow version no longer works with shap deep explainer")
    def test_explain_model_local_keras_classification(self, tabular_explainer):
        X, y = shap.datasets.adult()
        x_train, x_test, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=7)
        # Fit a DNN keras model
        model = create_keras_classifier(x_train.values, y_train)
        test_logger.info('Running explain local for test_explain_model_local_keras_classification')
        self._explain_model_local_dnn_classification_common(tabular_explainer, model, x_train,
                                                            x_test, y_train, X.columns.values)

    def test_explain_model_local_pytorch_classification(self, tabular_explainer):
        X, y = shap.datasets.adult()
        x_train, x_test, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=7)
        # Fit a DNN pytorch model
        model = create_pytorch_classifier(x_train.values, y_train)
        test_logger.info('Running explain local for test_explain_model_local_pytorch_classification')
        self._explain_model_local_dnn_classification_common(tabular_explainer, model, x_train,
                                                            x_test, y_train, X.columns.values)

    def _explain_model_local_dnn_regression_common(self, tabular_explainer, model, x_train,
                                                   x_test, y_train, features):
        # Create tabular explainer
        exp = tabular_explainer(model, x_train, features=features, model_task=ModelTask.Regression)
        explanation = exp.explain_local(x_test)
        assert explanation.local_importance_values is not None
        assert len(explanation.local_importance_values) == len(x_test)
        assert explanation.num_examples == len(x_test)
        assert len(explanation.local_importance_values[0]) == len(features)
        assert explanation.num_features == len(features)

    @pytest.mark.skip(reason="latest tensorflow version no longer works with shap deep explainer")
    def test_explain_model_local_keras_regression(self, housing, tabular_explainer):
        x_train = housing[DatasetConstants.X_TRAIN]
        x_test = housing[DatasetConstants.X_TEST]
        # Fit a DNN keras model
        model = create_keras_regressor(x_train, housing[DatasetConstants.Y_TRAIN])
        test_logger.info('Running explain local for test_explain_model_local_keras_regression')
        self._explain_model_local_dnn_regression_common(tabular_explainer, model, x_train,
                                                        x_test, housing[DatasetConstants.Y_TRAIN],
                                                        housing[DatasetConstants.FEATURES])

    def test_explain_model_local_pytorch_regression(self, housing, tabular_explainer):
        x_train = housing[DatasetConstants.X_TRAIN]
        x_test = housing[DatasetConstants.X_TEST]
        # Fit a DNN pytorch model
        model = create_pytorch_regressor(x_train, housing[DatasetConstants.Y_TRAIN])
        test_logger.info('Running explain local for test_explain_model_local_pytorch_regression')
        self._explain_model_local_dnn_regression_common(tabular_explainer, model, x_train,
                                                        x_test, housing[DatasetConstants.Y_TRAIN],
                                                        housing[DatasetConstants.FEATURES])

    def test_explain_model_local_kernel_regression(self, housing, tabular_explainer):
        # Fit a linear regression model
        model = create_sklearn_linear_regressor(housing[DatasetConstants.X_TRAIN], housing[DatasetConstants.Y_TRAIN])

        # Create tabular explainer
        exp = tabular_explainer(model, housing[DatasetConstants.X_TRAIN], features=housing[DatasetConstants.FEATURES])
        test_logger.info('Running explain local for test_explain_model_local_kernel_regression')
        explanation = exp.explain_local(housing[DatasetConstants.X_TEST])
        assert explanation.local_importance_values is not None
        assert len(explanation.local_importance_values) == len(housing[DatasetConstants.X_TEST])
        assert explanation.num_examples == len(housing[DatasetConstants.X_TEST])
        assert len(explanation.local_importance_values[0]) == len(housing[DatasetConstants.FEATURES])
        assert explanation.num_features == len(housing[DatasetConstants.FEATURES])
        self.verify_top_rows_local_features_with_and_without_top_k(explanation,
                                                                   self.housing_local_features_first_five_lr)

    def test_explain_model_linear_regression(self, housing, tabular_explainer):
        # Fit a linear regression model
        model = create_sklearn_linear_regressor(housing[DatasetConstants.X_TRAIN],
                                                housing[DatasetConstants.Y_TRAIN],
                                                pipeline=True)

        # Create tabular explainer
        exp = tabular_explainer(model, housing[DatasetConstants.X_TRAIN], features=housing[DatasetConstants.FEATURES])
        test_logger.info('Running explain global for test_explain_model_linear_regression')
        explanation = exp.explain_global(housing[DatasetConstants.X_TEST])
        self.verify_housing_overall_features_lr(explanation.get_ranked_global_names(),
                                                explanation.get_ranked_global_values())

    def test_explain_model_subset_classification_dense(self, verify_tabular):
        verify_tabular.verify_explain_model_subset_classification_dense()

    def test_explain_model_subset_regression_sparse(self, verify_tabular):
        verify_tabular.verify_explain_model_subset_regression_sparse()

    def test_explain_model_subset_classification_sparse(self, verify_tabular):
        verify_tabular.verify_explain_model_subset_classification_sparse()

    def test_explain_model_with_sampling_regression_sparse(self, verify_tabular):
        verify_tabular.verify_explain_model_with_sampling_regression_sparse()

    def test_explain_raw_feats_regression(self, housing, tabular_explainer):
        # verify that no errors get thrown when calling get_raw_feat_importances
        x_train = housing[DatasetConstants.X_TRAIN][DATA_SLICE]
        x_test = housing[DatasetConstants.X_TEST][DATA_SLICE]
        y_train = housing[DatasetConstants.Y_TRAIN][DATA_SLICE]

        model = create_sklearn_linear_regressor(x_train, y_train)

        explainer = tabular_explainer(model, x_train)

        global_explanation = explainer.explain_global(x_test)
        local_explanation = explainer.explain_local(x_test)
        # 0th raw feature maps to 1 and 3 generated features, 1st raw feature maps to 0th and 2nd gen. features
        raw_feat_indices = [[1, 3], [0, 2]]
        num_generated_cols = x_train.shape[1]
        feature_map = _get_feature_map_from_indices_list(raw_feat_indices, num_raw_cols=2,
                                                         num_generated_cols=num_generated_cols)
        global_raw_importances = global_explanation.get_raw_feature_importances([feature_map])
        assert len(global_raw_importances) == len(raw_feat_indices), ('length of global importances '
                                                                      'does not match number of features')
        local_raw_importances = local_explanation.get_raw_feature_importances([feature_map])
        assert len(local_raw_importances) == x_test.shape[0], ('length of local importances does not match number '
                                                               'of samples')

    def test_explain_raw_feats_classification(self, iris, tabular_explainer):
        # verify that no errors get thrown when calling get_raw_feat_importances
        x_train = iris[DatasetConstants.X_TRAIN]
        x_test = iris[DatasetConstants.X_TEST]
        y_train = iris[DatasetConstants.Y_TRAIN]

        model = create_sklearn_random_forest_classifier(x_train, y_train)

        explainer = tabular_explainer(model, x_train)
        global_explanation = explainer.explain_global(x_test)
        local_explanation = explainer.explain_local(x_test)
        raw_feat_indices = [[1, 3], [0, 2]]
        num_generated_cols = x_train.shape[1]
        # Create a feature map for only two features
        feature_map = _get_feature_map_from_indices_list(raw_feat_indices, num_raw_cols=2,
                                                         num_generated_cols=num_generated_cols)
        global_raw_importances = global_explanation.get_raw_feature_importances([feature_map])
        assert len(global_raw_importances) == len(raw_feat_indices), \
            'length of global importances does not match number of features'
        local_raw_importances = local_explanation.get_raw_feature_importances([feature_map])
        assert len(local_raw_importances) == len(iris[DatasetConstants.CLASSES]), \
            'length of local importances does not match number of classes'

    def test_explain_raw_feats_titanic(self, tabular_explainer):
        titanic_url = ('https://raw.githubusercontent.com/amueller/'
                       'scipy-2017-sklearn/091d371/notebooks/datasets/titanic3.csv')
        data = pd.read_csv(titanic_url)
        # fill missing values
        data = data.fillna(method="ffill")
        data = data.fillna(method="bfill")
        numeric_features = ['age', 'fare']
        categorical_features = ['embarked', 'sex', 'pclass']
        y = data['survived'].values
        X = data[categorical_features + numeric_features]
        # Split data into train and test
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        def conv(X):
            if isinstance(X, pd.Series):
                return X.values
            return X

        many_to_one_transformer = FunctionTransformer(lambda x: conv(x.sum(axis=1)).reshape(-1, 1))
        many_to_many_transformer = FunctionTransformer(lambda x: np.hstack(
            (conv(np.prod(x, axis=1)).reshape(-1, 1), conv(np.prod(x, axis=1)**2).reshape(-1, 1))
        ))
        transformations = ColumnTransformer([
            ("age_fare_1", Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), ["age", "fare"]),
            ("age_fare_2", many_to_one_transformer, ["age", "fare"]),
            ("age_fare_3", many_to_many_transformer, ["age", "fare"]),
            ("embarked", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy='constant', fill_value='missing')),
                ("encoder", OneHotEncoder(sparse=False))]), ["embarked"]),
            ("sex_pclass", OneHotEncoder(sparse=False), ["sex", "pclass"])
        ])
        clf = Pipeline(steps=[('preprocessor', transformations),
                              ('classifier', LogisticRegression(solver='lbfgs'))])
        clf.fit(x_train, y_train)
        explainer = tabular_explainer(clf.steps[-1][1],
                                      initialization_examples=x_train,
                                      features=x_train.columns,
                                      transformations=transformations,
                                      allow_all_transformations=True)
        explainer.explain_global(x_test)
        explainer.explain_local(x_test)

    @pytest.mark.parametrize('drop', ['first', 'if_binary'])
    def test_explain_with_ohe_drop_column(self, tabular_explainer, drop):
        attritionData = retrieve_dataset('WA_Fn-UseC_-HR-Employee-Attrition.csv')
        # Dropping Employee count as all values are 1 and hence attrition is independent of this feature
        attritionData = attritionData.drop(['EmployeeCount'], axis=1)
        # Dropping Employee Number since it is merely an identifier
        attritionData = attritionData.drop(['EmployeeNumber'], axis=1)

        attritionData = attritionData.drop(['Over18'], axis=1)

        # Since all values are 80
        attritionData = attritionData.drop(['StandardHours'], axis=1)

        # Converting target variables from string to numerical values
        target_map = {'Yes': 1, 'No': 0}
        attritionData["Attrition_numerical"] = attritionData["Attrition"].apply(lambda x: target_map[x])
        target = attritionData["Attrition_numerical"]

        attritionXData = attritionData.drop(['Attrition_numerical', 'Attrition'], axis=1)

        # creating dummy columns for each categorical feature
        categorical = []
        for col, value in attritionXData.items():
            if value.dtype == 'object':
                categorical.append(col)

        # store the numerical columns
        numerical = attritionXData.columns.difference(categorical)

        # We create the preprocessing pipelines for both numeric and categorical data.
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='error', drop=drop))])

        transformations = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numerical),
                ('cat', categorical_transformer, categorical)])

        x_train, x_test, y_train, y_test = train_test_split(attritionXData,
                                                            target,
                                                            test_size=0.2,
                                                            random_state=0,
                                                            stratify=target)

        # Append classifier to preprocessing pipeline.
        # Now we have a full prediction pipeline.
        clf = Pipeline(steps=[('preprocessor', transformations),
                              ('classifier', LGBMClassifier())])

        clf.fit(x_train, y_train)

        explainer = tabular_explainer(clf.steps[-1][1],
                                      initialization_examples=x_train,
                                      features=attritionXData.columns,
                                      classes=['Leaving', 'Staying'],
                                      transformations=transformations)
        global_explanation = explainer.explain_global(x_test)
        local_shape = global_explanation._local_importance_values.shape
        num_rows_expected = len(x_test)
        num_cols = len(x_test.columns)
        assert local_shape == (2, num_rows_expected, num_cols)
        assert len(global_explanation.global_importance_values) == num_cols
        assert global_explanation.num_features == num_cols

    def test_explain_with_transformations_list_classification(self, verify_tabular):
        verify_tabular.verify_explain_model_transformations_list_classification()

    def test_explain_with_transformations_column_transformer_classification(self, verify_tabular):
        verify_tabular.verify_explain_model_transformations_column_transformer_classification()

    def test_explain_with_transformations_list_regression(self, verify_tabular):
        verify_tabular.verify_explain_model_transformations_list_regression()

    def test_explain_with_transformations_column_transformer_regression(self, verify_tabular):
        verify_tabular.verify_explain_model_transformations_column_transformer_regression()

    def test_explain_model_categorical(self, verify_tabular):
        verify_tabular.verify_explain_model_categorical()

    def test_explain_model_pandas_string(self, tabular_explainer):
        np.random.seed(777)
        num_rows = 100
        num_ints = 10
        num_cols = 4
        split_ratio = 0.2
        A = np.random.randint(num_ints, size=num_rows)
        B = np.random.random(size=num_rows)
        C = np.random.randn(num_rows)
        cat = np.random.choice(['New York', 'San Francisco', 'Los Angeles',
                                'Atlanta', 'Denver', 'Chicago', 'Miami', 'DC', 'Boston'], 100)
        label = np.random.choice([0, 1], num_rows)
        df = pd.DataFrame(data={'A': A, 'B': B, 'C': C, 'cat': cat, 'label': label})
        df.cat = df.cat.astype('category')
        X = df.drop('label', axis=1)
        y = df.label

        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=split_ratio)

        clf = create_lightgbm_classifier(x_train, y_train)

        explainer = tabular_explainer(clf, initialization_examples=x_train, features=x_train.columns)
        global_explanation = explainer.explain_global(x_test)
        local_shape = global_explanation._local_importance_values.shape
        num_rows_expected = split_ratio * num_rows
        assert local_shape == (2, num_rows_expected, num_cols)
        assert len(global_explanation.global_importance_values) == num_cols
        assert global_explanation.num_features == num_cols

    def test_explain_model_classification_with_predict_only(self, tabular_explainer):
        X, y = shap.datasets.adult()
        x_train, x_test, y_train, _ = train_test_split(X, y, test_size=0.003, random_state=7)
        # Fit a tree model
        model = create_sklearn_random_forest_classifier(x_train, y_train)

        # Wrap the model in a predict-only API
        wrapped_model = wrap_classifier_without_proba(model)

        # Create tabular explainer
        exp = tabular_explainer(wrapped_model, x_train, features=X.columns.values, model_task=ModelTask.Classification)
        test_logger.info('Running explain global for test_explain_model_classification_with_predict_only')
        explanation = exp.explain_global(x_test)
        # Validate predicted y values are boolean
        assert np.all(np.isin(explanation.eval_y_predicted, [0, 1]))

    def test_tabular_explainer_get_explainers(self):
        non_gpu_explainers = _get_uninitialized_explainers(use_gpu=False)
        assert non_gpu_explainers == [TreeExplainer, DeepExplainer, LinearExplainer, KernelExplainer]
        gpu_explainers = _get_uninitialized_explainers(use_gpu=True)
        assert gpu_explainers == [GPUKernelExplainer]

    @pytest.mark.skip(reason="latest tensorflow version no longer works with shap deep explainer")
    def test_explain_model_keras_regressor(self, housing, tabular_explainer):
        # verify that no errors get thrown when calling get_raw_feat_importances
        x_train = housing[DatasetConstants.X_TRAIN][DATA_SLICE]
        x_test = housing[DatasetConstants.X_TEST][DATA_SLICE]
        y_train = housing[DatasetConstants.Y_TRAIN][DATA_SLICE]

        model = create_scikit_keras_regressor(x_train, y_train)

        explainer = tabular_explainer(model, x_train)

        global_explanation = explainer.explain_global(x_test)
        local_explanation = explainer.explain_local(x_test)
        global_importance_values = global_explanation.global_importance_values
        num_feats = x_train.shape[1]
        num_samples = x_train.shape[0]
        assert len(global_importance_values) == num_feats, ('length of global importances '
                                                            'does not match number of features')
        local_importance_values = local_explanation.local_importance_values
        assert len(local_importance_values) == num_samples, ('length of local importances does not match number '
                                                             'of samples')

    @pytest.mark.skip(reason="latest tensorflow version no longer works with shap deep explainer")
    def test_explain_model_keras_classifier(self, iris, tabular_explainer):
        # verify that no errors get thrown when calling get_raw_feat_importances
        x_train = iris[DatasetConstants.X_TRAIN][DATA_SLICE]
        x_test = iris[DatasetConstants.X_TEST][DATA_SLICE]
        y_train = iris[DatasetConstants.Y_TRAIN][DATA_SLICE]

        model = create_scikit_keras_classifier(x_train, y_train)

        explainer = tabular_explainer(model, x_train)

        global_explanation = explainer.explain_global(x_test)
        local_explanation = explainer.explain_local(x_test)
        global_importance_values = global_explanation.global_importance_values
        num_feats = x_train.shape[1]
        num_classes = 2
        assert len(global_importance_values) == num_feats, ('length of global importances '
                                                            'does not match number of features')
        local_importance_values = local_explanation.local_importance_values
        assert len(local_importance_values) == num_classes, ('length of local importances does not match number '
                                                             'of classes')

    @pytest.mark.skip(reason="latest tensorflow version no longer works with shap deep explainer")
    def test_explain_model_batch_dataset(self, housing, tabular_explainer):
        X_train = housing[DatasetConstants.X_TRAIN]
        X_test = housing[DatasetConstants.X_TEST][DATA_SLICE]
        y_train = housing[DatasetConstants.Y_TRAIN]
        y_test = housing[DatasetConstants.Y_TEST][DATA_SLICE]
        features = housing[DatasetConstants.FEATURES]
        X_train_df = pd.DataFrame(X_train, columns=list(features))
        X_test_df = pd.DataFrame(X_test, columns=list(features))
        inp = (dict(X_train_df), y_train)
        inp_ds = tf.data.Dataset.from_tensor_slices(inp).batch(32)
        val = (dict(X_test_df), y_test)
        val_ds = tf.data.Dataset.from_tensor_slices(val).batch(32)
        model = create_tf_model(inp_ds, val_ds, features)
        wrapped_dataset = DatasetWrapper(val_ds)
        wrapped_model = wrap_model(model, wrapped_dataset, model_task='regression')

        explainer = tabular_explainer(wrapped_model, inp_ds)

        global_explanation = explainer.explain_global(wrapped_dataset)
        local_explanation = explainer.explain_local(wrapped_dataset)
        global_importance_values = global_explanation.global_importance_values
        num_rows = X_test_df.shape[0]
        num_feats = X_test_df.shape[1]
        assert len(global_importance_values) == num_feats, ('length of global importances '
                                                            'does not match number of features')
        local_importance_values = local_explanation.local_importance_values
        assert len(local_importance_values) == num_rows, ('length of local importances does not match number '
                                                          'of rows')

    def verify_adult_overall_features(self, ranked_global_names, ranked_global_values):
        # Verify order of features
        test_logger.info(ranked_global_names)
        test_logger.info("length of ranked_global_values: %s", str(len(ranked_global_values)))
        exp_features = ['Relationship', 'Marital Status', 'Education-Num', 'Capital Gain',
                        'Age', 'Hours per week', 'Capital Loss', 'Sex', 'Occupation',
                        'Country', 'Race', 'Workclass']
        np.testing.assert_array_equal(ranked_global_names, exp_features)
        assert len(ranked_global_values) == len(exp_features)

    def verify_adult_per_class_features(self, ranked_per_class_names, ranked_per_class_values):
        # Verify order of features
        test_logger.info(ranked_per_class_names)
        test_logger.info("shape of ranked_per_class_values: %s", str(len(ranked_per_class_values)))
        exp_features = [['Relationship', 'Marital Status', 'Education-Num', 'Capital Gain', 'Age', 'Hours per week',
                         'Capital Loss', 'Sex', 'Occupation', 'Country', 'Race', 'Workclass'],
                        ['Relationship', 'Marital Status', 'Education-Num', 'Capital Gain', 'Age', 'Hours per week',
                         'Capital Loss', 'Sex', 'Occupation', 'Country', 'Race', 'Workclass']]
        np.testing.assert_array_equal(ranked_per_class_names, exp_features)
        assert len(ranked_per_class_values) == len(exp_features)
        assert len(ranked_per_class_values[0]) == len(exp_features[0])

    def verify_iris_overall_features(self, ranked_global_names, ranked_global_values, verify_tabular):
        # Verify order of features
        test_logger.info("length of ranked_global_values: %s", str(len(ranked_global_values)))
        exp_features = verify_tabular.iris_overall_expected_features
        np.testing.assert_array_equal(ranked_global_names, exp_features)
        assert len(ranked_global_values) == 4

    def verify_iris_overall_features_no_names(self, ranked_global_names, ranked_global_values):
        # Verify order of features
        test_logger.info("length of ranked_global_values: %s", str(len(ranked_global_values)))
        exp_features = [2, 3, 0, 1]
        np.testing.assert_array_equal(ranked_global_names, exp_features)
        assert len(ranked_global_values) == len(exp_features)

    def verify_iris_per_class_features(self, ranked_per_class_names, ranked_per_class_values):
        # Verify order of features
        exp_features = self.iris_per_class_expected_features
        np.testing.assert_array_equal(ranked_per_class_names, exp_features)
        assert len(ranked_per_class_values) == 3
        assert len(ranked_per_class_values[0]) == 4

    def verify_iris_per_class_features_no_names(self, ranked_per_class_names, ranked_per_class_values):
        # Verify order of features
        exp_features = [[2, 3, 1, 0],
                        [2, 3, 0, 1],
                        [2, 3, 0, 1]]
        np.testing.assert_array_equal(ranked_per_class_names, exp_features)
        assert len(ranked_per_class_values) == len(exp_features)
        assert len(ranked_per_class_values[0]) == len(exp_features[0])

    def verify_housing_overall_features_rf(self, ranked_global_names, ranked_global_values):
        # Note: the order seems to differ from one machine to another, so we won't validate exact order
        test_logger.info("length of ranked_global_values: %s", str(len(ranked_global_values)))
        assert ranked_global_names[0] == 'MedInc'
        assert len(ranked_global_values) == 8

    def verify_housing_overall_features_lr(self, ranked_global_names, ranked_global_values):
        # Verify order of features
        test_logger.info("length of ranked_global_values: %s", str(len(ranked_global_values)))
        exp_features = ['Latitude', 'Longitude', 'MedInc', 'AveRooms', 'HouseAge',
                        'AveBedrms', 'AveOccup', 'Population']
        np.testing.assert_array_equal(ranked_global_names, exp_features)
        assert len(ranked_global_values) == len(exp_features)

    def verify_top_rows_local_features_with_and_without_top_k(self, explanation, local_features,
                                                              is_classification=False, top_rows=5):
        if is_classification:
            ranked_local_names = explanation.get_ranked_local_names()
            classes = list(range(len(ranked_local_names)))
            # Get top rows
            top_rows_local_names = np.array(ranked_local_names)[classes, :top_rows].tolist()
            top_rows_local_names_k_2 = np.array(explanation.get_ranked_local_names(top_k=2))[classes, :top_rows]
            top_rows_local_names_k_2 = top_rows_local_names_k_2.tolist()
            # Validate against reference data
            assert top_rows_local_names == local_features
            # Validate topk parameter works correctly
            assert top_rows_local_names_k_2 == np.array(local_features)[classes, :top_rows, :2].tolist()
        else:
            # Get top rows
            top_rows_local_names = explanation.get_ranked_local_names()[:top_rows]
            # Validate against reference data
            assert top_rows_local_names == local_features
            top_rows_local_names_k_2 = explanation.get_ranked_local_names(top_k=2)[:top_rows]
            # Validate topk parameter works correctly
            assert top_rows_local_names_k_2 == np.array(local_features)[:, :2].tolist()

    def verify_global_mli_data(self, data, explanation, features, is_full=False):
        if is_full:
            assert InterpretData.MLI in data
            full_data_scores = data[InterpretData.MLI][0][InterpretData.VALUE][InterpretData.SCORES]
            assert full_data_scores == explanation.local_importance_values
        else:
            assert InterpretData.MLI in data
            assert InterpretData.NAMES in data
            assert InterpretData.SCORES in data
            assert data[InterpretData.NAMES] == features
            assert data[InterpretData.SCORES] == explanation.global_importance_values

    def verify_local_mli_data(self, data):
        assert data == {InterpretData.MLI: []}

    @property
    def iris_per_class_expected_features(self):
        return [['petal length', 'petal width', 'sepal width', 'sepal length'],
                ['petal length', 'petal width', 'sepal length', 'sepal width'],
                ['petal length', 'petal width', 'sepal length', 'sepal width']]

    @property
    def housing_local_features_first_five_rf(self):
        return [['AveRooms', 'Latitude', 'HouseAge', 'Population', 'Longitude', 'AveBedrms', 'AveOccup', 'MedInc'],
                ['MedInc', 'HouseAge', 'Latitude', 'Population', 'Longitude', 'AveBedrms', 'AveRooms', 'AveOccup'],
                ['AveOccup', 'MedInc', 'HouseAge', 'Latitude', 'Population', 'Longitude', 'AveBedrms', 'AveRooms'],
                ['AveOccup', 'AveRooms', 'HouseAge', 'Latitude', 'Population', 'Longitude', 'AveBedrms', 'MedInc'],
                ['MedInc', 'Population', 'HouseAge', 'Longitude', 'AveBedrms', 'Latitude', 'AveRooms', 'AveOccup']]

    @property
    def housing_local_features_first_five_lr(self):
        return [['Latitude', 'AveRooms', 'HouseAge', 'AveBedrms', 'AveOccup', 'Population', 'Longitude', 'MedInc'],
                ['Latitude', 'MedInc', 'AveRooms', 'HouseAge', 'Population', 'AveOccup', 'AveBedrms', 'Longitude'],
                ['Longitude', 'HouseAge', 'AveRooms', 'AveOccup', 'MedInc', 'Population', 'AveBedrms', 'Latitude'],
                ['Longitude', 'AveRooms', 'AveBedrms', 'AveOccup', 'Population', 'HouseAge', 'MedInc', 'Latitude'],
                ['MedInc', 'Longitude', 'Population', 'AveOccup', 'HouseAge', 'AveBedrms', 'AveRooms', 'Latitude']]

    @property
    def adult_local_features_first_three_rf(self):
        return [[['Relationship', 'Education-Num', 'Capital Gain', 'Sex', 'Hours per week',
                  'Capital Loss', 'Occupation', 'Race', 'Workclass', 'Country', 'Age',
                  'Marital Status'],
                 ['Relationship', 'Capital Gain', 'Capital Loss', 'Occupation', 'Race',
                  'Workclass', 'Country', 'Sex', 'Age', 'Hours per week', 'Marital Status',
                  'Education-Num'],
                 ['Age', 'Education-Num', 'Capital Gain', 'Hours per week', 'Capital Loss',
                  'Occupation', 'Workclass', 'Race', 'Country', 'Sex', 'Marital Status',
                  'Relationship']],
                [['Marital Status', 'Age', 'Country', 'Workclass', 'Race', 'Occupation',
                  'Capital Loss', 'Hours per week', 'Sex', 'Capital Gain', 'Education-Num',
                  'Relationship'],
                 ['Education-Num', 'Marital Status', 'Hours per week', 'Age', 'Sex',
                  'Country', 'Workclass', 'Race', 'Occupation', 'Capital Loss',
                  'Capital Gain', 'Relationship'],
                 ['Relationship', 'Marital Status', 'Sex', 'Country', 'Race', 'Workclass',
                  'Occupation', 'Capital Loss', 'Hours per week', 'Capital Gain',
                  'Education-Num', 'Age']]]
