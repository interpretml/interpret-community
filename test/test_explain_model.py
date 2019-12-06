# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import pytest

# Tests for model explainability SDK
import numpy as np
import scipy as sp
import shap
import logging
import pandas as pd

from interpret_community.common.policy import SamplingPolicy

from common_utils import create_sklearn_random_forest_classifier, create_sklearn_svm_classifier, \
    create_sklearn_random_forest_regressor, create_sklearn_linear_regressor, create_keras_classifier, \
    create_keras_regressor, create_lightgbm_classifier, create_pytorch_classifier, create_pytorch_regressor, \
    create_xgboost_classifier
from raw_explain.utils import _get_feature_map_from_indices_list
from interpret_community.common.constants import ModelTask
from constants import DatasetConstants

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer

from constants import owner_email_tools_and_ux
from datasets import retrieve_dataset

test_logger = logging.getLogger(__name__)
test_logger.setLevel(logging.DEBUG)

DATA_SLICE = slice(10)


@pytest.mark.owner(email=owner_email_tools_and_ux)
@pytest.mark.usefixtures('clean_dir')
class TestTabularExplainer(object):
    def test_working(self):
        assert True

    def test_pandas_with_feature_names(self, iris, tabular_explainer, verify_tabular):
        # create pandas dataframes
        x_train = pd.DataFrame(data=iris[DatasetConstants.X_TRAIN], columns=iris[DatasetConstants.FEATURES])
        x_test = pd.DataFrame(data=iris[DatasetConstants.X_TEST], columns=iris[DatasetConstants.FEATURES])
        # Fit an SVM model
        model = create_sklearn_svm_classifier(x_train, iris[DatasetConstants.Y_TRAIN])

        exp = tabular_explainer(model,
                                x_train,
                                features=iris[DatasetConstants.FEATURES],
                                classes=iris[DatasetConstants.CLASSES])
        test_logger.info("Running explain global for test_pandas_with_feature_names")
        explanation = exp.explain_global(x_test)
        ranked_global_values = explanation.get_ranked_global_values()
        ranked_per_class_values = explanation.get_ranked_per_class_values()
        ranked_global_names = explanation.get_ranked_global_names()
        ranked_per_class_names = explanation.get_ranked_per_class_names()

        self.verify_iris_overall_features(ranked_global_names, ranked_global_values, verify_tabular)
        self.verify_iris_per_class_features(ranked_per_class_names, ranked_per_class_values)

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

    def test_explain_model_local_regression_dnn(self, verify_tabular):
        verify_tabular.verify_explain_model_local_regression_dnn()

    def test_explanation_get_feature_importance_dict(self, iris, tabular_explainer):
        x_train = pd.DataFrame(data=iris[DatasetConstants.X_TRAIN], columns=iris[DatasetConstants.FEATURES])
        x_test = pd.DataFrame(data=iris[DatasetConstants.X_TEST], columns=iris[DatasetConstants.FEATURES])
        # Fit an SVM model
        model = create_sklearn_svm_classifier(x_train, iris[DatasetConstants.Y_TRAIN])

        exp = tabular_explainer(model, x_train, classes=iris[DatasetConstants.CLASSES])
        test_logger.info("Running explain global for test_pandas_no_feature_names")
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

        exp = tabular_explainer(model, iris[DatasetConstants.X_TRAIN], features=iris[DatasetConstants.FEATURES],
                                classes=iris[DatasetConstants.CLASSES])

        local_explanation = exp.explain_local(iris[DatasetConstants.X_TEST][0])

        assert len(local_explanation.local_importance_values) == len(iris[DatasetConstants.CLASSES])
        assert local_explanation.num_classes == len(iris[DatasetConstants.CLASSES])
        assert len(local_explanation.local_importance_values[0]) == len(iris[DatasetConstants.FEATURES])
        assert local_explanation.num_features == len(iris[DatasetConstants.FEATURES])

        local_rank = local_explanation.get_local_importance_rank()
        assert len(local_rank) == len(iris[DatasetConstants.CLASSES])
        assert len(local_rank[0]) == len(iris[DatasetConstants.FEATURES])

        ranked_names = local_explanation.get_ranked_local_names()
        assert len(ranked_names) == len(iris[DatasetConstants.CLASSES])
        assert len(ranked_names[0]) == len(iris[DatasetConstants.FEATURES])

        ranked_values = local_explanation.get_ranked_local_values()
        assert len(ranked_values) == len(iris[DatasetConstants.CLASSES])
        assert len(ranked_values[0]) == len(iris[DatasetConstants.FEATURES])

    def test_explain_multi_local_instance_classification(self, iris, tabular_explainer):
        # Fit an SVM model
        model = create_sklearn_svm_classifier(iris[DatasetConstants.X_TRAIN], iris[DatasetConstants.Y_TRAIN])

        exp = tabular_explainer(model, iris[DatasetConstants.X_TRAIN], features=iris[DatasetConstants.FEATURES],
                                classes=iris[DatasetConstants.CLASSES])

        local_explanation = exp.explain_local(iris[DatasetConstants.X_TEST])

        assert len(local_explanation.local_importance_values) == len(iris[DatasetConstants.CLASSES])
        assert local_explanation.num_classes == len(iris[DatasetConstants.CLASSES])
        assert len(local_explanation.local_importance_values[0]) == len(iris[DatasetConstants.X_TEST])
        assert local_explanation.num_examples == len(iris[DatasetConstants.X_TEST])
        assert len(local_explanation.local_importance_values[0][0]) == len(iris[DatasetConstants.FEATURES])
        assert local_explanation.num_features == len(iris[DatasetConstants.FEATURES])

        local_rank = local_explanation.get_local_importance_rank()
        assert len(local_rank) == len(iris[DatasetConstants.CLASSES])
        assert len(local_rank[0]) == len(iris[DatasetConstants.X_TEST])
        assert len(local_rank[0][0]) == len(iris[DatasetConstants.FEATURES])

        ranked_names = local_explanation.get_ranked_local_names()
        assert len(ranked_names) == len(iris[DatasetConstants.CLASSES])
        assert len(ranked_names[0]) == len(iris[DatasetConstants.X_TEST])
        assert len(ranked_names[0][0]) == len(iris[DatasetConstants.FEATURES])

        ranked_values = local_explanation.get_ranked_local_values()
        assert len(ranked_values) == len(iris[DatasetConstants.CLASSES])
        assert len(ranked_values[0]) == len(iris[DatasetConstants.X_TEST])
        assert len(ranked_values[0][0]) == len(iris[DatasetConstants.FEATURES])

    def test_explain_single_local_instance_regression(self, boston, tabular_explainer):
        # Fit an SVM model
        model = create_sklearn_random_forest_regressor(boston[DatasetConstants.X_TRAIN],
                                                       boston[DatasetConstants.Y_TRAIN])

        exp = tabular_explainer(model, boston[DatasetConstants.X_TRAIN], features=boston[DatasetConstants.FEATURES])

        local_explanation = exp.explain_local(boston[DatasetConstants.X_TEST][0])

        assert len(local_explanation.local_importance_values) == len(boston[DatasetConstants.FEATURES])
        assert local_explanation.num_features == len(boston[DatasetConstants.FEATURES])

        local_rank = local_explanation.get_local_importance_rank()
        assert len(local_rank) == len(boston[DatasetConstants.FEATURES])

        ranked_names = local_explanation.get_ranked_local_names()
        assert len(ranked_names) == len(boston[DatasetConstants.FEATURES])

        ranked_values = local_explanation.get_ranked_local_values()
        assert len(ranked_values) == len(boston[DatasetConstants.FEATURES])

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
        test_logger.info("Running explain global for test_explain_model_local")
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
        test_logger.info("Running explain global for test_explain_model_local")
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
        x_train, x_test, y_train, _ = self.create_msx_data(0.1)
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
        background = sp.sparse.csr_matrix(shape, dtype=x_train.dtype)

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

    def test_explain_model_lightgbm_binary(self, tabular_explainer):
        X, y = shap.datasets.adult()
        x_train, x_test, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=7)
        # Fit a tree model
        model = create_lightgbm_classifier(x_train, y_train)

        classes = ["<50k", ">50k"]
        # Create tabular explainer
        exp = tabular_explainer(model, x_train, features=X.columns.values,
                                classes=classes)
        test_logger.info('Running explain global for test_explain_model_lightgbm_binary')
        explanation = exp.explain_global(x_test)
        assert len(explanation.local_importance_values[0]) == len(x_test)
        assert len(explanation.local_importance_values) == len(classes)

    def _explain_model_dnn_common(self, tabular_explainer, model, x_train, x_test, y_train, features):
        # Create tabular explainer
        exp = tabular_explainer(model, x_train.values, features=features, model_task=ModelTask.Classification)
        policy = SamplingPolicy(allow_eval_sampling=True)
        exp.explain_global(x_test.values, sampling_policy=policy)

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

    def test_explain_model_random_forest_regression(self, boston, tabular_explainer):
        # Fit a random forest regression model
        model = create_sklearn_random_forest_regressor(boston[DatasetConstants.X_TRAIN],
                                                       boston[DatasetConstants.Y_TRAIN])

        # Create tabular explainer
        exp = tabular_explainer(model, boston[DatasetConstants.X_TRAIN], features=boston[DatasetConstants.FEATURES])
        test_logger.info('Running explain global for test_explain_model_random_forest_regression')
        explanation = exp.explain_global(boston[DatasetConstants.X_TEST])
        self.verify_boston_overall_features_rf(explanation.get_ranked_global_names(),
                                               explanation.get_ranked_global_values())

    def test_explain_model_local_tree_regression(self, boston, tabular_explainer):
        # Fit a random forest regression model
        model = create_sklearn_random_forest_regressor(boston[DatasetConstants.X_TRAIN],
                                                       boston[DatasetConstants.Y_TRAIN])

        # Create tabular explainer
        exp = tabular_explainer(model, boston[DatasetConstants.X_TRAIN], features=boston[DatasetConstants.FEATURES])
        test_logger.info('Running explain local for test_explain_model_local_tree_regression')
        explanation = exp.explain_local(boston[DatasetConstants.X_TEST])
        assert explanation.local_importance_values is not None
        assert len(explanation.local_importance_values) == len(boston[DatasetConstants.X_TEST])
        assert explanation.num_examples == len(boston[DatasetConstants.X_TEST])
        assert len(explanation.local_importance_values[0]) == len(boston[DatasetConstants.FEATURES])
        assert explanation.num_features == len(boston[DatasetConstants.FEATURES])
        self.verify_top_rows_local_features_with_and_without_top_k(explanation,
                                                                   self.boston_local_features_first_five_rf)

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
        test_logger.info('Running explain local for test_explain_model_local_keras_classification')
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

    def test_explain_model_local_keras_regression(self, boston, tabular_explainer):
        x_train = boston[DatasetConstants.X_TRAIN]
        x_test = boston[DatasetConstants.X_TEST]
        # Fit a DNN keras model
        model = create_keras_regressor(x_train, boston[DatasetConstants.Y_TRAIN])
        test_logger.info('Running explain local for test_explain_model_local_keras_regression')
        self._explain_model_local_dnn_regression_common(tabular_explainer, model, x_train,
                                                        x_test, boston[DatasetConstants.Y_TRAIN],
                                                        boston[DatasetConstants.FEATURES])

    def test_explain_model_local_pytorch_regression(self, boston, tabular_explainer):
        x_train = boston[DatasetConstants.X_TRAIN]
        x_test = boston[DatasetConstants.X_TEST]
        # Fit a DNN pytorch model
        model = create_pytorch_regressor(x_train, boston[DatasetConstants.Y_TRAIN])
        test_logger.info('Running explain local for test_explain_model_local_pytorch_regression')
        self._explain_model_local_dnn_regression_common(tabular_explainer, model, x_train,
                                                        x_test, boston[DatasetConstants.Y_TRAIN],
                                                        boston[DatasetConstants.FEATURES])

    def test_explain_model_local_kernel_regression(self, boston, tabular_explainer):
        # Fit a linear regression model
        model = create_sklearn_linear_regressor(boston[DatasetConstants.X_TRAIN], boston[DatasetConstants.Y_TRAIN])

        # Create tabular explainer
        exp = tabular_explainer(model, boston[DatasetConstants.X_TRAIN], features=boston[DatasetConstants.FEATURES])
        test_logger.info('Running explain local for test_explain_model_regression')
        explanation = exp.explain_local(boston[DatasetConstants.X_TEST])
        assert explanation.local_importance_values is not None
        assert len(explanation.local_importance_values) == len(boston[DatasetConstants.X_TEST])
        assert explanation.num_examples == len(boston[DatasetConstants.X_TEST])
        assert len(explanation.local_importance_values[0]) == len(boston[DatasetConstants.FEATURES])
        assert explanation.num_features == len(boston[DatasetConstants.FEATURES])
        self.verify_top_rows_local_features_with_and_without_top_k(explanation,
                                                                   self.boston_local_features_first_five_lr)

    def test_explain_model_linear_regression(self, boston, tabular_explainer):
        # Fit a linear regression model
        model = create_sklearn_linear_regressor(boston[DatasetConstants.X_TRAIN],
                                                boston[DatasetConstants.Y_TRAIN],
                                                pipeline=True)

        # Create tabular explainer
        exp = tabular_explainer(model, boston[DatasetConstants.X_TRAIN], features=boston[DatasetConstants.FEATURES])
        test_logger.info('Running explain global for test_explain_model_regression')
        explanation = exp.explain_global(boston[DatasetConstants.X_TEST])
        self.verify_boston_overall_features_lr(explanation.get_ranked_global_names(),
                                               explanation.get_ranked_global_values())

    def test_explain_model_subset_classification_dense(self, verify_tabular):
        verify_tabular.verify_explain_model_subset_classification_dense()

    def test_explain_model_subset_regression_sparse(self, verify_tabular):
        verify_tabular.verify_explain_model_subset_regression_sparse()

    def test_explain_model_subset_classification_sparse(self, verify_tabular):
        verify_tabular.verify_explain_model_subset_classification_sparse()

    def test_explain_model_with_sampling_regression_sparse(self, verify_tabular):
        verify_tabular.verify_explain_model_with_sampling_regression_sparse()

    def test_explain_raw_feats_regression(self, boston, tabular_explainer):
        # verify that no errors get thrown when calling get_raw_feat_importances
        x_train = boston[DatasetConstants.X_TRAIN][DATA_SLICE]
        x_test = boston[DatasetConstants.X_TEST][DATA_SLICE]
        y_train = boston[DatasetConstants.Y_TRAIN][DATA_SLICE]

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

    def create_msx_data(self, test_size):
        sparse_matrix = retrieve_dataset('msx_transformed_2226.npz')
        sparse_matrix_x = sparse_matrix[:, :sparse_matrix.shape[1] - 2]
        sparse_matrix_y = sparse_matrix[:, (sparse_matrix.shape[1] - 2):(sparse_matrix.shape[1] - 1)]
        return train_test_split(sparse_matrix_x, sparse_matrix_y, test_size=test_size, random_state=7)

    def verify_adult_overall_features(self, ranked_global_names, ranked_global_values):
        # Verify order of features
        test_logger.info(ranked_global_names)
        test_logger.info("length of ranked_global_values: %s", str(len(ranked_global_values)))
        exp_features = ['Relationship', 'Marital Status', 'Education-Num', 'Capital Gain',
                        'Age', 'Hours per week', 'Capital Loss', 'Sex', 'Occupation',
                        'Country', 'Race', 'Workclass']
        np.testing.assert_array_equal(ranked_global_names, exp_features)
        assert(len(ranked_global_values) == len(exp_features))

    def verify_adult_per_class_features(self, ranked_per_class_names, ranked_per_class_values):
        # Verify order of features
        test_logger.info(ranked_per_class_names)
        test_logger.info("shape of ranked_per_class_values: %s", str(len(ranked_per_class_values)))
        exp_features = [['Relationship', 'Marital Status', 'Education-Num', 'Capital Gain', 'Age', 'Hours per week',
                         'Capital Loss', 'Sex', 'Occupation', 'Country', 'Race', 'Workclass'],
                        ['Relationship', 'Marital Status', 'Education-Num', 'Capital Gain', 'Age', 'Hours per week',
                         'Capital Loss', 'Sex', 'Occupation', 'Country', 'Race', 'Workclass']]
        np.testing.assert_array_equal(ranked_per_class_names, exp_features)
        assert(len(ranked_per_class_values) == len(exp_features))
        assert(len(ranked_per_class_values[0]) == len(exp_features[0]))

    def verify_iris_overall_features(self, ranked_global_names, ranked_global_values, verify_tabular):
        # Verify order of features
        test_logger.info("length of ranked_global_values: %s", str(len(ranked_global_values)))
        exp_features = verify_tabular.iris_overall_expected_features
        np.testing.assert_array_equal(ranked_global_names, exp_features)
        assert(len(ranked_global_values) == 4)

    def verify_iris_overall_features_no_names(self, ranked_global_names, ranked_global_values):
        # Verify order of features
        test_logger.info("length of ranked_global_values: %s", str(len(ranked_global_values)))
        exp_features = [2, 3, 0, 1]
        np.testing.assert_array_equal(ranked_global_names, exp_features)
        assert(len(ranked_global_values) == len(exp_features))

    def verify_iris_per_class_features(self, ranked_per_class_names, ranked_per_class_values):
        # Verify order of features
        exp_features = self.iris_per_class_expected_features
        np.testing.assert_array_equal(ranked_per_class_names, exp_features)
        assert(len(ranked_per_class_values) == 3)
        assert(len(ranked_per_class_values[0]) == 4)

    def verify_iris_per_class_features_no_names(self, ranked_per_class_names, ranked_per_class_values):
        # Verify order of features
        exp_features = [[2, 3, 1, 0],
                        [2, 3, 0, 1],
                        [2, 3, 0, 1]]
        np.testing.assert_array_equal(ranked_per_class_names, exp_features)
        assert(len(ranked_per_class_values) == len(exp_features))
        assert(len(ranked_per_class_values[0]) == len(exp_features[0]))

    def verify_boston_overall_features_rf(self, ranked_global_names, ranked_global_values):
        # Note: the order seems to differ from one machine to another, so we won't validate exact order
        test_logger.info("length of ranked_global_values: %s", str(len(ranked_global_values)))
        assert(ranked_global_names[0] == 'RM')
        assert(len(ranked_global_values) == 13)

    def verify_boston_overall_features_lr(self, ranked_global_names, ranked_global_values):
        # Verify order of features
        test_logger.info("length of ranked_global_values: %s", str(len(ranked_global_values)))
        exp_features = ['RM', 'RAD', 'DIS', 'LSTAT', 'TAX', 'PTRATIO', 'NOX', 'CRIM', 'B', 'ZN', 'AGE',
                        'CHAS', 'INDUS']
        np.testing.assert_array_equal(ranked_global_names, exp_features)
        assert(len(ranked_global_values) == len(exp_features))

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

    @property
    def iris_per_class_expected_features(self):
        return [['petal length', 'petal width', 'sepal width', 'sepal length'],
                ['petal length', 'petal width', 'sepal length', 'sepal width'],
                ['petal length', 'petal width', 'sepal length', 'sepal width']]

    @property
    def boston_local_features_first_five_rf(self):
        return [['LSTAT', 'CRIM', 'B', 'AGE', 'INDUS', 'RAD', 'CHAS', 'ZN', 'TAX', 'DIS', 'PTRATIO', 'NOX', 'RM'],
                ['LSTAT', 'CRIM', 'NOX', 'AGE', 'B', 'TAX', 'INDUS', 'RAD', 'CHAS', 'ZN', 'PTRATIO', 'DIS', 'RM'],
                ['LSTAT', 'NOX', 'CRIM', 'AGE', 'TAX', 'B', 'INDUS', 'RAD', 'CHAS', 'ZN', 'PTRATIO', 'DIS', 'RM'],
                ['LSTAT', 'CRIM', 'NOX', 'AGE', 'B', 'RAD', 'INDUS', 'CHAS', 'ZN', 'TAX', 'PTRATIO', 'DIS', 'RM'],
                ['DIS', 'INDUS', 'RAD', 'CHAS', 'ZN', 'AGE', 'B', 'TAX', 'PTRATIO', 'NOX', 'CRIM', 'RM', 'LSTAT']]

    @property
    def boston_local_features_first_five_lr(self):
        return [['RAD', 'CHAS', 'DIS', 'RM', 'B', 'INDUS', 'CRIM', 'LSTAT', 'AGE', 'ZN', 'PTRATIO', 'TAX', 'NOX'],
                ['TAX', 'LSTAT', 'NOX', 'CRIM', 'B', 'AGE', 'INDUS', 'CHAS', 'ZN', 'RAD', 'PTRATIO', 'RM', 'DIS'],
                ['TAX', 'NOX', 'CRIM', 'B', 'AGE', 'LSTAT', 'INDUS', 'CHAS', 'ZN', 'RM', 'RAD', 'PTRATIO', 'DIS'],
                ['LSTAT', 'TAX', 'B', 'CRIM', 'NOX', 'AGE', 'INDUS', 'CHAS', 'ZN', 'DIS', 'RAD', 'RM', 'PTRATIO'],
                ['RAD', 'DIS', 'INDUS', 'CHAS', 'ZN', 'AGE', 'RM', 'PTRATIO', 'NOX', 'TAX', 'LSTAT', 'B', 'CRIM']]

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
