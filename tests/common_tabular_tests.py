# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

# Common tests for tabular explainers
from enum import Enum

import numpy as np
import pandas as pd
import pytest
import shap
from common_utils import (create_cancer_data, create_energy_data,
                          create_iris_data, create_keras_multiclass_classifier,
                          create_keras_regressor, create_msx_data,
                          create_multiclass_sparse_newsgroups_data,
                          create_pandas_only_svm_classifier,
                          create_pytorch_multiclass_classifier,
                          create_pytorch_regressor,
                          create_sklearn_linear_regressor,
                          create_sklearn_logistic_regressor,
                          create_sklearn_random_forest_classifier,
                          create_sklearn_random_forest_regressor,
                          create_sklearn_svm_classifier,
                          create_xgboost_classifier)
from constants import ModelType
from datasets import retrieve_dataset
from interpret_community.common.constants import (ExplainParams, InterpretData,
                                                  ModelTask, ShapValuesOutput)
from interpret_community.common.explanation_utils import _summarize_data
from interpret_community.common.policy import SamplingPolicy
from lightgbm import LGBMRegressor
from scipy.sparse import csr_matrix
from scipy.special import expit
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from test_serialize_explanation import verify_serialization
from transformation_utils import (get_transformations_from_col_transformer,
                                  get_transformations_many_to_many,
                                  get_transformations_one_to_many_greater,
                                  get_transformations_one_to_many_smaller)

DATA_SLICE = slice(10)


TOLERANCE = 1e-2
PROBABILITY_TOLERANCE = 1e-5
SURROGATE_MODEL = 'surrogate_model'


class TransformationType(Enum):
    TransformationsList = 1
    ColumnTransformer = 2


class VerifyTabularTests(object):
    def __init__(self, test_logger, create_explainer, specify_policy=True):
        self.test_logger = test_logger
        self.create_explainer = create_explainer
        self.specify_policy = specify_policy

    def _verify_explain_model_transformations_classification(self, transformation_type, get_transformations,
                                                             create_model, true_labels_required,
                                                             allow_all_transformations=False):
        x_train, x_test, y_train, y_test, feature_names, classes = create_iris_data()
        x_train = pd.DataFrame(x_train, columns=feature_names)
        x_test = pd.DataFrame(x_test, columns=feature_names)

        # Fit an SVM model
        col_transformer = get_transformations(feature_names)
        x_train_transformed = col_transformer.fit_transform(x_train)
        if transformation_type == TransformationType.TransformationsList:
            transformations = get_transformations_from_col_transformer(col_transformer)
        else:
            transformations = col_transformer

        if create_model is None:
            model = create_sklearn_svm_classifier(x_train_transformed, y_train)
        else:
            model = create_model(x_train_transformed, y_train)

        explainer = self.create_explainer(model, x_train, features=feature_names,
                                          transformations=transformations, classes=classes,
                                          allow_all_transformations=allow_all_transformations)
        if true_labels_required:
            global_explanation = explainer.explain_global(x_test, y_test)
        else:
            global_explanation = explainer.explain_global(x_test)
            local_explanation = explainer.explain_local(x_test)
            verify_serialization(local_explanation, extra_path='local', exist_ok=True)
            feat_imps_local = np.array(local_explanation.local_importance_values)
            assert feat_imps_local.shape[-1] == len(feature_names)
            assert local_explanation.num_features == len(feature_names)
            per_class_values = global_explanation.get_ranked_per_class_values()
            assert len(per_class_values) == len(classes)
            assert global_explanation.num_classes == len(classes)
            assert len(per_class_values[0]) == len(feature_names)
            assert global_explanation.num_features == len(feature_names)
            assert len(global_explanation.get_ranked_per_class_names()[0]) == len(feature_names)
            feat_imps_global_local = np.array(global_explanation.local_importance_values)
            assert feat_imps_global_local.shape[-1] == len(feature_names)
            assert local_explanation.is_raw

        assert global_explanation.is_raw
        assert len(global_explanation.get_ranked_global_values()) == len(feature_names)
        assert global_explanation.num_features == len(feature_names)
        assert len(global_explanation.get_ranked_global_names()) == len(feature_names)
        assert (global_explanation.classes == classes).all()

        assert global_explanation.features == feature_names

        feat_imps_global = np.array(global_explanation.global_importance_values)

        assert feat_imps_global.shape[-1] == len(feature_names)

        verify_serialization(global_explanation, extra_path='global', exist_ok=True)

    def _verify_explain_model_transformations_regression(self, transformations_type, get_transformations,
                                                         create_model, true_labels_required,
                                                         allow_all_transformations=False):
        x_train, x_test, y_train, y_test, feature_names = create_energy_data()

        col_transformer = get_transformations(feature_names)
        x_train_transformed = col_transformer.fit_transform(x_train)

        if transformations_type == TransformationType.TransformationsList:
            transformations = get_transformations_from_col_transformer(col_transformer)
        else:
            transformations = col_transformer

        if create_model is None:
            model = create_sklearn_linear_regressor(x_train_transformed, y_train)
        else:
            model = create_model(x_train_transformed, y_train)

        explainer = self.create_explainer(model, x_train, features=feature_names, transformations=transformations,
                                          allow_all_transformations=allow_all_transformations)

        if true_labels_required:
            global_explanation = explainer.explain_global(x_test, y_test)
        else:
            global_explanation = explainer.explain_global(x_test)
            local_explanation = explainer.explain_local(x_test)
            assert local_explanation.is_raw
            assert np.array(local_explanation.local_importance_values).shape[-1] == len(feature_names)
            assert np.array(global_explanation.local_importance_values).shape[-1] == len(feature_names)
            assert local_explanation.num_features == len(feature_names)
            assert global_explanation.num_features == local_explanation.num_features

        assert global_explanation.is_raw
        assert np.array(global_explanation.global_importance_values).shape[-1] == len(feature_names)
        assert global_explanation.num_features == len(feature_names)

    def verify_explanation_top_k_bottom_k(self, explanation, is_per_class, is_local):
        K = 3
        global_values_whole = explanation.get_ranked_global_values()
        global_values_top_k = explanation.get_ranked_global_values(top_k=K)
        assert K == len(global_values_top_k)
        assert global_values_top_k == global_values_whole[:3]

        if is_per_class:
            per_class_values_whole = explanation.get_ranked_per_class_values()
            per_class_values_top_k = explanation.get_ranked_per_class_values(top_k=K)
            assert len(per_class_values_whole) == len(per_class_values_top_k)
            assert K == len(per_class_values_top_k[0])
            assert per_class_values_top_k[0] == per_class_values_whole[0][:3]

        if is_local:
            local_names_whole = explanation.get_ranked_local_names()
            local_names_top_k = explanation.get_ranked_local_names(top_k=K)
            assert len(local_names_whole) == len(local_names_top_k)
            assert len(local_names_whole[0]) == len(local_names_top_k[0])
            assert K == len(local_names_top_k[0][0])
            assert local_names_top_k[0][0] == local_names_whole[0][0][:3]

    def _verify_explain_model_local_common(self, model, x_train, x_test, y_train, y_test,
                                           feature_names, target_names, expected_overall_features=None,
                                           expected_per_class_features=None,
                                           is_per_class=True, include_evaluation_examples=True,
                                           include_local=True, has_explain_local=True,
                                           true_labels_required=False, num_overall_features_equal=-1):
        # Create tabular explainer
        explainer = self.create_explainer(model, x_train, features=feature_names, classes=target_names,
                                          model_task=ModelTask.Classification)
        self.test_logger.info('Running explain global for verify_explain_model_local')
        if include_evaluation_examples:
            if not include_local:
                # If include local is false (non-default), specify param
                explanation = explainer.explain_global(x_test, include_local=include_local)
            else:
                if true_labels_required:
                    explanation = explainer.explain_global(x_test, y_test)
                else:
                    explanation = explainer.explain_global(x_test)
        else:
            explanation = explainer.explain_global()
        assert not explanation.is_raw
        # Validate data has global info
        global_data = explanation.data(key=-1)
        assert InterpretData.OVERALL in global_data
        ranked_global_values = explanation.get_ranked_global_values()
        ranked_global_names = explanation.get_ranked_global_names()
        # Note: DNNs may be too random to validate here
        if expected_overall_features is not None:
            self.verify_iris_overall_features(ranked_global_names,
                                              ranked_global_values,
                                              expected_overall_features,
                                              num_overall_features_equal)
        if is_per_class:
            ranked_per_class_values = explanation.get_ranked_per_class_values()
            ranked_per_class_names = explanation.get_ranked_per_class_names()
            # Note: DNNs may be too random to validate here
            if expected_per_class_features is not None:
                self.verify_iris_per_class_features(ranked_per_class_names,
                                                    ranked_per_class_values,
                                                    expected_per_class_features)
        if has_explain_local:
            explanation_local = explainer.explain_local(x_test)
            # Validate there is a local explanation per class in multiclass case
            assert np.array(explanation_local.local_importance_values).shape[0] == len(target_names)
            assert explanation_local.num_classes == len(target_names)
            # Validate data has local info
            local_data = explanation_local.data(key=-1)
            assert InterpretData.SPECIFIC in local_data
            local_data_0 = explanation_local.data(key=0)
            for key in [InterpretData.NAMES, InterpretData.SCORES, InterpretData.TYPE]:
                assert key in local_data_0

    def verify_explain_model_local(self, expected_overall_features, expected_per_class_features=None,
                                   is_per_class=True, include_evaluation_examples=True,
                                   include_local=True, has_explain_local=True, true_labels_required=False,
                                   num_overall_features_equal=-1):
        x_train, x_test, y_train, y_test, feature_names, target_names = create_iris_data()
        # Fit an SVM model
        model = create_sklearn_svm_classifier(x_train, y_train)
        self._verify_explain_model_local_common(model, x_train, x_test, y_train, y_test,
                                                feature_names, target_names, expected_overall_features,
                                                expected_per_class_features=expected_per_class_features,
                                                is_per_class=is_per_class,
                                                include_evaluation_examples=include_evaluation_examples,
                                                include_local=include_local,
                                                has_explain_local=has_explain_local,
                                                true_labels_required=true_labels_required,
                                                num_overall_features_equal=num_overall_features_equal)

    def verify_explain_model_local_dnn(self, is_per_class=True, include_evaluation_examples=True,
                                       include_local=True, has_explain_local=True, true_labels_required=False,
                                       num_overall_features_equal=-1):
        x_train, x_test, y_train, y_test, feature_names, target_names = create_iris_data()
        # Fit a keras dnn classification model
        model = create_keras_multiclass_classifier(x_train, y_train)
        self._verify_explain_model_local_common(model, x_train, x_test, y_train, y_test,
                                                feature_names, target_names,
                                                expected_overall_features=None,
                                                expected_per_class_features=None,
                                                is_per_class=is_per_class,
                                                include_evaluation_examples=include_evaluation_examples,
                                                include_local=include_local,
                                                has_explain_local=has_explain_local,
                                                true_labels_required=true_labels_required,
                                                num_overall_features_equal=num_overall_features_equal)
        # Similar but now for pytorch multiclass model as well
        model = create_pytorch_multiclass_classifier(x_train, y_train)
        self._verify_explain_model_local_common(model, x_train, x_test, y_train, y_test,
                                                feature_names, target_names,
                                                expected_overall_features=None,
                                                expected_per_class_features=None,
                                                is_per_class=is_per_class,
                                                include_evaluation_examples=include_evaluation_examples,
                                                include_local=include_local,
                                                has_explain_local=has_explain_local,
                                                true_labels_required=true_labels_required,
                                                num_overall_features_equal=num_overall_features_equal)

    def _verify_explain_model_local_regression_common(self, model, x_train, x_test, y_train, y_test,
                                                      feature_names, include_evaluation_examples=True,
                                                      include_local=True, has_explain_local=True,
                                                      true_labels_required=False):
        # Create tabular explainer
        explainer = self.create_explainer(model, x_train, features=feature_names, model_task=ModelTask.Regression)
        self.test_logger.info('Running explain global for verify_explain_model_local_regression')
        if include_evaluation_examples:
            if not include_local:
                # If include local is false (non-default), specify param
                explanation = explainer.explain_global(x_test, include_local=include_local)
            else:
                if true_labels_required:
                    explanation = explainer.explain_global(x_test, y_test)
                else:
                    explanation = explainer.explain_global(x_test)
        else:
            explanation = explainer.explain_global()
        ranked_global_values = explanation.get_ranked_global_values()
        ranked_global_names = explanation.get_ranked_global_names()
        self.verify_energy_overall_features(ranked_global_names, ranked_global_values)
        if has_explain_local:
            explanation_local = explainer.explain_local(x_test)
            # Validate there is an explanation per row (without class) in regression case
            assert np.array(explanation_local.local_importance_values).shape[0] == len(x_test)
            assert explanation_local.num_examples == len(x_test)

    def verify_explain_model_local_regression(self, include_evaluation_examples=True, include_local=True,
                                              has_explain_local=True, true_labels_required=False):
        x_train, x_test, y_train, y_test, feature_names = create_energy_data()
        # Fit a linear model
        model = create_sklearn_linear_regressor(x_train, y_train)
        self._verify_explain_model_local_regression_common(model, x_train, x_test, y_train, y_test,
                                                           feature_names,
                                                           include_evaluation_examples=include_evaluation_examples,
                                                           include_local=include_local,
                                                           has_explain_local=has_explain_local,
                                                           true_labels_required=true_labels_required)

    def verify_explain_model_local_regression_dnn(self, include_evaluation_examples=True, include_local=True,
                                                  has_explain_local=True, true_labels_required=False):
        x_train, x_test, y_train, y_test, feature_names = create_energy_data()
        # Note: we normalize data and labels to prevent pytorch from failing
        # with NaN loss due to large values
        y_scaler = MinMaxScaler()
        y_train_values = y_train.values.reshape(-1, 1)
        y_scaler.fit(y_train_values)
        y_train = y_scaler.transform(y_train_values).flatten()
        y_test = y_scaler.transform(y_test.values.reshape(-1, 1)).flatten()
        x_scaler = MinMaxScaler()
        x_scaler.fit(x_train)
        x_train = x_scaler.transform(x_train)
        x_test = x_scaler.transform(x_test)
        # Fit a dnn keras regression model
        model = create_keras_regressor(x_train, y_train)
        self._verify_explain_model_local_regression_common(model, x_train, x_test, y_train, y_test,
                                                           feature_names,
                                                           include_evaluation_examples=include_evaluation_examples,
                                                           include_local=include_local,
                                                           has_explain_local=has_explain_local,
                                                           true_labels_required=true_labels_required)
        # Similar but now for a pytorch model as well
        model = create_pytorch_regressor(x_train, y_train)
        self._verify_explain_model_local_regression_common(model, x_train, x_test, y_train, y_test,
                                                           feature_names,
                                                           include_evaluation_examples=include_evaluation_examples,
                                                           include_local=include_local,
                                                           has_explain_local=has_explain_local,
                                                           true_labels_required=true_labels_required)

    def verify_explain_model_local_single(self):
        x_train, x_test, y_train, _, feature_names, target_names = create_iris_data()
        # Fit an SVM model
        model = create_sklearn_svm_classifier(x_train, y_train)
        explainer = self.create_explainer(model, x_train, features=feature_names, classes=target_names)
        explainer.explain_local(x_test[0])

    def verify_explain_model_pandas_input(self, include_evaluation_examples=True, include_local=True,
                                          has_explain_local=True, true_labels_required=False):
        x_train, x_test, y_train, y_test, feature_names, target_names = create_iris_data()
        x_train = pd.DataFrame(x_train, columns=feature_names)
        x_test = pd.DataFrame(x_test, columns=feature_names)
        # Fit an SVM model that only accepts pandas input
        pipeline = create_pandas_only_svm_classifier(x_train, y_train)
        explainer = self.create_explainer(pipeline, x_train, features=feature_names, classes=target_names)
        if include_evaluation_examples:
            if not include_local:
                # If include local is false (non-default), specify param
                explanation = explainer.explain_global(x_test, include_local=include_local)
            else:
                if true_labels_required:
                    explanation = explainer.explain_global(x_test, y_test)
                else:
                    explanation = explainer.explain_global(x_test)
                assert explanation.num_features == len(feature_names)
        else:
            explanation = explainer.explain_global()
        assert len(explanation.global_importance_values) == len(feature_names)
        if has_explain_local:
            explanation_local = explainer.explain_local(x_test)
            assert np.array(explanation_local.local_importance_values).shape[1] == len(x_test)
            assert explanation_local.num_examples == len(x_test)

    def verify_explain_model_int_features(self, is_per_class=True, include_evaluation_examples=True):
        x_train, x_test, y_train, _, feature_names, target_names = create_cancer_data()
        # Fit an SVM model
        model = create_sklearn_svm_classifier(x_train, y_train)

        # Create tabular explainer
        explainer = self.create_explainer(model, x_train, features=feature_names, classes=target_names)
        self.test_logger.info('Running explain global for verify_explain_model_int_features')
        if include_evaluation_examples:
            explanation = explainer.explain_global(x_test)
        else:
            explanation = explainer.explain_global()
        assert len(explanation.get_ranked_global_names()) == len(feature_names)
        if is_per_class:
            ranked_per_class_values = explanation.get_ranked_per_class_values()
            assert len(ranked_per_class_values) == len(target_names)
        explanation_local = explainer.explain_local(x_test)
        # Validate there is a local explanation per class for binary case
        assert np.array(explanation_local.local_importance_values).shape[0] == 2

    def verify_explain_model_npz_linear(self, include_evaluation_examples=True, true_labels_required=False):
        # run explain model on a real sparse dataset from the field
        x_train, x_test, y_train, y_test = create_msx_data(0.05)
        x_train = x_train[DATA_SLICE]
        x_test = x_test[DATA_SLICE]
        y_train = y_train[DATA_SLICE]
        y_test = y_test[DATA_SLICE]
        # Fit a linear regression model
        model = create_sklearn_linear_regressor(x_train, y_train.toarray().flatten())
        # Create tabular explainer
        explainer = self.create_explainer(model, x_train)
        self.test_logger.info('Running explain global for verify_explain_model_npz_linear')
        if self.specify_policy:
            policy = {ExplainParams.SAMPLING_POLICY: SamplingPolicy(allow_eval_sampling=True)}
        else:
            policy = {}
        if include_evaluation_examples:
            if true_labels_required:
                explainer.explain_global(x_test, y_test, **policy)
            else:
                explainer.explain_global(x_test, **policy)
        else:
            explainer.explain_global(**policy)

    def verify_explain_model_sparse(self, summarize_background=True, include_evaluation_examples=True,
                                    true_labels_required=False, include_local=True):
        X, y = retrieve_dataset('a1a.svmlight')
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
        x_train = x_train[DATA_SLICE]
        x_test = x_test[DATA_SLICE]
        y_train = y_train[DATA_SLICE]
        y_test = y_test[DATA_SLICE]
        # Fit a linear regression model
        model = create_sklearn_linear_regressor(x_train, y_train)
        if summarize_background:
            _, cols = x_train.shape
            shape = 1, cols
            background = csr_matrix(shape, dtype=x_train.dtype)
        else:
            background = x_train

        # Create tabular explainer
        explainer = self.create_explainer(model, background)
        self.test_logger.info('Running explain global for verify_explain_model_sparse')
        if self.specify_policy:
            policy = {ExplainParams.SAMPLING_POLICY: SamplingPolicy(allow_eval_sampling=True)}
        else:
            policy = {}
        if not include_local:
            policy[ExplainParams.INCLUDE_LOCAL] = include_local
        if include_evaluation_examples:
            if true_labels_required:
                explainer.explain_global(x_test, y_test, **policy)
            else:
                explainer.explain_global(x_test, **policy)
        else:
            explainer.explain_global(**policy)

    def verify_explain_model_hashing(self, summarize_background=True, include_evaluation_examples=True,
                                     true_labels_required=False, include_local=True, batch_size=None):
        # verifies we can run on very sparse data
        x_train, x_test, y_train, y_test, target_names, _ = create_multiclass_sparse_newsgroups_data()
        x_train = x_train[DATA_SLICE]
        x_test = x_test[DATA_SLICE]
        y_train = y_train[DATA_SLICE]
        y_test = y_test[DATA_SLICE]
        # Fit a logistic regression model for the multiclass classification data
        model = create_sklearn_logistic_regressor(x_train, y_train)
        self.test_logger.info('Running explain global for verify_explain_model_hashing')
        if summarize_background:
            background = _summarize_data(x_train)
        else:
            background = x_train
        # Create tabular explainer
        explainer = self.create_explainer(
            model, background, model_task=ModelTask.Classification, classes=target_names)
        if self.specify_policy:
            policy = {ExplainParams.SAMPLING_POLICY: SamplingPolicy(allow_eval_sampling=True)}
        else:
            policy = {}
        if not include_local:
            policy[ExplainParams.INCLUDE_LOCAL] = include_local
        if batch_size is not None:
            policy[ExplainParams.BATCH_SIZE] = batch_size
        if include_evaluation_examples:
            if true_labels_required:
                explainer.explain_global(x_test, y_test, **policy)
            else:
                explainer.explain_global(x_test, **policy)
        else:
            explainer.explain_global(**policy)

    def verify_explain_model_with_summarize_data(self, expected_overall_features, expected_per_class_features=None,
                                                 num_overall_features_equal=-1):
        x_train, x_test, y_train, _, feature_names, target_names = create_iris_data()
        # Fit an SVM model
        model = create_sklearn_svm_classifier(x_train, y_train)

        # Create tabular explainer
        summary = _summarize_data(x_train, 10)
        explainer = self.create_explainer(model, summary, features=feature_names, classes=target_names)
        self.test_logger.info('Running explain global for verify_explain_model_with_summarize_data')
        summary = _summarize_data(x_train, 10)
        explanation = explainer.explain_global(x_test)
        self.verify_iris_overall_features(explanation.get_ranked_global_names(),
                                          explanation.get_ranked_global_values(),
                                          expected_overall_features,
                                          num_overall_features_equal)
        self.verify_iris_per_class_features(explanation.get_ranked_per_class_names(),
                                            explanation.get_ranked_per_class_values(),
                                            expected_per_class_features)

    def verify_explain_model_subset_classification_dense(self, is_local=True,
                                                         true_labels_required=False):
        # Verify explaining a subset of the features
        X, y = shap.datasets.adult()
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=7)
        # Fit a tree model
        model = create_sklearn_logistic_regressor(x_train, y_train)

        # Create tabular explainer
        classes = [" <=50K", " >50K"]
        explainer = self.create_explainer(model, x_train, features=list(range(x_train.shape[1])), classes=classes)
        self.test_logger.info('Running explain global for verify_explain_model_subset_classification_dense')
        # Get most important features
        if true_labels_required:
            o16n_explanation = explainer.explain_global(x_test, y_test)
        else:
            o16n_explanation = explainer.explain_global(x_test)
        ranked_global_names = o16n_explanation.get_ranked_global_names()
        column_subset = ranked_global_names[:5]
        # Run explain model again but this time only on the feature subset and on a single row
        x_test_row = x_test.values[0, :].reshape((1, x_test.values[0, :].shape[0]))
        explainer = self.create_explainer(model, x_train, features=X.columns.values,
                                          explain_subset=column_subset, classes=classes)
        if true_labels_required:
            explainer.explain_global(x_test_row, y_test[0:1])
            # Run it again but for multiple rows (the entire test set)
            explainer.explain_global(x_test, y_test)
        else:
            explainer.explain_global(x_test_row)
            # Run it again but for multiple rows (the entire test set)
            explainer.explain_global(x_test)

    def verify_explain_model_subset_regression_sparse(self, is_local=True,
                                                      true_labels_required=False):
        # Verify explaining a subset of the features, but on sparse regression data
        x_train, x_test, y_train, y_test = create_msx_data(0.01)
        DATA_SLICE = slice(100)
        x_train = x_train[DATA_SLICE]
        x_test = x_test[DATA_SLICE]
        y_train = y_train[DATA_SLICE]
        y_test = y_test[DATA_SLICE]
        # Fit a linear regression model
        model = create_sklearn_linear_regressor(x_train, y_train.toarray().flatten())
        # Create tabular explainer
        explainer = self.create_explainer(model, x_train, features=list(range(x_train.shape[1])))
        self.test_logger.info('Running explain global for verify_explain_model_subset_regression_sparse')
        # Get most important features
        if true_labels_required:
            o16n_explanation = explainer.explain_global(x_test, y_test)
        else:
            o16n_explanation = explainer.explain_global(x_test)
        ranked_global_names = o16n_explanation.get_ranked_global_names()
        column_subset = ranked_global_names[:5]
        # Run explain model again but this time only on the feature subset and on a single row
        x_test_row = x_test[0, :]
        explainer = self.create_explainer(model, x_train, explain_subset=column_subset)
        if true_labels_required:
            explainer.explain_global(x_test_row, y_test[0:1])
        else:
            explainer.explain_global(x_test_row)
        # Run it again but for multiple rows (the entire test set)
        if true_labels_required:
            explanation_subset = explainer.explain_global(x_test, y_test)
        else:
            explanation_subset = explainer.explain_global(x_test)
        if is_local:
            local_importance_values = o16n_explanation.local_importance_values
            local_importance_values_subset = explanation_subset.local_importance_values
            # Compare results to initial explanation
            res = np.isclose(local_importance_values_subset,
                             np.array(local_importance_values)[:, column_subset], 0.2, 0.1)
            total_in_threshold = np.sum(res)
            total_elems = res.shape[0] * res.shape[1]
            correct_ratio = total_in_threshold / total_elems
            # Surprisingly, they are almost identical!
            assert correct_ratio > 0.9

    def verify_explain_model_subset_classification_sparse(self, is_local=True,
                                                          true_labels_required=False):
        # verifies explaining on a subset of features with sparse classification data
        x_train, x_test, y_train, y_test, classes, _ = create_multiclass_sparse_newsgroups_data()
        x_train = x_train[DATA_SLICE]
        x_test = x_test[DATA_SLICE]
        y_train = y_train[DATA_SLICE]
        y_test = y_test[DATA_SLICE]
        # Fit a logistic regression classification model
        model = create_sklearn_logistic_regressor(x_train, y_train)
        # Create tabular explainer
        explainer = self.create_explainer(model, x_train, features=list(range(x_train.shape[1])), classes=classes)
        self.test_logger.info('Running explain global for verify_explain_model_subset_classification_sparse')
        # Get most important features
        if true_labels_required:
            o16n_explanation = explainer.explain_global(x_test, y_test)
        else:
            o16n_explanation = explainer.explain_global(x_test)
        ranked_global_names = o16n_explanation.get_ranked_global_names()
        column_subset = ranked_global_names[:5]
        # Run explain model again but this time only on the feature subset and on a single row
        x_test_row = x_test[0, :]
        explainer = self.create_explainer(model, x_train, explain_subset=column_subset, classes=classes)
        if true_labels_required:
            explainer.explain_global(x_test_row, y_test[0:1])
            # Run it again but for multiple rows (the entire test set)
            explanation_subset = explainer.explain_global(x_test, y_test)
        else:
            explainer.explain_global(x_test_row)
            # Run it again but for multiple rows (the entire test set)
            explanation_subset = explainer.explain_global(x_test)
        if is_local:
            local_importance_values = o16n_explanation.local_importance_values
            local_importance_values_subset = explanation_subset.local_importance_values
            # Compare results to initial explanation
            for i in range(len(local_importance_values_subset)):
                res = np.isclose(local_importance_values_subset[i],
                                 np.array(local_importance_values[i])[:, column_subset], 0.2, 0.1)
                total_in_threshold = np.sum(res)
                total_elems = res.shape[0] * res.shape[1]
                correct_ratio = total_in_threshold / total_elems
                # Surprisingly, they are almost identical!
                assert correct_ratio > 0.9

    def verify_explain_model_with_sampling_regression_sparse(self, true_labels_required=False):
        # Verify that evaluation dataset can be downsampled
        x_train, x_test, y_train, y_test = create_msx_data(0.2)
        x_train = x_train[DATA_SLICE]
        x_test = x_test[DATA_SLICE]
        y_train = y_train[DATA_SLICE]
        y_test = y_test[DATA_SLICE]
        # Fit a linear regression model
        model = create_sklearn_linear_regressor(x_train, y_train.toarray().flatten())
        # Create tabular explainer
        explainer = self.create_explainer(model, x_train, features=list(range(x_train.shape[1])))
        self.test_logger.info('Running explain global for '
                              'verify_explain_model_with_sampling_regression_sparse')
        # Sample the evaluation dataset with multiple runs of KMeans
        if self.specify_policy:
            policy = {ExplainParams.SAMPLING_POLICY: SamplingPolicy(allow_eval_sampling=True)}
        else:
            policy = {}
        # Get most important features
        if true_labels_required:
            explainer.explain_global(x_test[:5], y_test[:5], **policy)
        else:
            explainer.explain_global(x_test[:5], **policy)

    def verify_explain_model_throws_on_bad_classifier_and_classes(self):
        # Verify that explain model throws when specifying a classifier without predict_proba and classes parameter
        x_train, x_test, y_train, y_test, feature_names, target_names = create_iris_data()
        # Fit an SVM model, but specify that it should not define a predict_proba function
        model = create_sklearn_svm_classifier(x_train, y_train, probability=False)
        self.test_logger.info('Running explain model for verify_explain_model_throws_on_bad_classifier_and_classes')
        with pytest.raises(
                ValueError,
                match="Classes is specified but model does not define predict_proba, only predict."):
            self.create_explainer(model, x_train, features=feature_names, classes=target_names)

    def verify_explain_model_throws_on_bad_pipeline_and_classes(self):
        # Verify that explain model throws when specifying a predict pipeline and classes parameter
        x_train, x_test, y_train, y_test, feature_names, target_names = create_iris_data()
        # Fit an SVM model, but specify that it should not define a predict_proba function
        model = create_sklearn_svm_classifier(x_train, y_train, probability=False)
        self.test_logger.info('Running explain model for verify_explain_model_throws_on_bad_pipeline_and_classes')
        with pytest.raises(
                ValueError,
                match="Classes is specified but function was predict, not predict_proba."):
            self.create_explainer(model.predict, x_train, is_function=True,
                                  features=feature_names, classes=target_names)

    def verify_explain_model_throws_on_classifier_and_no_classes(self):
        # Verify that explain model throws when specifying a classifier but no classes parameter
        x_train, x_test, y_train, _, feature_names, _ = create_iris_data()
        # Fit an SVM model
        model = create_sklearn_svm_classifier(x_train, y_train)
        self.test_logger.info('Running explain model for verify_explain_model_throws_on_classifier_and_no_classes')
        with pytest.raises(
                ValueError,
                match="LIME Explainer requires classes to be specified if using a classification model"):
            self.create_explainer(model, x_train, features=feature_names)

    def verify_explain_model_transformations_list_classification(self, create_model=None,
                                                                 true_labels_required=False):
        self._verify_explain_model_transformations_classification(
            TransformationType.TransformationsList, get_transformations_one_to_many_smaller,
            create_model, true_labels_required)
        self._verify_explain_model_transformations_classification(
            TransformationType.TransformationsList, get_transformations_one_to_many_greater,
            create_model, true_labels_required)
        self._verify_explain_model_transformations_classification(
            TransformationType.TransformationsList, get_transformations_many_to_many,
            create_model, true_labels_required, allow_all_transformations=True
        )

    def verify_explain_model_transformations_column_transformer_classification(self, create_model=None,
                                                                               true_labels_required=False):
        self._verify_explain_model_transformations_classification(
            TransformationType.ColumnTransformer, get_transformations_one_to_many_smaller,
            create_model, true_labels_required)
        self._verify_explain_model_transformations_classification(
            TransformationType.ColumnTransformer, get_transformations_one_to_many_greater,
            create_model, true_labels_required)
        self._verify_explain_model_transformations_classification(
            TransformationType.ColumnTransformer, get_transformations_many_to_many,
            create_model, true_labels_required, allow_all_transformations=True
        )

    def verify_explain_model_transformations_list_regression(self, create_model=None,
                                                             true_labels_required=False):
        self._verify_explain_model_transformations_regression(
            TransformationType.TransformationsList, get_transformations_one_to_many_smaller,
            create_model, true_labels_required)
        self._verify_explain_model_transformations_regression(
            TransformationType.TransformationsList, get_transformations_one_to_many_greater,
            create_model, true_labels_required)
        self._verify_explain_model_transformations_regression(
            TransformationType.TransformationsList, get_transformations_many_to_many,
            create_model, true_labels_required, allow_all_transformations=True
        )

    def verify_explain_model_transformations_column_transformer_regression(self, create_model=None,
                                                                           true_labels_required=False):
        self._verify_explain_model_transformations_regression(
            TransformationType.ColumnTransformer, get_transformations_one_to_many_smaller,
            create_model, true_labels_required)
        self._verify_explain_model_transformations_regression(
            TransformationType.ColumnTransformer, get_transformations_one_to_many_greater,
            create_model, true_labels_required)
        self._verify_explain_model_transformations_regression(
            TransformationType.ColumnTransformer, get_transformations_many_to_many,
            create_model, true_labels_required, allow_all_transformations=True
        )

    def verify_explain_model_shap_values_multiclass(self, shap_values_output=ShapValuesOutput.DEFAULT,
                                                    model_type=ModelType.DEFAULT):
        x_train, x_test, y_train, _, feature_names, target_names = create_iris_data()
        if model_type == ModelType.XGBOOST:
            # Fit an XGBoost model
            model = create_xgboost_classifier(x_train, y_train)
        elif model_type == ModelType.TREE:
            # Fit an sklearn tree-based model
            model = create_sklearn_random_forest_classifier(x_train, y_train)
        else:
            # Fit an SVM model
            model = create_sklearn_svm_classifier(x_train, y_train)

        # Create tabular explainer
        kwargs = {}
        kwargs[ExplainParams.SHAP_VALUES_OUTPUT] = shap_values_output
        explainer = self.create_explainer(model, x_train, features=feature_names, classes=target_names, **kwargs)
        self.test_logger.info('Running explain global for verify_explain_model_shap_values_multiclass')
        explanation = explainer.explain_global(x_test)
        is_probability = shap_values_output != ShapValuesOutput.DEFAULT
        self.validate_explanation(explanation, is_multiclass=True, is_probability=is_probability)
        # validate explanation has init_data on it in mimic explainer case (note there is none for TreeExplainer)
        if hasattr(explainer, SURROGATE_MODEL):
            assert explanation.init_data is not None

    def verify_explain_model_shap_values_binary(self, shap_values_output=ShapValuesOutput.DEFAULT,
                                                model_type=ModelType.DEFAULT):
        x_train, x_test, y_train, _, feature_names, target_names = create_cancer_data()
        if model_type == ModelType.XGBOOST:
            # Fit an XGBoost model, which is handled in special way currently
            model = create_xgboost_classifier(x_train, y_train)
        elif model_type == ModelType.TREE:
            # Fit an sklearn tree-based model
            model = create_sklearn_random_forest_classifier(x_train, y_train)
        else:
            # Fit an SVM model
            model = create_sklearn_svm_classifier(x_train, y_train)

        # Create tabular explainer
        kwargs = {}
        kwargs[ExplainParams.SHAP_VALUES_OUTPUT] = shap_values_output
        explainer = self.create_explainer(model, x_train, features=feature_names, classes=target_names, **kwargs)
        self.test_logger.info('Running explain global for verify_explain_model_shap_values_binary')
        explanation = explainer.explain_global(x_test)
        is_probability = shap_values_output != ShapValuesOutput.DEFAULT
        has_surrogate_model = hasattr(explainer, SURROGATE_MODEL)
        has_explainer_probas = not has_surrogate_model and shap_values_output == ShapValuesOutput.PROBABILITY
        if shap_values_output == ShapValuesOutput.TEACHER_PROBABILITY or has_explainer_probas:
            model_output = model.predict_proba(x_test)
        else:
            if has_surrogate_model:
                predictions = explainer.surrogate_model.predict(x_test)
            elif str(type(model)).endswith("XGBClassifier'>"):
                predictions = model.predict(x_test, output_margin=True)
            else:
                predictions = model.predict(x_test)
            model_output = expit(predictions)
            model_output = np.stack((1 - model_output, model_output), axis=-1)
        self.validate_explanation(explanation, is_probability=is_probability, model_output=model_output)

    def verify_explain_model_shap_values_regression(self, shap_values_output=ShapValuesOutput.DEFAULT,
                                                    model_type=ModelType.DEFAULT):
        x_train, x_test, y_train, y_test, feature_names = create_energy_data()
        if model_type == ModelType.TREE:
            # Fit a tree-based model
            model = create_sklearn_random_forest_regressor(x_train, y_train)
        else:
            # Fit a linear model
            model = create_sklearn_linear_regressor(x_train, y_train)

        # Create tabular explainer
        kwargs = {}
        kwargs[ExplainParams.SHAP_VALUES_OUTPUT] = shap_values_output
        explainer = self.create_explainer(model, x_train, features=feature_names, **kwargs)

        self.test_logger.info('Running explain global for verify_explain_model_shap_values_regression')
        explanation = explainer.explain_global(x_test)
        model_output = model.predict(x_test)
        self.validate_explanation(explanation, is_probability=False, is_regression=True,
                                  model_output=model_output)

    def verify_explain_model_categorical(self, pass_categoricals=False, verify_same_shape=False):
        headers = ["symboling", "normalized_losses", "make", "fuel_type", "aspiration",
                   "num_doors", "body_style", "drive_wheels", "engine_location",
                   "wheel_base", "length", "width", "height", "curb_weight",
                   "engine_type", "num_cylinders", "engine_size", "fuel_system",
                   "bore", "stroke", "compression_ratio", "horsepower", "peak_rpm",
                   "city_mpg", "highway_mpg", "price"]
        df = retrieve_dataset('imports-85.csv', header=None, names=headers, na_values="?")
        df_y = df['price']
        df_X = df.drop(columns='price')
        df_train_X, df_test_X, df_train_y, df_test_y = train_test_split(df_X, df_y, test_size=0.2, random_state=7)
        # Encode strings to ordinal values
        categorical_col_names = list(df_train_X.select_dtypes(include='object').columns)
        categorical_col_indices = [df_train_X.columns.get_loc(col_name) for col_name in categorical_col_names]
        kwargs = {'num_leaves': 31, 'num_trees': 100, 'objective': 'regression',
                  'categorical_feature': categorical_col_indices}
        lgbm_regressor = LGBMRegressor(**kwargs)
        # Impute the x and y values
        imp_X = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        imp_y = SimpleImputer(missing_values=np.nan, strategy='mean')
        # reshape to 2D array since SimpleImputer can't work on 1D array
        df_train_y = df_train_y.values.reshape(df_train_y.shape[0], 1)
        imp_y.fit(df_train_y)
        imp_df_y = imp_y.transform(df_train_y)
        imp_X.fit(df_train_X)
        imp_train_X = pd.DataFrame(imp_X.transform(df_train_X))

        class CustomTextTransformer(BaseEstimator, TransformerMixin):
            def __init__(self):
                return

            def fit(self, X, y=None):
                return self

            def transform(self, X):
                return X.astype('U')

        custom_text = CustomTextTransformer()
        encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
        ct1 = ColumnTransformer([('cu', custom_text, categorical_col_indices)], remainder='passthrough')
        ct2 = ColumnTransformer([('ord', encoder, slice(0, len(categorical_col_indices)))], remainder='passthrough')
        pipeline = Pipeline([('cu', ct1), ('ct', ct2), ('lgbm', lgbm_regressor)])
        pipeline.fit(imp_train_X, imp_df_y[:, 0])
        if pass_categoricals:
            explainer = self.create_explainer(pipeline, imp_train_X, categorical_features=categorical_col_indices)
        else:
            explainer = self.create_explainer(pipeline, imp_train_X)
        explanation = explainer.explain_global(imp_X.transform(df_test_X))
        verify_serialization(explanation, exist_ok=True)
        # validate explanation when passing categoricals is for raw features for some specific explainers
        if pass_categoricals and verify_same_shape:
            assert np.array(explanation.local_importance_values).shape[1] == df_train_X.shape[1]

    def validate_explanation(self, explanation, is_multiclass=False, is_probability=False,
                             is_regression=False, model_output=None):
        verify_serialization(explanation, exist_ok=True)
        if is_regression:
            for idx, row in enumerate(explanation.local_importance_values):
                features = np.array(row)
                sum_features = np.sum(features)
                if len(explanation.expected_values) > 1:
                    expected_value = explanation.expected_values[idx]
                else:
                    expected_value = explanation.expected_values
                total_sum = expected_value + sum_features
                # Verify the sum of the expected values and feature importance values
                # matches the teacher model's output
                assert abs(model_output[idx] - total_sum) < TOLERANCE
        else:
            for class_idx, class_explanation in enumerate(explanation.local_importance_values):
                for row_idx, row in enumerate(class_explanation):
                    features = np.array(row)
                    sum_features = np.sum(features)
                    if isinstance(explanation.expected_values, list):
                        expected_value = explanation.expected_values[class_idx]
                    else:
                        expected_value = explanation.expected_values
                    total_sum = expected_value + sum_features
                    # Verify sum of expected values and feature importance values
                    # with inverse logit applied is a probability
                    if not is_probability:
                        predicted_probability = expit(total_sum)
                    else:
                        predicted_probability = total_sum
                    ubound = 1.0 + PROBABILITY_TOLERANCE
                    lbound = 0.0 - PROBABILITY_TOLERANCE
                    assert predicted_probability <= ubound
                    assert predicted_probability >= lbound
                    if model_output is not None:
                        assert abs(predicted_probability - model_output[row_idx, class_idx]) < TOLERANCE

    def verify_energy_overall_features(self,
                                       ranked_global_names,
                                       ranked_global_values):
        # Verify order of features
        self.test_logger.info("length of ranked_global_values: %s", str(len(ranked_global_values)))
        assert len(ranked_global_values) == len(ranked_global_values)
        assert len(ranked_global_values) == 8

    def verify_iris_overall_features(self,
                                     ranked_global_names,
                                     ranked_global_values,
                                     expected_overall_features,
                                     num_overall_features_equal=-1):
        # Verify order of features
        self.test_logger.info("length of ranked_global_values: %s", str(len(ranked_global_values)))
        if num_overall_features_equal < 0:
            np.testing.assert_array_equal(ranked_global_names, expected_overall_features)
        else:
            np.testing.assert_array_equal(ranked_global_names[0:num_overall_features_equal - 1],
                                          expected_overall_features[0:num_overall_features_equal - 1])
        assert len(ranked_global_values) == 4

    def verify_iris_per_class_features(self,
                                       ranked_per_class_names,
                                       ranked_per_class_values,
                                       expected_per_class_features):
        # Verify order of features
        np.testing.assert_array_equal(ranked_per_class_names, expected_per_class_features)
        assert len(ranked_per_class_values) == np.array(expected_per_class_features).shape[0]
        assert len(ranked_per_class_values[0]) == np.array(expected_per_class_features).shape[1]

    @property
    def iris_overall_expected_features(self):
        return ['petal length', 'petal width', 'sepal length', 'sepal width']
