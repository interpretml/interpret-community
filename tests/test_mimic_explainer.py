# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

# Tests for MIMIC Explainer
import json
import logging
import uuid
from sys import platform

import numpy as np
import pandas as pd
import pytest
import scipy
from common_utils import (LIGHTGBM_METHOD, LINEAR_METHOD,
                          create_binary_classification_dataset,
                          create_cancer_data, create_iris_data,
                          create_lightgbm_regressor,
                          create_pytorch_single_output_classifier,
                          create_timeseries_data)
from constants import ModelType, owner_email_tools_and_ux
from datasets import retrieve_dataset
from interpret_community.common.constants import ModelTask, ShapValuesOutput
from interpret_community.common.exception import ScenarioNotSupportedException
from interpret_community.mimic.mimic_explainer import SMALL_DATA_THRESHOLD
from interpret_community.mimic.models.lightgbm_model import \
    LGBMExplainableModel
from interpret_community.mimic.models.linear_model import (
    LinearExplainableModel, SGDExplainableModel)
from interpret_community.mimic.models.tree_model import \
    DecisionTreeExplainableModel
from models import (DataFrameTestModel,
                    PredictAsDataFrameClassificationTestModel,
                    PredictAsDataFrameREgressionTestModel, SkewedTestModel)
from sklearn import datasets
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (LinearRegression, LogisticRegression,
                                  SGDClassifier)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

test_logger = logging.getLogger(__name__)
test_logger.setLevel(logging.DEBUG)

LGBM_MODEL_IDX = 0
SGD_MODEL_IDX = 2
MACOS_PLATFORM = 'darwin'


@pytest.mark.owner(email=owner_email_tools_and_ux)
@pytest.mark.usefixtures('_clean_dir')
class TestMimicExplainer(object):
    def test_working(self):
        assert True

    def test_explain_model_local(self, verify_mimic_classifier):
        iris_overall_expected_features = self.iris_overall_expected_features
        iris_per_class_expected_features = self.iris_per_class_expected_features
        num_overall_features_equal = 2
        is_macos = platform.startswith(MACOS_PLATFORM)
        # Don't check per class features on MACOS
        is_per_class = not is_macos
        for idx, verifier in enumerate(verify_mimic_classifier):
            verifier.verify_explain_model_local(iris_overall_expected_features[idx],
                                                iris_per_class_expected_features[idx],
                                                num_overall_features_equal=num_overall_features_equal,
                                                is_per_class=is_per_class)

    def test_explain_model_local_dnn(self, verify_mimic_classifier):
        for verifier in verify_mimic_classifier:
            verifier.verify_explain_model_local_dnn()

    def test_explain_model_local_without_evaluation_examples(self, verify_mimic_classifier):
        iris_overall_expected_features = self.iris_overall_expected_features_without_evaluation
        is_macos = platform.startswith(MACOS_PLATFORM)
        if is_macos:
            num_overall_features_equal = 1
        else:
            num_overall_features_equal = -1
        for idx, verifier in enumerate(verify_mimic_classifier):
            verifier.verify_explain_model_local(iris_overall_expected_features[idx],
                                                is_per_class=False,
                                                include_evaluation_examples=False,
                                                num_overall_features_equal=num_overall_features_equal)

    def test_explain_model_local_without_include_local(self, verify_mimic_classifier):
        iris_overall_expected_features = self.iris_overall_expected_features
        iris_per_class_expected_features = self.iris_per_class_expected_features
        num_overall_features_equal = 2
        is_macos = platform.startswith(MACOS_PLATFORM)
        # Don't check per class features on MACOS
        is_per_class = not is_macos
        for idx, verifier in enumerate(verify_mimic_classifier):
            verifier.verify_explain_model_local(iris_overall_expected_features[idx],
                                                iris_per_class_expected_features[idx],
                                                include_local=False,
                                                is_per_class=is_per_class,
                                                num_overall_features_equal=num_overall_features_equal)

    def test_explain_model_local_regression_without_include_local(self, verify_mimic_regressor):
        for verifier in verify_mimic_regressor:
            verifier.verify_explain_model_local_regression(include_local=False)

    def test_explain_model_local_regression_dnn(self, verify_mimic_regressor):
        for verifier in verify_mimic_regressor:
            verifier.verify_explain_model_local_regression_dnn()

    def test_explain_model_pandas_input(self, verify_mimic_classifier):
        for verifier in verify_mimic_classifier:
            verifier.verify_explain_model_pandas_input()

    def test_explain_model_pandas_input_without_include_local(self, verify_mimic_classifier):
        for verifier in verify_mimic_classifier:
            verifier.verify_explain_model_pandas_input(include_local=False)

    def test_explain_model_pandas_input_without_evaluation_examples(self, verify_mimic_classifier):
        for verifier in verify_mimic_classifier:
            verifier.verify_explain_model_pandas_input(include_local=False, include_evaluation_examples=False)

    def test_explain_model_int_features(self, verify_mimic_regressor):
        for verifier in verify_mimic_regressor:
            verifier.verify_explain_model_int_features(is_per_class=False)

    def test_explain_model_int_features_without_evaluation_examples(self, verify_mimic_regressor):
        for verifier in verify_mimic_regressor:
            verifier.verify_explain_model_int_features(is_per_class=False, include_evaluation_examples=False)

    def test_explain_model_npz_linear(self, verify_sparse_mimic):
        for verifier in verify_sparse_mimic:
            verifier.verify_explain_model_npz_linear()

    def test_explain_model_npz_linear_without_evaluation_examples(self, verify_sparse_mimic):
        for verifier in verify_sparse_mimic:
            verifier.verify_explain_model_npz_linear(include_evaluation_examples=False)

    def test_explain_model_sparse(self, verify_sparse_mimic):
        for verifier in verify_sparse_mimic:
            verifier.verify_explain_model_sparse(summarize_background=False)

    def test_explain_model_sparse_without_evaluation_examples(self, verify_sparse_mimic):
        for verifier in verify_sparse_mimic:
            verifier.verify_explain_model_sparse(summarize_background=False,
                                                 include_evaluation_examples=False)

    def test_explain_model_sparse_without_include_local(self, verify_sparse_mimic):
        for verifier in verify_sparse_mimic:
            verifier.verify_explain_model_sparse(summarize_background=False,
                                                 include_local=False)

    def test_explain_model_hashing(self, verify_sparse_mimic):
        for verifier in verify_sparse_mimic:
            verifier.verify_explain_model_hashing(summarize_background=False)

    def test_explain_model_hashing_without_evaluation_examples(self, verify_sparse_mimic):
        for verifier in verify_sparse_mimic:
            verifier.verify_explain_model_hashing(summarize_background=False,
                                                  include_evaluation_examples=False)

    @pytest.mark.parametrize(("batch_size"), [None, 1, 100])
    @pytest.mark.parametrize(("include_evaluation_examples"), [True, False])
    def test_explain_model_hashing_without_include_local(self, verify_sparse_mimic,
                                                         batch_size,
                                                         include_evaluation_examples):
        for verifier in verify_sparse_mimic:
            verifier.verify_explain_model_hashing(summarize_background=False,
                                                  include_evaluation_examples=include_evaluation_examples,
                                                  include_local=False,
                                                  batch_size=batch_size)

    def test_explain_model_subset_classification_dense(self, verify_mimic_regressor):
        for verifier in verify_mimic_regressor:
            verifier.verify_explain_model_subset_classification_dense(is_local=False)

    def test_explain_model_subset_regression_sparse(self, verify_sparse_mimic):
        for verifier in verify_sparse_mimic:
            verifier.verify_explain_model_subset_regression_sparse(is_local=False)

    def test_explain_model_subset_classification_sparse(self, verify_sparse_mimic):
        for verifier in verify_sparse_mimic:
            verifier.verify_explain_model_subset_classification_sparse(is_local=False)

    def test_explain_model_with_sampling_regression_sparse(self, verify_sparse_mimic):
        for verifier in verify_sparse_mimic:
            verifier.verify_explain_model_with_sampling_regression_sparse()

    def test_explain_model_local_single(self, verify_sparse_mimic):
        for verifier in verify_sparse_mimic:
            verifier.verify_explain_model_local_single()

    def test_explain_model_with_special_args(self, verify_mimic_special_args):
        num_overall_features_equal = -1
        is_macos = platform.startswith(MACOS_PLATFORM)
        if is_macos:
            num_overall_features_equal = 2
        # Don't check per class features on MACOS
        is_per_class = not is_macos
        for idx, verifier in enumerate(verify_mimic_special_args):
            iris_overall_expected_features = self.iris_overall_expected_features_special_args
            iris_per_class_expected_features = self.iris_per_class_expected_features_special_args
            # retrying 4 times in case this test fails due to a lightgbm bug
            for _ in range(4):
                try:
                    verifier.verify_explain_model_local(iris_overall_expected_features[idx],
                                                        iris_per_class_expected_features[idx],
                                                        is_per_class=is_per_class,
                                                        num_overall_features_equal=num_overall_features_equal)
                    break
                except json.decoder.JSONDecodeError:
                    pass

    def test_explain_with_transformations_list_classification(self, verify_mimic_classifier):
        for verifier in verify_mimic_classifier:
            verifier.verify_explain_model_transformations_list_classification()

    def test_explain_with_transformations_column_transformer_classification(self, verify_mimic_classifier):
        for verifier in verify_mimic_classifier:
            verifier.verify_explain_model_transformations_column_transformer_classification()

    def test_explain_with_transformations_list_regression(self, verify_mimic_regressor):
        for verifier in verify_mimic_regressor:
            verifier.verify_explain_model_transformations_list_regression()

    def test_explain_with_transformations_column_transformer_regression(self, verify_mimic_regressor):
        for verifier in verify_mimic_regressor:
            verifier.verify_explain_model_transformations_column_transformer_regression()

    def test_explain_model_shap_values_binary(self, verify_mimic_regressor):
        for verifier in verify_mimic_regressor:
            verifier.verify_explain_model_shap_values_binary()

    def test_explain_model_shap_values_binary_xgboost(self, verify_mimic_regressor):
        for verifier in verify_mimic_regressor:
            verifier.verify_explain_model_shap_values_binary(model_type=ModelType.XGBOOST)

    def test_explain_model_shap_values_multiclass(self, verify_mimic_classifier):
        for verifier in verify_mimic_classifier:
            verifier.verify_explain_model_shap_values_multiclass()

    def test_explain_model_shap_values_binary_proba(self, verify_mimic_classifier):
        # Note: only LGBMExplainableModel supports conversion to probabilities for now
        verify_mimic_classifier[LGBM_MODEL_IDX].verify_explain_model_shap_values_binary(ShapValuesOutput.PROBABILITY)

    def test_explain_model_shap_values_binary_proba_xgboost(self, verify_mimic_classifier):
        # Note: only LGBMExplainableModel supports conversion to probabilities for now
        verify_mimic_classifier[LGBM_MODEL_IDX].verify_explain_model_shap_values_binary(ShapValuesOutput.PROBABILITY,
                                                                                        model_type=ModelType.XGBOOST)

    def test_explain_model_shap_values_multiclass_proba(self, verify_mimic_classifier):
        # Note: only LGBMExplainableModel supports conversion to probabilities for now
        verify_mimic = verify_mimic_classifier[LGBM_MODEL_IDX]
        verify_mimic.verify_explain_model_shap_values_multiclass(ShapValuesOutput.PROBABILITY)

    def test_explain_model_shap_values_binary_teacher_proba(self, verify_mimic_classifier):
        # Note: only LGBMExplainableModel supports conversion to probabilities for now
        verify_mimic = verify_mimic_classifier[LGBM_MODEL_IDX]
        verify_mimic.verify_explain_model_shap_values_binary(ShapValuesOutput.TEACHER_PROBABILITY)

    def test_explain_model_shap_values_binary_teacher_proba_xgboost(self, verify_mimic_classifier):
        # Note: only LGBMExplainableModel supports conversion to probabilities for now
        verify_mimic = verify_mimic_classifier[LGBM_MODEL_IDX]
        verify_mimic.verify_explain_model_shap_values_binary(ShapValuesOutput.TEACHER_PROBABILITY,
                                                             model_type=ModelType.XGBOOST)

    def test_explain_model_shap_values_multiclass_teacher_proba(self, verify_mimic_classifier):
        # Note: only LGBMExplainableModel supports conversion to probabilities for now
        verify_mimic = verify_mimic_classifier[LGBM_MODEL_IDX]
        verify_mimic.verify_explain_model_shap_values_multiclass(ShapValuesOutput.TEACHER_PROBABILITY)

    def test_explain_model_shap_values_regression_teacher(self, verify_mimic_regressor):
        # Note: only LGBMExplainableModel supports conversion to probabilities for now
        verify_mimic = verify_mimic_regressor[LGBM_MODEL_IDX]
        verify_mimic.verify_explain_model_shap_values_regression(ShapValuesOutput.TEACHER_PROBABILITY)

    def _validate_model_serialization(self, model, x_train, x_test, mimic_explainer):
        explainable_model = LGBMExplainableModel
        explainer = mimic_explainer(model, x_train, explainable_model, max_num_of_augmentations=10)
        global_explanation = explainer.explain_global(x_test, include_local=False)
        # Save the explainer to dictionary
        properties = explainer._save()
        # Restore from dictionary
        deserialized_explainer = mimic_explainer._load(model, properties)
        # validate we didn't miss any properties on the current explainable model
        for key in explainer.__dict__:
            if key not in deserialized_explainer.__dict__:
                raise Exception('Key {} missing from serialized mimic explainable model'.format(key))
        # Run explain global on deserialized guy
        de_global_explanation = deserialized_explainer.explain_global(x_test, include_local=False)
        np.testing.assert_array_equal(global_explanation.global_importance_values,
                                      de_global_explanation.global_importance_values)
        assert global_explanation.method == LIGHTGBM_METHOD

    def test_explain_model_categorical(self, verify_mimic_regressor):
        for idx, verifier in enumerate(verify_mimic_regressor):
            verify_same_shape = idx == LGBM_MODEL_IDX
            verifier.verify_explain_model_categorical(pass_categoricals=True,
                                                      verify_same_shape=verify_same_shape)

    @pytest.mark.parametrize(("sample_cnt_per_grain", "grains_dict"), [
        (240, {}),
        (20, {'fruit': ['apple', 'grape'], 'store': [100, 200, 50]})])
    def test_dataframe_model(self, mimic_explainer, sample_cnt_per_grain, grains_dict):
        X, _ = create_timeseries_data(sample_cnt_per_grain, 'time', 'y', grains_dict)
        model = DataFrameTestModel(X.copy())
        model = Pipeline([('test', model)])
        features = list(X.columns.values) + list(X.index.names)
        model_task = ModelTask.Unknown
        kwargs = {'explainable_model_args': {'n_jobs': 1}, 'augment_data': False, 'reset_index': True}
        if grains_dict:
            kwargs['categorical_features'] = ['fruit']
        mimic_explainer(model, X, LGBMExplainableModel, features=features, model_task=model_task, **kwargs)

    def _timeseries_generated_data(self):
        # Load diabetes data and convert to data frame
        x, y = datasets.load_diabetes(return_X_y=True)
        nrows, ncols = x.shape
        column_names = [str(i) for i in range(ncols)]
        X = pd.DataFrame(x, columns=column_names)

        # Add an arbitrary time axis
        time_column_name = "Date" + str(uuid.uuid4())
        dates = pd.date_range('1980-01-01', periods=nrows, freq='MS')
        X[time_column_name] = dates
        index_keys = [time_column_name]
        X.set_index(index_keys, inplace=True)

        # Split into train and test sets
        test_frac = 0.2
        cutoff_index = int(np.floor((1.0 - test_frac) * nrows))

        X_train = X.iloc[:cutoff_index]
        y_train = y[:cutoff_index]
        X_test = X.iloc[cutoff_index:]
        y_test = y[cutoff_index:]

        return X_train, X_test, y_train, y_test, time_column_name

    def test_datetime_features(self, mimic_explainer):
        X_train, x_test, _, _, _ = self._timeseries_generated_data()
        kwargs = {'reset_index': 'reset'}
        model = DataFrameTestModel(X_train.copy())
        features = list(X_train.columns.values) + list(X_train.index.names)
        mimic_explainer(model, X_train, LGBMExplainableModel, features=features, **kwargs)
        # Note: need to fix column names after featurization as more columns are added to surrogate model

    def test_datetime_features_ignore(self, mimic_explainer):
        # Validate we throw when reset_index is set to ignore
        X_train, x_test, _, _, _ = self._timeseries_generated_data()
        kwargs = {'reset_index': 'ignore'}
        model = DataFrameTestModel(X_train.copy())
        features = list(X_train.columns.values)
        # Validate we hit the assertion error on the DataFrameTestModel for checking the presence of index column
        with pytest.raises(AssertionError):
            mimic_explainer(model, X_train, LGBMExplainableModel, features=features, **kwargs)
        # Validate we don't hit error if we disable the index column asserts
        model = DataFrameTestModel(X_train.copy(), assert_index_present=False)
        explainer = mimic_explainer(model, X_train, LGBMExplainableModel, features=features, **kwargs)
        explanation = explainer.explain_global(x_test)
        assert explanation.method == LIGHTGBM_METHOD

    def test_datetime_features_already_featurized(self, mimic_explainer):
        # Validate we still passthrough underlying index to teacher model
        # even if we don't use it for surrogate model
        X_train, x_test, _, _, _ = self._timeseries_generated_data()
        kwargs = {'reset_index': 'reset_teacher'}
        model = DataFrameTestModel(X_train.copy())
        features = list(X_train.columns.values)
        explainer = mimic_explainer(model, X_train, LGBMExplainableModel, features=features, **kwargs)
        explanation = explainer.explain_global(x_test)
        assert explanation.method == LIGHTGBM_METHOD

    def test_explain_model_imbalanced_classes(self, mimic_explainer):
        x_train = retrieve_dataset('unbalanced_dataset.npz')
        model = SkewedTestModel()
        model_predictions = model.predict(x_train)
        # Assert the model's predictions are skewed
        assert len(np.unique(model_predictions)) == 2
        explainable_model = LGBMExplainableModel
        explainer = mimic_explainer(model, x_train, explainable_model, max_num_of_augmentations=10)
        global_explanation = explainer.explain_global(x_train, include_local=True)
        # There should be an explanation per feature
        assert len(global_explanation.global_importance_values) == 1585
        # We should get back an explanation for each class
        assert len(global_explanation.local_importance_values) == 3
        # Get the underlying multiclass model
        surrogate_predictions = explainer.surrogate_model.model.predict(x_train)
        assert len(np.unique(surrogate_predictions)) == 2
        assert len(np.unique(model_predictions)) == 2
        assert np.isclose(surrogate_predictions, model_predictions).all()
        assert global_explanation.method == LIGHTGBM_METHOD

    def test_explain_model_small_data(self, mimic_explainer):
        num_features = 5
        num_rows = 100
        test_size = 0.8
        X, y = make_regression(n_samples=num_rows, n_features=num_features)
        x_train, x_test, y_train, _ = train_test_split(X, y, test_size=test_size, random_state=42)
        assert x_train.shape[0] < SMALL_DATA_THRESHOLD
        # Fit a regression model
        lin = LinearRegression()
        model = lin.fit(x_train, y_train)
        explainable_model = LGBMExplainableModel
        explainer = mimic_explainer(model, x_train, explainable_model)
        global_explanation = explainer.explain_global(x_test, include_local=True)
        # Verify feature importances are non-zero despite
        # the small training data size
        assert np.count_nonzero(global_explanation._local_importance_values) > 0

    def test_explain_raw_feats_regression(self, mimic_explainer):
        # verify that no errors get thrown when calling get_raw_feat_importances
        num_features = 19999
        num_rows = 1000
        test_size = 0.2
        X, y = make_regression(n_samples=num_rows, n_features=num_features)
        x_train, x_test, y_train, _ = train_test_split(X, y, test_size=test_size, random_state=42)

        lin = LinearRegression()
        scaler_transformer = Pipeline(steps=[('scaler', StandardScaler())])
        transformations = [(list(range(num_features)), scaler_transformer)]
        clf = Pipeline(steps=[('preprocessor', scaler_transformer), ('regressor', lin)])
        model = clf.fit(x_train, y_train)
        explainable_model = LGBMExplainableModel
        explainer = mimic_explainer(model, x_train, explainable_model,
                                    transformations=transformations, augment_data=False)
        global_explanation = explainer.explain_global(x_test)
        local_explanation = explainer.explain_local(x_test)
        # There should be an explanation per feature
        assert len(global_explanation.global_importance_values) == num_features
        # There should be an explanation for each row
        assert len(local_explanation.local_importance_values) == num_rows * test_size

    def _verify_predictions_and_replication_metric(self, mimic_explainer, data):
        predictions_main_model = mimic_explainer._get_teacher_model_predictions(data)
        predictions_surrogate_model = mimic_explainer._get_surrogate_model_predictions(data)
        replication_score = mimic_explainer.get_surrogate_model_replication_measure(data)

        assert predictions_main_model is not None
        assert predictions_surrogate_model is not None
        if mimic_explainer.classes is not None:
            assert mimic_explainer.classes == np.unique(predictions_main_model).tolist()
            assert mimic_explainer.classes == np.unique(predictions_surrogate_model).tolist()
        assert replication_score is not None
        assert isinstance(replication_score, float)

        if mimic_explainer.classes is None:
            with pytest.raises(
                    ScenarioNotSupportedException,
                    match="Replication measure for regression surrogate not supported "
                          "because of single instance in training data"):
                mimic_explainer.get_surrogate_model_replication_measure(
                    data[0].reshape(1, len(data[0])))

    def test_explain_model_string_classes(self, mimic_explainer):
        adult_census_income = retrieve_dataset('AdultCensusIncome.csv', skipinitialspace=True)
        X = adult_census_income.drop(['income'], axis=1)
        y = adult_census_income[['income']]
        features = X.columns.values.tolist()
        classes = y['income'].unique().tolist()
        pipe_cfg = {
            'num_cols': X.dtypes[X.dtypes == 'int64'].index.values.tolist(),
            'cat_cols': X.dtypes[X.dtypes == 'object'].index.values.tolist(),
        }
        num_pipe = Pipeline([
            ('num_imputer', SimpleImputer(strategy='median')),
            ('num_scaler', StandardScaler())
        ])
        cat_pipe = Pipeline([
            ('cat_imputer', SimpleImputer(strategy='constant', fill_value='?')),
            ('cat_encoder', OneHotEncoder(handle_unknown='ignore', sparse=False))
        ])
        feat_pipe = ColumnTransformer([
            ('num_pipe', num_pipe, pipe_cfg['num_cols']),
            ('cat_pipe', cat_pipe, pipe_cfg['cat_cols'])
        ])
        X_train = X.copy()
        y_train = y.copy()
        X_train.reset_index(drop=True, inplace=True)
        y_train.reset_index(drop=True, inplace=True)
        X_train = feat_pipe.fit_transform(X_train)
        model = SGDClassifier()
        model = model.fit(X_train, y_train['income'])
        model_task = ModelTask.Classification
        explainer = mimic_explainer(model, X.iloc[:1000], LinearExplainableModel,
                                    augment_data=True, max_num_of_augmentations=10,
                                    features=features, classes=classes, model_task=model_task,
                                    transformations=feat_pipe)
        global_explanation = explainer.explain_global(X.iloc[:1000])
        assert global_explanation.method == LINEAR_METHOD

        self._verify_predictions_and_replication_metric(explainer, X.iloc[:1000])

    def test_linear_explainable_model_regression(self, mimic_explainer):
        num_features = 3
        x_train = np.array([['a', 'E', 'x'], ['c', 'D', 'y']])
        y_train = np.array([1, 2])
        lin = LinearRegression()
        one_hot_transformer = Pipeline(steps=[('one-hot', OneHotEncoder())])
        transformations = [(list(range(num_features)), one_hot_transformer)]
        clf = Pipeline(steps=[('preprocessor', one_hot_transformer), ('regressor', lin)])
        model = clf.fit(x_train, y_train)
        explainable_model = LinearExplainableModel
        explainer = mimic_explainer(model.named_steps['regressor'], x_train, explainable_model,
                                    transformations=transformations, augment_data=False,
                                    explainable_model_args={'sparse_data': True}, features=['f1', 'f2', 'f3'])
        global_explanation = explainer.explain_global(x_train)
        assert global_explanation.method == LINEAR_METHOD

        self._verify_predictions_and_replication_metric(explainer, x_train)

    @pytest.mark.parametrize('if_multiclass', [True, False])
    @pytest.mark.parametrize('raw_feature_transformations', [True, False])
    @pytest.mark.parametrize('explainable_model', [LinearExplainableModel, LGBMExplainableModel])
    def test_sparse_explainable_model_classification(self, mimic_explainer, if_multiclass,
                                                     raw_feature_transformations,
                                                     explainable_model):
        n_samples = 100
        n_cat_features = 15

        cat_feature_names = [f'cat_feature_{i}' for i in range(n_cat_features)]
        cat_features = np.random.choice(['a', 'b', 'c', 'd'], (n_samples, n_cat_features))

        data_x = pd.DataFrame(cat_features, columns=cat_feature_names)
        data_y = np.random.choice(['0', '1'], n_samples)
        if if_multiclass:
            data_y = np.random.choice([0, 1, 2, 3], n_samples)
            classes = [0, 1, 2, 3]
        else:
            data_y = np.random.choice([0, 1], n_samples)
            classes = [0, 1]

        # prepare feature encoders
        cat_feature_encoders = [OneHotEncoder().fit(cat_features[:, i].reshape(-1, 1)) for i in range(n_cat_features)]

        # fit binary classification model
        encoded_cat_features = [cat_feature_encoders[i].transform(cat_features[:, i].reshape(-1, 1)) for i in
                                range(n_cat_features)]
        encoded_cat_features = scipy.sparse.hstack(encoded_cat_features, format='csr')

        model = LogisticRegression(random_state=42).fit(encoded_cat_features, data_y)

        # generate explanation
        cat_transformations = [([cat_feature_name], encoder) for cat_feature_name, encoder in
                               zip(cat_feature_names, cat_feature_encoders)]

        if explainable_model == LinearExplainableModel:
            explainable_model_args = {'sparse_data': True}
        else:
            explainable_model_args = {}

        if raw_feature_transformations:
            explainer = mimic_explainer(model=model,
                                        initialization_examples=data_x,
                                        explainable_model=explainable_model,
                                        explainable_model_args=explainable_model_args,
                                        augment_data=False,
                                        features=cat_feature_names,
                                        classes=classes,
                                        transformations=cat_transformations,
                                        model_task=ModelTask.Classification)
            global_explanation = explainer.explain_global(evaluation_examples=data_x)
        else:
            explainer = mimic_explainer(model=model,
                                        initialization_examples=encoded_cat_features,
                                        explainable_model=explainable_model,
                                        explainable_model_args=explainable_model_args,
                                        augment_data=False,
                                        classes=classes,
                                        model_task=ModelTask.Classification)
            global_explanation = explainer.explain_global(evaluation_examples=encoded_cat_features)

        if explainable_model == LinearExplainableModel:
            assert global_explanation.method == LINEAR_METHOD
        else:
            assert global_explanation.method == LIGHTGBM_METHOD
        if raw_feature_transformations:
            assert len(global_explanation.get_feature_importance_dict()) == data_x.shape[1]
        else:
            assert len(global_explanation.get_feature_importance_dict()) == encoded_cat_features.shape[1]
        sorted_local_importance_values = global_explanation.get_ranked_local_values()[0]
        sorted_local_importance_names = global_explanation.get_ranked_local_names()[0]
        assert len(sorted_local_importance_values) == len(sorted_local_importance_names)
        if if_multiclass:
            if raw_feature_transformations:
                self._verify_predictions_and_replication_metric(explainer, data_x)
            else:
                self._verify_predictions_and_replication_metric(explainer, encoded_cat_features)

    def test_dense_wide_data(self, mimic_explainer):
        # use 6000 rows instead for real performance testing
        data = np.random.randn(50, 40000)
        feature_names = [f'f_{i}' for i in range(39999)]
        column_names = [f'f_{i}' for i in range(39999)]
        column_names.append('label')
        dataframe1 = pd.DataFrame(data, columns=column_names)
        df_y = dataframe1['label']
        df_X = dataframe1.drop(columns='label')
        # Fit a lightgbm regression model
        model = create_lightgbm_regressor(df_X, df_y)
        explainable_model = LGBMExplainableModel
        explainer = mimic_explainer(model, df_X, explainable_model, augment_data=False,
                                    features=feature_names)
        global_explanation = explainer.explain_global(df_X)
        assert global_explanation.method == LIGHTGBM_METHOD

    @property
    def iris_overall_expected_features(self):
        return [['petal length', 'petal width', 'sepal width', 'sepal length'],
                ['petal length', 'petal width', 'sepal length', 'sepal width'],
                ['petal length', 'petal width', 'sepal length', 'sepal width'],
                ['petal length', 'petal width', 'sepal width', 'sepal length']]

    @property
    def iris_overall_expected_features_special_args(self):
        return [['petal length', 'petal width', 'sepal width', 'sepal length'],
                ['petal length', 'petal width', 'sepal length', 'sepal width'],
                ['petal length', 'petal width', 'sepal length', 'sepal width'],
                ['petal length', 'petal width', 'sepal width', 'sepal length']]

    @property
    def iris_overall_expected_features_without_evaluation(self):
        return [['petal length', 'sepal length', 'petal width', 'sepal width'],
                ['petal length', 'petal width', 'sepal length', 'sepal width'],
                ['petal length', 'petal width', 'sepal length', 'sepal width'],
                ['petal length', 'petal width', 'sepal width', 'sepal length']]

    @property
    def iris_per_class_expected_features(self):
        return [[['petal length', 'petal width', 'sepal length', 'sepal width'],
                 ['petal length', 'petal width', 'sepal length', 'sepal width'],
                 ['petal width', 'petal length', 'sepal width', 'sepal length']],
                [['petal length', 'petal width', 'sepal width', 'sepal length'],
                 ['petal width', 'petal length', 'sepal width', 'sepal length'],
                 ['petal length', 'petal width', 'sepal length', 'sepal width']],
                [['petal length', 'petal width', 'sepal width', 'sepal length'],
                 ['petal width', 'sepal width', 'petal length', 'sepal length'],
                 ['petal length', 'petal width', 'sepal length', 'sepal width']],
                [['petal length', 'petal width', 'sepal width', 'sepal length'],
                 ['petal length', 'petal width', 'sepal width', 'sepal length'],
                 ['petal length', 'petal width', 'sepal width', 'sepal length']]]

    @property
    def iris_per_class_expected_features_special_args(self):
        return [[['petal length', 'petal width', 'sepal length', 'sepal width'],
                 ['petal length', 'petal width', 'sepal length', 'sepal width'],
                 ['petal length', 'petal width', 'sepal width', 'sepal length']],
                [['petal length', 'petal width', 'sepal width', 'sepal length'],
                 ['petal width', 'petal length', 'sepal width', 'sepal length'],
                 ['petal length', 'petal width', 'sepal length', 'sepal width']],
                [['petal length', 'petal width', 'sepal width', 'sepal length'],
                 ['petal length', 'petal width', 'sepal width', 'sepal length'],
                 ['petal length', 'petal width', 'sepal length', 'sepal width']],
                [['petal length', 'petal width', 'sepal width', 'sepal length'],
                 ['petal length', 'petal width', 'sepal width', 'sepal length'],
                 ['petal length', 'petal width', 'sepal width', 'sepal length']]]


@pytest.mark.owner(email=owner_email_tools_and_ux)
@pytest.mark.usefixtures('_clean_dir')
class TestMimicExplainerWrappedModels(object):
    def test_working(self):
        assert True

    @pytest.mark.parametrize('if_predictions_as_dataframe', [True, False])
    @pytest.mark.parametrize('explainable_model', [LGBMExplainableModel,
                                                   LinearExplainableModel,
                                                   DecisionTreeExplainableModel,
                                                   SGDExplainableModel])
    def test_explain_model_binary_classification_with_different_format_predictions(
            self, mimic_explainer, if_predictions_as_dataframe, explainable_model):
        x_train, y_train, x_test, y_test, classes = create_binary_classification_dataset()
        model = LogisticRegression(random_state=42).fit(x_train, y_train)
        model.fit(x_train, y_train)

        model = PredictAsDataFrameClassificationTestModel(
            model, return_predictions_as_dataframe=if_predictions_as_dataframe)
        kwargs = {}
        explainer = mimic_explainer(model, x_train, explainable_model, **kwargs)
        global_explanation = explainer.explain_global(evaluation_examples=x_test)
        assert global_explanation is not None

    @pytest.mark.parametrize('if_predictions_as_dataframe', [True, False])
    @pytest.mark.parametrize('explainable_model', [LGBMExplainableModel,
                                                   LinearExplainableModel,
                                                   DecisionTreeExplainableModel,
                                                   SGDExplainableModel])
    def test_explain_model_multiclass_classification_with_different_format_predictions(
            self, mimic_explainer, if_predictions_as_dataframe, explainable_model):
        x_train, x_test, y_train, y_test, _, classes = create_iris_data()
        model = LogisticRegression(random_state=42).fit(x_train, y_train)
        model.fit(x_train, y_train)

        model = PredictAsDataFrameClassificationTestModel(
            model, return_predictions_as_dataframe=if_predictions_as_dataframe)

        kwargs = {}
        explainer = mimic_explainer(model, x_train, explainable_model, **kwargs)
        global_explanation = explainer.explain_global(evaluation_examples=x_test)
        assert global_explanation is not None

    @pytest.mark.parametrize('if_predictions_as_dataframe', [True, False])
    @pytest.mark.parametrize('explainable_model', [LGBMExplainableModel,
                                                   LinearExplainableModel,
                                                   DecisionTreeExplainableModel,
                                                   SGDExplainableModel])
    def test_explain_model_regression_with_different_format_predictions(
            self, mimic_explainer, if_predictions_as_dataframe, explainable_model):
        num_features = 3
        x_train = np.array([['a', 'E', 'x'], ['c', 'D', 'y']])
        y_train = np.array([1, 2])
        lin = LinearRegression()
        one_hot_transformer = Pipeline(steps=[('one-hot', OneHotEncoder())])
        transformations = [(list(range(num_features)), one_hot_transformer)]
        clf = Pipeline(steps=[('preprocessor', one_hot_transformer), ('regressor', lin)])
        model = clf.fit(x_train, y_train)
        model = PredictAsDataFrameREgressionTestModel(model.named_steps['regressor'],
                                                      if_predictions_as_dataframe)
        explainer = mimic_explainer(model, x_train, explainable_model,
                                    transformations=transformations, augment_data=False,
                                    explainable_model_args={}, features=['f1', 'f2', 'f3'])
        global_explanation = explainer.explain_global(x_train)
        assert global_explanation is not None

    def test_mimic_pytorch_binary_single_output(self, mimic_explainer):
        x_train, x_test, y_train, _, feature_names, _ = create_cancer_data()
        # Fit a pytorch DNN model
        model = create_pytorch_single_output_classifier(x_train.values, y_train)
        test_logger.info('Running explain global for test_mimic_pytorch_binary_single_output')
        model_task = ModelTask.Classification
        explainer = mimic_explainer(model, x_train, LGBMExplainableModel,
                                    features=feature_names, model_task=model_task)
        global_explanation = explainer.explain_global(x_train)
        assert global_explanation is not None
        predicted_y = explainer.model.predict(x_train)
        # assert not all predictions zeros
        assert np.any(predicted_y)
        # also assert there are many nonzeros predicted
        assert np.count_nonzero(predicted_y) > predicted_y.shape[0] / 4
