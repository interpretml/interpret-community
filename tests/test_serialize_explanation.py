# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Tests the Explanation JSON serializer"""

import collections.abc
import logging
import os

import numpy as np
import pandas as pd
import pytest
from common_utils import (create_msx_data, create_sklearn_linear_regressor,
                          create_sklearn_svm_classifier)
from constants import DatasetConstants, owner_email_tools_and_ux
from interpret_community.common.constants import ExplainParams
from interpret_community.dataset.dataset_wrapper import DatasetWrapper
from interpret_community.explanation import load_explanation, save_explanation
from interpret_community.mimic.models.lightgbm_model import \
    LGBMExplainableModel
from scipy.sparse import issparse

test_logger = logging.getLogger(__name__)


@pytest.fixture(scope='class')
def iris_svm_model(iris):
    # uses iris DatasetConstants
    model = create_sklearn_svm_classifier(iris[DatasetConstants.X_TRAIN], iris[DatasetConstants.Y_TRAIN])
    return model


@pytest.mark.owner(email=owner_email_tools_and_ux)
@pytest.mark.usefixtures('_clean_dir')
class TestSerializeExplanation(object):

    def test_save_explanation(self, iris, tabular_explainer, iris_svm_model):
        explainer = tabular_explainer(iris_svm_model,
                                      iris[DatasetConstants.X_TRAIN],
                                      features=iris[DatasetConstants.FEATURES])
        explanation = explainer.explain_local(iris[DatasetConstants.X_TEST])
        save_explanation(explanation, os.path.join('brand', 'new', 'path'))

    def test_save_and_load_explanation_local_only(self, iris, tabular_explainer, iris_svm_model):
        explainer = tabular_explainer(iris_svm_model,
                                      iris[DatasetConstants.X_TRAIN],
                                      features=iris[DatasetConstants.FEATURES])
        explanation = explainer.explain_local(iris[DatasetConstants.X_TEST])
        verify_serialization(explanation, assert_numpy_types=True)

    def test_save_and_load_explanation_global_only(self, iris, tabular_explainer, iris_svm_model):
        explainer = tabular_explainer(iris_svm_model,
                                      iris[DatasetConstants.X_TRAIN],
                                      features=iris[DatasetConstants.FEATURES])
        explanation = explainer.explain_global(iris[DatasetConstants.X_TEST], include_local=False)
        verify_serialization(explanation, assert_numpy_types=True)

    def test_save_and_load_explanation_global_and_local(self, iris, tabular_explainer, iris_svm_model):
        explainer = tabular_explainer(iris_svm_model,
                                      iris[DatasetConstants.X_TRAIN],
                                      features=iris[DatasetConstants.FEATURES])
        explanation = explainer.explain_global(iris[DatasetConstants.X_TEST])
        verify_serialization(explanation, assert_numpy_types=True)

    def test_save_and_load_explanation_global_and_local_multiple_times(
            self, iris, tabular_explainer, iris_svm_model):
        explainer = tabular_explainer(iris_svm_model,
                                      iris[DatasetConstants.X_TRAIN],
                                      features=iris[DatasetConstants.FEATURES])
        explanation = explainer.explain_global(iris[DatasetConstants.X_TEST])
        verify_serialization(explanation, assert_numpy_types=True, num_times=5)

    def test_save_and_load_sparse_explanation(self, mimic_explainer):
        x_train, x_test, y_train, y_test = create_msx_data(0.05)
        # Fit a linear regression model
        model = create_sklearn_linear_regressor(x_train, y_train.toarray().flatten())
        explainable_model = LGBMExplainableModel
        explainer = mimic_explainer(model, x_train, explainable_model, augment_data=False)
        explanation = explainer.explain_global(x_test)
        verify_serialization(explanation)


@pytest.mark.owner(email=owner_email_tools_and_ux)
class TestSerializeExplanationBackcompat(object):
    def test_old_load_explanation_backcompat(self, iris, tabular_explainer, iris_svm_model):
        explainer = tabular_explainer(iris_svm_model,
                                      iris[DatasetConstants.X_TRAIN],
                                      features=iris[DatasetConstants.FEATURES])
        explanation = explainer.explain_global(iris[DatasetConstants.X_TEST], include_local=False)
        loaded_explanation = load_explanation(os.path.join('.', 'tests', 'backcompat_explanation'))
        explanation._id = loaded_explanation._id
        _assert_explanation_equivalence(explanation, loaded_explanation, rtol=0.03, atol=0.002)
        _assert_numpy_explanation_types(explanation, loaded_explanation, rtol=0.03, atol=0.002)


def _generate_old_explanation(iris, tabular_explainer, iris_svm_model):
    # Code used to generate an explanation from an older version
    explainer = tabular_explainer(iris_svm_model,
                                  iris[DatasetConstants.X_TRAIN],
                                  features=iris[DatasetConstants.FEATURES])
    explanation = explainer.explain_global(iris[DatasetConstants.X_TEST], include_local=False)
    path = os.path.join('.', 'tests', 'backcompat_explanation')
    save_explanation(explanation, path, exist_ok=False)


def _assert_explanation_equivalence(actual, expected, rtol=None, atol=None):
    # get the non-null properties in the expected explanation
    paramkeys = filter(lambda x, expected=expected: hasattr(expected, getattr(ExplainParams, x)),
                       list(ExplainParams.get_serializable()))
    for paramkey in paramkeys:
        param = getattr(ExplainParams, paramkey)
        actual_value = getattr(actual, param, None)
        expected_value = getattr(expected, param, None)
        if isinstance(actual_value, DatasetWrapper):
            if isinstance(actual_value.original_dataset, np.ndarray):
                actual_dataset = actual_value.original_dataset.tolist()
            else:
                actual_dataset = actual_value.original_dataset
            if isinstance(expected_value.original_dataset, np.ndarray):
                expected_dataset = expected_value.original_dataset.tolist()
            else:
                expected_dataset = expected_value.original_dataset
            if issparse(actual_dataset) and issparse(expected_dataset):
                _assert_sparse_data_equivalence(actual_dataset, expected_dataset, rtol=rtol, atol=atol)
            else:
                _assert_allclose_or_eq(actual_dataset, expected_dataset, rtol=rtol, atol=atol)
        elif isinstance(actual_value, (np.ndarray, collections.abc.Sequence)):
            _assert_allclose_or_eq(actual_value, expected_value, rtol=rtol, atol=atol)
        elif isinstance(actual_value, pd.DataFrame) and isinstance(expected_value, pd.DataFrame):
            _assert_allclose_or_eq(actual_value.values, expected_value.values, rtol=rtol, atol=atol)
        elif issparse(actual_value) and issparse(expected_value):
            _assert_sparse_data_equivalence(actual_value, expected_value, rtol=rtol, atol=atol)
        else:
            assert actual_value == expected_value


def _assert_allclose_or_eq(actual, expected, rtol=None, atol=None):
    if rtol is not None:
        try:
            return np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
        except TypeError:
            print("Caught type error, defaulting to regular compare")
    np.testing.assert_array_equal(actual, expected)


def _assert_sparse_data_equivalence(actual, expected, rtol=None, atol=None):
    _assert_allclose_or_eq(actual.data, expected.data, rtol=rtol, atol=atol)
    _assert_allclose_or_eq(actual.indices, expected.indices, rtol=rtol, atol=atol)
    _assert_allclose_or_eq(actual.indptr, expected.indptr, rtol=rtol, atol=atol)
    _assert_allclose_or_eq(actual.shape, expected.shape, rtol=rtol, atol=atol)


def _assert_numpy_explanation_types(actual, expected, rtol=None, atol=None):
    # assert "_" variables equivalence
    if hasattr(actual, ExplainParams.get_private(ExplainParams.LOCAL_IMPORTANCE_VALUES)):
        assert isinstance(actual._local_importance_values, np.ndarray)
        assert isinstance(expected._local_importance_values, np.ndarray)
        if rtol is None:
            np.testing.assert_array_equal(actual._local_importance_values,
                                          expected._local_importance_values)
        else:
            np.testing.assert_allclose(actual._local_importance_values,
                                       expected._local_importance_values,
                                       rtol=rtol,
                                       atol=atol)
    if hasattr(actual, ExplainParams.get_private(ExplainParams.EVAL_DATA)):
        assert isinstance(actual._eval_data, np.ndarray)
        assert isinstance(expected._eval_data, np.ndarray)
        if rtol is None:
            np.testing.assert_array_equal(actual._eval_data, expected._eval_data)
        else:
            np.testing.assert_allclose(actual._eval_data,
                                       expected._eval_data,
                                       rtol=rtol,
                                       atol=atol)


# performs serialization and de-serialization for any explanation
# tests to verify that the de-serialized result is equivalent to the original
# exposed outside this module to allow any test involving an explanation to
# incorporate serialization testing
def verify_serialization(
        explanation, extra_path=None, exist_ok=False, assert_numpy_types=False,
        num_times=1):
    loaded_explanation = explanation
    for index in range(num_times):
        path = os.path.join('brand', 'new', 'path') + str(index)
        if extra_path is not None:
            path = os.path.join(path, extra_path)
        save_explanation(loaded_explanation, path, exist_ok=exist_ok)
        loaded_explanation = load_explanation(path)
        _assert_explanation_equivalence(explanation, loaded_explanation)
        if assert_numpy_types:
            _assert_numpy_explanation_types(explanation, loaded_explanation)
