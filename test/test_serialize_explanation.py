# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Tests the Explanation JSON serializer"""

import collections.abc
import pytest
import logging
import numpy as np
import pandas as pd
import os

from interpret_community.common.constants import ExplainParams
from interpret_community.explanation.explanation import save_explanation, load_explanation
from common_utils import create_sklearn_svm_classifier
from constants import DatasetConstants
from constants import owner_email_tools_and_ux
from interpret_community.dataset.dataset_wrapper import DatasetWrapper
try:
    from shap.common import DenseData
except:
    from shap.utils._legacy import DenseData

test_logger = logging.getLogger(__name__)


@pytest.fixture(scope='class')
def iris_svm_model(iris):
    # uses iris DatasetConstants
    model = create_sklearn_svm_classifier(iris[DatasetConstants.X_TRAIN], iris[DatasetConstants.Y_TRAIN])
    yield model


@pytest.mark.owner(email=owner_email_tools_and_ux)
@pytest.mark.usefixtures('clean_dir')
class TestSerializeExplanation(object):

    def test_save_explanation(self, iris, tabular_explainer, iris_svm_model):
        explainer = tabular_explainer(iris_svm_model,
                                      iris[DatasetConstants.X_TRAIN],
                                      features=iris[DatasetConstants.FEATURES])
        explanation = explainer.explain_local(iris[DatasetConstants.X_TEST])
        save_explanation(explanation, 'brand/new/path')

    def test_save_and_load_explanation_local_only(self, iris, tabular_explainer, iris_svm_model):
        explainer = tabular_explainer(iris_svm_model,
                                      iris[DatasetConstants.X_TRAIN],
                                      features=iris[DatasetConstants.FEATURES])
        explanation = explainer.explain_local(iris[DatasetConstants.X_TEST])
        verify_serialization(explanation)

    def test_save_and_load_explanation_global_only(self, iris, tabular_explainer, iris_svm_model):
        explainer = tabular_explainer(iris_svm_model,
                                      iris[DatasetConstants.X_TRAIN],
                                      features=iris[DatasetConstants.FEATURES])
        explanation = explainer.explain_global(iris[DatasetConstants.X_TEST], include_local=False)
        verify_serialization(explanation)

    def test_save_and_load_explanation_global_and_local(self, iris, tabular_explainer, iris_svm_model):
        explainer = tabular_explainer(iris_svm_model,
                                      iris[DatasetConstants.X_TRAIN],
                                      features=iris[DatasetConstants.FEATURES])
        explanation = explainer.explain_global(iris[DatasetConstants.X_TEST], include_local=False)
        verify_serialization(explanation)


def _assert_explanation_equivalence(actual, expected):
    # get the non-null properties in the expected explanation
    paramkeys = filter(lambda x, expected=expected: hasattr(expected, getattr(ExplainParams, x)),
                       list(ExplainParams.get_serializable()))
    for paramkey in paramkeys:
        param = getattr(ExplainParams, paramkey)
        actual_value = getattr(actual, param, None)
        expected_value = getattr(expected, param, None)
        if isinstance(actual_value, DatasetWrapper) or isinstance(actual_value, DenseData):
            if isinstance(actual_value.original_dataset, np.ndarray):
                actual_dataset = actual_value.original_dataset.tolist()
            else:
                actual_dataset = actual_value.original_dataset
            if isinstance(expected_value.original_dataset, np.ndarray):
                expected_dataset = expected_value.original_dataset.tolist()
            else:
                expected_dataset = expected_value.original_dataset
            np.testing.assert_array_equal(actual_dataset, expected_dataset)
        elif isinstance(actual_value, (np.ndarray, collections.abc.Sequence)):
            np.testing.assert_array_equal(actual_value, expected_value)
        elif isinstance(actual_value, pd.DataFrame) and isinstance(expected_value, pd.DataFrame):
            np.testing.assert_array_equal(actual_value.values, expected_value.values)
        else:
            assert actual_value == expected_value


# performs serialization and de-serialization for any explanation
# tests to verify that the de-serialized result is equivalent to the original
# exposed outside this module to allow any test involving an explanation to
# incorporate serialization testing
def verify_serialization(explanation, extra_path=None, exist_ok=False):
    path = 'brand/new/path'
    if extra_path is not None:
        path = os.path.join(path, extra_path)
    save_explanation(explanation, path, exist_ok=exist_ok)
    loaded_explanation = load_explanation(path)
    _assert_explanation_equivalence(explanation, loaded_explanation)
