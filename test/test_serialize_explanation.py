# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Tests the Explanation JSON serializer"""

import collections.abc
import pytest
import logging
import json
import numpy as np
import pandas as pd

from interpret_community.common.constants import ExplainParams
from interpret_community.mimic.mimic_explainer import MimicExplainer
from interpret_community.mimic.models.lightgbm_model import LGBMExplainableModel
from interpret_community.explanation.explanation import save_explanation, load_explanation
from common_utils import create_sklearn_svm_classifier
from constants import DatasetConstants
from constants import owner_email_tools_and_ux
from interpret_community.dataset.dataset_wrapper import DatasetWrapper
from shap.common import DenseData

test_logger = logging.getLogger(__name__)


@pytest.fixture(scope='class')
def iris_svm_model(iris):
    # uses iris DatasetConstants
    model = create_sklearn_svm_classifier(iris[DatasetConstants.X_TRAIN], iris[DatasetConstants.Y_TRAIN])
    yield model


@pytest.mark.owner(email=owner_email_tools_and_ux)
@pytest.mark.usefixtures('clean_dir')
class TestSerializeExplanation(object):

    def test_json_serialize_mimic_no_features(self, iris, iris_svm_model):
        explainer = MimicExplainer(iris_svm_model,
                                   iris[DatasetConstants.X_TRAIN],
                                   LGBMExplainableModel,
                                   max_num_of_augmentations=10,
                                   classes=iris[DatasetConstants.CLASSES].tolist())
        explanation = explainer.explain_global()
        verify_serialization(explanation)

    def test_json_serialize_local_explanation_classification(self, iris, tabular_explainer, iris_svm_model):
        explainer = tabular_explainer(iris_svm_model,
                                      iris[DatasetConstants.X_TRAIN],
                                      features=iris[DatasetConstants.FEATURES])
        explanation = explainer.explain_local(iris[DatasetConstants.X_TEST])
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
def verify_serialization(explanation):
    paramkeys = ['MODEL_TYPE', 'MODEL_TASK', 'METHOD', 'FEATURES', 'CLASSES']
    log_items = dict()
    for paramkey in paramkeys:
        param = getattr(ExplainParams, paramkey)
        value = getattr(explanation, param, None)
        if value is not None:
            if isinstance(value, np.ndarray):
                log_items[param] = value.tolist()
            else:
                log_items[param] = value
    comment = json.dumps(log_items)
    test_logger.setLevel(logging.INFO)
    test_logger.info("validating serialization of explanation:\n%s", comment)
    expljson = save_explanation(explanation)
    deserialized_explanation = load_explanation(expljson)
    _assert_explanation_equivalence(deserialized_explanation, explanation)
