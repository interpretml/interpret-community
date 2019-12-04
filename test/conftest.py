# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import pytest
import logging
import tempfile
import os
from common_utils import create_iris_data, create_boston_data, create_simple_titanic_data, \
    create_complex_titanic_data
from constants import DatasetConstants
from interpret_community.tabular_explainer import TabularExplainer
from common_tabular_tests import VerifyTabularTests
from interpret_community.mimic.mimic_explainer import MimicExplainer
from interpret_community.mimic.models.lightgbm_model import LGBMExplainableModel
from interpret_community.mimic.models.linear_model import LinearExplainableModel, SGDExplainableModel
from interpret_community.mimic.models.tree_model import DecisionTreeExplainableModel
test_logger = logging.getLogger(__name__)
test_logger.setLevel(logging.DEBUG)


def pytest_itemcollected(item):
    if not item.get_closest_marker("domain"):
        item.add_marker(pytest.mark.domain(["explain", "model"]))


@pytest.fixture()
def clean_dir():
    new_path = tempfile.mkdtemp()
    print("tmp test directory: " + new_path)
    os.chdir(new_path)


@pytest.fixture(scope='session')
def iris():
    x_train, x_test, y_train, y_test, features, classes = create_iris_data()
    yield {
        DatasetConstants.X_TRAIN: x_train,
        DatasetConstants.X_TEST: x_test,
        DatasetConstants.Y_TRAIN: y_train,
        DatasetConstants.Y_TEST: y_test,
        DatasetConstants.FEATURES: features,
        DatasetConstants.CLASSES: classes
    }


@pytest.fixture(scope='session')
def boston():
    x_train, x_test, y_train, y_test, features = create_boston_data()
    yield {
        DatasetConstants.X_TRAIN: x_train,
        DatasetConstants.X_TEST: x_test,
        DatasetConstants.Y_TRAIN: y_train,
        DatasetConstants.Y_TEST: y_test,
        DatasetConstants.FEATURES: features
    }


@pytest.fixture(scope='session')
def titanic_simple():
    x_train, x_test, y_train, y_test, numeric, categorical = create_simple_titanic_data()
    yield {
        DatasetConstants.X_TRAIN: x_train,
        DatasetConstants.X_TEST: x_test,
        DatasetConstants.Y_TRAIN: y_train,
        DatasetConstants.Y_TEST: y_test,
        DatasetConstants.NUMERIC: numeric,
        DatasetConstants.CATEGORICAL: categorical
    }


@pytest.fixture(scope='session')
def titanic_complex():
    x_train, x_test, y_train, y_test = create_complex_titanic_data()
    yield {
        DatasetConstants.X_TRAIN: x_train,
        DatasetConstants.X_TEST: x_test,
        DatasetConstants.Y_TRAIN: y_train,
        DatasetConstants.Y_TEST: y_test
    }


@pytest.fixture(scope='session')
def tabular_explainer():
    return TabularExplainer


@pytest.fixture(scope='class')
def verify_tabular():
    def create_explainer(model, x_train, **kwargs):
        return TabularExplainer(model, x_train, **kwargs)

    return VerifyTabularTests(test_logger, create_explainer)


@pytest.fixture(scope='session')
def mimic_explainer():
    return MimicExplainer


def generate_create_method(explainable_model, is_sparse=False, explainable_model_args={}):
    if is_sparse:
        def create_explainer(model, x_train, **kwargs):
            return MimicExplainer(model, x_train, explainable_model, max_num_of_augmentations=10,
                                  explainable_model_args=explainable_model_args.copy(), **kwargs)
    else:
        def create_explainer(model, x_train, **kwargs):
            return MimicExplainer(model, x_train, explainable_model,
                                  explainable_model_args=explainable_model_args.copy(), **kwargs)
    return create_explainer


def get_mimic_explainers(classifier=True):
    verify_mimic = []
    explainers = [LGBMExplainableModel, LinearExplainableModel, SGDExplainableModel, DecisionTreeExplainableModel]
    if classifier:
        explainable_models_args = [{}, {'solver': 'liblinear'}, {}, {}]
    else:
        explainable_models_args = [{}, {}, {}, {}]
    for explainer, explainable_model_args in zip(explainers, explainable_models_args):
        generated_create_explainer = generate_create_method(explainer, explainable_model_args=explainable_model_args)
        verify_mimic.append(VerifyTabularTests(test_logger, generated_create_explainer, specify_policy=False))
    return verify_mimic


@pytest.fixture(scope='session')
def verify_mimic_classifier():
    return get_mimic_explainers(classifier=True)


@pytest.fixture(scope='session')
def verify_mimic_regressor():
    return get_mimic_explainers(classifier=False)


@pytest.fixture(scope='session')
def verify_sparse_mimic():
    verify_sparse_mimic = []
    # Note: linear explainers don't work on sparse data yet
    sparse_explainers = [LGBMExplainableModel, LinearExplainableModel, SGDExplainableModel]
    for sparse_explainer in sparse_explainers:
        generated_sparse_create_explainer = generate_create_method(sparse_explainer, is_sparse=True)
        verify_sparse_mimic.append(VerifyTabularTests(test_logger, generated_sparse_create_explainer,
                                                      specify_policy=False))
    return verify_sparse_mimic


@pytest.fixture(scope='session')
def verify_mimic_special_args():
    # Validation of special args passed to underlying lightgbm model
    lgbm_explainable_model_args = {'num_leaves': 64, 'learning_rate': 0.2, 'n_estimators': 200}
    lgbm_create_explainer = generate_create_method(LGBMExplainableModel, is_sparse=False,
                                                   explainable_model_args=lgbm_explainable_model_args)

    # Validation of special args passed to underlying linear and sgd model
    linear_explainable_model_args = {'fit_intercept': False, 'solver': 'liblinear'}
    linear_create_explainer = generate_create_method(LinearExplainableModel, is_sparse=False,
                                                     explainable_model_args=linear_explainable_model_args)
    sgd_explainable_model_args = {'fit_intercept': False, 'alpha': 0.001, 'early_stopping': True}
    sgd_create_explainer = generate_create_method(SGDExplainableModel, is_sparse=False,
                                                  explainable_model_args=sgd_explainable_model_args)

    tree_explainable_model_args = {'min_samples_split': 4, 'min_samples_leaf': 2}
    tree_create_explainer = generate_create_method(DecisionTreeExplainableModel, is_sparse=False,
                                                   explainable_model_args=tree_explainable_model_args)

    verify_mimic_special_args = []
    for create_explainer in [lgbm_create_explainer, linear_create_explainer,
                             sgd_create_explainer, tree_create_explainer]:
        verify_mimic_special_args.append(VerifyTabularTests(test_logger, create_explainer, specify_policy=False))
    return verify_mimic_special_args
