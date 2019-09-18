# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

# Tests for kernel, tree and deep explainers.

import pytest
import logging

from lightgbm import LGBMClassifier, LGBMRegressor

from interpret_community.shap.kernel_explainer import KernelExplainer
from interpret_community.shap.tree_explainer import TreeExplainer
from interpret_community.shap.deep_explainer import DeepExplainer
from interpret_community.shap.linear_explainer import LinearExplainer
from common_tabular_tests import VerifyTabularTests
from common_utils import create_keras_multiclass_classifier, create_keras_regressor, \
    create_sklearn_linear_regressor, create_sklearn_logistic_regressor
from constants import owner_email_tools_and_ux

test_logger = logging.getLogger(__name__)
test_logger.setLevel(logging.INFO)


@pytest.mark.owner(email=owner_email_tools_and_ux)
@pytest.mark.usefixtures("clean_dir")
class TestKernelExplainer(object):
    def setup_class(self):
        def create_explainer(model, x_train, **kwargs):
            return KernelExplainer(model, x_train, **kwargs)

        self._verify_tabular = VerifyTabularTests(test_logger, create_explainer)

    def test_kernel_explainer_raw_transformations_list_classification(self):
        self._verify_tabular.verify_explain_model_transformations_list_classification()

    def test_kernel_explainer_raw_transformations_column_transformer_classification(self):
        self._verify_tabular.verify_explain_model_transformations_column_transformer_classification()

    def test_kernel_explainer_raw_transformations_list_regression(self):
        self._verify_tabular.verify_explain_model_transformations_list_regression()

    def test_kernel_explainer_raw_transformations_column_transformer_regression(self):
        self._verify_tabular.verify_explain_model_transformations_list_regression()


@pytest.mark.owner(email=owner_email_tools_and_ux)
@pytest.mark.usefixtures("clean_dir")
class TestDeepExplainer(object):
    def setup_class(self):
        def create_explainer(model, x_train, **kwargs):
            return DeepExplainer(model, x_train, **kwargs)

        self._verify_tabular = VerifyTabularTests(test_logger, create_explainer)

    def _get_create_model(self, classification):
        if classification:
            train_fn = create_keras_multiclass_classifier
        else:
            train_fn = create_keras_regressor

        def create_model(x, y):
            return train_fn(x, y)
        return create_model

    def test_deep_explainer_raw_transformations_list_classification(self):
        self._verify_tabular.verify_explain_model_transformations_list_classification(self._get_create_model(
            classification=True))

    def test_deep_explainer_raw_transformations_column_transformer_classification(self):
        self._verify_tabular.verify_explain_model_transformations_column_transformer_classification(
            self._get_create_model(classification=True))

    def test_deep_explainer_raw_transformations_list_regression(self):
        self._verify_tabular.verify_explain_model_transformations_list_regression(self._get_create_model(
            classification=False))

    def test_deep_explainer_raw_transformations_column_transformer_regression(self):
        self._verify_tabular.verify_explain_model_transformations_column_transformer_regression(
            self._get_create_model(classification=False))


@pytest.mark.owner(email=owner_email_tools_and_ux)
@pytest.mark.usefixtures("clean_dir")
class TestTreeExplainer(object):
    def setup_class(self):
        def create_explainer(model, x_train, **kwargs):
            return TreeExplainer(model, **kwargs)

        self._verify_tabular = VerifyTabularTests(test_logger, create_explainer)

    def _get_create_model(self, classification):
        if classification:
            model = LGBMClassifier()
        else:
            model = LGBMRegressor()

        def create_model(x, y):
            return model.fit(x, y)

        return create_model

    def test_tree_explainer_raw_transformations_list_classification(self):
        self._verify_tabular.verify_explain_model_transformations_list_classification(self._get_create_model(
            classification=True))

    def test_tree_explainer_raw_transformations_column_transformer_classification(self):
        self._verify_tabular.verify_explain_model_transformations_column_transformer_classification(
            self._get_create_model(classification=True))

    def test_tree_explainer_raw_transformations_list_regression(self):
        self._verify_tabular.verify_explain_model_transformations_list_regression(self._get_create_model(
            classification=False))

    def test_tree_explainer_raw_transformations_column_transformer_regression(self):
        self._verify_tabular.verify_explain_model_transformations_list_regression(self._get_create_model(
            classification=False))


@pytest.mark.owner(email=owner_email_tools_and_ux)
@pytest.mark.usefixtures("clean_dir")
class TestLinearExplainer(object):
    def setup_class(self):
        def create_explainer(model, x_train, **kwargs):
            return LinearExplainer(model, x_train, **kwargs)

        self._verify_tabular = VerifyTabularTests(test_logger, create_explainer)

    def _get_create_model(self, classification):
        if classification:
            train_fn = create_sklearn_logistic_regressor
        else:
            train_fn = create_sklearn_linear_regressor

        def create_model(x, y):
            return train_fn(x, y)

        return create_model

    def test_linear_explainer_raw_transformations_list_classification(self):
        self._verify_tabular.verify_explain_model_transformations_list_classification(self._get_create_model(
            classification=True))

    def test_linear_explainer_raw_transformations_column_transformer_classification(self):
        self._verify_tabular.verify_explain_model_transformations_column_transformer_classification(
            self._get_create_model(classification=True))

    def test_linear_explainer_raw_transformations_list_regression(self):
        self._verify_tabular.verify_explain_model_transformations_list_regression(self._get_create_model(
            classification=False))

    def test_linear_explainer_raw_transformations_column_transformer_regression(self):
        self._verify_tabular.verify_explain_model_transformations_list_regression(self._get_create_model(
            classification=False))
