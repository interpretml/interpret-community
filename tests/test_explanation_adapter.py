# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import logging

import pytest
import shap
from captum.attr import IntegratedGradients
from common_utils import (create_pytorch_classifier, create_pytorch_regressor,
                          create_sklearn_random_forest_classifier,
                          create_sklearn_random_forest_regressor)
from constants import DatasetConstants, owner_email_tools_and_ux
from explanation_utils import (
    validate_global_classification_explanation_shape,
    validate_global_regression_explanation_shape,
    validate_local_classification_explanation_shape,
    validate_local_regression_explanation_shape)
from interpret_community.adapter import ExplanationAdapter
from sklearn.model_selection import train_test_split
from test_serialize_explanation import verify_serialization

try:
    import torch
except ImportError:
    pass

test_logger = logging.getLogger(__name__)


@pytest.mark.owner(email=owner_email_tools_and_ux)
@pytest.mark.usefixtures('_clean_dir')
class TestExplanationAdapter(object):

    def test_working(self):
        assert True

    @pytest.mark.parametrize(("include_local"), [True, False])
    @pytest.mark.parametrize(("include_expected_values"), [True, False])
    def test_explanation_adapter_shap_classifier(self, include_local,
                                                 include_expected_values):
        X, y = shap.datasets.adult()
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=7)
        model = create_sklearn_random_forest_classifier(x_train.values, y_train)
        explainer = shap.KernelExplainer(model.predict_proba, shap.kmeans(x_train.values, 10))
        shap_values = explainer.shap_values(x_test.values, nsamples=1000)
        adapter = ExplanationAdapter(features=list(x_train.columns), classification=True)
        expected_values = None
        if include_expected_values:
            expected_values = explainer.expected_value
        global_explanation = adapter.create_global(shap_values, evaluation_examples=x_test.values,
                                                   include_local=include_local,
                                                   expected_values=expected_values)
        verify_serialization(global_explanation, exist_ok=True)
        validate_global_classification_explanation_shape(global_explanation, x_test,
                                                         include_local=include_local)
        local_explanation = adapter.create_local(shap_values, evaluation_examples=x_test.values,
                                                 expected_values=expected_values)
        verify_serialization(local_explanation, exist_ok=True)
        validate_local_classification_explanation_shape(local_explanation, x_test)

    @pytest.mark.parametrize(("include_local"), [True, False])
    @pytest.mark.parametrize(("include_expected_values"), [True, False])
    def test_explanation_adapter_shap_regressor(self, housing, include_local,
                                                include_expected_values):
        x_train = housing[DatasetConstants.X_TRAIN]
        x_test = housing[DatasetConstants.X_TEST]
        features = housing[DatasetConstants.FEATURES]
        model = create_sklearn_random_forest_regressor(x_train,
                                                       housing[DatasetConstants.Y_TRAIN])
        explainer = shap.KernelExplainer(model.predict, shap.kmeans(x_train, 10))
        shap_values = explainer.shap_values(x_test, nsamples=1000)
        adapter = ExplanationAdapter(features=features, classification=False)
        expected_values = None
        if include_expected_values:
            expected_values = explainer.expected_value
        global_explanation = adapter.create_global(shap_values, evaluation_examples=x_test,
                                                   include_local=include_local,
                                                   expected_values=expected_values)
        verify_serialization(global_explanation, exist_ok=True)
        validate_global_regression_explanation_shape(global_explanation, x_test,
                                                     include_local=include_local)
        local_explanation = adapter.create_local(shap_values, evaluation_examples=x_test,
                                                 expected_values=expected_values)
        verify_serialization(local_explanation, exist_ok=True)
        validate_local_regression_explanation_shape(local_explanation, x_test)

    def test_explanation_adapter_captum_classifier(self):
        X, y = shap.datasets.adult()
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=7)
        # Fit a pytorch DNN model
        model = create_pytorch_classifier(x_train.values, y_train)
        ig = IntegratedGradients(model)
        attributions, delta = ig.attribute(torch.Tensor(x_test.values).float(),
                                           target=torch.Tensor(y_test).long(),
                                           return_convergence_delta=True)
        adapter = ExplanationAdapter(features=list(x_train.columns), classification=True)
        global_explanation = adapter.create_global(attributions, evaluation_examples=x_test.values)
        verify_serialization(global_explanation, exist_ok=True)
        validate_global_classification_explanation_shape(global_explanation, x_test)
        local_explanation = adapter.create_local(attributions, evaluation_examples=x_test.values)
        verify_serialization(local_explanation, exist_ok=True)
        validate_local_classification_explanation_shape(local_explanation, x_test)

    def test_explanation_adapter_captum_regressor(self, housing):
        x_train = housing[DatasetConstants.X_TRAIN]
        x_test = housing[DatasetConstants.X_TEST]
        features = housing[DatasetConstants.FEATURES]
        # Fit a pytorch DNN model
        model = create_pytorch_regressor(x_train, housing[DatasetConstants.Y_TRAIN])
        ig = IntegratedGradients(model)
        attributions, delta = ig.attribute(torch.Tensor(x_test).float(),
                                           return_convergence_delta=True)
        adapter = ExplanationAdapter(features=features, classification=False)
        global_explanation = adapter.create_global(attributions, evaluation_examples=x_test)
        verify_serialization(global_explanation, exist_ok=True)
        validate_global_regression_explanation_shape(global_explanation, x_test)
        local_explanation = adapter.create_local(attributions, evaluation_examples=x_test)
        verify_serialization(local_explanation, exist_ok=True)
        validate_local_regression_explanation_shape(local_explanation, x_test)
