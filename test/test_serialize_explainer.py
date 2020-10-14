# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Tests serializing the explainers or models"""

import pytest
import logging
from joblib import dump
from os import path
import shap

from common_utils import create_sklearn_svm_classifier, create_scikit_cancer_data
from constants import owner_email_tools_and_ux
from interpret.ext.blackbox import TabularExplainer, MimicExplainer
from interpret.ext.glassbox import LGBMExplainableModel
from interpret_community.common.constants import ModelTask

test_logger = logging.getLogger(__name__)

LIGHTGBM_METHOD = 'mimic.lightgbm'


@pytest.mark.owner(email=owner_email_tools_and_ux)
@pytest.mark.usefixtures('clean_dir')
class TestSerializeExplainer(object):

    def test_serialize_kernel(self):
        test_logger.info("Running test_serialize_kernel to validate inner explainer and wrapped model serialization")
        x_train, _, y_train, _, feature_names, target_names = create_scikit_cancer_data()
        model = create_sklearn_svm_classifier(x_train, y_train)
        explainer = TabularExplainer(model,
                                     x_train,
                                     features=feature_names,
                                     classes=target_names)
        # Validate wrapped model and inner explainer can be serialized
        model_name = 'wrapped_model.joblib'
        explainer_name = 'inner_explainer.joblib'
        with open(explainer_name, 'wb') as stream:
            dump(explainer.explainer.explainer, stream)
        with open(model_name, 'wb') as stream:
            dump(explainer.model.predict_proba, stream)
        assert path.exists(model_name)
        assert path.exists(explainer_name)

    def test_serialize_mimic_lightgbm(self):
        test_logger.info("Running test_serialize_mimic_lightgbm to validate serializing explainer with lightgbm model")
        x_train, x_test, y_train, _, feature_names, target_names = create_scikit_cancer_data()
        model = create_sklearn_svm_classifier(x_train, y_train)
        model_task = ModelTask.Unknown
        kwargs = {'explainable_model_args': {'n_jobs': 1}, 'augment_data': False, 'reset_index': True}
        explainer = MimicExplainer(model, x_train, LGBMExplainableModel, features=feature_names,
                                   model_task=model_task, classes=target_names, **kwargs)
        explanation = explainer.explain_global(x_test)
        assert explanation.method == LIGHTGBM_METHOD

        tree_explainer = shap.TreeExplainer(explainer.surrogate_model.model)

        # Validate wrapped model, surrogate, and tree explainer with surrogate can be serialized
        model_name = 'wrapped_model.joblib'
        surrogate_name = 'surrogate_model.joblib'
        tree_explainer_name = 'tree_explainer_model.joblib'
        with open(model_name, 'wb') as stream:
            dump(explainer.model.predict_proba, stream)
        with open(surrogate_name, 'wb') as stream:
            dump(explainer.surrogate_model.model, stream)
        with open(tree_explainer_name, 'wb') as stream:
            dump(tree_explainer, stream)
        assert path.exists(model_name)
        assert path.exists(surrogate_name)
        assert path.exists(tree_explainer_name)
