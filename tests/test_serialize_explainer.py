# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Tests serializing the explainers or models"""

import logging
import time
from os import path

import numpy as np
import pytest
import shap
from common_utils import (LIGHTGBM_METHOD, create_cancer_data,
                          create_energy_data, create_iris_data,
                          create_scikit_cancer_data,
                          create_sklearn_linear_regressor,
                          create_sklearn_svm_classifier, get_mimic_method)
from constants import owner_email_tools_and_ux
from interpret.ext.blackbox import MimicExplainer, TabularExplainer
from interpret_community.common.constants import ModelTask
from interpret_community.mimic.models.lightgbm_model import \
    LGBMExplainableModel
from interpret_community.mimic.models.linear_model import (
    LinearExplainableModel, SGDExplainableModel)
from interpret_community.mimic.models.tree_model import \
    DecisionTreeExplainableModel
from joblib import dump, load
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

test_logger = logging.getLogger(__name__)


@pytest.mark.owner(email=owner_email_tools_and_ux)
@pytest.mark.usefixtures('_clean_dir')
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
            dump(explainer.model, stream)
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
            dump(explainer.model, stream)
        with open(surrogate_name, 'wb') as stream:
            dump(explainer.surrogate_model.model, stream)
        with open(tree_explainer_name, 'wb') as stream:
            dump(tree_explainer, stream)
        assert path.exists(model_name)
        assert path.exists(surrogate_name)
        assert path.exists(tree_explainer_name)

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

    def test_explain_model_serialization_multiclass(self, mimic_explainer):
        x_train, x_test, y_train, _, _, _ = create_iris_data()
        # Fit an SVM model
        model = create_sklearn_svm_classifier(x_train, y_train)
        self._validate_model_serialization(model, x_train, x_test, mimic_explainer)

    def test_explain_model_serialization_binary(self, mimic_explainer):
        x_train, x_test, y_train, _, _, _ = create_cancer_data()
        # Fit an SVM model
        model = create_sklearn_svm_classifier(x_train, y_train)
        self._validate_model_serialization(model, x_train, x_test, mimic_explainer)

    def test_explain_model_serialization_regression(self, mimic_explainer):
        x_train, x_test, y_train, _, feature_names = create_energy_data()
        # Fit a linear model
        model = create_sklearn_linear_regressor(x_train, y_train)
        self._validate_model_serialization(model, x_train, x_test, mimic_explainer)


@pytest.mark.owner(email=owner_email_tools_and_ux)
@pytest.mark.usefixtures('_clean_dir')
class TestPickleUnpickleMimicExplainer(object):
    def _verify_explanations(self, explainer, test_data, surrogate_model_method):
        explanation = explainer.explain_global(test_data)
        assert explanation is not None
        assert explanation.method == surrogate_model_method

    def pickle_unpickle_explainer(self, explainer):
        explainer_pickle_file_name = 'explainer' + str(time.time()) + '.pkl'
        try:
            dump(explainer, explainer_pickle_file_name)
            loaded_explainer = load(explainer_pickle_file_name)
            return loaded_explainer
        except Exception as e:
            raise e

    @pytest.mark.parametrize("surrogate_model", [LGBMExplainableModel, DecisionTreeExplainableModel,
                                                 LinearExplainableModel, SGDExplainableModel])
    def test_pickle_unpickle_mimic_explainer_classification(self, surrogate_model):
        x_train, x_test, y_train, _, feature_names, target_names = create_scikit_cancer_data()
        model = create_sklearn_svm_classifier(x_train, y_train)
        model_task = ModelTask.Unknown
        surrogate_model = surrogate_model
        explainer = MimicExplainer(model, x_train, surrogate_model, features=feature_names,
                                   model_task=model_task, classes=target_names)

        self._verify_explanations(explainer, x_test, get_mimic_method(surrogate_model))
        recovered_explainer = self.pickle_unpickle_explainer(explainer)
        self._verify_explanations(recovered_explainer, x_test, get_mimic_method(surrogate_model))

    @pytest.mark.parametrize("surrogate_model", [LGBMExplainableModel, DecisionTreeExplainableModel,
                                                 LinearExplainableModel, SGDExplainableModel])
    def test_pickle_unpickle_mimic_explainer_regression(self, surrogate_model):
        num_features = 100
        num_rows = 1000
        test_size = 0.2
        X, y = make_regression(n_samples=num_rows, n_features=num_features)
        x_train, x_test, y_train, _ = train_test_split(X, y, test_size=test_size, random_state=42)

        model = LinearRegression()
        model.fit(x_train, y_train)
        surrogate_model = surrogate_model
        explainer = MimicExplainer(model, x_train, surrogate_model)

        self._verify_explanations(explainer, x_test, get_mimic_method(surrogate_model))
        recovered_explainer = self.pickle_unpickle_explainer(explainer)
        self._verify_explanations(recovered_explainer, x_test, get_mimic_method(surrogate_model))
