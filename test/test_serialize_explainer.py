# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Tests serializing the explainers or models"""

import pytest
import logging
from joblib import dump
from os import path

from common_utils import create_sklearn_svm_classifier, create_scikit_cancer_data
from constants import owner_email_tools_and_ux
from interpret.ext.blackbox import TabularExplainer

test_logger = logging.getLogger(__name__)


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
