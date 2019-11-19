# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import pytest

# Tests for Glassbox Models
import logging
import numpy as np

from constants import owner_email_tools_and_ux, DatasetConstants

test_logger = logging.getLogger(__name__)
test_logger.setLevel(logging.DEBUG)

LGBM_MODEL_IDX = 0
SGD_MODEL_IDX = 2


@pytest.mark.owner(email=owner_email_tools_and_ux)
@pytest.mark.usefixtures('clean_dir')
class TestGlassboxModels(object):
    def test_working(self):
        assert True

    def test_train_glassbox_classifier(self, classification_glassbox_models, iris):
        # Fit a glassbox model
        for init_glassbox_model in classification_glassbox_models:
            glassbox_model = init_glassbox_model()
            glassbox_model.fit(iris[DatasetConstants.X_TRAIN], iris[DatasetConstants.Y_TRAIN])
            y_pred = glassbox_model.predict(iris[DatasetConstants.X_TEST])
            assert len(y_pred) == len(iris[DatasetConstants.X_TEST])
            local_explanation = glassbox_model.explain_local(iris[DatasetConstants.X_TEST])
            global_explanation = glassbox_model.explain_global()
            assert np.array(local_explanation.local_importance_values).shape[0] == len(iris[DatasetConstants.CLASSES])
            assert np.array(local_explanation.local_importance_values).shape[1] == len(iris[DatasetConstants.X_TEST])
            assert local_explanation.num_examples == len(iris[DatasetConstants.X_TEST])
            assert len(global_explanation.global_importance_values) == len(iris[DatasetConstants.FEATURES])

    def test_train_glassbox_regressor(self, regression_glassbox_models, boston):
        # Fit a glassbox model
        for init_glassbox_model in regression_glassbox_models:
            glassbox_model = init_glassbox_model()
            glassbox_model.fit(boston[DatasetConstants.X_TRAIN], boston[DatasetConstants.Y_TRAIN])
            y_pred = glassbox_model.predict(boston[DatasetConstants.X_TEST])
            assert len(y_pred) == len(boston[DatasetConstants.X_TEST])
            local_explanation = glassbox_model.explain_local(boston[DatasetConstants.X_TEST])
            global_explanation = glassbox_model.explain_global()
            assert np.array(local_explanation.local_importance_values).shape[0] == len(boston[DatasetConstants.X_TEST])
            assert local_explanation.num_examples == len(boston[DatasetConstants.X_TEST])
            assert len(global_explanation.global_importance_values) == len(boston[DatasetConstants.FEATURES])
