import os

import mlflow
import pytest
from common_utils import create_sklearn_random_forest_classifier
from constants import DatasetConstants, owner_email_tools_and_ux
from interpret_community.mlflow.mlflow import get_explanation, log_explanation
from test_serialize_explanation import _assert_explanation_equivalence

TEST_DOWNLOAD = 'test_download'
TEST_EXPERIMENT = 'test_experiment'
TEST_EXPLANATION = 'test_explanation'


@pytest.mark.owner(email=owner_email_tools_and_ux)
@pytest.mark.usefixtures('_clean_dir')
class TestMlflow(object):

    def test_working(self):
        assert True

    def test_upload_as_model(self, iris, tabular_explainer, tracking_uri):
        mlflow.set_tracking_uri(tracking_uri)
        x_train = iris[DatasetConstants.X_TRAIN]
        x_test = iris[DatasetConstants.X_TEST]
        y_train = iris[DatasetConstants.Y_TRAIN]

        model = create_sklearn_random_forest_classifier(x_train, y_train)

        explainer = tabular_explainer(model, x_train)
        global_explanation = explainer.explain_global(x_test)
        mlflow.set_experiment(TEST_EXPERIMENT)
        with mlflow.start_run() as run:
            log_explanation(TEST_EXPLANATION, global_explanation)
            os.makedirs(TEST_DOWNLOAD, exist_ok=True)
            run_id = run.info.run_id
        downloaded_explanation_mlflow = get_explanation(run_id, TEST_EXPLANATION)
        _assert_explanation_equivalence(global_explanation, downloaded_explanation_mlflow)

    def test_upload_two_explanations(self, iris, tabular_explainer, tracking_uri):
        mlflow.set_tracking_uri(tracking_uri)
        x_train = iris[DatasetConstants.X_TRAIN]
        x_test = iris[DatasetConstants.X_TEST]
        y_train = iris[DatasetConstants.Y_TRAIN]

        model = create_sklearn_random_forest_classifier(x_train, y_train)

        explainer = tabular_explainer(model, x_train)
        global_explanation = explainer.explain_global(x_test)
        local_explanation = explainer.explain_local(x_test)
        mlflow.set_experiment(TEST_EXPERIMENT)
        with mlflow.start_run() as run:
            log_explanation('global_explanation', global_explanation)
            log_explanation('local_explanation', local_explanation)
            os.makedirs(TEST_DOWNLOAD, exist_ok=True)
            run_id = run.info.run_id
        downloaded_explanation_mlflow = get_explanation(run_id, 'global_explanation')
        _assert_explanation_equivalence(global_explanation, downloaded_explanation_mlflow)
