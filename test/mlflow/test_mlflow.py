import os
import pytest
import mlflow

from common_utils import create_sklearn_random_forest_classifier

from interpret_community.mlflow.mlflow import _log_explanation, log_explanation
from interpret_community.explanation.explanation import load_explanation
from constants import owner_email_tools_and_ux, DatasetConstants
from test_serialize_explanation import _assert_explanation_equivalence


TEST_EXPLANATION = 'test_explanation'
TEST_EXPERIMENT = 'test_experiment'


@pytest.mark.owner(email=owner_email_tools_and_ux)
# @pytest.mark.usefixtures('clean_dir')
class TestMlflow(object):

    def test_working(self):
        assert True

    def test_basic_upload(self, iris, tabular_explainer):
        x_train = iris[DatasetConstants.X_TRAIN]
        x_test = iris[DatasetConstants.X_TEST]
        y_train = iris[DatasetConstants.Y_TRAIN]

        model = create_sklearn_random_forest_classifier(x_train, y_train)

        explainer = tabular_explainer(model, x_train)
        global_explanation = explainer.explain_global(x_test)
        mlflow.set_experiment(TEST_EXPERIMENT)
        client = mlflow.tracking.MlflowClient()
        with mlflow.start_run() as run:
            _log_explanation(TEST_EXPLANATION, global_explanation)
            os.mkdir(TEST_EXPLANATION)
            download_path = client.download_artifacts(run.info.run_uuid, '', dst_path=TEST_EXPLANATION)
        downloaded_explanation = load_explanation(download_path)
        _assert_explanation_equivalence(global_explanation, downloaded_explanation)

    def test_upload_to_tracking_store(self, iris, tabular_explainer, tracking_uri):
        mlflow.set_tracking_uri(tracking_uri)
        x_train = iris[DatasetConstants.X_TRAIN]
        x_test = iris[DatasetConstants.X_TEST]
        y_train = iris[DatasetConstants.Y_TRAIN]

        model = create_sklearn_random_forest_classifier(x_train, y_train)

        explainer = tabular_explainer(model, x_train)
        global_explanation = explainer.explain_global(x_test)
        mlflow.set_experiment(TEST_EXPERIMENT)
        client = mlflow.tracking.MlflowClient()
        with mlflow.start_run() as run:
            _log_explanation(TEST_EXPLANATION, global_explanation)
            os.mkdir(TEST_EXPLANATION)
            download_path = client.download_artifacts(run.info.run_uuid, '', dst_path=TEST_EXPLANATION)
        downloaded_explanation_mlflow = load_explanation(download_path)
        _assert_explanation_equivalence(global_explanation, downloaded_explanation_mlflow)

    def test_upload_as_model(self, iris, tabular_explainer, tracking_uri):
        mlflow.set_tracking_uri(tracking_uri)
        x_train = iris[DatasetConstants.X_TRAIN]
        x_test = iris[DatasetConstants.X_TEST]
        y_train = iris[DatasetConstants.Y_TRAIN]

        model = create_sklearn_random_forest_classifier(x_train, y_train)

        explainer = tabular_explainer(model, x_train)
        global_explanation = explainer.explain_global(x_test)
        mlflow.set_experiment(TEST_EXPERIMENT)
        client = mlflow.tracking.MlflowClient()
        with mlflow.start_run() as run:
            log_explanation(TEST_EXPLANATION, global_explanation)
            os.mkdir(TEST_EXPLANATION)
            download_path = client.download_artifacts(run.info.run_uuid, '', dst_path=TEST_EXPLANATION)
        downloaded_explanation_mlflow = load_explanation(download_path)
        _assert_explanation_equivalence(global_explanation, downloaded_explanation_mlflow)
