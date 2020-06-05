import os
import pytest
import mlflow

from azureml.core import Workspace, Experiment, Run

from common_utils import create_sklearn_random_forest_classifier

from interpret_community.mlflow.mlflow import _log_explanation
from interpret_community.explanation.explanation import load_explanation
from constants import owner_email_tools_and_ux, DatasetConstants
from test_serialize_explanation import _assert_explanation_equivalence


@pytest.mark.owner(email=owner_email_tools_and_ux)
@pytest.mark.usefixtures('clean_dir')
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
        mlflow.set_experiment('test_experiment')
        client = mlflow.tracking.MlflowClient()
        with mlflow.start_run() as run:
            _log_explanation('test_explanation', global_explanation)
            artifacts = client.list_artifacts(run.info.run_uuid)
            os.mkdir('test_explanation')
            download_path = client.download_artifacts(run.info.run_uuid, '', dst_path='test_explanation')
        downloaded_explanation = load_explanation(download_path)
        _assert_explanation_equivalence(global_explanation, downloaded_explanation)

    def test_upload_to_azure(self, iris, tabular_explainer):
        ws = Workspace.from_config()
        mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
        x_train = iris[DatasetConstants.X_TRAIN]
        x_test = iris[DatasetConstants.X_TEST]
        y_train = iris[DatasetConstants.Y_TRAIN]

        model = create_sklearn_random_forest_classifier(x_train, y_train)

        explainer = tabular_explainer(model, x_train)
        global_explanation = explainer.explain_global(x_test)
        mlflow.set_experiment('test_experiment')
        client = mlflow.tracking.MlflowClient()
        with mlflow.start_run() as run:
            _log_explanation('test_explanation', global_explanation)
            artifacts = client.list_artifacts(run.info.run_uuid)
            os.mkdir('test_explanation')
            download_path = client.download_artifacts(run.info.run_uuid, '', dst_path='test_explanation')
        downloaded_explanation_mlflow = load_explanation(download_path)
        _assert_explanation_equivalence(global_explanation, downloaded_explanation_mlflow)
        azure_experiment = Experiment(ws, 'test_experiment')
        azure_run = Run(azure_experiment, run.info.run_uuid)
        azure_run.download_files(prefix='test_explanation', output_directory='azure_download')
        downloaded_explanation_azure = load_explanation('test_explanation')
        _assert_explanation_equivalence(global_explanation, downloaded_explanation_azure)
