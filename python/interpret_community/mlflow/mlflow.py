import os
from tempfile import TemporaryDirectory

import interpret_community
import yaml

from ..explanation.explanation import (_get_explanation_metadata,
                                       load_explanation, save_explanation)


def _load_pyfunc(path):
    """Load the explanation from the given path.

    :param path: The path from which to load the explanation.
    :type path: str
    """

    load_explanation(path)


def save_model(path, loader_module=None, data_path=None, conda_env=None, mlflow_model=None, **kwargs):
    """Save the explanation locally using the MLflow model format.

    This function is necessary for log_explanation to work properly.

    :param path: The destination path for the saved explanation.
    :type path: str
    :param loader_module: The package that will be used to reload a serialized explanation. In this case,
        always interpret_community.mlflow.
    :type loader_module: str
    :param data_path: The path to the serialized explanation files.
    :type data_path: str
    :param conda_env: The path to a YAML file with basic Python environment information.
    :type conda_env: str
    :param mlflow_model: In our case, always None.
    :type mlflow_model: None
    :return: The MLflow model representation of the explanation.
    :rtype: mlflow.models.Model
    """

    try:
        import mlflow
        from mlflow.models import Model
        from mlflow.pyfunc.model import get_default_conda_env
        from mlflow.utils.file_utils import _copy_file_or_tree
    except ImportError as e:
        raise Exception("Could not log_explanation to mlflow. Missing mlflow dependency, "
                        "pip install mlflow to resolve the error: {}.".format(e))

    if os.path.exists(path):
        raise Exception(
            message="Path '{}' already exists".format(path))
    os.makedirs(path)

    if mlflow_model is None:
        mlflow_model = Model()

    data = None
    if data_path is not None:
        model_file = _copy_file_or_tree(src=data_path, dst=path, dst_dir="data")
        data = model_file

    conda_env_subpath = "mlflow_env.yml"
    if conda_env is None:
        conda_env = get_default_conda_env()
    elif not isinstance(conda_env, dict):
        with open(conda_env, "r") as f:
            conda_env = yaml.safe_load(f)
    with open(os.path.join(path, conda_env_subpath), "w") as f:
        yaml.safe_dump(conda_env, stream=f, default_flow_style=False)

    mlflow.pyfunc.add_to_model(
        mlflow_model, loader_module=loader_module, data=data, env=conda_env_subpath, **kwargs)
    mlflow_model.save(os.path.join(path, 'MLmodel'))
    return mlflow_model


def log_explanation(name, explanation):
    """Log the explanation to MLflow using MLflow model logging.

    :param name: The name of the explanation. Will be used as a directory name.
    :type name: str
    :param explanation: The explanation object to log.
    :type explanation: Explanation
    """

    try:
        from mlflow.models import Model
    except ImportError as e:
        raise Exception("Could not log_explanation to mlflow. Missing mlflow dependency, "
                        "pip install mlflow to resolve the error: {}.".format(e))
    import cloudpickle as pickle

    with TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, 'explanation')
        save_explanation(explanation, path, exist_ok=True)

        conda_env = {
            "name": "mlflow-env",
            "channels": ["defaults"],
            "dependencies": [
                "pip",
                {
                    "pip": [
                        "interpret-community=={}".format(interpret_community.__version__),
                        "cloudpickle=={}".format(pickle.__version__)]
                }
            ]
        }
        conda_path = os.path.join(tempdir, "conda.yaml")
        with open(conda_path, "w") as stream:
            yaml.dump(conda_env, stream)
        kwargs = {'interpret_community_metadata': _get_explanation_metadata(explanation)}
        Model.log(name,
                  flavor=interpret_community.mlflow,
                  loader_module='interpret_community.mlflow',
                  data_path=path,
                  conda_env=conda_path,
                  **kwargs)


def get_explanation(run_id, name):
    """Download and deserialize an explanation that has been logged to MLflow.

    :param run_id: The ID of the run the explanation was logged to.
    :type run_id: str
    :param name: The name given to the explanation when it was logged.
    :type name: str
    :return: The rehydrated explanation.
    :rtype: Explanation
    """

    try:
        import mlflow
    except ImportError as e:
        raise Exception("Could not get_explanation from mlflow. Missing mlflow dependency, "
                        "pip install mlflow to resolve the error: {}.".format(e))
    DOWNLOAD_DIR = 'exp_downloads'
    client = mlflow.tracking.MlflowClient()
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    client.download_artifacts(run_id, name, dst_path=DOWNLOAD_DIR)
    full_path = os.path.join(DOWNLOAD_DIR, name, 'data', 'explanation')
    return load_explanation(full_path)
