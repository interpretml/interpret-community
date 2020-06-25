from tempfile import TemporaryDirectory
import os
import yaml

import interpret_community
from ..explanation.explanation import save_explanation, load_explanation, _get_explanation_metadata


def _load_pyfunc(path):
    load_explanation(path)


def save_model(path, loader_module=None, data_path=None, conda_env=None, mlflow_model=None, **kwargs):
    try:
        import mlflow
        from mlflow.models import Model
        from mlflow.utils.file_utils import _copy_file_or_tree
        from mlflow.pyfunc.model import get_default_conda_env
    except ImportError as e:
        raise Exception("Could not log_model to mlflow. Missing mlflow dependency, "
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
    try:
        from mlflow.models import Model
    except ImportError as e:
        raise Exception("Could not log_model to mlflow. Missing mlflow dependency, "
                        "pip install mlflow to resolve the error: {}.".format(e))
    import cloudpickle as pickle

    with TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, 'exp')
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
