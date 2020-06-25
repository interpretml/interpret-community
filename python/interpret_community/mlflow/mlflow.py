import mlflow
from tempfile import TemporaryDirectory
import os
import yaml

from mlflow.models import Model
from mlflow.utils.file_utils import _copy_file_or_tree
from mlflow.pyfunc.model import get_default_conda_env

import interpret_community

from ..explanation.explanation import save_explanation, load_explanation, _get_explanation_metadata


def _log_explanation(name, explanation):
    with TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, 'explanation')
        save_explanation(explanation, path)
        mlflow.log_artifacts(path)


def _load_pyfunc(path):
    load_explanation(path)


def _save_model_with_loader_module_and_data_path(path, loader_module, data_path=None,
                                                 conda_env=None, mlflow_model=Model(), **kwargs):
    """
    Export model as a generic Python function model.
    :param path: The path to which to save the Python model.
    :param loader_module: The name of the Python module that is used to load the model
                          from ``data_path``. This module must define a method with the prototype
                          ``_load_pyfunc(data_path)``.
    :param data_path: Path to a file or directory containing model data.
    :param code_paths: A list of local filesystem paths to Python file dependencies (or directories
                      containing file dependencies). These files are *prepended* to the system
                      path before the model is loaded.
    :param conda_env: Either a dictionary representation of a Conda environment or the path to a
                      Conda environment yaml file. If provided, this decsribes the environment
                      this model should be run in.
    :return: Model configuration containing model info.
    """

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


def save_model(path, loader_module=None, data_path=None, conda_env=None, mlflow_model=None, **kwargs):
    first_argument_set = {
        "loader_module": loader_module,
        "data_path": data_path,
    }
    first_argument_set_specified = any([item is not None for item in first_argument_set.values()])

    # if os.path.exists(path):
    #     raise MlflowException(
    #         message="Path '{}' already exists".format(path),
    #         error_code=RESOURCE_ALREADY_EXISTS)
    # os.makedirs(path)
    os.makedirs(path, exist_ok=True)
    if mlflow_model is None:
        mlflow_model = Model()

    return _save_model_with_loader_module_and_data_path(
        path=path, loader_module=loader_module, data_path=data_path,
        conda_env=conda_env, mlflow_model=mlflow_model, **kwargs)


def log_explanation(name, explanation):
    try:
        import mlflow.pyfunc
    except ImportError as e:
        raise Exception("Could not log_model to mlflow. Missing mlflow dependency, pip install mlflow to resolve the error: {}.".format(e))
    import cloudpickle as pickle

    model_explanation = Model()
    with TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, 'exp')
        save_explanation(explanation, path, exist_ok=True)

        conda_env = {"name": "mlflow-env",
                     "channels": ["defaults"],
                     "dependencies": ["pip",
                                      {"pip": [
                                        # "interpret-community=={}".format(interpret_community.VERSION),
                                        "cloudpickle=={}".format(pickle.__version__)]
                                      }
                                     ]
                    }
        conda_path = os.path.join(tempdir, "conda.yaml")  # TODO Open issue and bug fix for dict support
        with open(conda_path, "w") as stream:
            yaml.dump(conda_env, stream)
        metadata = {}
        kwargs = {'interpret_community_metadata': _get_explanation_metadata(explanation)}
        Model.log(name,
                  flavor=interpret_community.mlflow,
                  loader_module='interpret_community.mlflow',
                  data_path=path,
                  conda_env=conda_path,
                  **kwargs)
