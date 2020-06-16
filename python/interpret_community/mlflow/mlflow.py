import mlflow
from tempfile import TemporaryDirectory
import os
import yaml

import interpret_community

from ..explanation.explanation import save_explanation, load_explanation


def _log_explanation(name, explanation):
    with TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, 'explanation')
        save_explanation(explanation, path)
        mlflow.log_artifacts(path)


def _load_pyfunc(path):
    load_explanation(path)


def log_explanation(name, explanation):
    try:
        import mlflow.pyfunc
    except ImportError as e:
        raise Exception("Could not log_model to mlflow. Missing mlflow dependency, pip install mlflow to resolve the error: {}.".format(e))
    import cloudpickle as pickle

    with TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, 'explanation')
        save_explanation(explanation, path)

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
        mlflow.pyfunc.log_model(explanation.id, loader_module="interpret_community.mlflow.mlflow", data_path=tempdir, conda_env=conda_path)
