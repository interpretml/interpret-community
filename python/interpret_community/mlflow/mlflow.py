import mlflow
from tempfile import TemporaryDirectory
import os

from ..explanation.explanation import save_explanation


def _log_explanation(name, explanation):
    with TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, 'explanation')
        save_explanation(explanation, path)
        mlflow.log_artifacts(path)
