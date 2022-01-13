# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Reimports helpful model wrapper and utils for implicitly rewrapping the model to conform to explainer contracts."""

from ml_wrappers.model import (WrappedClassificationModel, WrappedPytorchModel,
                               WrappedRegressionModel, _wrap_model, wrap_model)

__all__ = ['WrappedClassificationModel', 'WrappedPytorchModel',
           'WrappedRegressionModel', '_wrap_model', 'wrap_model']
