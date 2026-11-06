# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Defines a decorator for tabular data which wraps pandas dataframes, scipy and numpy arrays in a DatasetWrapper."""

from functools import wraps
from inspect import signature

import pandas as pd
from ml_wrappers import DatasetWrapper

from ..common.constants import ResetIndex
from ..common.warnings_suppressor import shap_warnings_suppressor

with shap_warnings_suppressor():
    import shap


def _validate_features(features, dataset, reset_index=ResetIndex.Ignore):
    """Validate feature names against the data before transformation or subsetting."""
    if features is not None:
        if isinstance(dataset, DatasetWrapper):
            dataset = dataset.original_dataset_with_type
        masker_type = getattr(getattr(shap, 'maskers', None), 'Masker', ())
        if isinstance(dataset, masker_type):
            dataset = dataset.data
        if str(type(dataset)).endswith(".DenseData'>"):
            dataset = dataset.data
        if isinstance(dataset, tuple) and len(dataset) == 2:
            num_features = len(dataset[0])
        elif isinstance(dataset, dict) and 'mean' in dataset:
            num_features = len(dataset['mean'])
        else:
            if not hasattr(dataset, 'shape'):
                dataset = wrap_dataset(dataset).dataset
            shape = dataset.shape
            num_features = shape[1] if len(shape) > 1 else shape[0]
            if isinstance(dataset, pd.DataFrame) and reset_index not in (ResetIndex.Ignore, ResetIndex.ResetTeacher):
                if not pd.RangeIndex(len(dataset)).equals(dataset.index):
                    num_features += dataset.index.nlevels
        if len(features) != num_features:
            raise ValueError(
                'The number of feature names passed in must be the same as the number of '
                'columns in the data. Number of features: {} feature names: {}'.format(
                    num_features, len(features)))


def tabular_decorator(explain_func):
    """Decorate an explanation function to wrap evaluation examples in a DatasetWrapper.

    :param explain_func: An explanation function where the first argument is a dataset.
    :type explain_func: explanation function
    """
    @wraps(explain_func)
    def explain_func_wrapper(self, evaluation_examples, *args, **kwargs):
        if not isinstance(evaluation_examples, DatasetWrapper):
            self._logger.debug('Eval examples not wrapped, wrapping')
            evaluation_examples = DatasetWrapper(evaluation_examples)
        # Older initialized explainers may only retain engineered feature names.
        features = None if hasattr(self, 'initialization_examples') else getattr(self, 'features', None)
        features = getattr(self, '_input_features', features)
        reset_index = ResetIndex.Ignore
        if getattr(self, 'transformations', None) is None:
            reset_index = getattr(self, 'reset_index', ResetIndex.Ignore)
        _validate_features(features, evaluation_examples, reset_index=reset_index)
        return explain_func(self, evaluation_examples, *args, **kwargs)
    return explain_func_wrapper


def wrap_dataset(dataset):
    if not isinstance(dataset, DatasetWrapper):
        dataset = DatasetWrapper(dataset)
    return dataset


def init_tabular_decorator(init_func):
    """Decorate a constructor to wrap initialization examples in a DatasetWrapper.

    Provided for convenience for tabular data explainers.

    :param init_func: Initialization constructor where the second argument is a dataset.
    :type init_func: Initialization constructor.
    """
    init_signature = signature(init_func)

    @wraps(init_func)
    def init_wrapper(self, model, initialization_examples, *args, **kwargs):
        initialization_examples = wrap_dataset(initialization_examples)
        arguments = init_signature.bind(
            self, model, initialization_examples, *args, **kwargs).arguments
        self._input_features = arguments.get('features')
        reset_index = ResetIndex.Ignore
        if arguments.get('transformations') is None:
            reset_index = arguments.get('reset_index', ResetIndex.Ignore)
        _validate_features(self._input_features, initialization_examples, reset_index=reset_index)
        return init_func(self, model, initialization_examples, *args, **kwargs)
    return init_wrapper
