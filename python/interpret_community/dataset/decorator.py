# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Defines a decorator for tabular data which wraps pandas dataframes, scipy and numpy arrays in a DatasetWrapper."""

from .dataset_wrapper import DatasetWrapper
from functools import wraps


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
        return explain_func(self, evaluation_examples, *args, **kwargs)
    return explain_func_wrapper


def init_tabular_decorator(init_func):
    """Decorate a constructor to wrap initialization examples in a DatasetWrapper.

    Provided for convenience for tabular data explainers.

    :param init_func: Initialization constructor where the second argument is a dataset.
    :type init_func: Initialization constructor.
    """
    @wraps(init_func)
    def init_wrapper(self, model, initialization_examples, *args, **kwargs):
        if not isinstance(initialization_examples, DatasetWrapper):
            initialization_examples = DatasetWrapper(initialization_examples)
        return init_func(self, model, initialization_examples, *args, **kwargs)
    return init_wrapper
