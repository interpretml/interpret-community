# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Defines helpful model wrapper and utils for implicitly rewrapping the model to conform to explainer contracts."""

import logging
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier

from .constants import ModelTask, SKLearn

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', 'Starting from version 2.2.1', UserWarning)


module_logger = logging.getLogger(__name__)
module_logger.setLevel(logging.INFO)


try:
    import torch
    import torch.nn as nn
except ImportError:
    module_logger.debug('Could not import torch, required if using a PyTorch model')


class _FunctionWrapper(object):
    """Wraps a function to reshape the input and output data.

    :param function: The prediction function to evaluate on the examples.
    :type function: function
    """

    def __init__(self, function):
        """Wraps a function to reshape the input and output data.

        :param function: The prediction function to evaluate on the examples.
        :type function: function
        """
        self._function = function

    def _function_input_1D_wrapper(self, dataset):
        """Wraps a function that reshapes the input dataset to be 2D from 1D.

        :param dataset: The model evaluation examples.
        :type dataset: numpy.array
        :return: A wrapped function.
        :rtype: function
        """
        if len(dataset.shape) == 1:
            dataset = dataset.reshape(1, -1)
        return self._function(dataset)

    def _function_flatten(self, dataset):
        """Wraps a function that flattens the input dataset from 2D to 1D.

        :param dataset: The model evaluation examples.
        :type dataset: numpy.array
        :return: A wrapped function.
        :rtype: function
        """
        return self._function(dataset).flatten()

    def _function_2D_two_cols_wrapper_2D_result(self, dataset):
        """Wraps a function that creates two columns, [1-p, p], from 2D array of one column evaluation result.

        :param dataset: The model evaluation examples.
        :type dataset: numpy.array
        :return: A wrapped function.
        :rtype: function
        """
        result = self._function(dataset)[:, 0]
        return np.stack([1 - result, result], axis=-1)

    def _function_2D_two_cols_wrapper_1D_result(self, dataset):
        """Wraps a function that creates two columns, [1-p, p], from evaluation result that is a 1D array.

        :param dataset: The model evaluation examples.
        :type dataset: numpy.array
        :return: A wrapped function.
        :rtype: function
        """
        result = self._function(dataset)
        return np.stack([1 - result, result], axis=-1)

    def _function_2D_one_col_wrapper(self, dataset):
        """Wraps a function that creates one column in rare edge case scenario for multiclass one-class result.

        :param dataset: The model evaluation examples.
        :type dataset: numpy.array
        :return: A wrapped function.
        :rtype: function
        """
        result = self._function(dataset)
        return result.reshape(result.shape[0], 1)


def _convert_to_two_cols(function, examples):
    """In classification case, convert the function's output to two columns if it outputs one column.

    :param function: The prediction function to evaluate on the examples.
    :type function: function
    :param examples: The model evaluation examples.
    :type examples: numpy.array or list
    :return: The function chosen from given model and classification domain.
    :rtype: (function, str)
    """
    # Add wrapper function to convert output to 2D array, check values to decide on whether
    # to throw, or create two columns [1-p, p], or leave just one in multiclass one-class edge-case
    result = function(examples)
    # If the function gives a 2D array of one column, we will need to reshape it prior to concat
    is_2d_result = len(result.shape) == 2
    convert_to_two_cols = False
    for value in result:
        if value < 0 or value > 1:
            raise Exception("Probability values outside of valid range: " + str(value))
        if value < 1:
            convert_to_two_cols = True
    wrapper = _FunctionWrapper(function)
    if convert_to_two_cols:
        # Create two cols, [1-p, p], from evaluation result
        if is_2d_result:
            return (wrapper._function_2D_two_cols_wrapper_2D_result, ModelTask.Classification)
        else:
            return (wrapper._function_2D_two_cols_wrapper_1D_result, ModelTask.Classification)
    else:
        if is_2d_result:
            return (function, ModelTask.Classification)
        else:
            return (wrapper._function_2D_one_col_wrapper, ModelTask.Classification)


class WrappedPytorchModel(object):
    """A class for wrapping a PyTorch model in the scikit-learn specification."""

    def __init__(self, model):
        """Initialize the PytorchModelWrapper with the model and evaluation function."""
        self._model = model
        # Set eval automatically for user for batchnorm and dropout layers
        self._model.eval()

    def predict(self, dataset):
        """Predict the output using the wrapped PyTorch model.

        :param dataset: The dataset to predict on.
        :type dataset: interpret_community.dataset.dataset_wrapper.DatasetWrapper
        """
        # Convert the data to pytorch Variable
        if isinstance(dataset, pd.DataFrame):
            dataset = dataset.values
        wrapped_dataset = torch.Tensor(dataset)
        with torch.no_grad():
            result = self._model(wrapped_dataset).numpy()
        # Reshape to 2D if output is 1D and input has one row
        if len(dataset.shape) == 1:
            result = result.reshape(1, -1)
        return result

    def predict_classes(self, dataset):
        """Predict the class using the wrapped PyTorch model.

        :param dataset: The dataset to predict on.
        :type dataset: interpret_community.dataset.dataset_wrapper.DatasetWrapper
        """
        # Convert the data to pytorch Variable
        if isinstance(dataset, pd.DataFrame):
            dataset = dataset.values
        wrapped_dataset = torch.Tensor(dataset)
        with torch.no_grad():
            result = self._model(wrapped_dataset)
        result_len = len(result.shape)
        if result_len == 1 or (result_len > 1 and result.shape[1] == 1):
            result = np.where(result.numpy() > 0.5, 1, 0)
        else:
            result = torch.max(result, 1)[1].numpy()
        # Reshape to 2D if output is 1D and input has one row
        if len(dataset.shape) == 1:
            result = result.reshape(1, -1)
        return result

    def predict_proba(self, dataset):
        """Predict the output probability using the wrapped PyTorch model.

        :param dataset: The dataset to predict_proba on.
        :type dataset: interpret_community.dataset.dataset_wrapper.DatasetWrapper
        """
        return self.predict(dataset)


class BaseWrappedModel(object):
    """A base class for WrappedClassificationModel and WrappedRegressionModel."""

    def __init__(self, model, eval_function, examples, model_task):
        """Initialize the WrappedClassificationModel with the model and evaluation function."""
        self._eval_function = eval_function
        self._model = model
        self._examples = examples
        self._model_task = model_task

    def __getstate__(self):
        """Influence how BaseWrappedModel is pickled.

        Removes _eval_function which may not be serializable.

        :return state: The state to be pickled, with _eval_function removed.
        :rtype: dict
        """
        odict = self.__dict__.copy()
        if self._examples is not None:
            del odict['_eval_function']
        return odict

    def __setstate__(self, state):
        """Influence how BaseWrappedModel is unpickled.

        Re-adds _eval_function which may not be serializable.

        :param dict: A dictionary of deserialized state.
        :type dict: dict
        """
        self.__dict__.update(state)
        if self._examples is not None:
            eval_function, _ = _eval_model(self._model, self._examples, self._model_task)
            self._eval_function = eval_function


class WrappedClassificationModel(BaseWrappedModel):
    """A class for wrapping a classification model."""

    def __init__(self, model, eval_function, examples=None):
        """Initialize the WrappedClassificationModel with the model and evaluation function."""
        super(WrappedClassificationModel, self).__init__(model, eval_function, examples, ModelTask.Classification)

    def predict(self, dataset):
        """Predict the output using the wrapped classification model.

        :param dataset: The dataset to predict on.
        :type dataset: interpret_community.dataset.dataset_wrapper.DatasetWrapper
        """
        is_sequential = str(type(self._model)).endswith("tensorflow.python.keras.engine.sequential.Sequential'>")
        if is_sequential or isinstance(self._model, WrappedPytorchModel):
            return self._model.predict_classes(dataset).flatten()
        preds = self._model.predict(dataset)
        if isinstance(preds, pd.DataFrame):
            preds = preds.values.ravel()
        # Handle possible case where the model has only a predict function and it outputs probabilities
        # Note this is different from WrappedClassificationWithoutProbaModel where there is no predict_proba
        # method but the predict method outputs classes
        has_predict_proba = hasattr(self._model, SKLearn.PREDICT_PROBA)
        if not has_predict_proba:
            if len(preds.shape) == 1:
                return np.argmax(preds)
            else:
                return np.argmax(preds, axis=1)
        return preds

    def predict_proba(self, dataset):
        """Predict the output probability using the wrapped model.

        :param dataset: The dataset to predict_proba on.
        :type dataset: interpret_community.dataset.dataset_wrapper.DatasetWrapper
        """
        proba_preds = self._eval_function(dataset)
        if isinstance(proba_preds, pd.DataFrame):
            proba_preds = proba_preds.values

        return proba_preds


class WrappedRegressionModel(BaseWrappedModel):
    """A class for wrapping a regression model."""

    def __init__(self, model, eval_function, examples=None):
        """Initialize the WrappedRegressionModel with the model and evaluation function."""
        super(WrappedRegressionModel, self).__init__(model, eval_function, examples, ModelTask.Regression)

    def predict(self, dataset):
        """Predict the output using the wrapped regression model.

        :param dataset: The dataset to predict on.
        :type dataset: interpret_community.dataset.dataset_wrapper.DatasetWrapper
        """
        preds = self._eval_function(dataset)
        if isinstance(preds, pd.DataFrame):
            preds = preds.values.ravel()

        return preds


class WrappedClassificationWithoutProbaModel(object):
    """A class for wrapping a classifier without a predict_proba method.

    Note: the classifier may not output numeric values for its predictions.
    We generate a trival boolean version of predict_proba
    """

    def __init__(self, model):
        """Initialize the WrappedClassificationWithoutProbaModel with the model."""
        self._model = model
        # Create a map from classes to index
        self._classes_to_index = {}
        for index, i in enumerate(self._model.classes_):
            self._classes_to_index[i] = index
        self._num_classes = len(self._model.classes_)

    def predict(self, dataset):
        """Predict the output using the wrapped regression model.

        :param dataset: The dataset to predict on.
        :type dataset: interpret_community.dataset.dataset_wrapper.DatasetWrapper
        """
        return self._model.predict(dataset)

    def predict_proba(self, dataset):
        """Predict the output probability using the wrapped model.

        :param dataset: The dataset to predict_proba on.
        :type dataset: interpret_community.dataset.dataset_wrapper.DatasetWrapper
        """
        predictions = self.predict(dataset)
        # Generate trivial boolean array for predictions
        probabilities = np.zeros((predictions.shape[0], self._num_classes))
        for row_idx, pred_class in enumerate(predictions):
            class_index = self._classes_to_index[pred_class]
            probabilities[row_idx, class_index] = 1
        return probabilities


def wrap_model(model, examples, model_task):
    """If needed, wraps the model in a common API based on model task and prediction function contract.

    :param model: The model to evaluate on the examples.
    :type model: model with a predict or predict_proba function.
    :param examples: The model evaluation examples.
    :type examples: interpret_community.dataset.dataset_wrapper.DatasetWrapper
    :param model_task: Optional parameter to specify whether the model is a classification or regression model.
        In most cases, the type of the model can be inferred based on the shape of the output, where a classifier
        has a predict_proba method and outputs a 2 dimensional array, while a regressor has a predict method and
        outputs a 1 dimensional array.
    :type model_task: str
    :return: The wrapper model.
    :rtype: model
    """
    return _wrap_model(model, examples, model_task, False)[0]


def _wrap_model(model, examples, model_task, is_function):
    """If needed, wraps the model or function in a common API based on model task and prediction function contract.

    :param model: The model or function to evaluate on the examples.
    :type model: function or model with a predict or predict_proba function
    :param examples: The model evaluation examples.
    :type examples: interpret_community.dataset.dataset_wrapper.DatasetWrapper
    :param model_task: Optional parameter to specify whether the model is a classification or regression model.
        In most cases, the type of the model can be inferred based on the shape of the output, where a classifier
        has a predict_proba method and outputs a 2 dimensional array, while a regressor has a predict method and
        outputs a 1 dimensional array.
    :type model_task: str
    :return: The function chosen from given model and chosen domain, or model wrapping the function and chosen domain.
    :rtype: (function, str) or (model, str)
    """
    if is_function:
        return _eval_function(model, examples, model_task)
    else:
        try:
            if isinstance(model, nn.Module):
                # Wrap the model in an extra layer that converts the numpy array
                # to pytorch Variable and adds predict and predict_proba functions
                model = WrappedPytorchModel(model)
        except (NameError, AttributeError):
            module_logger.debug('Could not import torch, required if using a pytorch model')
        if _classifier_without_proba(model):
            model = WrappedClassificationWithoutProbaModel(model)
        eval_function, eval_ml_domain = _eval_model(model, examples, model_task)
        if eval_ml_domain == ModelTask.Classification:
            return WrappedClassificationModel(model, eval_function, examples), eval_ml_domain
        else:
            return WrappedRegressionModel(model, eval_function, examples), eval_ml_domain


def _classifier_without_proba(model):
    """Returns True if the given model is a classifier without predict_proba, eg SGDClassifier.

    :param model: The model to evaluate on the examples.
    :type model: model with a predict or predict_proba function
    :return: True if the given model is a classifier without predict_proba.
    :rtype: bool
    """
    return isinstance(model, SGDClassifier) and not hasattr(model, SKLearn.PREDICT_PROBA)


def _eval_model(model, examples, model_task):
    """Return function from model and specify the ML Domain using model evaluation on examples.

    :param model: The model to evaluate on the examples.
    :type model: model with a predict or predict_proba function
    :param examples: The model evaluation examples.
    :type examples: interpret_community.dataset.dataset_wrapper.DatasetWrapper
    :param model_task: Optional parameter to specify whether the model is a classification or regression model.
        In most cases, the type of the model can be inferred based on the shape of the output, where a classifier
        has a predict_proba method and outputs a 2 dimensional array, while a regressor has a predict method and
        outputs a 1 dimensional array.
    :type model_task: str
    :return: The function chosen from given model and chosen domain.
    :rtype: (function, str)
    """
    # TODO: Add more model types here
    is_sequential = str(type(model)).endswith("tensorflow.python.keras.engine.sequential.Sequential'>")
    if is_sequential or isinstance(model, WrappedPytorchModel):
        if model_task == ModelTask.Regression:
            return _eval_function(model.predict, examples, ModelTask.Regression)
        result = model.predict_proba(examples.typed_wrapper_func(examples.dataset[0:1]))
        if result.shape[1] == 1 and model_task == ModelTask.Unknown:
            raise Exception("Please specify model_task to disambiguate model type since "
                            "result of calling function is 2D array of one column.")
        else:
            return _eval_function(model.predict_proba, examples, ModelTask.Classification)
    else:
        has_predict_proba = hasattr(model, SKLearn.PREDICT_PROBA)
        # Note: Allow user to override default to use predict method for regressor
        if has_predict_proba and model_task != ModelTask.Regression:
            return _eval_function(model.predict_proba, examples, model_task)
        else:
            return _eval_function(model.predict, examples, model_task)


def _eval_function(function, examples, model_task, wrapped=False):
    """Return function and specify the ML Domain using function evaluation on examples.

    :param function: The prediction function to evaluate on the examples.
    :type function: function
    :param examples: The model evaluation examples.
    :type examples: interpret_community.dataset.dataset_wrapper.DatasetWrapper
    :param model_task: Optional parameter to specify whether the model is a classification or regression model.
        In most cases, the type of the model can be inferred based on the shape of the output, where a classifier
        has a predict_proba method and outputs a 2 dimensional array, while a regressor has a predict method and
        outputs a 1 dimensional array.
    :type model_task: str
    :param wrapped: Indicates if function has already been wrapped.
    :type wrapped: bool
    :return: The function chosen from given model and chosen domain.
    :rtype: (function, str)
    """
    # Try to run the function on a single example - if it doesn't work wrap
    # it in a function that converts a 1D array to 2D for those functions
    # that only support 2D arrays as input
    examples_dataset = examples.dataset
    if str(type(examples_dataset)).endswith(".DenseData'>"):
        examples_dataset = examples_dataset.data
    try:
        result = function(examples.typed_wrapper_func(examples_dataset[0]))
        if result is None:
            raise Exception("Wrapped function returned None in model wrapper when called on dataset")
    except Exception as ex:
        # If function has already been wrapped, re-throw error to prevent stack overflow
        if wrapped:
            raise ex
        wrapper = _FunctionWrapper(function)
        return _eval_function(wrapper._function_input_1D_wrapper, examples, model_task, wrapped=True)
    if len(result.shape) == 2:
        # If the result of evaluation the function is a 2D array of 1 column,
        # and they did not specify classifier or regressor, throw exception
        # to force the user to disambiguate the results.
        if result.shape[1] == 1:
            if model_task == ModelTask.Unknown:
                if isinstance(result, pd.DataFrame):
                    return (function, ModelTask.Regression)
                raise Exception("Please specify model_task to disambiguate model type since "
                                "result of calling function is 2D array of one column.")
            elif model_task == ModelTask.Classification:
                return _convert_to_two_cols(function, examples_dataset)
            else:
                # model_task == ModelTask.Regression
                # In case user specified a regressor but we have a 2D output with one column,
                # we want to flatten the function to 1D
                wrapper = _FunctionWrapper(function)
                return (wrapper._function_flatten, model_task)
        else:
            if model_task == ModelTask.Unknown or model_task == ModelTask.Classification:
                return (function, ModelTask.Classification)
            else:
                raise Exception("Invalid shape for prediction: "
                                "Regression function cannot output 2D array with multiple columns")
    elif len(result.shape) == 1:
        if model_task == ModelTask.Unknown:
            return (function, ModelTask.Regression)
        elif model_task == ModelTask.Classification:
            return _convert_to_two_cols(function, examples_dataset)
        return (function, model_task)
    elif len(result.shape) == 0:
        # single value returned, flatten output array
        wrapper = _FunctionWrapper(function)
        return (wrapper._function_flatten, model_task)
    raise Exception("Failed to wrap function, may require custom wrapper for input function or model")
