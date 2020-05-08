# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Defines the black box explainer API, which can either take in a black box model or function."""

import numpy as np
from scipy.sparse import issparse, isspmatrix_csr, csr_matrix
from functools import wraps

from .base_explainer import BaseExplainer
from .aggregate import init_aggregator_decorator
from .constants import ModelTask, ExplainParams
from .chained_identity import ChainedIdentity

from .._internal.raw_explain.raw_explain_utils import get_datamapper_and_transformed_data
from ..dataset.dataset_wrapper import DatasetWrapper


class _Wrapper(object):
    """Internal wrapper class, used to get around issues with pickle serialization.

    :param initialization_examples: A matrix of feature vector examples (# examples x # features) for
        initializing the explainer.
    :type initialization_examples: numpy.array or pandas.DataFrame or iml.datatypes.DenseData or
        scipy.sparse.csr_matrix
    :param function: The function to explain.
    :type function: function that accepts a 2d ndarray
    """

    def __init__(self, initialization_examples, function):
        """Initialize the Wrapper, used to get around issues with pickle serialization.

        :param initialization_examples: A matrix of feature vector examples (# examples x # features) for
            initializing the explainer.
        :type initialization_examples: numpy.array or pandas.DataFrame or iml.datatypes.DenseData or
            scipy.sparse.csr_matrix
        :param function: The function to explain.
        :type function: function that accepts a 2d ndarray
        """
        self.typed_wrapper_func = initialization_examples.typed_wrapper_func
        self.function = function

    def wrapped_function(self, dataset):
        """Combine the prediction function and type cast function into a single function.

        :param dataset: A matrix of feature vector examples (# examples x # features).
        :type dataset: numpy.array or pandas.DataFrame or scipy.sparse.csr_matrix
        """
        return self.function(self.typed_wrapper_func(dataset))


class BlackBoxMixin(ChainedIdentity):
    """Mixin for black box models or functions.

    :param model: The model to explain or function if is_function is True.
    :type model: model that implements sklearn.predict or sklearn.predict_proba or function that accepts a 2d ndarray
    :param is_function: Default is False. Set to True if passing sklearn.predict or sklearn.predict_proba
        function instead of model.
    :type is_function: bool
    """

    def __init__(self, model, is_function=False, model_task=ModelTask.Unknown, **kwargs):
        """Initialize the BlackBoxMixin.

        :param model: The model to explain or function if is_function is True.
        :type model: model that implements sklearn.predict or sklearn.predict_proba or function that accepts a 2d
            ndarray
        :param is_function: Default is False. Set to True if passing sklearn.predict or sklearn.predict_proba
            function instead of model.
        :type is_function: bool
        :param model_task: Optional parameter to specify whether the model is a classification or regression model.
            In most cases, the type of the model can be inferred based on the shape of the output, where a classifier
            has a predict_proba method and outputs a 2 dimensional array, while a regressor has a predict method and
            outputs a 1 dimensional array.
        :type model_task: str
        """
        super(BlackBoxMixin, self).__init__(**kwargs)
        self._logger.debug('Initializing BlackBoxMixin')
        # If true, this is a classification model
        self.predict_proba_flag = hasattr(model, "predict_proba")

        if is_function:
            self._logger.debug('Function passed in, no model')
            self.function = model
            self.model = None
        else:
            self._logger.debug('Model passed in')
            self.model = model
            if self.predict_proba_flag:
                self.function = self.model.predict_proba
                # Allow user to override default to use predict method for regressor
                if model_task == ModelTask.Regression:
                    self.function = self.model.predict
            else:
                errMsg = 'predict_proba not supported by given model, assuming regression model and trying predict'
                self._logger.debug(errMsg)
                # try predict instead since this is likely a regression scenario
                self.function = self.model.predict
                # If classifier, but there is no predict_proba method, throw an exception
                if model_task == ModelTask.Classification:
                    raise Exception("No predict_proba method on model which has model_task='classifier'")

    def _get_ys_dict(self, evaluation_examples, transformations=None, allow_all_transformations=False):
        """Get the predicted ys to be incorporated into a kwargs dictionary.
        :param evaluation_examples: The same ones we usually work with, must be able to be passed into the
            model or function.
        :type evaluation_examples: numpy.array or pandas.DataFrame or scipy.sparse.csr_matrix
        :param transformations: See documentation on any explainer.
        :type transformations: sklearn.compose.ColumnTransformer or list[tuple]
        :param allow_all_transformations: Allow many to many and many to one transformations
        :type allow_all_transformations: bool
        :return: The dictionary with none, one, or both of predicted ys and predicted proba ys for eval
            examples.
        :rtype: dict
        """
        if transformations is not None:
            _, evaluation_examples = get_datamapper_and_transformed_data(
                examples=evaluation_examples,
                transformations=transformations,
                allow_all_transformations=allow_all_transformations
            )
        if isinstance(evaluation_examples, DatasetWrapper):
            evaluation_examples = evaluation_examples.original_dataset_with_type
        if len(evaluation_examples.shape) == 1:
            evaluation_examples = evaluation_examples.reshape(1, -1)
        ys_dict = {}
        if self.model is not None:
            ys_dict[ExplainParams.EVAL_Y_PRED] = self.model.predict(evaluation_examples)
            if self.predict_proba_flag:
                ys_dict[ExplainParams.EVAL_Y_PRED_PROBA] = self.model.predict_proba(evaluation_examples)
        else:
            predictions = self.function(evaluation_examples)
            pred_function = ExplainParams.EVAL_Y_PRED
            if len(predictions.shape) > 1:
                pred_function = ExplainParams.EVAL_Y_PRED_PROBA
            ys_dict[pred_function] = predictions
        return ys_dict


class BlackBoxExplainer(BaseExplainer, BlackBoxMixin):
    """The base class for black box models or functions.

    :param model: The model to explain or function if is_function is True.
    :type model: model that implements sklearn.predict or sklearn.predict_proba or function that accepts a 2d ndarray
    :param is_function: Default is false. Set to True if passing sklearn.predict or sklearn.predict_proba
        function instead of model.
    :type is_function: bool
    """

    def __init__(self, model, is_function=False, model_task=ModelTask.Unknown, **kwargs):
        """Initialize the BlackBoxExplainer.

        :param model: The model to explain or function if is_function is True.
        :type model: model that implements sklearn.predict or sklearn.predict_proba or function that accepts a 2d
            ndarray
        :param is_function: Default is False. Set to True if passing sklearn.predict or sklearn.predict_proba
            function instead of model.
        :type is_function: bool
        """
        super(BlackBoxExplainer, self).__init__(model, is_function=is_function,
                                                model_task=model_task, **kwargs)
        self._logger.debug('Initializing BlackBoxExplainer')


def init_blackbox_decorator(init_func):
    """Decorate a constructor to wrap initialization examples in a DatasetWrapper.

    Provided for convenience for tabular data explainers.

    :param init_func: Initialization constructor where the second argument is a dataset.
    :type init_func: Initialization constructor.
    """
    init_func = init_aggregator_decorator(init_func)

    @wraps(init_func)
    def init_wrapper(self, model, *args, **kwargs):
        self.explainer = None
        self.current_index_list = [0]
        self.original_data_ref = [None]
        return init_func(self, model, *args, **kwargs)

    return init_wrapper


def add_prepare_function_and_summary_method(cls):
    """Decorate blackbox explainer to allow aggregating local explanations to global.

    Adds two protected methods _function_subset_wrapper and _prepare_function_and_summary to
    the blackbox explainer.  The former creates a wrapper around the prediction function for
    explaining subsets of features in the evaluation samples dataset.  The latter calls the
    former to create a wrapper and also computes the summary background dataset for the explainer.
    """
    def _function_subset_wrapper(self, original_data_ref, explain_subset, f, current_index_list):
        """Create a wrapper around the prediction function.

        See more details on wrapper.

        :return: The wrapper around the prediction function.
        """
        def wrapper(data):
            """Private wrapper around the prediction function.

            Adds back in the removed columns when using the explain_subset parameter.
            We tile the original evaluation row by the number of samples generated
            and replace the subset of columns the user specified with the result from shap,
            which is the input data passed to the wrapper.

            :return: The prediction function wrapped by a helper method.
            """
            # If list is empty, just return the original data, as this is the background case
            original_data = original_data_ref[0]
            idx = current_index_list[0]
            tiles = int(data.shape[0])
            evaluation_row = original_data[idx]
            if issparse(evaluation_row):
                if not isspmatrix_csr(evaluation_row):
                    evaluation_row = evaluation_row.tocsr()
                nnz = evaluation_row.nnz
                rows, cols = evaluation_row.shape
                rows *= tiles
                shape = rows, cols
                if nnz == 0:
                    examples = csr_matrix(shape, dtype=evaluation_row.dtype).tolil()
                else:
                    new_indptr = np.arange(0, rows * nnz + 1, nnz)
                    new_data = np.tile(evaluation_row.data, rows)
                    new_indices = np.tile(evaluation_row.indices, rows)
                    examples = csr_matrix((new_data, new_indices, new_indptr),
                                          shape=shape).tolil()
            else:
                examples = np.tile(original_data[idx], tiles).reshape((data.shape[0], original_data.shape[1]))
            examples[:, explain_subset] = data
            return f(examples)
        return wrapper

    def _prepare_function_and_summary(self, function, original_data_ref,
                                      current_index_list, explain_subset=None, **kwargs):
        if explain_subset:
            # Note: need to take subset before compute summary
            self.initialization_examples.take_subset(explain_subset)
        self.initialization_examples.compute_summary(**kwargs)
        # Add wrapper on top of function to convert to original dataset type
        wrapper = _Wrapper(self.initialization_examples, function)
        wrapped_function = wrapper.wrapped_function

        # Score the model on the given dataset
        if explain_subset:
            if original_data_ref[0] is None:
                # This is only used for construction; not used during general computation
                original_data_ref[0] = self.initialization_examples.original_dataset
            wrapped_function = self._function_subset_wrapper(original_data_ref, explain_subset,
                                                             wrapped_function, current_index_list)
        summary = self.initialization_examples.dataset
        return wrapped_function, summary

    setattr(cls, '_function_subset_wrapper', _function_subset_wrapper)
    setattr(cls, '_prepare_function_and_summary', _prepare_function_and_summary)
    return cls
