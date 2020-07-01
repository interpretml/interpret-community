# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Defines the PFIExplainer for computing global explanations on black box models or functions.

The PFIExplainer uses permutation feature importance to compute a score for each column
given a model based on how the output metric varies as each column is randomly permuted.
Although very fast for computing global explanations, PFI does not support local explanations
and can be inaccurate when there are feature interactions.
"""

import numpy as np
from scipy.sparse import issparse, isspmatrix_csc, SparseEfficiencyWarning
from sklearn.metrics import mean_absolute_error, explained_variance_score, mean_squared_error, \
    mean_squared_log_error, median_absolute_error, r2_score, average_precision_score, f1_score, \
    fbeta_score, precision_score, recall_score
import logging

from ..common.base_explainer import GlobalExplainer
from ..common.blackbox_explainer import BlackBoxMixin
from ..common.model_wrapper import _wrap_model, WrappedPytorchModel
from .._internal.raw_explain.raw_explain_utils import get_datamapper_and_transformed_data, \
    transform_with_datamapper
from ..dataset.decorator import tabular_decorator
from ..explanation.explanation import _create_raw_feats_global_explanation, \
    _get_raw_explainer_create_explanation_kwargs, _create_global_explanation
from ..common.constants import ExplainParams, ExplainType, ModelTask, Extension
from ..common.explanation_utils import _order_imp
from .metric_constants import MetricConstants, error_metrics
from ..common.progress import get_tqdm

# Although we get a sparse efficiency warning when using csr matrix format for setting the
# values, if we use lil scikit-learn converts the matrix to csr which has much worse performance
import warnings
from functools import wraps


module_logger = logging.getLogger(__name__)
module_logger.setLevel(logging.INFO)


try:
    import torch.nn as nn
except ImportError:
    module_logger.debug('Could not import torch, required if using a pytorch model')

TRUE_LABELS = 'true_labels'


def labels_decorator(explain_func):
    """Decorate PFI explainer to throw better error message if true_labels not passed.

    :param explain_func: PFI explanation function.
    :type explain_func: explanation function
    """
    @wraps(explain_func)
    def explain_func_wrapper(self, evaluation_examples, *args, **kwargs):
        # NOTE: true_labels can either be in args or kwargs
        if not args and TRUE_LABELS not in kwargs:
            raise TypeError("PFI explainer requires true_labels parameter to be passed in for explain_global")
        return explain_func(self, evaluation_examples, *args, **kwargs)
    return explain_func_wrapper


class PFIExplainer(GlobalExplainer, BlackBoxMixin):
    available_explanations = [Extension.GLOBAL]
    explainer_type = Extension.BLACKBOX

    """Defines the Permutation Feature Importance Explainer for explaining black box models or functions.

    :param model: The black box model or function (if is_function is True) to be explained. Also known
        as the teacher model.
    :type model: model that implements sklearn.predict or sklearn.predict_proba or function that accepts a 2d ndarray
    :param is_function: Default is False. Set to True if passing function instead of model.
    :type is_function: bool
    :param metric: The metric name or function to evaluate the permutation.
        Note that if a metric function is provided, a higher value must be better.
        Otherwise, take the negative of the function or set is_error_metric to True.
        By default, if no metric is provided, F1 Score is used for binary classification,
        F1 Score with micro average is used for multiclass classification and mean
        absolute error is used for regression.
    :type metric: str or function that accepts two arrays, y_true and y_pred.
    :param metric_args: Optional arguments for metric function.
    :type metric_args: dict
    :param is_error_metric: If custom metric function is provided, set to True if a higher
        value of the metric is better.
    :type is_error_metric: bool
    :param explain_subset: List of feature indexes. If specified, only selects a subset of the
        features in the evaluation dataset for explanation. For permutation feature importance,
        we can shuffle, score and evaluate on the specified indexes when this parameter is set.
        This argument is not supported when transformations are set.
    :type explain_subset: list[int]
    :param features: A list of feature names.
    :type features: list[str]
    :param classes: Class names as a list of strings. The order of the class names should match
        that of the model output.  Only required if explaining classifier.
    :type classes: list[str]
    :param transformations: sklearn.compose.ColumnTransformer or a list of tuples describing the column name and
        transformer. When transformations are provided, explanations are of the features before the transformation.
        The format for a list of transformations is same as the one here:
        https://github.com/scikit-learn-contrib/sklearn-pandas.

        If you are using a transformation that is not in the list of sklearn.preprocessing transformations that
        are supported by the `interpret-community <https://github.com/interpretml/interpret-community>`_
        package, then this parameter cannot take a list of more than one column as input for the transformation.
        You can use the following sklearn.preprocessing  transformations with a list of columns since these are
        already one to many or one to one: Binarizer, KBinsDiscretizer, KernelCenterer, LabelEncoder, MaxAbsScaler,
        MinMaxScaler, Normalizer, OneHotEncoder, OrdinalEncoder, PowerTransformer, QuantileTransformer,
        RobustScaler, StandardScaler.

        Examples for transformations that work::

            [
                (["col1", "col2"], sklearn_one_hot_encoder),
                (["col3"], None) #col3 passes as is
            ]
            [
                (["col1"], my_own_transformer),
                (["col2"], my_own_transformer),
            ]

        An example of a transformation that would raise an error since it cannot be interpreted as one to many::

            [
                (["col1", "col2"], my_own_transformer)
            ]

        The last example would not work since the interpret-community package can't determine whether
        my_own_transformer gives a many to many or one to many mapping when taking a sequence of columns.
    :type transformations: sklearn.compose.ColumnTransformer or list[tuple]
    :param allow_all_transformations: Allow many to many and many to one transformations.
    :type allow_all_transformations: bool
    :param seed: Random number seed for shuffling.
    :type seed: int
    :param for_classifier_use_predict_proba: If specifying a model instead of a function, and the model
        is a classifier, set to True instead of the default False to use predict_proba instead of
        predict when calculating the metric.
    :type for_classifier_use_predict_proba: bool
    :param show_progress: Default to 'True'. Determines whether to display the explanation status bar
        when using PFIExplainer.
    :type show_progress: bool
    :param model_task: Optional parameter to specify whether the model is a classification or regression model.
        In most cases, the type of the model can be inferred based on the shape of the output, where a classifier
        has a predict_proba method and outputs a 2 dimensional array, while a regressor has a predict method and
        outputs a 1 dimensional array.
    :type model_task: str
    """

    def __init__(self, model, is_function=False, metric=None, metric_args=None, is_error_metric=False,
                 explain_subset=None, features=None, classes=None, transformations=None,
                 allow_all_transformations=False, seed=0, for_classifier_use_predict_proba=False,
                 show_progress=True, model_task=ModelTask.Unknown, **kwargs):
        """Initialize the PFIExplainer.

        :param model: The black box model or function (if is_function is True) to be explained. Also known
            as the teacher model.
        :type model: model that implements sklearn.predict or sklearn.predict_proba or function that accepts a 2d
            ndarray
        :param is_function: Default is False. Set to True if passing function instead of model.
        :type is_function: bool
        :param metric: The metric name or function to evaluate the permutation.
            Note that if a metric function is provided, a higher value must be better.
            Otherwise, take the negative of the function or set is_error_metric to True.
            By default, if no metric is provided, F1 Score is used for binary classification,
            F1 Score with micro average is used for multiclass classification and mean
            absolute error is used for regression.
        :type metric: str or function that accepts two arrays, y_true and y_pred.
        :param metric_args: Optional arguments for metric function.
        :type metric_args: dict
        :param is_error_metric: If custom metric function is provided, set to True if a higher
            value of the metric is better.
        :type is_error_metric: bool
        :param explain_subset: List of feature indexes. If specified, only selects a subset of the
            features in the evaluation dataset for explanation. For permutation feature importance,
            we can shuffle, score and evaluate on the specified indexes when this parameter is set.
            This argument is not supported when transformations are set.
        :type explain_subset: list[int]
        :param features: A list of feature names.
        :type features: list[str]
        :param classes: Class names as a list of strings. The order of the class names should match
            that of the model output.  Only required if explaining classifier.
        :type classes: list[str]
        :param transformations: sklearn.compose.ColumnTransformer or a list of tuples describing the column name and
            transformer. When transformations are provided, explanations are of the features before the transformation.
            The format for a list of transformations is same as the one here:
            https://github.com/scikit-learn-contrib/sklearn-pandas.

            If you are using a transformation that is not in the list of sklearn.preprocessing transformations that
            are supported by the `interpret-community <https://github.com/interpretml/interpret-community>`_
            package, then this parameter cannot take a list of more than one column as input for the transformation.
            You can use the following sklearn.preprocessing  transformations with a list of columns since these are
            already one to many or one to one: Binarizer, KBinsDiscretizer, KernelCenterer, LabelEncoder, MaxAbsScaler,
            MinMaxScaler, Normalizer, OneHotEncoder, OrdinalEncoder, PowerTransformer, QuantileTransformer,
            RobustScaler, StandardScaler.

            Examples for transformations that work::

                [
                    (["col1", "col2"], sklearn_one_hot_encoder),
                    (["col3"], None) #col3 passes as is
                ]
                [
                    (["col1"], my_own_transformer),
                    (["col2"], my_own_transformer),
                ]

            An example of a transformation that would raise an error since it cannot be interpreted as one to many::

                [
                    (["col1", "col2"], my_own_transformer)
                ]

            The last example would not work since the interpret-community package can't determine whether
            my_own_transformer gives a many to many or one to many mapping when taking a sequence of columns.
        :type transformations: sklearn.compose.ColumnTransformer or list[tuple]
        :param allow_all_transformations: Allow many to many and many to one transformations
        :type allow_all_transformations: bool
        :param seed: Random number seed for shuffling.
        :type seed: int
        :param for_classifier_use_predict_proba: If specifying a model instead of a function, and the model
            is a classifier, set to True instead of the default False to use predict_proba instead of
            predict when calculating the metric.
        :type for_classifier_use_predict_proba: bool
        :param show_progress: Default to 'True'.  Determines whether to display the explanation status bar
            when using PFIExplainer.
        :type show_progress: bool
        :param model_task: Optional parameter to specify whether the model is a classification or regression model.
            In most cases, the type of the model can be inferred based on the shape of the output, where a classifier
            has a predict_proba method and outputs a 2 dimensional array, while a regressor has a predict method and
            outputs a 1 dimensional array.
        :type model_task: str
        """
        log_pytorch_missing = False
        try:
            if isinstance(model, nn.Module):
                # Wrap the model in an extra layer that converts the numpy array
                # to pytorch Variable and adds predict and predict_proba functions
                model = WrappedPytorchModel(model)
        except (NameError, AttributeError):
            log_pytorch_missing = True
        super(PFIExplainer, self).__init__(model, is_function=is_function, **kwargs)
        # Note: we can't log debug until after init has been called to create the logger
        if log_pytorch_missing:
            self._logger.debug('Could not import torch, required if using a pytorch model')
        self._logger.debug('Initializing PFIExplainer')

        if transformations is not None and explain_subset is not None:
            raise ValueError("explain_subset not supported with transformations")

        self._datamapper = None
        if transformations is not None:
            self._datamapper, _ = get_datamapper_and_transformed_data(
                transformations=transformations, allow_all_transformations=allow_all_transformations)

        self._method = 'pfi'
        self.features = features
        self.classes = classes
        self.transformations = transformations
        self.explain_subset = explain_subset
        self.show_progress = show_progress
        self.seed = seed
        self.metric = metric
        self.metric_args = metric_args
        self.for_classifier_use_predict_proba = for_classifier_use_predict_proba
        self.is_error_metric = is_error_metric
        self.model_task = model_task
        if self.metric_args is None:
            self.metric_args = {}
        # If no metric specified, pick a default based on whether this is for classification or regression
        if metric is None:
            if model_task != ModelTask.Regression and self.predict_proba_flag:
                self.metric = f1_score
                self.metric_args = {'average': 'micro'}
            else:
                self.metric = mean_absolute_error
        # If the metric is a string, substitute it with the corresponding evaluation function
        metric_to_func = {MetricConstants.MEAN_ABSOLUTE_ERROR: mean_absolute_error,
                          MetricConstants.EXPLAINED_VARIANCE_SCORE: explained_variance_score,
                          MetricConstants.MEAN_SQUARED_ERROR: mean_squared_error,
                          MetricConstants.MEAN_SQUARED_LOG_ERROR: mean_squared_log_error,
                          MetricConstants.MEDIAN_ABSOLUTE_ERROR: median_absolute_error,
                          MetricConstants.R2_SCORE: r2_score,
                          MetricConstants.AVERAGE_PRECISION_SCORE: average_precision_score,
                          MetricConstants.F1_SCORE: f1_score,
                          MetricConstants.FBETA_SCORE: fbeta_score,
                          MetricConstants.PRECISION_SCORE: precision_score,
                          MetricConstants.RECALL_SCORE: recall_score}
        if metric is str:
            try:
                self.metric = metric_to_func[metric]
                self.is_error_metric = metric in error_metrics
            except: # noqa
                raise Exception('Metric \'{}\' not in supported list of metrics, please pass function instead'
                                .format(metric))
        if self.classes is not None and not self.predict_proba_flag:
            if self.model is None:
                error = 'Classes is specified but function was predict, not predict_proba.'
            else:
                error = 'Classes is specified but model does not define predict_proba, only predict.'
            raise ValueError(error)

    def _add_metric(self, predict_function, shuffled_dataset, true_labels,
                    base_metric, global_importance_values, idx):
        """Compute and add the metric to the global importance values array.

        :param predict_function: The prediction function.
        :type predict_function: function
        :param shuffled_dataset: The shuffled dataset to predict on.
        :type shuffled_dataset: scipy.csr or numpy.ndarray
        :param true_labels: The true labels.
        :type true_labels: numpy.ndarray
        :param base_metric: Base metric for unshuffled dataset.
        :type base_metric: float
        :param global_importance_values: Pre-allocated array of global importance values.
        :type global_importance_values: numpy.ndarray
        """
        shuffled_prediction = predict_function(shuffled_dataset)
        if issparse(shuffled_prediction):
            shuffled_prediction = shuffled_prediction.toarray()
        metric = self.metric(true_labels, shuffled_prediction, **self.metric_args)
        importance_score = base_metric - metric
        # Flip the sign of the metric if this is an error metric
        if self.is_error_metric:
            importance_score *= -1
        global_importance_values[idx] = importance_score

    def _compute_sparse_metric(self, dataset, col_idx, subset_idx, random_indexes, shuffled_dataset,
                               predict_function, true_labels, base_metric, global_importance_values):
        """Shuffle a sparse dataset column and compute the feature importance metric.

        :param dataset: Dataset used as a reference point for getting column indexes per row.
        :type dataset: scipy.csc
        :param col_idx: The column index.
        :type col_idx: int
        :param subset_idx: The subset index.
        :type subset_idx: int
        :param random_indexes: Generated random indexes.
        :type random_indexes: numpy.ndarray
        :param shuffled_dataset: The dataset to shuffle.
        :type shuffled_dataset: scipy.csr
        :param predict_function: The prediction function.
        :type predict_function: function
        :param true_labels: The true labels.
        :type true_labels: numpy.ndarray
        :param base_metric: Base metric for unshuffled dataset.
        :type base_metric: float
        :param global_importance_values: Pre-allocated array of global importance values.
        :type global_importance_values: numpy.ndarray
        """
        # Get non zero column indexes
        indptr = dataset.indptr
        indices = dataset.indices
        col_nz_indices = indices[indptr[col_idx]:indptr[col_idx + 1]]
        # Sparse optimization: If all zeros, skip the column!  Shuffling won't make a difference to metric.
        if col_nz_indices.size == 0:
            return
        data = dataset.data
        # Replace non-zero indexes with shuffled indexes
        col_random_indexes = random_indexes[0:len(col_nz_indices)]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', SparseEfficiencyWarning)
            # Shuffle the sparse column indexes
            shuffled_dataset[col_random_indexes, col_idx] = shuffled_dataset[col_nz_indices, col_idx].T
            # Get set difference and zero-out indexes that had a value but now should be zero
            difference_nz_random = list(set(col_nz_indices).difference(set(col_random_indexes)))
            difference_random_nz = list(set(col_random_indexes).difference(set(col_nz_indices)))
            # Set values that should not be sparse explicitly to zeros
            shuffled_dataset[difference_nz_random, col_idx] = np.zeros((len(difference_nz_random)),
                                                                       dtype=data.dtype)
            if self.explain_subset:
                idx = subset_idx
            else:
                idx = col_idx
            self._add_metric(predict_function, shuffled_dataset, true_labels,
                             base_metric, global_importance_values, idx)
            # Restore column back to previous state by undoing shuffle
            shuffled_dataset[col_nz_indices, col_idx] = shuffled_dataset[col_random_indexes, col_idx].T
            shuffled_dataset[difference_random_nz, col_idx] = np.zeros((len(difference_random_nz)),
                                                                       dtype=data.dtype)

    def _compute_dense_metric(self, dataset, col_idx, subset_idx, random_indexes,
                              predict_function, true_labels, base_metric, global_importance_values):
        """Shuffle a dense dataset column and compute the feature importance metric.

        :param dataset: Dataset used as a reference point for getting column indexes per row.
        :type dataset: numpy.ndarray
        :param col_idx: The column index.
        :type col_idx: int
        :param subset_idx: The subset index.
        :type subset_idx: int
        :param random_indexes: Generated random indexes.
        :type random_indexes: numpy.ndarray
        :param predict_function: The prediction function.
        :type predict_function: function
        :param true_labels: The true labels.
        :type true_labels: numpy.ndarray
        :param base_metric: Base metric for unshuffled dataset.
        :type base_metric: float
        :param global_importance_values: Pre-allocated array of global importance values.
        :type global_importance_values: numpy.ndarray
        """
        # Create a copy of the original dataset
        shuffled_dataset = np.array(dataset, copy=True)
        # Shuffle one of the columns in place
        shuffled_dataset[:, col_idx] = shuffled_dataset[random_indexes, col_idx]
        if self.explain_subset:
            idx = subset_idx
        else:
            idx = col_idx
        self._add_metric(predict_function, shuffled_dataset, true_labels,
                         base_metric, global_importance_values, idx)

    def _get_explain_global_kwargs(self, evaluation_examples, true_labels):
        """Get the kwargs for explain_global to create a global explanation.

        :param evaluation_examples: A matrix of feature vector examples (# examples x # features) on which to
            explain the model's output.
        :type evaluation_examples: numpy.array or pandas.DataFrame or scipy.sparse.csr_matrix
        :param true_labels: An array of true labels used for reference to compute the evaluation metric
            for base case and after each permutation.
        :type true_labels: numpy.array or pandas.DataFrame
        :return: Args for explain_global.
        :rtype: dict
        """
        classification = self.predict_proba_flag
        kwargs = {ExplainParams.METHOD: ExplainType.PFI}
        if classification:
            kwargs[ExplainParams.CLASSES] = self.classes
            kwargs[ExplainParams.MODEL_TASK] = ExplainType.CLASSIFICATION
        else:
            kwargs[ExplainParams.MODEL_TASK] = ExplainType.REGRESSION

        if self.model is not None:
            kwargs[ExplainParams.MODEL_TYPE] = str(type(self.model))
        else:
            kwargs[ExplainParams.MODEL_TYPE] = ExplainType.FUNCTION
        dataset = evaluation_examples.dataset
        typed_wrapper_func = evaluation_examples.typed_wrapper_func

        kwargs[ExplainParams.NUM_FEATURES] = evaluation_examples.num_features

        def generate_predict_function():
            if self.model is not None:
                wrapped_model, _ = _wrap_model(self.model, evaluation_examples, self.model_task, False)
                if self.for_classifier_use_predict_proba:
                    def model_predict_proba_func(dataset):
                        return wrapped_model.predict_proba(typed_wrapper_func(dataset))
                    return model_predict_proba_func
                else:
                    def model_predict_func(dataset):
                        return wrapped_model.predict(typed_wrapper_func(dataset))
                    return model_predict_func
            else:
                wrapped_function, _ = _wrap_model(self.function, evaluation_examples, self.model_task, True)

                def user_defined_or_default_predict_func(dataset):
                    return wrapped_function(typed_wrapper_func(dataset))
                return user_defined_or_default_predict_func

        predict_function = generate_predict_function()
        # Score the model on the given dataset
        prediction = predict_function(dataset)
        # The scikit-learn metrics can't handle sparse arrays
        if issparse(true_labels):
            true_labels = true_labels.toarray()
        if issparse(prediction):
            prediction = prediction.toarray()
        # Evaluate the model with given metric on the dataset
        base_metric = self.metric(true_labels, prediction, **self.metric_args)
        # Ensure we get the same results when shuffling
        if self.seed is not None:
            np.random.seed(self.seed)
        if self.explain_subset:
            # When specifying a subset, only shuffle and score on the columns specified
            column_indexes = self.explain_subset
            global_importance_values = np.zeros(len(self.explain_subset))
        else:
            column_indexes = range(dataset.shape[1])
            global_importance_values = np.zeros(dataset.shape[1])
        tqdm = get_tqdm(self._logger, self.show_progress)
        if issparse(dataset):
            # Create a dataset for shuffling
            # Although lil matrix is better for changing sparsity structure, scikit-learn
            # converts matrixes back to csr for prediction which is much more expensive
            shuffled_dataset = dataset.tocsr(copy=True)
            # Convert to csc format if not already for faster column index access
            if not isspmatrix_csc(dataset):
                dataset = dataset.tocsc()
            # Get max NNZ across all columns
            dataset_nnz = dataset.getnnz(axis=0)
            maxnnz = max(dataset_nnz)
            column_indexes = np.unique(np.intersect1d(dataset.nonzero()[1], column_indexes))
            # Choose random, shuffled n of k indexes
            random_indexes = np.random.choice(dataset.shape[0], maxnnz, replace=False)
            # Shuffle all sparse columns
            for subset_idx, col_idx in tqdm(enumerate(column_indexes)):
                self._compute_sparse_metric(dataset, col_idx, subset_idx, random_indexes, shuffled_dataset,
                                            predict_function, true_labels, base_metric, global_importance_values)
        else:
            num_rows = dataset.shape[0]
            random_indexes = np.random.choice(num_rows, num_rows, replace=False)
            for subset_idx, col_idx in tqdm(enumerate(column_indexes)):
                self._compute_dense_metric(dataset, col_idx, subset_idx, random_indexes, predict_function,
                                           true_labels, base_metric, global_importance_values)
        order = _order_imp(global_importance_values)
        kwargs[ExplainParams.EXPECTED_VALUES] = None
        kwargs[ExplainParams.CLASSIFICATION] = classification
        kwargs[ExplainParams.GLOBAL_IMPORTANCE_VALUES] = global_importance_values
        kwargs[ExplainParams.GLOBAL_IMPORTANCE_RANK] = order
        kwargs[ExplainParams.FEATURES] = evaluation_examples.get_features(features=self.features,
                                                                          explain_subset=self.explain_subset)
        return kwargs

    @labels_decorator
    @tabular_decorator
    def explain_global(self, evaluation_examples, true_labels):
        """Globally explains the blackbox model using permutation feature importance.

        Note this will not include per class feature importances or local feature importances.

        :param evaluation_examples: A matrix of feature vector examples (# examples x # features) on which to
            explain the model's output through permutation feature importance.
        :type evaluation_examples: numpy.array or pandas.DataFrame or scipy.sparse.csr_matrix
        :param true_labels: An array of true labels used for reference to compute the evaluation metric
            for base case and after each permutation.
        :type true_labels: numpy.array or pandas.DataFrame
        :return: A model explanation object. It is guaranteed to be a GlobalExplanation.
            If the model is a classifier (has predict_proba), it will have the properties of ClassesMixin.
        :rtype: DynamicGlobalExplanation
        """
        if self._datamapper is not None:
            evaluation_examples = transform_with_datamapper(evaluation_examples, self._datamapper)
        kwargs = self._get_explain_global_kwargs(evaluation_examples, true_labels)
        kwargs[ExplainParams.EVAL_DATA] = evaluation_examples

        explanation = _create_global_explanation(**kwargs)

        # if transformations have been passed, then return raw features explanation
        raw_kwargs = _get_raw_explainer_create_explanation_kwargs(kwargs=kwargs)
        return explanation if self._datamapper is None else _create_raw_feats_global_explanation(
            explanation, feature_maps=[self._datamapper.feature_map], features=self.features, **raw_kwargs)
