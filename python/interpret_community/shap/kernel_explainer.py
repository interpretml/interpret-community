# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Defines the KernelExplainer for computing explanations on black box models or functions."""

import numpy as np

from ..common.blackbox_explainer import BlackBoxExplainer, add_prepare_function_and_summary_method, \
    init_blackbox_decorator
from ..common.aggregate import add_explain_global_method
from ..common.explanation_utils import _convert_to_list, _append_shap_values_instance, \
    _convert_single_instance_to_multi
from ..common.model_wrapper import _wrap_model
from ..common.constants import Defaults, Attributes, ExplainParams, ExplainType, ModelTask, \
    Extension
from ..explanation.explanation import _create_local_explanation
from ..dataset.dataset_wrapper import DatasetWrapper
from ..dataset.decorator import tabular_decorator, init_tabular_decorator
from ..explanation.explanation import _create_raw_feats_local_explanation, \
    _get_raw_explainer_create_explanation_kwargs
from .kwargs_utils import _get_explain_global_kwargs
from .._internal.raw_explain.raw_explain_utils import get_datamapper_and_transformed_data, \
    transform_with_datamapper

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', 'Starting from version 2.2.1', UserWarning)
    import shap


@add_prepare_function_and_summary_method
@add_explain_global_method
class KernelExplainer(BlackBoxExplainer):
    available_explanations = [Extension.GLOBAL, Extension.LOCAL]
    explainer_type = Extension.BLACKBOX

    """The Kernel Explainer for explaining black box models or functions.

    :param model: The model to explain or function if is_function is True.
    :type model: model that implements sklearn.predict or sklearn.predict_proba or function that accepts a 2d ndarray
    :param initialization_examples: A matrix of feature vector examples (# examples x # features) for
        initializing the explainer.
    :type initialization_examples: numpy.array or pandas.DataFrame or iml.datatypes.DenseData or
        scipy.sparse.csr_matrix
    :param is_function: Default is False. Set to True if passing function instead of a model.
    :type is_function: bool
    :param explain_subset: List of feature indices. If specified, only selects a subset of the
        features in the evaluation dataset for explanation, which will speed up the explanation
        process when number of features is large and the user already knows the set of interested
        features. The subset can be the top-k features from the model summary.
    :type explain_subset: list[int]
    :param nsamples: Default to 'auto'. Number of times to re-evaluate the model when
        explaining each prediction. More samples lead to lower variance estimates of the
        feature importance values, but incur more computation cost. When 'auto' is provided,
        the number of samples is computed according to a heuristic rule.
    :type nsamples: 'auto' or int
    :param features: A list of feature names.
    :type features: list[str]
    :param classes: Class names as a list of strings. The order of the class names should match
        that of the model output. Only required if explaining classifier.
    :type classes: list[str]
    :param nclusters: Number of means to use for approximation. A dataset is summarized with nclusters mean
        samples weighted by the number of data points they each represent. When the number of initialization
        examples is larger than (10 x nclusters), those examples will be summarized with k-means where
        k = nclusters.
    :type nclusters: int
    :param show_progress: Default to 'True'. Determines whether to display the explanation status bar
        when using shap_values from the KernelExplainer.
    :type show_progress: bool
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
    :param model_task: Optional parameter to specify whether the model is a classification or regression model.
        In most cases, the type of the model can be inferred based on the shape of the output, where a classifier
        has a predict_proba method and outputs a 2 dimensional array, while a regressor has a predict method and
        outputs a 1 dimensional array.
    :type model_task: str
    """

    @init_tabular_decorator
    @init_blackbox_decorator
    def __init__(self, model, initialization_examples, is_function=False, explain_subset=None,
                 nsamples=Defaults.AUTO, features=None, classes=None, nclusters=10,
                 show_progress=True, transformations=None, allow_all_transformations=False,
                 model_task=ModelTask.Unknown, **kwargs):
        """Initialize the KernelExplainer.

        :param model: The model to explain or function if is_function is True.
        :type model: model that implements sklearn.predict or sklearn.predict_proba or function that accepts a 2d
            ndarray
        :param initialization_examples: A matrix of feature vector examples (# examples x # features) for
            initializing the explainer.
        :type initialization_examples: numpy.array or pandas.DataFrame or iml.datatypes.DenseData or
            scipy.sparse.csr_matrix
        :param is_function: Default is False. Set to True if passing function instead of a model.
        :type is_function: bool
        :param explain_subset: List of feature indices. If specified, only selects a subset of the
            features in the evaluation dataset for explanation, which will speed up the explanation
            process when number of features is large and the user already knows the set of interested
            features. The subset can be the top-k features from the model summary.
        :type explain_subset: list[int]
        :param nsamples: Default to 'auto'. Number of times to re-evaluate the model when
            explaining each prediction. More samples lead to lower variance estimates of the
            feature importance values, but incur more computation cost. When 'auto' is provided,
            the number of samples is computed according to a heuristic rule.
        :type nsamples: 'auto' or int
        :param features: A list of feature names.
        :type features: list[str]
        :param classes: Class names as a list of strings. The order of the class names should match
            that of the model output.  Only required if explaining classifier.
        :type classes: list[str]
        :param nclusters: Number of means to use for approximation. A dataset is summarized with nclusters mean
            samples weighted by the number of data points they each represent. When the number of initialization
            examples is larger than (10 x nclusters), those examples will be summarized with k-means where
            k = nclusters.
        :type nclusters: int
        :param show_progress: Default to 'True'. Determines whether to display the explanation status bar
            when using shap_values from the KernelExplainer.
        :type show_progress: bool
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
        :param model_task: Optional parameter to specify whether the model is a classification or regression model.
            In most cases, the type of the model can be inferred based on the shape of the output, where a classifier
            has a predict_proba method and outputs a 2 dimensional array, while a regressor has a predict method and
            outputs a 1 dimensional array.
        :type model_task: str
        """
        self._datamapper = None
        if transformations is not None:
            self._datamapper, initialization_examples = get_datamapper_and_transformed_data(
                examples=initialization_examples, transformations=transformations,
                allow_all_transformations=allow_all_transformations)
        # string-index the initialization examples
        self._column_indexer = initialization_examples.string_index()
        wrapped_model, eval_ml_domain = _wrap_model(model, initialization_examples, model_task, is_function)
        super(KernelExplainer, self).__init__(wrapped_model, is_function=is_function,
                                              model_task=eval_ml_domain, **kwargs)
        self._logger.debug('Initializing KernelExplainer')
        self._method = 'shap.kernel'
        self.initialization_examples = initialization_examples
        self.features = features
        self.classes = classes
        self.nclusters = nclusters
        self.explain_subset = explain_subset
        self.show_progress = show_progress
        self.nsamples = nsamples
        self.transformations = transformations
        self._allow_all_transformations = allow_all_transformations
        self._reset_evaluation_background(self.function, **kwargs)

    def _reset_evaluation_background(self, function, **kwargs):
        """Modify the explainer to use the new evaluation example for background data.

        Note: when constructing an explainer, an evaluation example is not available and hence the initialization
        data is used.

        :param function: Function.
        :type function: Function that accepts a 2d ndarray
        """
        function, summary = self._prepare_function_and_summary(function, self.original_data_ref,
                                                               self.current_index_list,
                                                               explain_subset=self.explain_subset, **kwargs)
        self.explainer = shap.KernelExplainer(function, summary)

    def _reset_wrapper(self):
        self.explainer = None
        self.current_index_list = [0]
        self.original_data_ref = [None]
        self.initialization_examples = DatasetWrapper(self.initialization_examples.original_dataset)

    @tabular_decorator
    def explain_global(self, evaluation_examples, sampling_policy=None,
                       include_local=True, batch_size=Defaults.DEFAULT_BATCH_SIZE):
        """Explain the model globally by aggregating local explanations to global.

        :param evaluation_examples: A matrix of feature vector examples (# examples x # features) on which
            to explain the model's output.
        :type evaluation_examples: numpy.array or pandas.DataFrame or scipy.sparse.csr_matrix
        :param sampling_policy: Optional policy for sampling the evaluation examples.  See documentation on
            SamplingPolicy for more information.
        :type sampling_policy: SamplingPolicy
        :param include_local: Include the local explanations in the returned global explanation.
            If include_local is False, will stream the local explanations to aggregate to global.
        :type include_local: bool
        :param batch_size: If include_local is False, specifies the batch size for aggregating
            local explanations to global.
        :type batch_size: int
        :return: A model explanation object. It is guaranteed to be a GlobalExplanation which also has the properties
            of LocalExplanation and ExpectedValuesMixin. If the model is a classifier, it will have the properties of
            PerClassMixin.
        :rtype: DynamicGlobalExplanation
        """
        kwargs = _get_explain_global_kwargs(sampling_policy, ExplainType.SHAP_KERNEL, include_local, batch_size)
        kwargs[ExplainParams.INIT_DATA] = self.initialization_examples
        original_evaluation_examples = evaluation_examples.typed_dataset
        kwargs[ExplainParams.EVAL_DATA] = original_evaluation_examples
        ys_dict = self._get_ys_dict(original_evaluation_examples,
                                    transformations=self.transformations,
                                    allow_all_transformations=self._allow_all_transformations)
        kwargs.update(ys_dict)
        kwargs[ExplainParams.NUM_FEATURES] = evaluation_examples.num_features

        return self._explain_global(evaluation_examples, **kwargs)

    def _get_explain_local_kwargs(self, evaluation_examples):
        """Get the kwargs for explain_local to create a local explanation.

        :param evaluation_examples: A matrix of feature vector examples (# examples x # features) on which
            to explain the model's output.
        :type evaluation_examples: numpy.array or pandas.DataFrame or scipy.sparse.csr_matrix
        :return: Args for explain_local.
        :rtype: dict
        """
        original_evaluation_examples = evaluation_examples.typed_dataset
        if self._datamapper is not None:
            evaluation_examples = transform_with_datamapper(evaluation_examples, self._datamapper)

        if self._column_indexer:
            evaluation_examples.apply_indexer(self._column_indexer)
        # Compute subset info prior
        if self.explain_subset:
            evaluation_examples.take_subset(self.explain_subset)

        # sample the evaluation examples
        if self.sampling_policy is not None and self.sampling_policy.allow_eval_sampling:
            sampling_method = self.sampling_policy.sampling_method
            max_dim_clustering = self.sampling_policy.max_dim_clustering
            evaluation_examples.sample(max_dim_clustering, sampling_method=sampling_method)
        kwargs = {ExplainParams.METHOD: ExplainType.SHAP_KERNEL}
        if self.classes is not None:
            kwargs[ExplainParams.CLASSES] = self.classes
        kwargs[ExplainParams.FEATURES] = evaluation_examples.get_features(features=self.features,
                                                                          explain_subset=self.explain_subset)
        original_evaluation = evaluation_examples.original_dataset
        kwargs[ExplainParams.NUM_FEATURES] = evaluation_examples.num_features
        evaluation_examples = evaluation_examples.dataset

        self._logger.debug('Running KernelExplainer')

        if self.explain_subset:
            # Need to reset state before and after explaining a subset of data with wrapper function
            self._reset_wrapper()
            self.original_data_ref[0] = original_evaluation
            self.current_index_list.append(0)
            output_shap_values = None
            for ex_idx, example in enumerate(evaluation_examples):
                self.current_index_list[0] = ex_idx
                # Note: when subsetting with KernelExplainer, for correct results we need to
                # set the background to be the evaluation data for columns not specified in subset
                self._reset_evaluation_background(self.function, nclusters=self.nclusters)
                tmp_shap_values = self.explainer.shap_values(example, silent=not self.show_progress,
                                                             nsamples=self.nsamples)
                if output_shap_values is None:
                    output_shap_values = _convert_single_instance_to_multi(tmp_shap_values)
                else:
                    output_shap_values = _append_shap_values_instance(output_shap_values, tmp_shap_values)
            # Need to reset state before and after explaining a subset of data with wrapper function
            self._reset_wrapper()
            shap_values = output_shap_values
        else:
            shap_values = self.explainer.shap_values(evaluation_examples, silent=not self.show_progress,
                                                     nsamples=self.nsamples)

        classification = isinstance(shap_values, list)
        expected_values = None
        if hasattr(self.explainer, Attributes.EXPECTED_VALUE):
            expected_values = np.array(self.explainer.expected_value)
        local_importance_values = _convert_to_list(shap_values)
        if classification:
            kwargs[ExplainParams.MODEL_TASK] = ExplainType.CLASSIFICATION
        else:
            kwargs[ExplainParams.MODEL_TASK] = ExplainType.REGRESSION
        if self.model is not None:
            kwargs[ExplainParams.MODEL_TYPE] = str(type(self.model))
        else:
            kwargs[ExplainParams.MODEL_TYPE] = ExplainType.FUNCTION
        kwargs[ExplainParams.LOCAL_IMPORTANCE_VALUES] = np.array(local_importance_values)
        kwargs[ExplainParams.EXPECTED_VALUES] = expected_values
        kwargs[ExplainParams.CLASSIFICATION] = classification
        kwargs[ExplainParams.INIT_DATA] = self.initialization_examples
        kwargs[ExplainParams.EVAL_DATA] = original_evaluation_examples
        ys_dict = self._get_ys_dict(original_evaluation_examples,
                                    transformations=self.transformations,
                                    allow_all_transformations=self._allow_all_transformations)
        kwargs.update(ys_dict)
        return kwargs

    @tabular_decorator
    def explain_local(self, evaluation_examples):
        """Explain the function locally by using SHAP's KernelExplainer.

        :param evaluation_examples: A matrix of feature vector examples (# examples x # features) on which
            to explain the model's output.
        :type evaluation_examples: DatasetWrapper
        :return: A model explanation object. It is guaranteed to be a LocalExplanation which also has the properties
            of ExpectedValuesMixin. If the model is a classifier, it will have the properties of the ClassesMixin.
        :rtype: DynamicLocalExplanation
        """
        kwargs = self._get_explain_local_kwargs(evaluation_examples)
        explanation = _create_local_explanation(**kwargs)

        if self._datamapper is None:
            return explanation
        else:
            # if transformations have been passed, then return raw features explanation
            raw_kwargs = _get_raw_explainer_create_explanation_kwargs(kwargs=kwargs)
            return _create_raw_feats_local_explanation(explanation, feature_maps=[self._datamapper.feature_map],
                                                       features=self.features, **raw_kwargs)
