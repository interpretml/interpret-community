# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Defines the LIMEExplainer for computing explanations on black box models using LIME."""

import numpy as np
from interpret_community._internal.raw_explain.raw_explain_utils import (
    get_datamapper_and_transformed_data, transform_with_datamapper)
from interpret_community.common.aggregate import add_explain_global_method
from interpret_community.common.blackbox_explainer import (
    BlackBoxExplainer, add_prepare_function_and_summary_method,
    init_blackbox_decorator)
from interpret_community.common.constants import (Defaults, ExplainParams,
                                                  ExplainType,
                                                  ExplanationParams, Extension,
                                                  ModelTask)
from interpret_community.common.model_wrapper import _wrap_model
from interpret_community.common.progress import get_tqdm
from interpret_community.dataset.decorator import (init_tabular_decorator,
                                                   tabular_decorator)
from interpret_community.explanation.explanation import (
    _create_local_explanation, _create_raw_feats_local_explanation,
    _get_raw_explainer_create_explanation_kwargs)

# Soft dependency for LIME
try:
    lime_import_failed = False
    from lime.lime_tabular import LimeTabularExplainer
except ImportError:
    lime_import_failed = True

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', 'Starting from version 2.2.1', UserWarning)


@add_prepare_function_and_summary_method
@add_explain_global_method
class LIMEExplainer(BlackBoxExplainer):
    available_explanations = [Extension.GLOBAL, Extension.LOCAL]
    explainer_type = Extension.BLACKBOX

    """Defines the LIME Explainer for explaining black box models or functions.

    :param model: The model to explain or function if is_function is True.
        A model that implements sklearn.predict or sklearn.predict_proba or function that accepts a 2d
        ndarray.
    :type model: object
    :param initialization_examples: A matrix of feature vector examples (# examples x # features) for
        initializing the explainer.
    :type initialization_examples: numpy.array or pandas.DataFrame or scipy.sparse.csr_matrix
    :param is_function: Default set to false, set to True if passing function instead of model.
    :type is_function: bool
    :param explain_subset: List of feature indices. If specified, only selects a subset of the
        features in the evaluation dataset for explanation. The subset can be the top-k features
        from the model summary.
    :type explain_subset: list[int]
    :param nclusters: Number of means to use for approximation. A dataset is summarized with nclusters mean
        samples weighted by the number of data points they each represent. When the number of initialization
        examples is larger than (10 x nclusters), those examples will be summarized with k-means where
        k = nclusters.
    :type nclusters: int
    :param features: A list of feature names.
    :type features: list[str]
    :param classes: Class names as a list of strings. The order of the class names should match
        that of the model output.  Only required if explaining classifier.
    :type classes: list[str]
    :param verbose: If true, uses verbose logging in LIME.
    :type verbose: bool
    :param categorical_features: Categorical feature names or indexes.
        If names are passed, they will be converted into indexes first.
    :type categorical_features: Union[list[str], list[int]]
    :param show_progress: Default to 'True'.  Determines whether to display the explanation status bar
        when using LIMEExplainer.
    :type show_progress: bool
    :param transformations: sklearn.compose.ColumnTransformer or a list of tuples describing the column name and
    transformer. When transformations are provided, explanations are of the features before the transformation.
    The format for list of transformations is same as the one here:
    https://github.com/scikit-learn-contrib/sklearn-pandas.

    If the user is using a transformation that is not in the list of sklearn.preprocessing transformations that
    we support then we cannot take a list of more than one column as input for the transformation.
    A user can use the following sklearn.preprocessing  transformations with a list of columns since these are
    already one to many or one to one: Binarizer, KBinsDiscretizer, KernelCenterer, LabelEncoder, MaxAbsScaler,
    MinMaxScaler, Normalizer, OneHotEncoder, OrdinalEncoder, PowerTransformer, QuantileTransformer, RobustScaler,
    StandardScaler.

    Examples for transformations that work::

        [
            (["col1", "col2"], sklearn_one_hot_encoder),
            (["col3"], None) #col3 passes as is
        ]
        [
            (["col1"], my_own_transformer),
            (["col2"], my_own_transformer),
        ]

    Example of transformations that would raise an error since it cannot be interpreted as one to many::

        [
            (["col1", "col2"], my_own_transformer)
        ]

    This would not work since it is hard to make out whether my_own_transformer gives a many to many or one to
    many mapping when taking a sequence of columns.
    :type transformations: sklearn.compose.ColumnTransformer or list[tuple]
    :param allow_all_transformations: Allow many to many and many to one transformations
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
                 nclusters=10, features=None, classes=None, verbose=False,
                 categorical_features=[],  # noqa: B006
                 show_progress=True, transformations=None, allow_all_transformations=False,
                 model_task=ModelTask.Unknown, **kwargs):
        """Initialize the LIME Explainer.

        :param model: The model to explain or function if is_function is True.
            A model that implements sklearn.predict or sklearn.predict_proba or function that accepts a 2d
            ndarray.
        :type model: object
        :param initialization_examples: A matrix of feature vector examples (# examples x # features) for
            initializing the explainer.
        :type initialization_examples: numpy.array or pandas.DataFrame or scipy.sparse.csr_matrix
        :param is_function: Default set to false, set to True if passing function instead of model.
        :type is_function: bool
        :param explain_subset: List of feature indices. If specified, only selects a subset of the
            features in the evaluation dataset for explanation. The subset can be the top-k features
            from the model summary.
        :type explain_subset: list[int]
        :param nclusters: Number of means to use for approximation. A dataset is summarized with nclusters mean
            samples weighted by the number of data points they each represent. When the number of initialization
            examples is larger than (10 x nclusters), those examples will be summarized with k-means where
            k = nclusters.
        :type nclusters: int
        :param features: A list of feature names.
        :type features: list[str]
        :param classes: Class names as a list of strings. The order of the class names should match
            that of the model output.  Only required if explaining classifier.
        :type classes: list[str]
        :param verbose: If true, uses verbose logging in LIME.
        :type verbose: bool
        :param categorical_features: Categorical feature names or indexes.
            If names are passed, they will be converted into indexes first.
        :type categorical_features: Union[list[str], list[int]]
        :param show_progress: Default to 'True'.  Determines whether to display the explanation status bar
            when using LIMEExplainer.
        :type show_progress: bool
        :param transformations: sklearn.compose.ColumnTransformer or a list of tuples describing the column name and
        transformer. When transformations are provided, explanations are of the features before the transformation.
        The format for list of transformations is same as the one here:
        https://github.com/scikit-learn-contrib/sklearn-pandas.

        If the user is using a transformation that is not in the list of sklearn.preprocessing transformations that
        we support then we cannot take a list of more than one column as input for the transformation.
        A user can use the following sklearn.preprocessing  transformations with a list of columns since these are
        already one to many or one to one: Binarizer, KBinsDiscretizer, KernelCenterer, LabelEncoder, MaxAbsScaler,
        MinMaxScaler, Normalizer, OneHotEncoder, OrdinalEncoder, PowerTransformer, QuantileTransformer, RobustScaler,
        StandardScaler.

        Examples for transformations that work::

            [
                (["col1", "col2"], sklearn_one_hot_encoder),
                (["col3"], None) #col3 passes as is
            ]
            [
                (["col1"], my_own_transformer),
                (["col2"], my_own_transformer),
            ]

        Example of transformations that would raise an error since it cannot be interpreted as one to many::

            [
                (["col1", "col2"], my_own_transformer)
            ]

        This would not work since it is hard to make out whether my_own_transformer gives a many to many or one to
        many mapping when taking a sequence of columns.
        :type transformations: sklearn.compose.ColumnTransformer or list[tuple]
        :param allow_all_transformations: Allow many to many and many to one transformations
        :type allow_all_transformations: bool
        :param model_task: Optional parameter to specify whether the model is a classification or regression model.
            In most cases, the type of the model can be inferred based on the shape of the output, where a classifier
            has a predict_proba method and outputs a 2 dimensional array, while a regressor has a predict method and
            outputs a 1 dimensional array.
        :type model_task: str
        """
        if lime_import_failed:
            raise Exception("Could not import LIME, required for LIMEExplainer")
        self._datamapper = None
        if transformations is not None:
            self._datamapper, initialization_examples = get_datamapper_and_transformed_data(
                examples=initialization_examples, transformations=transformations,
                allow_all_transformations=allow_all_transformations)
        wrapped_model, eval_ml_domain = _wrap_model(model, initialization_examples, model_task, is_function)
        super(LIMEExplainer, self).__init__(wrapped_model, is_function=is_function,
                                            model_task=eval_ml_domain, **kwargs)
        self._logger.debug('Initializing LIMEExplainer')

        self._method = 'lime'
        self.initialization_examples = initialization_examples
        self.classification = False
        self.features = initialization_examples.get_features(features=features)
        self.classes = classes
        self.nclusters = nclusters
        self.explain_subset = explain_subset
        self.show_progress = show_progress
        self.transformations = transformations
        # If categorical_features is a list of string column names instead of indexes, make sure to convert to indexes
        if not all(isinstance(categorical_feature, int) for categorical_feature in categorical_features):
            categorical_features = initialization_examples.get_column_indexes(self.features, categorical_features)
        # Index the categorical string columns
        self._column_indexer = initialization_examples.string_index(columns=categorical_features)
        function, summary = self._prepare_function_and_summary(self.function, self.original_data_ref,
                                                               self.current_index_list, nclusters=nclusters,
                                                               explain_subset=explain_subset, **kwargs)
        if str(type(summary)).endswith(".DenseData'>"):
            summary = summary.data
        self._lime_feature_names = [str(i) for i in range(summary.shape[1])]
        result = function(summary[0].reshape((1, -1)))
        # If result is 2D array, this is classification scenario, otherwise regression
        if len(result.shape) == 2:
            self.classification = True
            mode = ExplainType.CLASSIFICATION
        elif len(result.shape) == 1:
            self.classification = False
            mode = ExplainType.REGRESSION
        else:
            raise Exception('Invalid function specified, does not conform to specifications on prediction')
        self.explainer = LimeTabularExplainer(summary, feature_names=self._lime_feature_names, class_names=classes,
                                              categorical_features=categorical_features, verbose=verbose,
                                              mode=mode, discretize_continuous=False)
        self.explainer.function = function
        if self.classes is None and self.classification:
            raise ValueError('LIME Explainer requires classes to be specified if using a classification model')
        if self.classes is not None and not self.classification:
            if self.model is None:
                error = 'Classes is specified but function was predict, not predict_proba.'
            else:
                error = 'Classes is specified but model does not define predict_proba, only predict.'
            raise ValueError(error)

    @tabular_decorator
    def explain_global(self, evaluation_examples, sampling_policy=None,
                       include_local=True, batch_size=Defaults.DEFAULT_BATCH_SIZE):
        """Explain the model globally by aggregating local explanations to global.

        :param evaluation_examples: A matrix of feature vector examples (# examples x # features) on which
            to explain the model's output.
        :type evaluation_examples: numpy.array or pandas.DataFrame or scipy.sparse.csr_matrix
        :param sampling_policy: Optional policy for sampling the evaluation examples.  See documentation on
            SamplingPolicy for more information.
        :type sampling_policy: interpret_community.common.policy.SamplingPolicy
        :param include_local: Include the local explanations in the returned global explanation.
            If include_local is False, will stream the local explanations to aggregate to global.
        :type include_local: bool
        :param batch_size: If include_local is False, specifies the batch size for aggregating
            local explanations to global.
        :type batch_size: int
        :return: A model explanation object containing the global explanation.
        :rtype: GlobalExplanation
        """
        kwargs = {ExplainParams.METHOD: ExplainType.LIME,
                  ExplainParams.SAMPLING_POLICY: sampling_policy,
                  ExplainParams.INCLUDE_LOCAL: include_local,
                  ExplainParams.BATCH_SIZE: batch_size}

        if self.classification:
            kwargs[ExplanationParams.CLASSES] = self.classes
            kwargs[ExplainType.MODEL_TASK] = ExplainType.CLASSIFICATION
        else:
            kwargs[ExplainType.MODEL_TASK] = ExplainType.REGRESSION

        kwargs[ExplainParams.EVAL_DATA] = evaluation_examples.typed_dataset

        return self._explain_global(evaluation_examples, **kwargs)

    @tabular_decorator
    def explain_local(self, evaluation_examples):
        """Explain the function locally by using LIME.

        :param evaluation_examples: A matrix of feature vector examples (# examples x # features) on which
            to explain the model's output.
        :type evaluation_examples: interpret_community.dataset.dataset_wrapper.DatasetWrapper
        :param features: A list of feature names.
        :type features: list[str]
        :param classes: Class names as a list of strings. The order of the class names should match
            that of the model output.  Only required if explaining classifier.
        :type classes: list[str]
        :return: A model explanation object containing the local explanation.
        :rtype: LocalExplanation
        """
        if self._datamapper is not None:
            evaluation_examples = transform_with_datamapper(evaluation_examples, self._datamapper)

        if self._column_indexer:
            evaluation_examples.apply_indexer(self._column_indexer)

        # Compute subset info prior
        if self.explain_subset:
            evaluation_examples.take_subset(self.explain_subset)

        # sample the evaluation examples
        # note: the sampled data is also used by KNN
        if self.sampling_policy is not None and self.sampling_policy.allow_eval_sampling:
            sampling_method = self.sampling_policy.sampling_method
            max_dim_clustering = self.sampling_policy.max_dim_clustering
            evaluation_examples.sample(max_dim_clustering, sampling_method=sampling_method)
        features = self.features
        if self.explain_subset:
            features = [features[i] for i in self.explain_subset]
        kwargs = {ExplainParams.METHOD: ExplainType.LIME}
        kwargs[ExplainParams.FEATURES] = features
        kwargs[ExplainParams.NUM_FEATURES] = evaluation_examples.num_features
        original_evaluation = evaluation_examples.original_dataset
        evaluation_examples = evaluation_examples.dataset
        if len(evaluation_examples.shape) == 1:
            evaluation_examples = evaluation_examples.reshape(1, -1)

        self._logger.debug('Running LIMEExplainer')
        if self.classification:
            kwargs[ExplanationParams.CLASSES] = self.classes
            kwargs[ExplainType.MODEL_TASK] = ExplainType.CLASSIFICATION
            num_classes = len(self.classes)
            labels = list(range(num_classes))
        else:
            kwargs[ExplainType.MODEL_TASK] = ExplainType.REGRESSION
            num_classes = 1
            labels = None
        lime_explanations = []

        tqdm = get_tqdm(self._logger, self.show_progress)

        if self.explain_subset:
            self.original_data_ref[0] = original_evaluation
            self.current_index_list.append(0)
            for ex_idx, example in tqdm(enumerate(evaluation_examples)):
                self.current_index_list[0] = ex_idx
                lime_explanations.append(self.explainer.explain_instance(example,
                                                                         self.explainer.function,
                                                                         labels=labels))
            self.current_index_list = [0]
        else:
            for _, example in tqdm(enumerate(evaluation_examples)):
                lime_explanations.append(self.explainer.explain_instance(example,
                                                                         self.explainer.function,
                                                                         labels=labels))
        if self.classification:
            lime_values = [None] * num_classes
            for lime_explanation in lime_explanations:
                for label in labels:
                    map_values = dict(lime_explanation.as_list(label=label))
                    if lime_values[label - 1] is None:
                        lime_values[label - 1] = [[map_values.get(feature, 0.0) for feature in
                                                   self._lime_feature_names]]
                    else:
                        lime_values[label - 1].append([map_values.get(feature, 0.0) for feature in
                                                       self._lime_feature_names])
        else:
            lime_values = None
            for lime_explanation in lime_explanations:
                map_values = dict(lime_explanation.as_list())
                if lime_values is None:
                    lime_values = [[map_values.get(feature, 0.0) for feature in self._lime_feature_names]]
                else:
                    lime_values.append([map_values.get(feature, 0.0) for feature in self._lime_feature_names])
        expected_values = None
        if self.model is not None:
            kwargs[ExplainParams.MODEL_TYPE] = str(type(self.model))
        else:
            kwargs[ExplainParams.MODEL_TYPE] = ExplainType.FUNCTION

        kwargs[ExplainParams.CLASSIFICATION] = self.classification
        kwargs[ExplainParams.LOCAL_IMPORTANCE_VALUES] = np.array(lime_values)
        kwargs[ExplainParams.EXPECTED_VALUES] = np.array(expected_values)
        kwargs[ExplainParams.EVAL_DATA] = original_evaluation

        explanation = _create_local_explanation(**kwargs)

        # if transformations have been passed, then return raw features explanation
        raw_kwargs = _get_raw_explainer_create_explanation_kwargs(kwargs=kwargs)
        return explanation if self._datamapper is None else _create_raw_feats_local_explanation(
            explanation, feature_maps=[self._datamapper.feature_map], features=self.features, **raw_kwargs)
