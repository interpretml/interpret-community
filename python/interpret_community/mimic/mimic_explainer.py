# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Defines the Mimic Explainer for computing explanations on black box models or functions.

The mimic explainer trains an explainable model to reproduce the output of the given black box model.
The explainable model is called a surrogate model and the black box model is called a teacher model.
Once trained to reproduce the output of the teacher model, the surrogate model's explanation can
be used to explain the teacher model.
"""

import json
import logging
import warnings

import numpy as np
from scipy.sparse import issparse
from sklearn.metrics import accuracy_score, r2_score

from .._internal.raw_explain.raw_explain_utils import (
    get_datamapper_and_transformed_data, transform_with_datamapper)
from ..common.blackbox_explainer import BlackBoxExplainer
from ..common.constants import (Defaults, ExplainableModelType, ExplainParams,
                                ExplainType, Extension, LightGBMParams,
                                MimicSerializationConstants, ModelTask,
                                ResetIndex, ShapValuesOutput)
from ..common.exception import ScenarioNotSupportedException
from ..common.explanation_utils import _order_imp
from ..common.model_wrapper import _wrap_model
from ..dataset.dataset_wrapper import DatasetWrapper
from ..dataset.decorator import init_tabular_decorator, tabular_decorator
from ..explanation.explanation import (
    _aggregate_global_from_local_explanation,
    _aggregate_streamed_local_explanations, _create_global_explanation,
    _create_local_explanation, _create_raw_feats_global_explanation,
    _create_raw_feats_local_explanation,
    _get_raw_explainer_create_explanation_kwargs)
from .model_distill import _inverse_soft_logit, _model_distill
from .models import LGBMExplainableModel

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', 'Starting from version 2.2.1', UserWarning)


class MimicExplainer(BlackBoxExplainer):
    available_explanations = [Extension.GLOBAL, Extension.LOCAL]
    explainer_type = Extension.BLACKBOX

    """The Mimic Explainer for explaining black box models or functions.

    :param model: The black box model or function (if is_function is True) to be explained. Also known
        as the teacher model.
        A model that implements sklearn.predict or sklearn.predict_proba or function that accepts a 2d ndarray.
    :type model: object
    :param initialization_examples: A matrix of feature vector examples (# examples x # features) for
        initializing the explainer.
    :type initialization_examples: numpy.array or pandas.DataFrame or scipy.sparse.csr_matrix
    :param explainable_model: The uninitialized surrogate model used to explain the black box model.
        Also known as the student model.
    :type explainable_model: interpret_community.mimic.models.BaseExplainableModel
    :param explainable_model_args: An optional map of arguments to pass to the explainable model
        for initialization.
    :type explainable_model_args: dict
    :param is_function: Default is False. Set to True if passing function instead of model.
    :type is_function: bool
    :param augment_data: If True, oversamples the initialization examples to improve surrogate
        model accuracy to fit teacher model. Useful for high-dimensional data where
        the number of rows is less than the number of columns.
    :type augment_data: bool
    :param max_num_of_augmentations: Maximum number of times we can increase the input data size.
    :type max_num_of_augmentations: int
    :param explain_subset: List of feature indices. If specified, only selects a subset of the
        features in the evaluation dataset for explanation. Note for mimic explainer this will
        not affect the execution time of getting the global explanation. This argument is not supported when
        transformations are set.
    :type explain_subset: list[int]
    :param features: A list of feature names.
    :type features: list[str]
    :param classes: Class names as a list of strings. The order of the class names should match
        that of the model output. Only required if explaining classifier.
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
    :param shap_values_output: The shap values output from the explainer.  Only applies to
        tree-based models that are in terms of raw feature values instead of probabilities.
        Can be default, probability or teacher_probability. If probability or teacher_probability
        are specified, we approximate the feature importance values as probabilities instead
        of using the default values. If teacher probability is specified, we use the probabilities
        from the teacher model as opposed to the surrogate model.
    :type shap_values_output: interpret_community.common.constants.ShapValuesOutput
    :param categorical_features: Categorical feature names or indexes.
        If names are passed, they will be converted into indexes first.
        Note if pandas indexes are categorical, you can either pass the name of the index or the index
        as if the pandas index was inserted at the end of the input dataframe.
    :type categorical_features: Union[list[str], list[int]]
    :param allow_all_transformations: Allow many to many and many to one transformations
    :type allow_all_transformations: bool
    :param model_task: Optional parameter to specify whether the model is a classification or regression model.
        In most cases, the type of the model can be inferred based on the shape of the output, where a classifier
        has a predict_proba method and outputs a 2 dimensional array, while a regressor has a predict method and
        outputs a 1 dimensional array.
    :type model_task: str
    :param reset_index: Uses the pandas DataFrame index column as part of the features when training
        the surrogate model.
    :type reset_index: str
    """

    @init_tabular_decorator
    def __init__(self, model, initialization_examples, explainable_model, explainable_model_args=None,
                 is_function=False, augment_data=True, max_num_of_augmentations=10, explain_subset=None,
                 features=None, classes=None, transformations=None, allow_all_transformations=False,
                 shap_values_output=ShapValuesOutput.DEFAULT, categorical_features=None,
                 model_task=ModelTask.Unknown, reset_index=ResetIndex.Ignore, **kwargs):
        """Initialize the MimicExplainer.

        :param model: The black box model or function (if is_function is True) to be explained.  Also known
            as the teacher model.
            A model that implements sklearn.predict or sklearn.predict_proba or function that accepts a 2d
            ndarray.
        :type model: object
        :param initialization_examples: A matrix of feature vector examples (# examples x # features) for
            initializing the explainer.
        :type initialization_examples: numpy.array or pandas.DataFrame or scipy.sparse.csr_matrix
        :param explainable_model: The uninitialized surrogate model used to explain the black box model.
            Also known as the student model.
        :type explainable_model: BaseExplainableModel
        :param explainable_model_args: An optional map of arguments to pass to the explainable model
            for initialization.
        :type explainable_model_args: dict
        :param is_function: Default is False. Set to True if passing function instead of model.
        :type is_function: bool
        :param augment_data: If True, oversamples the initialization examples to improve surrogate
            model accuracy to fit teacher model.  Useful for high-dimensional data where
            the number of rows is less than the number of columns.
        :type augment_data: bool
        :param max_num_of_augmentations: Maximum number of times we can increase the input data size.
        :type max_num_of_augmentations: int
        :param explain_subset: List of feature indices. If specified, only selects a subset of the
            features in the evaluation dataset for explanation. Note for mimic explainer this will
            not affect the execution time of getting the global explanation. This argument is not supported when
            transformations are set.
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
        :param shap_values_output: The shap values output from the explainer.  Only applies to
            tree-based models that are in terms of raw feature values instead of probabilities.
            Can be default, probability or teacher_probability. If probability or teacher_probability
            are specified, we approximate the feature importance values as probabilities instead
            of using the default values. If teacher probability is specified, we use the probabilities
            from the teacher model as opposed to the surrogate model.
        :type shap_values_output: interpret_community.common.constants.ShapValuesOutput
        :param categorical_features: Categorical feature names or indexes.
            If names are passed, they will be converted into indexes first.
            Note if pandas indexes are categorical, you can either pass the name of the index or the index
            as if the pandas index was inserted at the end of the input dataframe.
        :type categorical_features: Union[list[str], list[int]]
        :param allow_all_transformations: Allow many to many and many to one transformations
        :type allow_all_transformations: bool
        :param model_task: Optional parameter to specify whether the model is a classification or regression model.
            In most cases, the type of the model can be inferred based on the shape of the output, where a classifier
            has a predict_proba method and outputs a 2 dimensional array, while a regressor has a predict method and
            outputs a 1 dimensional array.
        :type model_task: str
        :param reset_index: Can be ignore, reset or reset_teacher.  By default we ignore the index column, but the
            user can override to reset it and make it a feature column that is then featurized to numeric. Or,
            when using reset_teacher, the user can reset it and ignore it during featurization but set it as
            the index when calling predict on the original model.  Only use reset_teacher if the index is already
            featurized as part of the data.
        :type reset_index: str
        """
        if transformations is not None and explain_subset is not None:
            raise ValueError("explain_subset not supported with transformations")
        self.reset_index = reset_index
        self._datamapper = None
        if transformations is not None:
            self._datamapper, new_initialization_examples = get_datamapper_and_transformed_data(
                examples=initialization_examples, transformations=transformations,
                allow_all_transformations=allow_all_transformations)
            # memory optimization to clear all internal references so they can be gc'ed as soon as possible
            initialization_examples._clear()
            initialization_examples = new_initialization_examples
        if reset_index != ResetIndex.Ignore:
            initialization_examples.reset_index()
        wrapped_model, eval_ml_domain = _wrap_model(model, initialization_examples, model_task, is_function)
        super(MimicExplainer, self).__init__(wrapped_model, is_function=is_function,
                                             model_task=eval_ml_domain, **kwargs)
        if explainable_model_args is None:
            explainable_model_args = {}
        if categorical_features is None:
            categorical_features = []
        self._logger.debug('Initializing MimicExplainer')

        # Get the feature names from the initialization examples
        self._init_features = initialization_examples.get_features(features=features)
        self.features = features
        # augment the data if necessary
        if augment_data:
            initialization_examples.augment_data(max_num_of_augmentations=max_num_of_augmentations)
            # get the original data with types and index column if reset_index != ResetIndex.Ignore
            original_training_data = initialization_examples.typed_dataset
        else:
            # get the original data with types and index column if reset_index != ResetIndex.Ignore
            # note: this is small memory optimization over above line if augment data = False
            original_training_data = initialization_examples.original_dataset_with_type

        # if index column should not be set on surrogate model, remove it
        if reset_index == ResetIndex.ResetTeacher:
            initialization_examples.set_index()

        # If categorical_features is a list of string column names instead of indexes, make sure to convert to indexes
        if not all(isinstance(categorical_feature, int) for categorical_feature in categorical_features):
            categorical_features = initialization_examples.get_column_indexes(self._init_features,
                                                                              categorical_features)

        # Featurize any timestamp columns
        # TODO: more sophisticated featurization
        self._timestamp_featurizer = initialization_examples.timestamp_featurizer()

        # If model is a linear model or isn't able to handle categoricals, one-hot-encode categoricals
        is_tree_model = explainable_model.explainable_model_type() == ExplainableModelType.TREE_EXPLAINABLE_MODEL_TYPE
        if is_tree_model and self._supports_categoricals(explainable_model):
            # Index the categorical string columns for training data
            self._column_indexer = initialization_examples.string_index(columns=categorical_features)
            self._one_hot_encoder = None
            if categorical_features:
                explainable_model_args[LightGBMParams.CATEGORICAL_FEATURE] = categorical_features
        else:
            # One-hot-encode categoricals for models that don't support categoricals natively
            self._column_indexer = initialization_examples.string_index(columns=categorical_features)
            self._one_hot_encoder = initialization_examples.one_hot_encode(columns=categorical_features)

        self.classes = classes
        self.explain_subset = explain_subset
        self.transformations = transformations
        self._shap_values_output = shap_values_output
        # Train the mimic model on the given model
        training_data = initialization_examples.dataset
        self.initialization_examples = initialization_examples
        if str(type(training_data)).endswith(".DenseData'>"):
            training_data = training_data.data

        explainable_model_args[ExplainParams.CLASSIFICATION] = self.predict_proba_flag
        if self._supports_shap_values_output(explainable_model):
            explainable_model_args[ExplainParams.SHAP_VALUES_OUTPUT] = shap_values_output
        self.surrogate_model = _model_distill(self.function, explainable_model, training_data,
                                              original_training_data, explainable_model_args)
        self._method = self.surrogate_model._method
        self._original_eval_examples = None
        self._allow_all_transformations = allow_all_transformations

    def _get_transformed_data(self, evaluation_examples):
        """Return the transformed data for some evaluation data.

        :param evaluation_examples: A matrix of feature vector examples (# examples x # features) on which to
            explain the model's output.  If specified, computes feature importance through aggregation.
        :type evaluation_examples: numpy.array or pandas.DataFrame or scipy.sparse.csr_matrix
        :return: Transformed data.
        :rtype: numpy.array or pandas.DataFrame or scipy.sparse.csr_matrix
        """
        if self.transformations is not None:
            _, transformed_evaluation_examples = get_datamapper_and_transformed_data(
                examples=evaluation_examples, transformations=self.transformations,
                allow_all_transformations=self._allow_all_transformations)
        else:
            transformed_evaluation_examples = evaluation_examples

        return transformed_evaluation_examples

    def _get_surrogate_model_predictions(self, evaluation_examples):
        """Return the predictions given by the surrogate model.

        :param evaluation_examples: A matrix of feature vector examples (# examples x # features) on which to
            explain the model's output.  If specified, computes feature importance through aggregation.
        :type evaluation_examples: numpy.array or pandas.DataFrame or scipy.sparse.csr_matrix
        :return: predictions of the surrogate model.
        :rtype: numpy.array
        """
        transformed_evaluation_examples = self._get_transformed_data(evaluation_examples)
        if self.classes is not None and len(self.classes) == 2:
            index_predictions = _inverse_soft_logit(self.surrogate_model.predict(transformed_evaluation_examples))
            actual_predictions = []
            for index in index_predictions:
                actual_predictions.append(self.classes[index])
            return np.array(actual_predictions)
        else:
            return self.surrogate_model.predict(transformed_evaluation_examples)

    def _get_teacher_model_predictions(self, evaluation_examples):
        """Return the predictions given by the teacher model.

        :param evaluation_examples: A matrix of feature vector examples (# examples x # features) on which to
            explain the model's output.  If specified, computes feature importance through aggregation.
        :type evaluation_examples: numpy.array or pandas.DataFrame or scipy.sparse.csr_matrix
        :return: predictions of the surrogate model.
        :rtype: numpy.array
        """
        transformed_evaluation_examples = self._get_transformed_data(evaluation_examples)
        return self.model.predict(transformed_evaluation_examples)

    def _supports_categoricals(self, explainable_model):
        return issubclass(explainable_model, LGBMExplainableModel)

    def _supports_shap_values_output(self, explainable_model):
        return issubclass(explainable_model, LGBMExplainableModel)

    def _get_explain_global_kwargs(self, evaluation_examples=None, include_local=True,
                                   batch_size=Defaults.DEFAULT_BATCH_SIZE):
        """Get the kwargs for explain_global to create a global explanation.

        :param evaluation_examples: A matrix of feature vector examples (# examples x # features) on which to
            explain the model's output.  If specified, computes feature importance through aggregation.
        :type evaluation_examples: numpy.array or pandas.DataFrame or scipy.sparse.csr_matrix
        :param include_local: Include the local explanations in the returned global explanation.
            If evaluation examples are specified and include_local is False, will stream the local
            explanations to aggregate to global.
        :type include_local: bool
        :param batch_size: If include_local is False, specifies the batch size for aggregating
            local explanations to global.
        :type batch_size: int
        :return: Args for explain_global.
        :rtype: dict
        """
        classification = self.predict_proba_flag
        kwargs = {ExplainParams.METHOD: self._get_method}
        if classification:
            kwargs[ExplainParams.CLASSES] = self.classes
        if evaluation_examples is not None:

            # Aggregate local explanation to global, either through computing the local
            # explanation and then aggregating or streaming the local explanation to global
            if include_local:
                # Get local explanation
                local_explanation = self._explain_local(evaluation_examples, from_global=True)
                kwargs[ExplainParams.LOCAL_EXPLANATION] = local_explanation
            else:
                if classification:
                    model_task = ModelTask.Classification
                else:
                    model_task = ModelTask.Regression
                if not isinstance(evaluation_examples, DatasetWrapper):
                    self._logger.debug('Eval examples not wrapped, wrapping')
                    evaluation_examples = DatasetWrapper(evaluation_examples)

                kwargs = _aggregate_streamed_local_explanations(self, evaluation_examples, model_task, self.features,
                                                                batch_size, **kwargs)
            return kwargs
        global_importance_values = self.surrogate_model.explain_global()
        order = _order_imp(global_importance_values)
        if classification:
            kwargs[ExplainParams.MODEL_TASK] = ExplainType.CLASSIFICATION
        else:
            kwargs[ExplainParams.MODEL_TASK] = ExplainType.REGRESSION
        if self.model is not None:
            kwargs[ExplainParams.MODEL_TYPE] = str(type(self.model))
        else:
            kwargs[ExplainParams.MODEL_TYPE] = ExplainType.FUNCTION
        kwargs[ExplainParams.EXPECTED_VALUES] = None
        kwargs[ExplainParams.CLASSIFICATION] = classification
        kwargs[ExplainParams.GLOBAL_IMPORTANCE_VALUES] = global_importance_values
        kwargs[ExplainParams.GLOBAL_IMPORTANCE_RANK] = order
        kwargs[ExplainParams.FEATURES] = self.features
        return kwargs

    def explain_global(self, evaluation_examples=None, include_local=True,
                       batch_size=Defaults.DEFAULT_BATCH_SIZE):
        """Globally explains the blackbox model using the surrogate model.

        If evaluation_examples are unspecified, retrieves global feature importance from explainable
        surrogate model.  Note this will not include per class feature importance. If evaluation_examples
        are specified, aggregates local explanations to global from the given evaluation_examples - which
        computes both global and per class feature importance.

        :param evaluation_examples: A matrix of feature vector examples (# examples x # features) on which to
            explain the model's output.  If specified, computes feature importance through aggregation.
        :type evaluation_examples: numpy.array or pandas.DataFrame or scipy.sparse.csr_matrix
        :param include_local: Include the local explanations in the returned global explanation.
            If evaluation examples are specified and include_local is False, will stream the local
            explanations to aggregate to global.
        :type include_local: bool
        :param batch_size: If include_local is False, specifies the batch size for aggregating
            local explanations to global.
        :type batch_size: int
        :return: A model explanation object. It is guaranteed to be a GlobalExplanation. If evaluation_examples are
            passed in, it will also have the properties of a LocalExplanation. If the model is a classifier (has
            predict_proba), it will have the properties of ClassesMixin, and if evaluation_examples were passed in it
            will also have the properties of PerClassMixin.
        :rtype: DynamicGlobalExplanation
        """
        if self._original_eval_examples is None:
            if isinstance(evaluation_examples, DatasetWrapper):
                self._original_eval_examples = evaluation_examples.original_dataset_with_type
            else:
                self._original_eval_examples = evaluation_examples
        kwargs = self._get_explain_global_kwargs(evaluation_examples=evaluation_examples, include_local=include_local,
                                                 batch_size=batch_size)
        kwargs[ExplainParams.INIT_DATA] = self.initialization_examples
        has_eval_examples = evaluation_examples is not None
        if has_eval_examples:
            kwargs[ExplainParams.EVAL_DATA] = self._original_eval_examples
            ys_dict = self._get_ys_dict(self._original_eval_examples,
                                        transformations=self.transformations,
                                        allow_all_transformations=self._allow_all_transformations)
            kwargs.update(ys_dict)
            if include_local:
                return _aggregate_global_from_local_explanation(**kwargs)
        explanation = _create_global_explanation(**kwargs)
        has_data_mapper = self._datamapper is not None

        if has_data_mapper:
            if not has_eval_examples:
                # If transformations have been passed, and no evaluation data, then return raw features explanation
                raw_kwargs = _get_raw_explainer_create_explanation_kwargs(kwargs=kwargs)
                return _create_raw_feats_global_explanation(explanation,
                                                            feature_maps=[self._datamapper.feature_map],
                                                            features=self.features,
                                                            **raw_kwargs)
            else:
                # Note this is a raw explanation already if transformations were passed
                explanation._is_raw = True
        # If eval examples passed, even if transformations is passed the explanation is
        # already in terms of raw data since it is aggregated from engineered.
        # If there is no data mapper then we should return the explanation directly.
        return explanation

    @property
    def _get_method(self):
        """Get the method for this explainer, or mimic with surrogate model type.

        :return: The method, or mimic with surrogate model type.
        :rtype: str
        """
        return "{}.{}".format(ExplainType.MIMIC, self._method)

    def _get_explain_local_kwargs(self, evaluation_examples, from_global):
        """Get the kwargs for explain_local to create a local explanation.

        :param evaluation_examples: A matrix of feature vector examples (# examples x # features) on which
            to explain the model's output.
        :type evaluation_examples: numpy.array or pandas.DataFrame or scipy.sparse.csr_matrix
        :param from_global: True if this is called from the explain_global API.
        :type from_global: bool
        :return: Args for explain_local.
        :rtype: dict
        """
        if self.reset_index != ResetIndex.Ignore:
            evaluation_examples.reset_index()
        kwargs = {}
        probabilities = None
        if self._shap_values_output == ShapValuesOutput.TEACHER_PROBABILITY:
            original_evaluation_examples = evaluation_examples.typed_dataset
            # Outputting shap values in terms of the probabilities of the teacher model
            probabilities = self.function(original_evaluation_examples)
        # if index column should not be set on surrogate model, remove it
        if self.reset_index == ResetIndex.ResetTeacher:
            evaluation_examples.set_index()
        if self._timestamp_featurizer:
            evaluation_examples.apply_timestamp_featurizer(self._timestamp_featurizer)
        if self._column_indexer:
            evaluation_examples.apply_indexer(self._column_indexer, bucket_unknown=True)
        if self._one_hot_encoder:
            evaluation_examples.apply_one_hot_encoder(self._one_hot_encoder)

        dataset = evaluation_examples.dataset

        kwargs[ExplainParams.NUM_FEATURES] = evaluation_examples.num_features

        local_importance_values = self.surrogate_model.explain_local(dataset, probabilities=probabilities)
        classification = isinstance(local_importance_values, list) or self.predict_proba_flag
        is_sparse = issparse(local_importance_values) or issparse(local_importance_values[0])
        expected_values = self.surrogate_model.expected_values
        kwargs[ExplainParams.METHOD] = self._get_method
        self.features = evaluation_examples.get_features(features=self.features)
        kwargs[ExplainParams.FEATURES] = self.features

        if self.predict_proba_flag:
            if not is_sparse:
                if self.surrogate_model.multiclass:
                    # For multiclass case, convert to array, but only if not sparse
                    local_importance_values = np.array(local_importance_values)
                else:
                    # TODO: Eventually move this back inside the surrogate model
                    # If binary case, we need to reformat the data to have importances per class
                    # and convert the expected values back to the original domain
                    local_importance_values = np.stack((-local_importance_values, local_importance_values))
            elif not self.surrogate_model.multiclass:
                # For binary classification sparse case we need to reformat the data
                # to have importance values per class
                local_importance_values = [-local_importance_values, local_importance_values]

        if classification:
            kwargs[ExplainParams.CLASSES] = self.classes
        # Reformat local_importance_values result if explain_subset specified
        if self.explain_subset:
            self._logger.debug('Getting subset of local_importance_values')
            if classification:
                if is_sparse and self.surrogate_model.multiclass:
                    for i in range(len(local_importance_values)):
                        local_importance_values[i] = local_importance_values[i][:, self.explain_subset]
                else:
                    local_importance_values = local_importance_values[:, :, self.explain_subset]
            else:
                local_importance_values = local_importance_values[:, self.explain_subset]
        if classification:
            kwargs[ExplainParams.MODEL_TASK] = ExplainType.CLASSIFICATION
        else:
            kwargs[ExplainParams.MODEL_TASK] = ExplainType.REGRESSION
        if self.model is not None:
            kwargs[ExplainParams.MODEL_TYPE] = str(type(self.model))
        else:
            kwargs[ExplainParams.MODEL_TYPE] = ExplainType.FUNCTION
        kwargs[ExplainParams.LOCAL_IMPORTANCE_VALUES] = local_importance_values
        kwargs[ExplainParams.EXPECTED_VALUES] = np.array(expected_values)
        kwargs[ExplainParams.CLASSIFICATION] = classification
        if not from_global:
            ys_dict = self._get_ys_dict(self._original_eval_examples,
                                        transformations=self.transformations,
                                        allow_all_transformations=self._allow_all_transformations)
            kwargs.update(ys_dict)
        return kwargs

    def _explain_local_helper(self, evaluation_examples, from_global):
        """Helper method to create the local explanation.

        :param evaluation_examples: A matrix of feature vector examples (# examples x # features) on which
            to explain the model's output.
        :type evaluation_examples: numpy.array or pandas.DataFrame or scipy.sparse.csr_matrix
        :param from_global: True if this is called from the explain_global API.
        :type from_global: bool
        :return: kwargs for building the model explanation object
        :rtype: dict
        """
        if not from_global:
            if self._original_eval_examples is None:
                if isinstance(evaluation_examples, DatasetWrapper):
                    self._original_eval_examples = evaluation_examples.original_dataset_with_type
                else:
                    self._original_eval_examples = evaluation_examples
        if self._datamapper is not None:
            new_evaluation_examples = transform_with_datamapper(evaluation_examples, self._datamapper)
            # memory optimization to clear all internal references so they can be gc'ed as soon as possible
            evaluation_examples._clear()
            evaluation_examples = new_evaluation_examples

        kwargs = self._get_explain_local_kwargs(evaluation_examples, from_global)
        if not from_global:
            kwargs[ExplainParams.INIT_DATA] = self.initialization_examples
            kwargs[ExplainParams.EVAL_DATA] = evaluation_examples
        return kwargs

    @tabular_decorator
    def _explain_local(self, evaluation_examples, from_global=False):
        """Locally explains the blackbox model using the surrogate model.

        :param evaluation_examples: A matrix of feature vector examples (# examples x # features) on which
            to explain the model's output.
        :type evaluation_examples: numpy.array or pandas.DataFrame or scipy.sparse.csr_matrix
        :param from_global: True if this is called from the explain_global API.
        :type from_global: bool
        :return: A model explanation object. It is guaranteed to be a LocalExplanation. If the model is a classifier,
            it will have the properties of the ClassesMixin.
        :rtype: DynamicLocalExplanation
        """
        kwargs = self._explain_local_helper(evaluation_examples, from_global)
        explanation = _create_local_explanation(**kwargs)

        # if transformations have been passed, then return raw features explanation
        raw_kwargs = _get_raw_explainer_create_explanation_kwargs(kwargs=kwargs)

        return explanation if self._datamapper is None else _create_raw_feats_local_explanation(
            explanation, feature_maps=[self._datamapper.feature_map], features=self.features, **raw_kwargs)

    @tabular_decorator
    def explain_local(self, evaluation_examples):
        """Locally explains the blackbox model using the surrogate model.

        :param evaluation_examples: A matrix of feature vector examples (# examples x # features) on which
            to explain the model's output.
        :type evaluation_examples: numpy.array or pandas.DataFrame or scipy.sparse.csr_matrix
        :return: A model explanation object. It is guaranteed to be a LocalExplanation. If the model is a classifier,
            it will have the properties of the ClassesMixin.
        :rtype: DynamicLocalExplanation
        """
        return self._explain_local(evaluation_examples)

    def _save(self):
        """Return a string dictionary representation of the mimic explainer.

        Currently only supported scenario is Mimic Explainer with LightGBM surrogate model.

        :return: A serialized dictionary representation of the mimic explainer.
        :rtype: dict
        """
        properties = {}
        # save all of the properties
        for key, value in self.__dict__.items():
            if key in MimicSerializationConstants.nonify_properties:
                properties[key] = None
            elif key in MimicSerializationConstants.save_properties:
                properties[key] = value._save()
            else:
                properties[key] = json.dumps(value)
        # return a dictionary of strings
        return properties

    @staticmethod
    def _load(model, properties):
        """Load a MimicExplainer from the given properties.

        Currently only supported scenario is Mimic Explainer with LightGBM surrogate model.

        :param model: The serialized ONNX model with a scikit-learn like API.
        :type model: ONNX model.
        :param properties: A serialized dictionary representation of the mimic explainer.
        :type properties: dict
        :return: The deserialized MimicExplainer.
        :rtype: interpret_community.mimic.MimicExplainer
        """
        # create the MimicExplainer without any properties using the __new__ function, similar to pickle
        mimic = MimicExplainer.__new__(MimicExplainer)
        # load all of the properties
        for key, value in properties.items():
            # Regenerate the properties on the fly
            if key in MimicSerializationConstants.nonify_properties:
                if key == MimicSerializationConstants.MODEL:
                    mimic.__dict__[key] = model
                elif key == MimicSerializationConstants.LOGGER:
                    parent = logging.getLogger(__name__)
                    mimic_identity = json.loads(properties[MimicSerializationConstants.IDENTITY])
                    mimic.__dict__[key] = parent.getChild(mimic_identity)
                elif key == MimicSerializationConstants.INITIALIZATION_EXAMPLES:
                    mimic.__dict__[key] = None
                elif key == MimicSerializationConstants.ORIGINAL_EVAL_EXAMPLES:
                    mimic.__dict__[key] = None
                elif key == MimicSerializationConstants.TIMESTAMP_FEATURIZER:
                    mimic.__dict__[key] = None
                elif key == MimicSerializationConstants.FUNCTION:
                    # TODO add third case if is_function was passed to mimic explainer
                    if json.loads(properties[MimicSerializationConstants.PREDICT_PROBA_FLAG]):
                        mimic.__dict__[key] = model.predict_proba
                    else:
                        mimic.__dict__[key] = model.predict
                else:
                    raise Exception("Unknown nonify key on deserialize in MimicExplainer: {}".format(key))
            elif key in MimicSerializationConstants.save_properties:
                mimic.__dict__[key] = LGBMExplainableModel._load(value)
            elif key in MimicSerializationConstants.enum_properties:
                # NOTE: If more enums added in future, will need to handle this differently
                mimic.__dict__[key] = ShapValuesOutput(json.loads(value))
            else:
                mimic.__dict__[key] = json.loads(value)
        if MimicSerializationConstants.ORIGINAL_EVAL_EXAMPLES not in mimic.__dict__:
            mimic.__dict__[MimicSerializationConstants.ORIGINAL_EVAL_EXAMPLES] = None
        if MimicSerializationConstants.TIMESTAMP_FEATURIZER not in mimic.__dict__:
            mimic.__dict__[MimicSerializationConstants.TIMESTAMP_FEATURIZER] = None
        if MimicSerializationConstants.RESET_INDEX not in mimic.__dict__:
            mimic.__dict__[MimicSerializationConstants.RESET_INDEX] = False
        if MimicSerializationConstants.ALLOW_ALL_TRANSFORMATIONS not in mimic.__dict__:
            mimic.__dict__[MimicSerializationConstants.ALLOW_ALL_TRANSFORMATIONS] = False
        return mimic

    def __getstate__(self):
        """Influence how MimicExplainer is pickled.

        Removes logger which is not serializable.

        :return state: The state to be pickled, with logger removed.
        :rtype: dict
        """
        odict = self.__dict__.copy()
        del odict['_logger']
        return odict

    def __setstate__(self, state):
        """Influence how MimicExplainer is unpickled.

        Re-adds logger which is not serializable.

        :param dict: A dictionary of deserialized state.
        :type dict: dict
        """
        self.__dict__.update(state)
        self._logger = logging.getLogger(__name__)

    def _get_surrogate_model_replication_measure(self, training_data):
        """Return the metric which tells how well the surrogate model replicates the teacher model.
        :param training_data: The data for getting the replication metric.
        :type training_data: numpy.array or pandas.DataFrame or scipy.sparse.csr_matrix
        :return: Metric that tells how well the surrogate model replicates the behavior of teacher model.
        :rtype: float
        """
        if self.classes is None and training_data.shape[0] == 1:
            raise ScenarioNotSupportedException(
                "Replication measure for regression surrogate not supported "
                "because of single instance in training data")

        surrogate_model_predictions = self._get_surrogate_model_predictions(training_data)
        teacher_model_predictions = self._get_teacher_model_predictions(training_data)

        if self.classes is not None:
            replication_measure = accuracy_score(teacher_model_predictions, surrogate_model_predictions)
        else:
            replication_measure = r2_score(teacher_model_predictions, surrogate_model_predictions)
        return replication_measure
