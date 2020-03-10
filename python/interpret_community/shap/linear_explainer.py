# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Defines the LinearExplainer for returning explanations for linear models."""

import numpy as np

from ..common.structured_model_explainer import StructuredInitModelExplainer
from ..common.explanation_utils import _fix_linear_explainer_shap_values
from ..common.aggregate import add_explain_global_method, init_aggregator_decorator
from ..common.constants import ExplainParams, Attributes, ExplainType, \
    Defaults, Extension, SHAPDefaults
from ..dataset.dataset_wrapper import DatasetWrapper
from ..dataset.decorator import tabular_decorator
from ..explanation.explanation import _create_local_explanation, \
    _create_raw_feats_local_explanation, _get_raw_explainer_create_explanation_kwargs
from .kwargs_utils import _get_explain_global_kwargs
from .._internal.raw_explain.raw_explain_utils import get_datamapper_and_transformed_data, \
    transform_with_datamapper

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', 'Starting from version 2.2.1', UserWarning)
    import shap


@add_explain_global_method
class LinearExplainer(StructuredInitModelExplainer):
    available_explanations = [Extension.GLOBAL, Extension.LOCAL]
    explainer_type = Extension.GREYBOX

    """Defines the LinearExplainer for returning explanations for linear models.

    :param model: The linear model to explain as the coefficient and intercept or scikit learn model.
    :type model: (coef, intercept) or sklearn.linear_model.*
    :param initialization_examples: A matrix of feature vector examples (# examples x # features) for
        initializing the explainer.
    :type initialization_examples: numpy.array or pandas.DataFrame or iml.datatypes.DenseData or
        scipy.sparse.csr_matrix
    :param explain_subset: List of feature indices. If specified, only selects a subset of the
        features in the evaluation dataset for explanation. The subset can be the top-k features
        from the model summary.
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
    """

    @init_aggregator_decorator
    def __init__(self, model, initialization_examples, explain_subset=None, features=None, classes=None,
                 transformations=None, allow_all_transformations=False, **kwargs):
        """Initialize the LinearExplainer.

        :param model: The linear model to explain as the coefficient and intercept or scikit learn model.
        :type model: (coef, intercept) or sklearn.linear_model.*
        :param initialization_examples: A matrix of feature vector examples (# examples x # features) for
            initializing the explainer.
        :type initialization_examples: numpy.array or pandas.DataFrame or iml.datatypes.DenseData or
            scipy.sparse.csr_matrix
        :param explain_subset: List of feature indices. If specified, only selects a subset of the
            features in the evaluation dataset for explanation. The subset can be the top-k features
            from the model summary.
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
        """
        self._datamapper = None
        if transformations is not None:
            self._datamapper, initialization_examples = get_datamapper_and_transformed_data(
                examples=initialization_examples, transformations=transformations,
                allow_all_transformations=allow_all_transformations)
        super(LinearExplainer, self).__init__(model, initialization_examples, **kwargs)
        self._logger.debug('Initializing LinearExplainer')
        self._method = 'shap.linear'
        self.explainer = shap.LinearExplainer(self.model, self.initialization_examples,
                                              feature_dependence=SHAPDefaults.INDEPENDENT)
        self.explain_subset = explain_subset
        self.features = features
        self.classes = classes
        self.transformations = transformations
        self._allow_all_transformations = allow_all_transformations

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
        kwargs = _get_explain_global_kwargs(sampling_policy, ExplainType.SHAP_LINEAR, include_local, batch_size)
        kwargs[ExplainParams.INIT_DATA] = self.initialization_examples
        kwargs[ExplainParams.EVAL_DATA] = evaluation_examples.original_dataset_with_type
        wrapped_evals = evaluation_examples
        if self.transformations is not None:
            _, evaluation_examples = get_datamapper_and_transformed_data(
                examples=evaluation_examples,
                transformations=self.transformations,
                allow_all_transformations=self._allow_all_transformations
            )
        if isinstance(evaluation_examples, DatasetWrapper):
            kwargs[ExplainParams.NUM_FEATURES] = evaluation_examples.num_features
            evaluation_examples = evaluation_examples.original_dataset_with_type
        if len(evaluation_examples.shape) == 1:
            evaluation_examples = evaluation_examples.reshape(1, -1)
        kwargs[ExplainParams.EVAL_Y_PRED] = self.model.predict(evaluation_examples)
        if hasattr(self.model, 'predict_proba'):
            kwargs[ExplainParams.EVAL_Y_PRED_PROBA] = self.model.predict_proba(evaluation_examples)
        return self._explain_global(wrapped_evals, **kwargs)

    def _get_explain_local_kwargs(self, evaluation_examples, original_evals):
        """Get the kwargs for explain_local to create a local explanation.

        :param evaluation_examples: A matrix of feature vector examples (# examples x # features) on which
            to explain the model's output.
        :type evaluation_examples: numpy.array or pandas.DataFrame or scipy.sparse.csr_matrix
        :param original_evals: The original data that the user passed into explain_local.
        :type original_evals: numpy.array or pandas.DataFrame or scipy.sparse.csr_matrix
        :return: Args for explain_local.
        :rtype: dict
        """
        self._logger.debug('Explaining linear model')
        kwargs = {ExplainParams.METHOD: ExplainType.SHAP_LINEAR}
        if self.classes is not None:
            kwargs[ExplainParams.CLASSES] = self.classes
        kwargs[ExplainParams.FEATURES] = evaluation_examples.get_features(features=self.features)
        kwargs[ExplainParams.NUM_FEATURES] = evaluation_examples.num_features
        evaluation_examples = evaluation_examples.dataset

        shap_values = self.explainer.shap_values(evaluation_examples)
        # Temporary fix for a bug in shap for regression models
        shap_values = _fix_linear_explainer_shap_values(self.model, shap_values)
        expected_values = None
        if hasattr(self.explainer, Attributes.EXPECTED_VALUE):
            self._logger.debug('Expected values available on explainer')
            expected_values = np.array(self.explainer.expected_value)
        classification = isinstance(shap_values, list)
        # Reformat shap values result if explain_subset specified
        if self.explain_subset:
            self._logger.debug('Getting subset of shap_values')
            if classification:
                for i in range(len(shap_values)):
                    shap_values[i] = shap_values[i][:, self.explain_subset]
            else:
                shap_values = shap_values[:, self.explain_subset]
        if classification:
            kwargs[ExplainParams.MODEL_TASK] = ExplainType.CLASSIFICATION
        else:
            kwargs[ExplainParams.MODEL_TASK] = ExplainType.REGRESSION
        kwargs[ExplainParams.MODEL_TYPE] = str(type(self.model))
        kwargs[ExplainParams.LOCAL_IMPORTANCE_VALUES] = np.array(shap_values)
        kwargs[ExplainParams.EXPECTED_VALUES] = expected_values
        kwargs[ExplainParams.CLASSIFICATION] = classification
        kwargs[ExplainParams.INIT_DATA] = self.initialization_examples
        kwargs[ExplainParams.EVAL_DATA] = original_evals
        if len(evaluation_examples.shape) == 1:
            evaluation_examples = evaluation_examples.reshape(1, -1)
        kwargs[ExplainParams.EVAL_Y_PRED] = self.model.predict(evaluation_examples)
        if hasattr(self.model, 'predict_proba'):
            kwargs[ExplainParams.EVAL_Y_PRED_PROBA] = self.model.predict_proba(evaluation_examples)
        return kwargs

    @tabular_decorator
    def explain_local(self, evaluation_examples):
        """Explain the model by using SHAP's linear explainer.

        :param evaluation_examples: A matrix of feature vector examples (# examples x # features) on which
            to explain the model's output.
        :type evaluation_examples: DatasetWrapper
        :return: A model explanation object. It is guaranteed to be a LocalExplanation which also has the properties
            of ExpectedValuesMixin. If the model is a classifier, it will have the properties of the ClassesMixin.
        :rtype: DynamicLocalExplanation
        """
        original_evals = evaluation_examples.original_dataset_with_type
        if self._datamapper is not None:
            evaluation_examples = transform_with_datamapper(evaluation_examples, self._datamapper)

        kwargs = self._get_explain_local_kwargs(evaluation_examples, original_evals)
        explanation = _create_local_explanation(**kwargs)

        if self._datamapper is None:
            return explanation
        else:
            # if transformations have been passed, then return raw features explanation
            raw_kwargs = _get_raw_explainer_create_explanation_kwargs(kwargs=kwargs)
            return _create_raw_feats_local_explanation(explanation, feature_maps=[self._datamapper.feature_map],
                                                       features=self.features, **raw_kwargs)
