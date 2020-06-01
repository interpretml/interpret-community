# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Defines the TreeExplainer for returning explanations for tree-based models."""

import numpy as np

from ..common.structured_model_explainer import PureStructuredModelExplainer
from ..common.explanation_utils import _get_dense_examples, _convert_to_list
from ..common.aggregate import add_explain_global_method, init_aggregator_decorator
from ..common.explanation_utils import _scale_tree_shap
from ..dataset.decorator import tabular_decorator
from ..explanation.explanation import _create_local_explanation, \
    _create_raw_feats_local_explanation, _get_raw_explainer_create_explanation_kwargs
from .kwargs_utils import _get_explain_global_kwargs
from ..common.constants import ExplainParams, Attributes, ExplainType, \
    ShapValuesOutput, Defaults, Extension
from .._internal.raw_explain.raw_explain_utils import get_datamapper_and_transformed_data, \
    transform_with_datamapper
from ..dataset.dataset_wrapper import DatasetWrapper

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', 'Starting from version 2.2.1', UserWarning)
    import shap


@add_explain_global_method
class TreeExplainer(PureStructuredModelExplainer):
    available_explanations = [Extension.GLOBAL, Extension.LOCAL]
    explainer_type = Extension.GREYBOX

    """The TreeExplainer for returning explanations for tree-based models.

    :param model: The tree model to explain.
    :type model: lightgbm, xgboost or scikit-learn tree model
    :param explain_subset: List of feature indices. If specified, only selects a subset of the
        features in the evaluation dataset for explanation. The subset can be the top-k features
        from the model summary.
    :type explain_subset: list[int]
    :param features: A list of feature names.
    :type features: list[str]
    :param classes: Class names as a list of strings. The order of the class names should match
        that of the model output. Only required if explaining classifier.
    :type classes: list[str]
    :param shap_values_output: The type of the output when using TreeExplainer.
        Currently only types 'default' and 'probability' are supported.  If 'probability'
        is specified, then the raw log-odds values are approximately scaled to probabilities from the TreeExplainer.
    :type shap_values_output: interpret_community.common.constants.ShapValuesOutput
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
    def __init__(self, model, explain_subset=None, features=None, classes=None,
                 shap_values_output=ShapValuesOutput.DEFAULT, transformations=None,
                 allow_all_transformations=False, **kwargs):
        """Initialize the TreeExplainer.

        :param model: The tree model to explain.
        :type model: lightgbm, xgboost or scikit-learn tree model
        :param explain_subset: List of feature indices. If specified, only selects a subset of the
            features in the evaluation dataset for explanation. The subset can be the top-k features
            from the model summary.
        :type explain_subset: list[int]
        :param features: A list of feature names.
        :type features: list[str]
        :param classes: Class names as a list of strings. The order of the class names should match
            that of the model output.  Only required if explaining classifier.
        :type classes: list[str]
        :param shap_values_output: The type of the output when using TreeExplainer.
            Currently only types 'default' and 'probability' are supported.  If 'probability'
            is specified, then the raw log-odds values are approximately scaled to probabilities from the
            TreeExplainer.
        :type shap_values_output: interpret_community.common.constants.ShapValuesOutput
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
            self._datamapper, _ = get_datamapper_and_transformed_data(
                transformations=transformations, allow_all_transformations=allow_all_transformations)

        super(TreeExplainer, self).__init__(model, **kwargs)
        self._logger.debug('Initializing TreeExplainer')
        self._method = 'shap.tree'
        self.explainer = shap.TreeExplainer(self.model)
        self.explain_subset = explain_subset
        self.features = features
        self.classes = classes
        self.transformations = transformations
        self._allow_all_transformations = allow_all_transformations
        self._shap_values_output = shap_values_output

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
        kwargs = _get_explain_global_kwargs(sampling_policy, ExplainType.SHAP_TREE, include_local, batch_size)
        kwargs[ExplainParams.EVAL_DATA] = evaluation_examples.original_dataset_with_type
        wrapped_evals = evaluation_examples
        if self.transformations is not None:
            _, evaluation_examples = get_datamapper_and_transformed_data(
                examples=evaluation_examples,
                transformations=self.transformations,
                allow_all_transformations=self._allow_all_transformations
            )
        typed_wrapper_func = None
        if isinstance(evaluation_examples, DatasetWrapper):
            kwargs[ExplainParams.NUM_FEATURES] = evaluation_examples.num_features
            typed_wrapper_func = evaluation_examples.typed_wrapper_func
            evaluation_examples = evaluation_examples.original_dataset_with_type

        if len(evaluation_examples.shape) == 1:
            # TODO: is this needed?
            evaluation_examples = evaluation_examples.reshape(1, -1)
        if typed_wrapper_func is not None:
            typed_evaluation_examples = typed_wrapper_func(evaluation_examples)
        else:
            typed_evaluation_examples = evaluation_examples
        kwargs[ExplainParams.EVAL_Y_PRED] = self.model.predict(typed_evaluation_examples)
        if hasattr(self.model, 'predict_proba'):
            kwargs[ExplainParams.EVAL_Y_PRED_PROBA] = self.model.predict_proba(typed_evaluation_examples)
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
        self._logger.debug('Explaining tree model')
        kwargs = {ExplainParams.METHOD: ExplainType.SHAP_TREE}
        if self.classes is not None:
            kwargs[ExplainParams.CLASSES] = self.classes
        kwargs[ExplainParams.FEATURES] = evaluation_examples.get_features(features=self.features)
        typed_wrapper_func = evaluation_examples.typed_wrapper_func
        kwargs[ExplainParams.NUM_FEATURES] = evaluation_examples.num_features
        evaluation_examples = evaluation_examples.dataset

        # for now convert evaluation examples to dense format if they are sparse
        # until TreeExplainer sparse support is added
        typed_dense_evaluation_examples = typed_wrapper_func(_get_dense_examples(evaluation_examples))
        shap_values = self.explainer.shap_values(typed_dense_evaluation_examples)
        expected_values = None
        if hasattr(self.explainer, Attributes.EXPECTED_VALUE):
            self._logger.debug('Expected values available on explainer')
            expected_values = np.array(self.explainer.expected_value)
        classification = isinstance(shap_values, list)
        if str(type(self.model)).endswith("XGBClassifier'>") and not classification:
            # workaround for XGBoost binary classifier output from SHAP
            classification = True
            shap_values = np.array((-shap_values, shap_values))
            expected_values = np.array((-expected_values, expected_values))
        if classification and self._shap_values_output == ShapValuesOutput.PROBABILITY:
            # Re-scale shap values to probabilities for classification case
            shap_values = _scale_tree_shap(shap_values, expected_values,
                                           self.model.predict_proba(typed_wrapper_func(evaluation_examples)))
        # Reformat shap values result if explain_subset specified
        if self.explain_subset:
            self._logger.debug('Getting subset of shap_values')
            if classification:
                for i in range(shap_values.shape[0]):
                    shap_values[i] = shap_values[i][:, self.explain_subset]
            else:
                shap_values = shap_values[:, self.explain_subset]
        local_importance_values = _convert_to_list(shap_values)
        if classification:
            kwargs[ExplainParams.MODEL_TASK] = ExplainType.CLASSIFICATION
        else:
            kwargs[ExplainParams.MODEL_TASK] = ExplainType.REGRESSION
        kwargs[ExplainParams.MODEL_TYPE] = str(type(self.model))
        kwargs[ExplainParams.LOCAL_IMPORTANCE_VALUES] = np.array(local_importance_values)
        kwargs[ExplainParams.EXPECTED_VALUES] = expected_values
        kwargs[ExplainParams.CLASSIFICATION] = classification
        kwargs[ExplainParams.EVAL_DATA] = original_evals
        if len(evaluation_examples.shape) == 1:
            evaluation_examples = evaluation_examples.reshape(1, -1)
        kwargs[ExplainParams.EVAL_Y_PRED] = self.model.predict(typed_wrapper_func(evaluation_examples))
        if hasattr(self.model, 'predict_proba'):
            kwargs[ExplainParams.EVAL_Y_PRED_PROBA] = self.model.predict_proba(typed_wrapper_func(evaluation_examples))
        return kwargs

    @tabular_decorator
    def explain_local(self, evaluation_examples):
        """Explain the model by using shap's tree explainer.

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
