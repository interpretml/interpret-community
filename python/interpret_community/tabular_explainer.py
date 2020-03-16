# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Defines the tabular explainer meta-api for returning the best explanation result based on the given model."""

from .common.base_explainer import BaseExplainer
from .common.structured_model_explainer import PureStructuredModelExplainer
from .dataset.decorator import tabular_decorator
from .common.constants import ExplainParams, Defaults, ModelTask, Extension
from .shap.tree_explainer import TreeExplainer
from .shap.deep_explainer import DeepExplainer
from .shap.kernel_explainer import KernelExplainer
from .shap.linear_explainer import LinearExplainer

InvalidExplainerErr = 'Could not find valid explainer to explain model'


class TabularExplainer(BaseExplainer):
    available_explanations = [Extension.GLOBAL, Extension.LOCAL]
    explainer_type = Extension.BLACKBOX

    """The tabular explainer meta-api for returning the best explanation result based on the given model.

    :param model: The model or pipeline to explain.
    :type model: model that implements sklearn.predict() or sklearn.predict_proba() or pipeline function that accepts
        a 2d ndarray
    :param initialization_examples: A matrix of feature vector examples (# examples x # features) for
        initializing the explainer.
    :type initialization_examples: numpy.array or pandas.DataFrame or iml.datatypes.DenseData or
        scipy.sparse.csr_matrix
    :param explain_subset: List of feature indices. If specified, only selects a subset of the
        features in the evaluation dataset for explanation, which will speed up the explanation
        process when number of features is large and the user already knows the set of interested
        features. The subset can be the top-k features from the model summary. This argument is not supported when
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

        If the user is using a transformation that is not in the list of sklearn.preprocessing transformations
        that are supported by the `interpret-community <https://github.com/interpretml/interpret-community>`_
        package, then this parameter cannot take a list of more than one column as input for the transformation.
        A user can use the following sklearn.preprocessing  transformations with a list of columns since these are
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

    def __init__(self, model, initialization_examples, explain_subset=None, features=None, classes=None,
                 transformations=None, allow_all_transformations=False, model_task=ModelTask.Unknown, **kwargs):
        """Initialize the TabularExplainer.

        :param model: The model or pipeline to explain.
        :type model: model that implements sklearn.predict or sklearn.predict_proba or pipeline function that accepts
            a 2d ndarray
        :param initialization_examples: A matrix of feature vector examples (# examples x # features) for
            initializing the explainer.
        :type initialization_examples: numpy.array or pandas.DataFrame or iml.datatypes.DenseData or
            scipy.sparse.csr_matrix
        :param explain_subset: List of feature indices. If specified, only selects a subset of the
            features in the evaluation dataset for explanation, which will speed up the explanation
            process when number of features is large and the user already knows the set of interested
            features. The subset can be the top-k features from the model summary. This argument is not supported when
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

            If the user is using a transformation that is not in the list of sklearn.preprocessing transformations
            that are supported by the `interpret-community <https://github.com/interpretml/interpret-community>`_
            package, then this parameter cannot take a list of more than one column as input for the transformation.
            A user can use the following sklearn.preprocessing  transformations with a list of columns since these are
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
        :param model_task: Optional parameter to specify whether the model is a classification or regression model.
            In most cases, the type of the model can be inferred based on the shape of the output, where a classifier
            has a predict_proba method and outputs a 2 dimensional array, while a regressor has a predict method and
            outputs a 1 dimensional array.
        :type model_task: str
        """
        super(TabularExplainer, self).__init__(**kwargs)
        self._logger.debug('Initializing TabularExplainer')

        if transformations is not None and explain_subset is not None:
            raise ValueError("explain_subset not supported with non-None transformations")

        self.model = model
        self.features = features
        self.classes = classes
        self.explain_subset = explain_subset
        self.transformations = transformations
        kwargs[ExplainParams.EXPLAIN_SUBSET] = self.explain_subset
        kwargs[ExplainParams.FEATURES] = features
        kwargs[ExplainParams.CLASSES] = classes
        uninitialized_explainers = self._get_uninitialized_explainers()
        is_valid = False
        last_exception = None
        for uninitialized_explainer in uninitialized_explainers:
            try:
                if issubclass(uninitialized_explainer, PureStructuredModelExplainer):
                    self.explainer = uninitialized_explainer(
                        self.model, transformations=transformations,
                        allow_all_transformations=allow_all_transformations, **kwargs)
                else:
                    # Note: Unlike DeepExplainer, LinearExplainer does not need model_task
                    if uninitialized_explainer != LinearExplainer:
                        kwargs[ExplainParams.MODEL_TASK] = model_task
                    else:
                        kwargs.pop(ExplainParams.MODEL_TASK, None)
                    self.explainer = uninitialized_explainer(
                        self.model, initialization_examples, transformations=transformations,
                        allow_all_transformations=allow_all_transformations,
                        **kwargs)
                self._method = self.explainer._method
                self._logger.info('Initialized valid explainer {} with args {}'.format(self.explainer, kwargs))
                is_valid = True
                break
            except Exception as ex:
                last_exception = ex
                self._logger.info('Failed to initialize explainer {} due to error: {}'
                                  .format(uninitialized_explainer, ex))
        if not is_valid:
            self._logger.info(InvalidExplainerErr)
            raise ValueError(InvalidExplainerErr) from last_exception

    def _get_uninitialized_explainers(self):
        """Return the uninitialized explainers used by the tabular explainer.

        :return: A list of the uninitialized explainers.
        :rtype: list
        """
        return [TreeExplainer, DeepExplainer, LinearExplainer, KernelExplainer]

    @tabular_decorator
    def explain_global(self, evaluation_examples, sampling_policy=None, include_local=True,
                       batch_size=Defaults.DEFAULT_BATCH_SIZE):
        """Globally explains the black box model or function.

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
        :return: A model explanation object. It is guaranteed to be a GlobalExplanation. If SHAP is used for the
            explanation, it will also have the properties of a LocalExplanation and the ExpectedValuesMixin. If the
            model does classification, it will have the properties of the PerClassMixin.
        :rtype: DynamicGlobalExplanation
        """
        kwargs = {ExplainParams.SAMPLING_POLICY: sampling_policy,
                  ExplainParams.INCLUDE_LOCAL: include_local,
                  ExplainParams.BATCH_SIZE: batch_size}
        return self.explainer.explain_global(evaluation_examples, **kwargs)

    @tabular_decorator
    def explain_local(self, evaluation_examples):
        """Locally explains the black box model or function.

        :param evaluation_examples: A matrix of feature vector examples (# examples x # features) on which
            to explain the model's output.
        :type evaluation_examples: numpy.array or pandas.DataFrame or scipy.sparse.csr_matrix
        :return: A model explanation object. It is guaranteed to be a LocalExplanation. If SHAP is used for the
            explanation, it will also have the properties of the ExpectedValuesMixin. If the model does
            classification, it will have the properties of the ClassesMixin.
        :rtype: DynamicLocalExplanation
        """
        return self.explainer.explain_local(evaluation_examples)
