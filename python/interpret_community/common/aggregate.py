# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Defines the aggregate explainer decorator for aggregating local explanations to global."""

from functools import wraps
from ..explanation.explanation import _aggregate_global_from_local_explanation, _create_global_explanation, \
    _aggregate_streamed_local_explanations
from .constants import ExplainParams, ModelTask, Defaults, ExplainType
from ..dataset.dataset_wrapper import DatasetWrapper
from .explanation_utils import _order_imp


def init_aggregator_decorator(init_func):
    """Decorate a constructor to wrap initialization examples in a DatasetWrapper.

    Provided for convenience for tabular data explainers.

    :param init_func: Initialization constructor where the second argument is a dataset.
    :type init_func: Initialization constructor.
    """
    @wraps(init_func)
    def init_wrapper(self, model, *args, **kwargs):
        self.sampling_policy = None
        return init_func(self, model, *args, **kwargs)
    return init_wrapper


def add_explain_global_method(cls):
    """Decorate an explainer to allow aggregating local explanations to global.

    Adds a protected method _explain_global that creates local explanations
    and then aggregates them to a global explanation by averaging.
    """
    def _get_explain_global_agg_kwargs(self, evaluation_examples, sampling_policy=None, **kwargs):
        """Create the arguments for aggregating local explanations to global.

        :param evaluation_examples: A matrix of feature vector examples (# examples x # features) on which
            to explain the model's output.
        :type evaluation_examples: numpy.array or pandas.DataFrame or scipy.sparse.csr_matrix
        :param sampling_policy: Optional policy for sampling the evaluation examples.  See documentation on
            SamplingPolicy for more information.
        :type sampling_policy: SamplingPolicy
        :return: Arguments for aggregating local to global.
        :rtype: dict
        """
        self.sampling_policy = sampling_policy
        if self.classes is not None:
            kwargs[ExplainParams.CLASSES] = self.classes
        # first get local explanation
        local_explanation = self.explain_local(evaluation_examples)
        kwargs[ExplainParams.LOCAL_EXPLANATION] = local_explanation
        return kwargs

    def _explain_global(self, evaluation_examples, sampling_policy=None, include_local=True,
                        batch_size=Defaults.DEFAULT_BATCH_SIZE, **kwargs):
        """Explains the model by aggregating local explanations to global.

        :param evaluation_examples: A matrix of feature vector examples (# examples x # features) on which
            to explain the model's output.
        :type evaluation_examples: numpy.array or pandas.DataFrame or scipy.sparse.csr_matrix
        :param sampling_policy: Optional policy for sampling the evaluation examples.  See documentation on
            SamplingPolicy for more information.
        :type sampling_policy: SamplingPolicy
        :param include_local: Include the local explanations in the returned global explanation.
            If evaluation examples are specified and include_local is False, will stream the local
            explanations to aggregate to global.
        :type include_local: bool
        :param batch_size: If include_local is False, specifies the batch size for aggregating
            local explanations to global.
        :type batch_size: int
        :return: A model explanation object containing the local and global explanation.
        :rtype: BaseExplanation
        """
        if include_local:
            kwargs = self._get_explain_global_agg_kwargs(evaluation_examples, sampling_policy=sampling_policy,
                                                         **kwargs)
            # Aggregate local explanation to global
            return _aggregate_global_from_local_explanation(**kwargs)
        else:
            if ExplainParams.CLASSIFICATION in kwargs:
                if kwargs[ExplainParams.CLASSIFICATION]:
                    model_task = ModelTask.Classification
                else:
                    model_task = ModelTask.Regression
            else:
                model_task = ModelTask.Unknown
            kwargs = _aggregate_streamed_local_explanations(self, evaluation_examples, model_task,
                                                            self.features, batch_size, **kwargs)
            return _create_global_explanation(**kwargs)
    setattr(cls, '_get_explain_global_agg_kwargs', _get_explain_global_agg_kwargs)
    setattr(cls, '_explain_global', _explain_global)
    return cls


def _get_explain_global_agg_kwargs(explainer, global_importance_values, classification,
                                   evaluation_examples=None, include_local=True,
                                   batch_size=Defaults.DEFAULT_BATCH_SIZE, model=None,
                                   features=None, classes=None, **kwargs):
    """Get the kwargs for explain_global to create a global explanation by aggregating from local.

    Handles the case where the evaluation examples may be optional for the global explanation.

    :param explainer: The explainer to generate the explanation from.
    :type explainer: BaseExplainer
    :param global_importance_values: Global importance values.
    :type global_importance_values: np.array
    :param classification: True if this is a classifier, otherwise regressor.
    :type classification: bool
    :param evaluation_examples: A matrix of feature vector examples (# examples x # features) on which to
        explain the model's output.  If specified, computes feature importances through aggregation.
    :type evaluation_examples: numpy.array or pandas.DataFrame or scipy.sparse.csr_matrix
    :param include_local: Include the local explanations in the returned global explanation.
        If evaluation examples are specified and include_local is False, will stream the local
        explanations to aggregate to global.
    :type include_local: bool
    :param batch_size: If include_local is False, specifies the batch size for aggregating
        local explanations to global.
    :type batch_size: int
    :param model: The model that is explained.  If none a function is explained.
    :type model: model
    :return: Args for explain_global.
    :rtype: dict
    """
    # Note kwargs must contain the method type
    if classification:
        kwargs[ExplainParams.CLASSES] = classes
    if evaluation_examples is not None:
        # Aggregate local explanation to global, either through computing the local
        # explanation and then aggregating or streaming the local explanation to global
        if include_local:
            # Get local explanation
            local_explanation = explainer.explain_local(evaluation_examples)
            kwargs[ExplainParams.LOCAL_EXPLANATION] = local_explanation
        else:
            if classification:
                model_task = ModelTask.Classification
            else:
                model_task = ModelTask.Regression
            if not isinstance(evaluation_examples, DatasetWrapper):
                explainer._logger.debug('Eval examples not wrapped, wrapping')
                evaluation_examples = DatasetWrapper(evaluation_examples)

            kwargs = _aggregate_streamed_local_explanations(explainer, evaluation_examples, model_task, features,
                                                            batch_size, **kwargs)
        return kwargs
    order = _order_imp(global_importance_values)
    if classification:
        kwargs[ExplainParams.MODEL_TASK] = ExplainType.CLASSIFICATION
    else:
        kwargs[ExplainParams.MODEL_TASK] = ExplainType.REGRESSION
    if model is not None:
        kwargs[ExplainParams.MODEL_TYPE] = str(type(model))
    else:
        kwargs[ExplainParams.MODEL_TYPE] = ExplainType.FUNCTION
    kwargs[ExplainParams.EXPECTED_VALUES] = None
    kwargs[ExplainParams.CLASSIFICATION] = classification
    kwargs[ExplainParams.GLOBAL_IMPORTANCE_VALUES] = global_importance_values
    kwargs[ExplainParams.GLOBAL_IMPORTANCE_RANK] = order
    kwargs[ExplainParams.FEATURES] = features
    return kwargs
