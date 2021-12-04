# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Defines an adapter for creating an interpret-community style explanation from other frameworks."""

import numpy as np
from interpret_community.common.constants import (Defaults, ExplainParams,
                                                  ExplainType, ModelTask)
from interpret_community.explanation.explanation import (
    ExpectedValuesMixin, _aggregate_global_from_local_explanation,
    _aggregate_streamed_local_explanations, _create_global_explanation,
    _create_local_explanation)


class ExplanationAdapter(object):
    """An adapter for creating an interpret-community explanation from local importance values.

    :param features: A list of feature names.
    :type features: list[str]
    :param classification: Indicates if this is a classification or regression explanation.
    :type classification: bool
    :param method: The explanation method used to explain the model (e.g., SHAP, LIME).
    :type method: str
    """

    def __init__(self, features=None, classification=False, method='Adapter'):
        """Create the explanation adapter.

        :param features: A list of feature names.
        :type features: list[str]
        :param classification: Indicates if this is a classification or regression explanation.
        :type classification: bool
        :param method: The explanation method used to explain the model (e.g., SHAP, LIME).
        :type method: str
        """
        self.classification = classification
        self.features = features
        self.method = method

    def create_local(self, local_importance_values, evaluation_examples=None, expected_values=None):
        """Create a local explanation from the list of local feature importance values.

        :param local_importance_values: The feature importance values.
        :type local_importance_values: numpy.array or scipy.sparse.csr_matrix or list[scipy.sparse.csr_matrix]
        :param evaluation_examples: A matrix of feature vector examples (# examples x # features) on which
            to explain the model's output.
        :type evaluation_examples: numpy.array or pandas.DataFrame or scipy.sparse.csr_matrix
        :param expected_values: The expected values of the model.
        :type expected_values: numpy.array
        """
        local_importance_values = np.array(local_importance_values)
        # handle the case that the local importance values have a 2d shape for classification scenario
        # and only specify the positive class
        if len(local_importance_values.shape) == 2 and self.classification:
            local_importance_values = np.array([-local_importance_values, local_importance_values])

        kwargs = {ExplainParams.METHOD: self.method}
        kwargs[ExplainParams.FEATURES] = self.features
        if self.classification:
            kwargs[ExplainParams.MODEL_TASK] = ExplainType.CLASSIFICATION
        else:
            kwargs[ExplainParams.MODEL_TASK] = ExplainType.REGRESSION
        kwargs[ExplainParams.LOCAL_IMPORTANCE_VALUES] = local_importance_values
        kwargs[ExplainParams.EXPECTED_VALUES] = expected_values
        kwargs[ExplainParams.CLASSIFICATION] = self.classification
        if evaluation_examples is not None:
            kwargs[ExplainParams.EVAL_DATA] = evaluation_examples
        return _create_local_explanation(**kwargs)

    def create_global(self, local_importance_values, evaluation_examples=None, expected_values=None,
                      include_local=True, batch_size=Defaults.DEFAULT_BATCH_SIZE):
        """Create a global explanation from the list of local feature importance values.

        :param local_importance_values: The feature importance values.
        :type local_importance_values: numpy.array or scipy.sparse.csr_matrix or list[scipy.sparse.csr_matrix]
        :param evaluation_examples: A matrix of feature vector examples (# examples x # features) on which
            to explain the model's output.
        :type evaluation_examples: numpy.array or pandas.DataFrame or scipy.sparse.csr_matrix
        :param expected_values: The expected values of the model.
        :type expected_values: numpy.array
        :param include_local: Include the local explanations in the returned global explanation.
            If include_local is False, will stream the local explanations to aggregate to global.
        :type include_local: bool
        :param batch_size: If include_local is False, specifies the batch size for aggregating
            local explanations to global.
        :type batch_size: int
        """
        local_explanation = self.create_local(local_importance_values, evaluation_examples, expected_values)
        kwargs = {ExplainParams.METHOD: self.method}
        kwargs[ExplainParams.FEATURES] = self.features
        if ExpectedValuesMixin._does_quack(local_explanation):
            kwargs[ExplainParams.EXPECTED_VALUES] = local_explanation.expected_values
        if include_local:
            kwargs[ExplainParams.LOCAL_EXPLANATION] = local_explanation
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
