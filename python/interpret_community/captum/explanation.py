# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Defines the interface to create an interpret-community style explanation from Captum."""

import numpy as np

from ..explanation.explanation import _create_local_explanation, _create_global_explanation, \
    _aggregate_global_from_local_explanation, _aggregate_streamed_local_explanations
from ..common.constants import ExplainParams, ExplainType, Defaults, ModelTask


class CaptumAdapter(object):
    def __init__(self, features=None, classification=False):
        self.classification = classification
        self.features = features

    def create_local(self, local_importance_values, evaluation_examples=None):
        kwargs = {ExplainParams.METHOD: ExplainType.CAPTUM}
        kwargs[ExplainParams.FEATURES] = self.features
        if self.classification:
            kwargs[ExplainParams.MODEL_TASK] = ExplainType.CLASSIFICATION
        else:
            kwargs[ExplainParams.MODEL_TASK] = ExplainType.REGRESSION
        kwargs[ExplainParams.LOCAL_IMPORTANCE_VALUES] = np.array(local_importance_values)
        kwargs[ExplainParams.EXPECTED_VALUES] = 0
        kwargs[ExplainParams.CLASSIFICATION] = self.classification
        if evaluation_examples is not None:
            kwargs[ExplainParams.EVAL_DATA] = evaluation_examples
        return _create_local_explanation(**kwargs)

    def create_global(self, local_importance_values, evaluation_examples=None, include_local=True, batch_size=Defaults.DEFAULT_BATCH_SIZE):
        local_explanation = self.create_local(local_importance_values, evaluation_examples)
        kwargs = {ExplainParams.METHOD: ExplainType.CAPTUM}
        kwargs[ExplainParams.FEATURES] = self.features
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

        return _create_global_explanation(local_explanation=local_explanation, classification=self.classification, **kwargs)
