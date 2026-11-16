# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Select a trained MimicExplainer using held-out surrogate fidelity."""

import logging

import numpy as np
import pandas as pd
from ml_wrappers import DatasetWrapper

from .mimic_explainer import MimicExplainer
from .models import (BaseExplainableModel, DecisionTreeExplainableModel,
                     LGBMExplainableModel, LinearExplainableModel,
                     SGDExplainableModel)

_logger = logging.getLogger(__name__)
_DEFAULT_CANDIDATES = (LGBMExplainableModel, LinearExplainableModel,
                       SGDExplainableModel, DecisionTreeExplainableModel)


def _raw_examples(examples):
    if isinstance(examples, DatasetWrapper):
        return examples.original_dataset_with_type
    return examples


def select_mimic_explainer(model, initialization_examples, evaluation_examples,
                           explainable_model=LGBMExplainableModel, explainable_model_args=None,
                           candidates=None, candidate_args=None, **kwargs):
    """Train candidate surrogates and return the most faithful existing MimicExplainer.

    Selection is separate from MimicExplainer's single-surrogate training. Each candidate
    is trained on initialization_examples only, using its own fitted preprocessing.
    Provide independent held-out evaluation_examples in the same raw feature schema;
    evaluating on training data favors overfitted surrogates.

    Classification scores compare probability-argmax class indices, the targets used
    for distillation, rather than display names or the teacher's custom predict threshold.
    Regression scores use sklearn's R2. Scoring reuses get_surrogate_model_replication_measure.
    Only finite scores compete; ties go to the first candidate. Failed candidates are
    logged and receive None. If every score is None, a new explainer using the caller's
    explainable_model and explainable_model_args is returned without scoring it.
    Failure to construct that fallback propagates to the caller.

    This performs a complete MimicExplainer construction, including teacher queries,
    for each candidate (and again for fallback if needed). The winning instance is
    returned without retraining; losing instances are not retained.

    :param model: Teacher model, or function when is_function=True is passed in kwargs.
    :param initialization_examples: Training ndarray, DataFrame, sparse matrix or DatasetWrapper.
    :param evaluation_examples: Held-out examples with at least two rows and matching columns.
    :param explainable_model: Surrogate class used only as fallback. Defaults to LightGBM.
    :param explainable_model_args: Constructor arguments used only for the fallback.
    :param candidates: Ordered iterable of distinct BaseExplainableModel subclasses.
        Defaults to LightGBM, linear, SGD and decision tree, in that order.
    :param candidate_args: Optional mapping from candidate class to its constructor arguments.
        Fallback arguments are never broadcast to the candidates.
    :param kwargs: Other MimicExplainer constructor arguments, shared by every construction.
    :return: (trained MimicExplainer, {candidate_class: float or None}).
        All-None scores indicate fallback. Inputs and model argument dictionaries are copied.
    :rtype: tuple
    """
    initialization_examples = _raw_examples(initialization_examples)
    evaluation_examples = _raw_examples(evaluation_examples)
    for name, examples in (('initialization_examples', initialization_examples),
                           ('evaluation_examples', evaluation_examples)):
        if not hasattr(examples, 'shape') or len(examples.shape) != 2 or examples.shape[1] == 0:
            raise ValueError('{} must be a two-dimensional feature matrix.'.format(name))
    if initialization_examples.shape[0] == 0 or evaluation_examples.shape[0] < 2:
        raise ValueError('Initialization data must be nonempty and evaluation data must have at least two rows.')
    if initialization_examples.shape[1] != evaluation_examples.shape[1]:
        raise ValueError('Initialization and evaluation data must have the same number of columns.')
    if isinstance(initialization_examples, pd.DataFrame) and isinstance(evaluation_examples, pd.DataFrame):
        if not initialization_examples.columns.equals(evaluation_examples.columns):
            raise ValueError('Initialization and evaluation DataFrame columns must match in the same order.')
    candidates = tuple(_DEFAULT_CANDIDATES if candidates is None else candidates)
    if not candidates or any(not isinstance(cls, type) or not issubclass(cls, BaseExplainableModel)
                             for cls in candidates):
        raise ValueError('Candidates must be a nonempty iterable of BaseExplainableModel subclasses.')
    if len(set(candidates)) != len(candidates):
        raise ValueError('Candidate classes must be distinct.')
    if not isinstance(explainable_model, type) or not issubclass(explainable_model, BaseExplainableModel):
        raise ValueError('The fallback must be a BaseExplainableModel subclass.')
    candidate_args = {} if candidate_args is None else candidate_args
    if any(cls not in candidates for cls in candidate_args):
        raise ValueError('candidate_args contains a class not present in candidates.')
    parameters = {cls: dict(candidate_args.get(cls, {})) for cls in candidates}
    fallback_args = dict(explainable_model_args or {})
    best_explainer = None
    best_score = None
    scores = {}
    for candidate in candidates:
        try:
            explainer = MimicExplainer(
                model, initialization_examples.copy(), candidate,
                explainable_model_args=parameters[candidate], **kwargs)
            score = explainer.get_surrogate_model_replication_measure(evaluation_examples)
            if not np.isfinite(score):
                _logger.warning('Surrogate %s produced a non-finite fidelity score.', candidate.__name__)
                score = None
        except Exception:
            _logger.warning('Unable to train or score surrogate %s.', candidate.__name__, exc_info=True)
            score = None
        scores[candidate] = score
        if score is not None and (best_score is None or score > best_score):
            best_explainer, best_score = explainer, score
    if best_explainer is None:
        best_explainer = MimicExplainer(
            model, initialization_examples.copy(), explainable_model,
            explainable_model_args=fallback_args, **kwargs)
    return best_explainer, scores
