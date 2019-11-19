# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Defines the base API for explainable models."""

from abc import ABCMeta, abstractmethod
from ...common.chained_identity import ChainedIdentity


def _get_initializer_args(surrogate_init_args):
    """Return a list of args to default values for the given function that are in the given argument dict.

    :param function: The function to retrieve the arguments from.
    :type function: Function
    :param surrogate_init_args: The arguments to initialize the surrogate model.
    :type surrogate_init_args: dict
    :return: A mapping from argument name to value for the surrogate model.
    :rtype: dict
    """
    # List of known args that child and base explainer are known to support
    base_model_args = ['_ident', '_parent_logger']
    # Pass all other args to underlying model
    return dict([(arg, surrogate_init_args.pop(arg))
                 for arg in surrogate_init_args.copy()
                 if arg not in base_model_args])


class BaseGlassboxModel(ChainedIdentity):
    """The base class for glassbox models."""

    __metaclass__ = ABCMeta

    def __init__(self, **kwargs):
        """Initialize the glassbox model."""
        super(BaseGlassboxModel, self).__init__(**kwargs)
        self._logger.debug('Initializing BaseGlassboxModel')

    @abstractmethod
    def fit(self, **kwargs):
        """Abstract method to fit the glassbox model."""
        pass

    @abstractmethod
    def explain_global(self, **kwargs):
        """Abstract method to get the global feature importances from the trained glassbox model."""
        pass

    @abstractmethod
    def explain_local(self, evaluation_examples, **kwargs):
        """Abstract method to get the local feature importances from the trained glassbox model."""
        pass

    @property
    @abstractmethod
    def model(self):
        """Abstract property to get the underlying model."""
        pass

    @staticmethod
    def explainable_model_type(self):
        """Retrieve the model type."""
        pass


class BaseGlassboxClassifier(BaseGlassboxModel):
    """The base class for glassbox classifiers."""

    __metaclass__ = ABCMeta

    def __init__(self, **kwargs):
        """Initialize the glassbox classifier."""
        super(BaseGlassboxClassifier, self).__init__(**kwargs)
        self._logger.debug('Initializing BaseGlassboxClassifier')

    @abstractmethod
    def predict(self, dataset, **kwargs):
        """Abstract method to predict labels using the explainable classifier."""
        pass

    @abstractmethod
    def predict_proba(self, dataset, **kwargs):
        """Abstract method to predict probabilities using the explainable classifier."""
        pass


class BaseGlassboxRegressor(BaseGlassboxModel):
    """The base class for glassbox regressors."""

    __metaclass__ = ABCMeta

    def __init__(self, **kwargs):
        """Initialize the glassbox model."""
        super(BaseGlassboxRegressor, self).__init__(**kwargs)
        self._logger.debug('Initializing BaseGlassboxRegressor')

    @abstractmethod
    def predict(self, dataset, **kwargs):
        """Abstract method to predict labels using the explainable regressor."""
        pass


class BaseExplainableModel(BaseGlassboxClassifier, BaseGlassboxRegressor):
    """The base class for models that can be explained as surrogates."""

    __metaclass__ = ABCMeta

    def __init__(self, **kwargs):
        """Initialize the Explainable Model."""
        super(BaseExplainableModel, self).__init__(**kwargs)
        self._logger.debug('Initializing BaseExplainableModel')

    @property
    @abstractmethod
    def expected_values(self):
        """Abstract property to get the expected values."""
        pass
