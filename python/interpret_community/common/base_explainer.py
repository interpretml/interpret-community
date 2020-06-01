# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Defines the base explainer API to create explanations."""

from abc import ABCMeta, abstractmethod
from .chained_identity import ChainedIdentity


class GlobalExplainer(ChainedIdentity):
    """The base class for explainers that create global explanations."""

    __metaclass__ = ABCMeta

    def __init__(self, *args, **kwargs):
        """Initialize the GlobalExplainer."""
        super(GlobalExplainer, self).__init__(*args, **kwargs)
        self._logger.debug('Initializing GlobalExplainer')

    @abstractmethod
    def explain_global(self, *args, **kwargs):
        """Abstract method to globally explain the given model.

        Note evaluation examples can be optional on derived classes since some explainers
        don't support it, for example MimicExplainer.

        :return: A model explanation object containing the global explanation.
        :rtype: GlobalExplanation
        """
        pass

    def __str__(self):
        """Get string representation of the explainer.

        :return: A string containing explainer name.
        :rtype: str
        """
        return "{}".format(self.__class__.__name__)


class LocalExplainer(ChainedIdentity):
    """The base class for explainers that create local explanations."""

    __metaclass__ = ABCMeta

    def __init__(self, *args, **kwargs):
        """Initialize the LocalExplainer."""
        super(LocalExplainer, self).__init__(*args, **kwargs)
        self._logger.debug('Initializing LocalExplainer')

    @abstractmethod
    def explain_local(self, evaluation_examples, **kwargs):
        """Abstract method to explain local instances.

        :param evaluation_examples: The evaluation examples.
        :type evaluation_examples: object
        :return: A model explanation object containing the local explanation.
        :rtype: LocalExplanation
        """
        pass

    def __str__(self):
        """Get string representation of the explainer.

        :return: A string containing explainer name.
        :rtype: str
        """
        return "{}".format(self.__class__.__name__)


class BaseExplainer(GlobalExplainer, LocalExplainer):
    """The base class for explainers that create global and local explanations."""

    __metaclass__ = ABCMeta

    def __init__(self, *args, **kwargs):
        """Initialize the BaseExplainer."""
        super(BaseExplainer, self).__init__(*args, **kwargs)
        self._logger.debug('Initializing BaseExplainer')
