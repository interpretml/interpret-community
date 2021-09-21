# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Defines the DEPRECATED Explanation dashboard class."""

import warnings


class ExplanationDashboard:
    """DEPRECATED Explanation Dashboard class, please use the Explanation Dashboard from raiwidgets package instead.

    Since this class is deprecated it will no longer display the widget.
    Please install raiwidgets from pypi by running:
    pip install --upgrade raiwidgets
    The dashboard can be run with the same parameters in the new namespace:
    from raiwidgets import ExplanationDashboard

    :param explanation: An object that represents an explanation.
    :type explanation: ExplanationMixin
    :param model: An object that represents a model. It is assumed that for the classification case
        it has a method of predict_proba() returning the prediction probabilities for each
        class and for the regression case a method of predict() returning the prediction value.
    :type model: object
    :param dataset:  A matrix of feature vector examples (# examples x # features), the same samples
        used to build the explanation. Overwrites any existing dataset on the explanation object. Must have fewer than
        10000 rows and fewer than 1000 columns.
    :type dataset: numpy.array or list[][]
    :param datasetX: Alias of the dataset parameter. If dataset is passed, this will have no effect. Must have fewer
        than 10000 rows and fewer than 1000 columns.
    :type datasetX: numpy.array or list[][]
    :param true_y: The true labels for the provided dataset. Overwrites any existing dataset on the
        explanation object.
    :type true_y: numpy.array or list[]
    :param classes: The class names.
    :type classes: numpy.array or list[]
    :param features: Feature names.
    :type features: numpy.array or list[]
    :param port: The port to use on locally hosted service.
    :type port: int
    :param use_cdn: Deprecated. Whether to load latest dashboard script from cdn, fall back to local script if False.
        .. deprecated:: 0.15.2
           Deprecated since 0.15.2, cdn has been removed.  Setting parameter to True or False will trigger warning.
    :type use_cdn: bool
    :param public_ip: Optional. If running on a remote vm, the external public ip address of the VM.
    :type public_ip: str
    :param with_credentials: Optional. If running on a remote vm, sets up CORS policy both on client and server.
    :type with_credentials: bool
    """

    def __init__(self, explanation, model=None, *, dataset=None,
                 true_y=None, classes=None, features=None, port=None,
                 datasetX=None, trueY=None, locale=None, public_ip=None,
                 with_credentials=False, use_cdn=None):
        warnings.warn("ExplanationDashboard in interpret-community package is deprecated and removed."
                      "Please use the ExplanationDashboard from raiwidgets package instead.")
