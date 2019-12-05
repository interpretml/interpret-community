# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Defines the Explanation dashboard class."""

from .ExplanationWidget import ExplanationWidget
from ._internal.constants import ExplanationDashboardInterface, WidgetRequestResponseConstants
from IPython.display import display
from scipy.sparse import issparse
import numpy as np
import pandas as pd


class ExplanationDashboard(object):
    """The dashboard class, wraps the dashboard component."""

    def __init__(self, explanationObject, model=None, *, datasetX=None, trueY=None, classes=None, features=None):
        """Initialize the Explanation Dashboard.

        :param explanationObject: An object that represents an explanation.
        :type explanationObject: ExplanationMixin
        :param model: An object that represents a model. It is assumed that for the classification case
            it has a method of predict_proba() returning the prediction probabilities for each
            class and for the regression case a method of predict() returning the prediction value.
        :type model: object
        :param datasetX:  A matrix of feature vector examples (# examples x # features), the same samples
            used to build the explanationObject. Will overwrite any set on explanation object already
        :type datasetX: numpy.array or list[][]
        :param trueY: The true labels for the provided dataset. Will overwrite any set on
            explanation object already
        :type trueY: numpy.array or list[]
        :param classes: The class names
        :type classes: numpy.array or list[]
        :param features: Feature names
        :type features: numpy.array or list[]
        """
        self._widget_instance = ExplanationWidget()
        self._model = model
        self._is_classifier = model is not None and hasattr(model, 'predict_proba') and \
            model.predict_proba is not None
        self._dataframeColumns = None
        dataArg = {}

        # List of explanations, key of explanation type is "explanation_type"
        self._mli_explanations = explanationObject.data(-1)["mli"]
        local_explanation = self._find_first_explanation(ExplanationDashboardInterface.MLI_LOCAL_EXPLANATION_KEY)
        global_explanation = self._find_first_explanation(ExplanationDashboardInterface.MLI_GLOBAL_EXPLANATION_KEY)
        ebm_explanation = self._find_first_explanation(ExplanationDashboardInterface.MLI_EBM_GLOBAL_EXPLANATION_KEY)
        dataset_explanation = self._find_first_explanation(ExplanationDashboardInterface.MLI_EXPLANATION_DATASET_KEY)

        predicted_y = None
        feature_length = None
        if dataset_explanation is not None:
            if datasetX is None:
                datasetX = dataset_explanation[ExplanationDashboardInterface.MLI_DATASET_X_KEY]
            if trueY is None:
                trueY = dataset_explanation[ExplanationDashboardInterface.MLI_DATASET_Y_KEY]

        if isinstance(datasetX, pd.DataFrame) and hasattr(datasetX, 'columns'):
            self._dataframeColumns = datasetX.columns
        try:
            list_dataset = self._convertToList(datasetX)
        except Exception:
            raise ValueError("Unsupported dataset type")
        if datasetX is not None and model is not None:
            try:
                predicted_y = model.predict(datasetX)
            except Exception:
                raise ValueError("Model does not support predict method for given dataset type")
            try:
                predicted_y = self._convertToList(predicted_y)
            except Exception:
                raise ValueError("Model prediction output of unsupported type")
        if predicted_y is not None:
            dataArg[ExplanationDashboardInterface.PREDICTED_Y] = predicted_y
        if list_dataset is not None:
            row_length, feature_length = np.shape(list_dataset)
            if row_length > 100000:
                raise ValueError("Exceeds maximum number of rows for visualization (100000)")
            if feature_length > 1000:
                raise ValueError("Exceeds maximum number of features for visualization (1000)")
            dataArg[ExplanationDashboardInterface.TRAINING_DATA] = list_dataset
            dataArg[ExplanationDashboardInterface.IS_CLASSIFIER] = self._is_classifier

        local_dim = None

        if trueY is not None and len(trueY) == row_length:
            dataArg[ExplanationDashboardInterface.TRUE_Y] = trueY

        if local_explanation is not None:
            try:
                local_explanation["scores"] = self._convertToList(local_explanation["scores"])
                local_explanation["intercept"] = self._convertToList(local_explanation["intercept"])
                dataArg[ExplanationDashboardInterface.LOCAL_EXPLANATIONS] = local_explanation
            except Exception:
                raise ValueError("Unsupported local explanation type")
            if list_dataset is not None:
                local_dim = np.shape(local_explanation["scores"])
                if len(local_dim) != 2 and len(local_dim) != 3:
                    raise ValueError("Local explanation expected to be a 2D or 3D list")
                if len(local_dim) == 2 and (local_dim[1] != feature_length or local_dim[0] != row_length):
                    raise ValueError("Shape mismatch: local explanation length differs from dataset")
                if len(local_dim) == 3 and (local_dim[2] != feature_length or local_dim[1] != row_length):
                    raise ValueError("Shape mismatch: local explanation length differs from dataset")
        if local_explanation is None and global_explanation is not None:
            try:
                global_explanation["scores"] = self._convertToList(global_explanation["scores"])
                if 'intercept' in global_explanation:
                    global_explanation["intercept"] = self._convertToList(global_explanation["intercept"])
                dataArg[ExplanationDashboardInterface.GLOBAL_EXPLANATION] = global_explanation
            except Exception:
                raise ValueError("Unsupported global explanation type")
        if ebm_explanation is not None:
            try:
                dataArg[ExplanationDashboardInterface.EBM_EXPLANATION] = ebm_explanation
            except Exception:
                raise ValueError("Unsupported ebm explanation type")

        if features is None and hasattr(explanationObject, 'features') and explanationObject.features is not None:
            features = explanationObject.features
        if features is not None:
            features = self._convertToList(features)
            if feature_length is not None and len(features) != feature_length:
                raise ValueError("Feature vector length mismatch: \
                    feature names length differs from local explanations dimension")
            dataArg[ExplanationDashboardInterface.FEATURE_NAMES] = features
        if classes is None and hasattr(explanationObject, 'classes') and explanationObject.classes is not None:
            classes = explanationObject.classes
        if classes is not None:
            classes = self._convertToList(classes)
            if local_dim is not None and len(classes) != local_dim[0]:
                raise ValueError("Class vector length mismatch: \
                    class names length differs from local explanations dimension")
            dataArg[ExplanationDashboardInterface.CLASS_NAMES] = classes
        if model is not None and hasattr(model, 'predict_proba') \
           and model.predict_proba is not None and datasetX is not None:
            try:
                probability_y = model.predict_proba(datasetX)
            except Exception:
                raise ValueError("Model does not support predict_proba method for given dataset type")
            try:
                probability_y = self._convertToList(probability_y)
            except Exception:
                raise ValueError("Model predict_proba output of unsupported type")
            dataArg[ExplanationDashboardInterface.PROBABILITY_Y] = probability_y
        dataArg[ExplanationDashboardInterface.HAS_MODEL] = model is not None
        self._widget_instance.value = dataArg
        self._widget_instance.observe(self._on_request, names=WidgetRequestResponseConstants.REQUEST)
        display(self._widget_instance)

    def _on_request(self, change):
        try:
            data = change.new[WidgetRequestResponseConstants.DATA]
            if self._dataframeColumns is not None:
                data = pd.DataFrame(data, columns=self._dataframeColumns)
            if (self._is_classifier):
                prediction = self._convertToList(self._model.predict_proba(data))
            else:
                prediction = self._convertToList(self._model.predict(data))
            self._widget_instance.response = {
                WidgetRequestResponseConstants.DATA: prediction,
                WidgetRequestResponseConstants.ID: change.new[WidgetRequestResponseConstants.ID]}
        except Exception:
            self._widget_instance.response = {
                WidgetRequestResponseConstants.ERROR: "Model threw exeption while predicting",
                WidgetRequestResponseConstants.DATA: [],
                WidgetRequestResponseConstants.ID: change.new[WidgetRequestResponseConstants.ID]}

    def _show(self):
        display(self._widget_instance)

    def _convertToList(self, array):
        if issparse(array):
            if array.shape[1] > 1000:
                raise ValueError("Exceeds maximum number of features for visualization (1000)")
            return array.toarray().tolist()
        if (isinstance(array, pd.DataFrame)):
            return array.values.tolist()
        if (isinstance(array, np.ndarray)):
            return array.tolist()
        return array

    def _find_first_explanation(self, key):
        new_array = [explanation for explanation
                     in self._mli_explanations
                     if explanation[ExplanationDashboardInterface.MLI_EXPLANATION_TYPE_KEY] == key]
        if len(new_array) > 0:
            return new_array[0]["value"]
        return None
