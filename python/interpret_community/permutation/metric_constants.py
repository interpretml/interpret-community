# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Defines metric constants for PFIExplainer."""

from enum import Enum


class MetricConstants(str, Enum):
    """The metric to use for PFIExplainer."""

    MEAN_ABSOLUTE_ERROR = 'mean_absolute_error'
    EXPLAINED_VARIANCE_SCORE = 'explained_variance_score'
    MEAN_SQUARED_ERROR = 'mean_squared_error'
    MEAN_SQUARED_LOG_ERROR = 'mean_squared_log_error'
    MEDIAN_ABSOLUTE_ERROR = 'median_absolute_error'
    R2_SCORE = 'r2_score'
    AVERAGE_PRECISION_SCORE = 'average_precision_score'
    F1_SCORE = 'f1_score'
    FBETA_SCORE = 'fbeta_score'
    PRECISION_SCORE = 'precision_score'
    RECALL_SCORE = 'recall_score'


# Note: Error metrics are those for which a higher score is worse, not better
# These should be given a negative value when computing the feature importance
error_metrics = {MetricConstants.MEAN_ABSOLUTE_ERROR, MetricConstants.MEAN_SQUARED_ERROR,
                 MetricConstants.MEAN_SQUARED_LOG_ERROR, MetricConstants.MEDIAN_ABSOLUTE_ERROR}
