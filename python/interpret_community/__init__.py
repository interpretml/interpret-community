# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Module for interpreting, including feature and class importance for blackbox, greybox and glassbox models.

You can use model interpretability to explain why a model model makes the predictions it does and help build
confidence in the model.
"""

from .tabular_explainer import TabularExplainer

__all__ = ["TabularExplainer"]

# Setup logging infrustructure
import logging
import os
# Only log to disk if environment variable specified
interpretc_logs = os.environ.get('INTERPRETC_LOGS')
if interpretc_logs is not None:
    logging.basicConfig(filename=interpretc_logs, level=logging.INFO)
    logging.info('Initializing logging file for interpret-community')
