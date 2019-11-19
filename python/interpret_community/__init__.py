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
import atexit
# Only log to disk if environment variable specified
interpretc_logs = os.environ.get('INTERPRETC_LOGS')
if interpretc_logs is not None:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    os.makedirs(os.path.dirname(interpretc_logs), exist_ok=True)
    handler = logging.FileHandler(interpretc_logs, mode='w')
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.info('Initializing logging file for interpret-community')
    def close_handler():
        handler.close()
        logger.removeHandler(handler)
    atexit.register(close_handler)
