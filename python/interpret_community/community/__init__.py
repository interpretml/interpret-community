# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Module for model interpretability, including feature and class importances for blackbox and greybox models."""

from .tabular_explainer import TabularExplainer

__all__ = ["TabularExplainer"]
