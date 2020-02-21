# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Module for model interpretability, including feature and class importance for blackbox and greybox models."""

from interpret_community.tabular_explainer import TabularExplainer

__all__ = ["TabularExplainer"]
