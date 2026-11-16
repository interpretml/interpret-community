# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Module for mimic explainer and explainable surrogate models."""
from .mimic_explainer import MimicExplainer
from .model_selection import select_mimic_explainer

__all__ = ["MimicExplainer", "select_mimic_explainer"]
