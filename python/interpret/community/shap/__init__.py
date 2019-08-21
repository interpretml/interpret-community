# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Module for shap-based blackbox and whitebox explainers."""
from .deep_explainer import DeepExplainer
from .kernel_explainer import KernelExplainer
from .tree_explainer import TreeExplainer

__all__ = ['DeepExplainer', 'KernelExplainer', 'TreeExplainer']
