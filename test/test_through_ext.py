# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Tests importing the explainers through extensions"""


def test_import_tabular():
    from interpret.ext.blackbox import TabularExplainer  # noqa


def test_import_kernel():
    from interpret.ext.blackbox import KernelExplainer  # noqa


def test_import_mimic():
    from interpret.ext.blackbox import MimicExplainer  # noqa


def test_import_pfi():
    from interpret.ext.blackbox import PFIExplainer  # noqa


def test_import_lime():
    from interpret.ext.blackbox import LIMEExplainer  # noqa


def test_import_linear():
    from interpret.ext.greybox import LinearExplainer  # noqa


def test_import_tree():
    from interpret.ext.greybox import TreeExplainer  # noqa


def test_import_deep():
    from interpret.ext.greybox import DeepExplainer  # noqa


def test_import_lgbm_explainable_model():
    from interpret.ext.glassbox import LGBMExplainableModel  # noqa


def test_import_linear_explainable_model():
    from interpret.ext.glassbox import LinearExplainableModel  # noqa


def test_import_sgd_explainable_model():
    from interpret.ext.glassbox import SGDExplainableModel  # noqa


def test_import_decision_tree_explainable_model():
    from interpret.ext.glassbox import DecisionTreeExplainableModel  # noqa
