# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Tests importing the explainers through extensions"""


def test_import():
    from interpret.ext.blackbox import TabularExplainer  # noqa
