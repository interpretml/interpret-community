# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Prevent disabled optional frameworks from loading during test collection."""

import importlib.abc
import importlib.machinery
import os
import sys

_FRAMEWORKS = {
    'INTERPRET_TEST_TENSORFLOW': ('tensorflow', 'keras'),
    'INTERPRET_TEST_PYTORCH': ('torch', 'torchvision', 'captum')
}
_BLOCKED = {
    module
    for environment_variable, modules in _FRAMEWORKS.items()
    if os.environ.get(environment_variable, '1') == '0'
    for module in modules
}


class _FrameworkBlocker(importlib.abc.MetaPathFinder,
                        importlib.abc.Loader):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.partition('.')[0] in _BLOCKED:
            return importlib.machinery.ModuleSpec(fullname, self)
        return None

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        raise ModuleNotFoundError(
            '{} disabled for this test process'.format(module.__name__),
            name=module.__name__)


if _BLOCKED:
    sys.meta_path.insert(0, _FrameworkBlocker())


def pytest_sessionstart(session):
    """Verify blocked frameworks were not imported before pytest startup."""
    loaded = _BLOCKED.intersection(sys.modules)
    if loaded:
        raise RuntimeError(
            'Disabled frameworks loaded before isolation: {}'.format(
                sorted(loaded)))
