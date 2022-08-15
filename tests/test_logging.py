# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Tests the logging mechanism in the library"""

import os

INTERPRET_C_LOGS = 'INTERPRET_C_LOGS'


def test_import_tabular():
    not_existing_dir = 'not_existing_dir'
    not_existing_path = os.path.join(not_existing_dir, 'interpret_community_log.txt')
    if os.path.exists(not_existing_path):
        os.remove(not_existing_path)
        os.rmdir(not_existing_dir)
    os.environ[INTERPRET_C_LOGS] = not_existing_path
    import importlib

    import interpret_community
    importlib.reload(interpret_community)
    assert os.path.exists(not_existing_path)

    del os.environ[INTERPRET_C_LOGS]
    importlib.reload(interpret_community)
    import logging
    importlib.reload(logging)
    try:
        os.remove(not_existing_path)
        os.rmdir(not_existing_dir)
    except PermissionError:
        pass
