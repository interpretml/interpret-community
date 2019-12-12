# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Tests the logging mechanism in the library"""

import os


def test_import_tabular():
    not_existing_dir = 'not_existing_dir'
    not_existing_path = not_existing_dir + '/interpret_community_log.txt'
    if os.path.exists(not_existing_path):
        os.remove(not_existing_path)
        os.rmdir(not_existing_dir)
    os.environ['INTERPRET_C_LOGS'] = not_existing_path
    import interpret_community
    import importlib
    importlib.reload(interpret_community)
    assert(os.path.exists(not_existing_path))

    try:
        os.remove(not_existing_path)
        os.rmdir(not_existing_dir)
    except PermissionError:
        pass
