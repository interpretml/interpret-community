# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Defines utility functions for serialization of data."""

from raiutils.data_processing import serialize_json_safe


def _serialize_json_safe(o):
    """
    Convert a value into something that is safe to parse into JSON.

    :param o: Object to make JSON safe.
    :return: New object
    """
    return serialize_json_safe(o)
