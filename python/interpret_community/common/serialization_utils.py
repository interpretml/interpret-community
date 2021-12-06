# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Defines utility functions for serialization of data."""

import datetime

import numpy as np


def _serialize_json_safe(o):
    """
    Convert a value into something that is safe to parse into JSON.

    :param o: Object to make JSON safe.
    :return: New object
    """
    if type(o) in {int, float, str, type(None)}:
        if isinstance(o, float):
            if np.isinf(o) or np.isnan(o):
                return 0
        return o
    elif isinstance(o, datetime.datetime):
        return o.__str__()
    elif isinstance(o, dict):
        return {k: _serialize_json_safe(v) for k, v in o.items()}
    elif isinstance(o, list):
        return [_serialize_json_safe(v) for v in o]
    elif isinstance(o, tuple):
        return tuple(_serialize_json_safe(v) for v in o)
    elif isinstance(o, np.ndarray):
        return _serialize_json_safe(o.tolist())
    else:
        # Attempt to convert Numpy type
        try:
            return o.item()
        except Exception:
            return o
