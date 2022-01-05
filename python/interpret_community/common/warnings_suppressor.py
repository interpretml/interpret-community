# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Suppresses warnings on imports."""

import os
import warnings

TF_CPP_MIN_LOG_LEVEL = 'TF_CPP_MIN_LOG_LEVEL'


class tf_warnings_suppressor(object):
    """Context manager to suppress warnings from tensorflow."""

    def __init__(self):
        """Initialize the tf_warnings_suppressor."""
        self._entered = False
        if TF_CPP_MIN_LOG_LEVEL in os.environ:
            self._default_tf_log_level = os.environ[TF_CPP_MIN_LOG_LEVEL]
        else:
            self._default_tf_log_level = '0'

    def __enter__(self):
        """Begins suppressing tensorflow warnings."""
        if self._entered:
            raise RuntimeError("Cannot enter %r twice" % self)
        self._entered = True
        os.environ[TF_CPP_MIN_LOG_LEVEL] = '2'

    def __exit__(self, *exc_info):
        """Finishes suppressing tensorflow warnings."""
        if not self._entered:
            raise RuntimeError("Cannot exit %r without entering first" % self)
        os.environ[TF_CPP_MIN_LOG_LEVEL] = self._default_tf_log_level


class shap_warnings_suppressor(object):
    """Context manager to suppress warnings from shap."""

    def __init__(self):
        """Initialize the shap_warnings_suppressor."""
        self._catch_warnings = warnings.catch_warnings()
        self._tf_warnings_suppressor = tf_warnings_suppressor()
        self._entered = False
        if TF_CPP_MIN_LOG_LEVEL in os.environ:
            self._default_tf_log_level = os.environ[TF_CPP_MIN_LOG_LEVEL]
        else:
            self._default_tf_log_level = '0'

    def __enter__(self):
        """Begins suppressing shap warnings."""
        if self._entered:
            raise RuntimeError("Cannot enter %r twice" % self)
        self._entered = True
        self._tf_warnings_suppressor.__enter__()
        log = self._catch_warnings.__enter__()
        warnings.filterwarnings('ignore', 'Starting from version 2.2.1', UserWarning)
        return log

    def __exit__(self, *exc_info):
        """Finishes suppressing shap warnings."""
        if not self._entered:
            raise RuntimeError("Cannot exit %r without entering first" % self)
        self._tf_warnings_suppressor.__exit__()
        self._catch_warnings.__exit__()
