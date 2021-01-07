# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Defines different types of exceptions that this package can raise."""


class ScenarioNotSupportedException(Exception):
    """An exception indicating that some scenario is not supported.

    :param exception_message: A message describing the error.
    :type exception_message: str
    """

    _error_code = "Unsupported scenario"
