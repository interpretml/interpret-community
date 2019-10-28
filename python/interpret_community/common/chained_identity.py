# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Defines a light-weight chained identity for logging."""

import logging


class ChainedIdentity(object):
    """The base class for logging information."""

    def __init__(self, **kwargs):
        """Initialize the ChainedIdentity."""
        self._logger = logging.getLogger("interpret_community").getChild(self.__class__.__name__)
        self._identity = self.__class__.__name__
        super(ChainedIdentity, self).__init__(**kwargs)
