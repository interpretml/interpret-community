import logging
import os

_logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)

class _LogWrapper:
    def __init__(self, description):
        self._description = description

    def __enter__(self):
        _logger.info("Starting %s", self._description)

    def __exit__(self, type, value, traceback):  # noqa: A002
        # raise exceptions if any occurred
        if value is not None:
            raise value
        _logger.info("Completed %s", self._description)
