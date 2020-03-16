# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Defines utilities for getting progress status for explanation."""


def _tqdm_func(*i, **kwargs):
    return i[0]


def get_tqdm(logger, show_progress):
    """Get the tqdm progress bar function.

    :param logger: The logger for logging info messages.
    :type logger: logger
    :param show_progress: Default to 'True'.  Determines whether to display the explanation status bar
        when using PFIExplainer.
    :type show_progress: bool
    :return: The tqdm (https://github.com/tqdm/tqdm) progress bar.
    :rtype: function
    """
    # This is used to get around code style build error F811
    tqdm = _tqdm_func
    if show_progress:
        try:
            from tqdm import tqdm
        except ImportError:
            logger.info('Failed to import tqdm to show progress bar')
    return tqdm
