# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Defines utilities for handling kwargs on SHAP-based explainers."""

from interpret_community.common.constants import ExplainParams


def _get_explain_global_kwargs(sampling_policy, method, include_local, batch_size):
    """Get the kwargs for explain_global.

    :param sampling_policy: Optional policy for sampling the evaluation examples. See documentation on
        SamplingPolicy for more information.
    :type sampling_policy: interpret_community.common.SamplingPolicy
    :param method: The explanation method used, e.g., shap_kernel, mimic, etc.
    :type method: str
    :param include_local: Whether a local explanation should be generated or only global
    :type include_local: bool
    :param batch_size: If include_local is False, specifies the batch size for aggregating
        local explanations to global.
    :type batch_size: int
    :return: Args for explain_global.
    :rtype: dict
    """
    kwargs = {ExplainParams.METHOD: method,
              ExplainParams.SAMPLING_POLICY: sampling_policy,
              ExplainParams.INCLUDE_LOCAL: include_local,
              ExplainParams.BATCH_SIZE: batch_size}
    return kwargs
