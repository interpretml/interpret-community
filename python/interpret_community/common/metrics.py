# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Defines metrics for validating model explanations."""

import numpy as np


def dcg(validate_order, ground_truth_order_relevance, top_values=10):
    """Compute the discounted cumulative gain (DCG).

    Compute the DCG as the sum of relevance scores penalized by the logarithmic position of the result.
    See https://en.wikipedia.org/wiki/Discounted_cumulative_gain for reference.

    :param validate_order: The order to validate.
    :type validate_order: list
    :param ground_truth_order_relevance: The ground truth relevancy of the documents to compare to.
    :type ground_truth_order_relevance: list
    :param top_values: Specifies the top values to compute the DCG for. The default is 10.
    :type top_values: int
    """
    # retrieve relevance score for each value in validation order
    relevance = np.vectorize(lambda x: ground_truth_order_relevance.get(x, 0))(validate_order[:top_values])
    gain = 2 ** relevance - 1
    discount = np.log2(np.arange(1, len(gain) + 1) + 1)
    sum_dcg = np.sum(gain / discount)
    return sum_dcg


def ndcg(validate_order, ground_truth_order, top_values=10):
    """Compute the normalized discounted cumulative gain (NDCG).

    Compute the NDCG as the ratio of the DCG for the validation order compared to the maximum DCG
    possible for the ground truth order.
    If the validation order is the same as the ground truth the NDCG will be the maximum of 1.0,
    and the least possible NDCG is 0.0.
    See https://en.wikipedia.org/wiki/Discounted_cumulative_gain for reference.

    :param validate_order: The order to validate for the documents. The values should be unique.
    :type validate_order: list
    :param ground_truth_order: The true order of the documents. The values should be unique.
    :type ground_truth_order: list
    :param top_values: Specifies the top values to compute the NDCG for. The default is 10.
    :type top_values: int
    """
    # Create map from true_order to "relevance" or reverse order index
    ground_truth_order_relevance = {}
    num_elems = len(ground_truth_order)
    for index, value in enumerate(ground_truth_order):
        # Compute the relevancy scores as a weighted reverse order in the given list.
        # Set the range of the relevance scores to be between 0 and 10
        # This is to prevent very large values when computing 2 ** relevance - 1
        ground_truth_order_relevance[value] = ((num_elems - index) / float(num_elems)) * 10.0
    # See https://en.wikipedia.org/wiki/Discounted_cumulative_gain for reference
    dcg_p = dcg(validate_order, ground_truth_order_relevance, top_values)
    idcg_p = dcg(ground_truth_order, ground_truth_order_relevance, top_values)
    ndcg = dcg_p / idcg_p
    return ndcg
