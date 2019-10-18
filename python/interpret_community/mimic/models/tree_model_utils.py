# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Defines utilities for tree-based explainable models."""

from scipy.special import expit
from ...common.constants import ShapValuesOutput
from ...common.explanation_utils import _scale_tree_shap


def _explain_local_tree_surrogate(tree_model, evaluation_examples, tree_explainer,
                                  shap_values_output, classification, probabilities,
                                  multiclass):
    """Locally explains the tree-based surrogate model.

    :param tree_model: A tree-based model.
    :type tree_model: Tree-based model with scikit-learn predict and predict_proba API.
    :param evaluation_examples: The evaluation examples to compute local feature importances for.
    :type evaluation_examples: numpy or scipy array
    :param tree_explainer: Tree explainer for the tree-based model.
    :type tree_explainer: TreeExplainer
    :param shap_values_output: The type of the output from explain_local when using TreeExplainer.
        Currently only types 'default', 'probability' and 'teacher_probability' are supported.  If
        'probability' is specified, then we approximately scale the raw log-odds values from the
        TreeExplainer to probabilities.
    :type shap_values_output: interpret_community.common.constants.ShapValuesOutput
    :param classification: Indicates if this is a classification or regression explanation.
    :type classification: bool
    :param probabilities: If output_type is probability, can specify the teacher model's
        probability for scaling the shap values.
        Note for regression case this will just be the output values of the teacher model.
    :type probabilities: numpy.ndarray
    :param multiclass: True if the tree_model is a multiclass model.
    :type multiclass: bool
    :return: The local explanation of feature importances.
    :rtype: Union[list, numpy.ndarray]
    """
    if len(evaluation_examples.shape) == 1:
        evaluation_examples = evaluation_examples.reshape(1, -1)
    # Note: For binary and multiclass case the expected values and shap values are in the units
    # of the raw predictions from the underlying model.
    # In binary case, we are using regressor on logit.  In multiclass case, shap TreeExplainer
    # outputs the margin instead of probabilities.
    shap_values = tree_explainer.shap_values(evaluation_examples)
    is_probability = shap_values_output == ShapValuesOutput.PROBABILITY
    is_teacher_probability = shap_values_output == ShapValuesOutput.TEACHER_PROBABILITY
    if is_probability or is_teacher_probability:
        expected_values = tree_explainer.expected_value
        if classification:
            expected_values = expit(expected_values)
            if probabilities is None:
                if is_teacher_probability:
                    raise Exception("Probabilities not specified for output type 'teacher_probability'")
                if multiclass:
                    probabilities = tree_model.predict_proba(evaluation_examples)
                else:
                    probabilities = expit(tree_model.predict(evaluation_examples))
                    probabilities = probabilities.reshape((probabilities.shape[0], 1))
            shap_values = _scale_tree_shap(shap_values, expected_values, probabilities)
        elif is_teacher_probability:
            # In regression case, the values are in terms of the surrogate model, so we transform to teacher model
            # note this is not in terms of probabilities, might need to figure out a better param name
            shap_values = _scale_tree_shap(shap_values, expected_values, probabilities)
    return shap_values


def _expected_values_tree_surrogate(tree_model, tree_explainer, shap_values_output, classification, multiclass):
    """Use TreeExplainer to get the expected values.

    :param tree_model: A tree-based model.
    :type tree_model: Tree-based model with scikit-learn predict and predict_proba API.
    :param tree_explainer: Tree explainer for the tree-based model.
    :type tree_explainer: TreeExplainer
    :param shap_values_output: The type of the output from explain_local when using TreeExplainer.
        Currently only types 'default', 'probability' and 'teacher_probability' are supported.  If
        'probability' is specified, then we approximately scale the raw log-odds values from the
        TreeExplainer to probabilities.
    :type shap_values_output: interpret_community.common.constants.ShapValuesOutput
    :param classification: Indicates if this is a classification or regression explanation.
    :type classification: bool
    :param multiclass: True if the tree_model is a multiclass model.
    :type multiclass: bool
    :return: The expected values of the tree-based model.
    :rtype: list
    """
    expected_values = tree_explainer.expected_value
    if classification:
        if not multiclass:
            expected_values = [-expected_values, expected_values]
        is_probability = shap_values_output == ShapValuesOutput.PROBABILITY
        is_teacher_probability = shap_values_output == ShapValuesOutput.TEACHER_PROBABILITY
        if is_probability or is_teacher_probability:
            return expit(expected_values)
    return expected_values
