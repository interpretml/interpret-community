# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

# Defines common test utilities for validating explanation objects

def validate_global_classification_explanation_shape(explanation, evaluation_examples,
                                                     num_classes=2, include_local=True):
    assert explanation._global_importance_values.shape[0] == evaluation_examples.shape[1]
    assert explanation._per_class_values.shape[0] == num_classes
    assert explanation._per_class_values.shape[1] == evaluation_examples.shape[1]
    if include_local:
        validate_local_classification_explanation_shape(explanation, evaluation_examples, num_classes)


def validate_local_classification_explanation_shape(explanation, evaluation_examples, num_classes=2):
    assert explanation._local_importance_values.shape[0] == num_classes
    assert explanation._local_importance_values.shape[1] == evaluation_examples.shape[0]
    assert explanation._local_importance_values.shape[2] == evaluation_examples.shape[1]


def validate_global_regression_explanation_shape(explanation, evaluation_examples,
                                                 num_classes=2, include_local=True):
    assert explanation._global_importance_values.shape[0] == evaluation_examples.shape[1]
    if include_local:
        validate_local_regression_explanation_shape(explanation, evaluation_examples, num_classes)


def validate_local_regression_explanation_shape(explanation, evaluation_examples, num_classes=2):
    assert explanation._local_importance_values.shape[0] == evaluation_examples.shape[0]
    assert explanation._local_importance_values.shape[1] == evaluation_examples.shape[1]
