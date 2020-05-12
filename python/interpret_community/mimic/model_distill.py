# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Utilities to train a surrogate model from teacher."""

import numpy as np
from scipy.sparse import issparse, isspmatrix_csr, vstack as sparse_vstack


def _soft_logit(values, clip_val=5):
    """Compute a soft logit on an iterable by bounding outputs to a min/max value.

    :param values: Iterable of numeric values to logit and clip.
    :type values: iter
    :param clip_val: Clipping threshold for logit output.
    :type clip_val: Union[Int, Float]
    """
    new_values = np.log(values / (1 - values))
    return np.clip(new_values, -clip_val, clip_val)


def _model_distill(teacher_model_predict_fn, uninitialized_surrogate_model, data, original_training_data,
                   explainable_model_args):
    """Teach a surrogate model to mimic a teacher model.

    :param teacher_model_predict_fn: Blackbox model's prediction function.
    :type teacher_model_predict_fn: function
    :param uninitialized_surrogate_model: Uninitialized model used to distill blackbox.
    :type uninitialized_surrogate_model: uninitialized model
    :param data: Representative data (or training data) to train distilled model.
    :type data: numpy.ndarray
    :param original_training_data: Representative data (or training data) to get predictions from teacher model.
    :type original_training_data: numpy.ndarray
    :param explainable_model_args: An optional map of arguments to pass to the explainable model
        for initialization.
    :type explainable_model_args: dict
    """
    # For regression, teacher_y is a real value whereas for classification it is a probability between 0 and 1
    teacher_y = teacher_model_predict_fn(original_training_data)
    multiclass = False
    training_labels = None
    is_classifier = len(teacher_y.shape) == 2
    # If the predict_proba function returned one column but this is a classifier, modify to [1-p, p]
    if is_classifier and teacher_y.shape[1] == 1:
        teacher_y = np.column_stack((1 - teacher_y, teacher_y))
    if is_classifier and teacher_y.shape[1] > 2:
        # If more than two classes, use multiclass surrogate
        multiclass = True
        # For multiclass case, we need to train on the class label
        training_labels = np.argmax(teacher_y, axis=1)
        unique_labels = set(np.unique(training_labels))
        if len(unique_labels) < teacher_y.shape[1]:
            # Get the missing labels
            missing_labels = set(range(teacher_y.shape[1])).difference(unique_labels)
            # Append some rows with the missing labels
            for missing_label in missing_labels:
                # Find max prob for missing label
                max_row_index = np.argmax(teacher_y[:, missing_label])
                # Append the extra label to data and y value
                training_labels = np.append(training_labels, missing_label)
                if issparse(data) and not isspmatrix_csr(data):
                    data = data.tocsr()
                vstack = sparse_vstack if issparse(data) else np.vstack
                data = vstack([data, data[max_row_index:max_row_index + 1, :]])
        surrogate_model = uninitialized_surrogate_model(multiclass=multiclass,
                                                        **explainable_model_args)
    else:
        surrogate_model = uninitialized_surrogate_model(**explainable_model_args)
    if is_classifier and teacher_y.shape[1] == 2:
        # Make sure output has only 1 dimension
        teacher_y = teacher_y[:, 1]
        # Transform to logit space and fit regression
        surrogate_model.fit(data, _soft_logit(teacher_y))
    else:
        # Use hard labels for regression or multiclass case
        if training_labels is None:
            training_labels = teacher_y
        surrogate_model.fit(data, training_labels)
    return surrogate_model
