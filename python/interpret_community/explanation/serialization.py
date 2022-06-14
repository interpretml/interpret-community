# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Defines the save and load explanation methods and helpers for serialization."""

import json
import os

import numpy as np
import pandas as pd
from ml_wrappers import DatasetWrapper
from scipy.sparse import csr_matrix, issparse

from ..common.constants import ExplainParams, ExplainType
from ..explanation import explanation as patched_explanation
from .explanation import (ClassesMixin, FeatureImportanceExplanation,
                          GlobalExplanation, LocalExplanation,
                          _create_global_explanation,
                          _create_local_explanation)

EXPLANATION_METADATA = 'explanation_metadata.json'
SERIALIZATION_VERSION = 'serialization_version'


def save_explanation(explanation, path, exist_ok=False):
    """Serialize the explanation.

    :param explanation: The Explanation to be serialized.
    :type explanation: Explanation
    :param path: The path to the directory in which the explanation will be saved. By default, must be a new directory
        to avoid overwriting any previous explanations. Set exist_ok to True to overrule this behavior.
    :type path: str
    :param exist_ok: If False (default), the path provided by the user must not already exist and will be created by
        this function. If True, a prexisting path may be passed. Any preexisting files whose names match those of the
        files that make up the explanation will be overwritten.
    :type exist_ok: bool
    """
    if os.path.exists(path) and not exist_ok:
        raise Exception('The directory specified by path already exists. '
                        'Please pass in a new directory or set exists_ok=True.')
    os.makedirs(path, exist_ok=True)

    # TODO replace with set of params from below
    uploadable_properties = [
        ExplainParams.FEATURES,
        ExplainParams.LOCAL_IMPORTANCE_VALUES,
        ExplainParams.EXPECTED_VALUES,
        ExplainParams.CLASSES,
        ExplainParams.GLOBAL_IMPORTANCE_NAMES,
        ExplainParams.GLOBAL_IMPORTANCE_RANK,
        ExplainParams.GLOBAL_IMPORTANCE_VALUES,
        ExplainParams.PER_CLASS_NAMES,
        ExplainParams.PER_CLASS_RANK,
        ExplainParams.PER_CLASS_VALUES,
        ExplainParams.INIT_DATA,
        ExplainParams.EVAL_DATA,
        ExplainParams.EVAL_Y_PRED,
        ExplainParams.EVAL_Y_PRED_PROBA
    ]
    # TODO will need to add viz data on top of this
    for prop in uploadable_properties:
        if hasattr(explanation, prop) and getattr(explanation, prop) is not None:
            value = getattr(explanation, prop)
            if isinstance(value, pd.DataFrame):
                value = value.values.tolist()
                metadata = 'DataFrame'
            elif isinstance(value, DatasetWrapper):
                value = value.original_dataset
                if issparse(value):
                    value = _convert_sparse_to_list(value)
                    metadata = 'DatasetWrapperSparse'
                else:
                    value = value.tolist()
                    metadata = 'DatasetWrapper'
            elif isinstance(value, np.ndarray):
                value = value.tolist()
                metadata = 'ndarray'
            elif isinstance(value, list):
                metadata = 'list'
            elif issparse(value):
                if prop != ExplainParams.LOCAL_IMPORTANCE_VALUES:
                    value = _convert_sparse_to_list(value)
                metadata = 'sparse'
            else:
                metadata = 'other'
            if prop == ExplainParams.LOCAL_IMPORTANCE_VALUES and explanation.is_local_sparse:
                value = _convert_sparse_data(value, explanation)
                metadata = 'sparse'
            data_dict = {
                'metadata': metadata,
                'data': value
            }
            filename = os.path.join(path, prop + '.json')
            with open(filename, 'w') as f:
                json.dump(data_dict, f)
    # create metadata file
    prop_dict = _get_explanation_metadata(explanation)

    filename = os.path.join(path, EXPLANATION_METADATA)
    with open(filename, 'w') as f:
        json.dump(prop_dict, f)


def load_explanation(path):
    """Deserialize the explanation.

    :param path: The path to the directory in which the explanation will be saved. By default, must be a new directory
        to avoid overwriting any previous explanations. Set exist_ok to True to overrule this behavior.
    :type path: str
    :return: The deserialized explanation.
    :rtype: Explanation
    """
    shared_params = [
        ExplainParams.EXPECTED_VALUES,
        ExplainParams.FEATURES,
        ExplainParams.CLASSES,
        ExplainParams.EVAL_DATA,
        ExplainParams.INIT_DATA,
        ExplainParams.EVAL_Y_PRED,
        ExplainParams.EVAL_Y_PRED_PROBA
    ]
    global_params = [
        ExplainParams.GLOBAL_IMPORTANCE_RANK,
        ExplainParams.GLOBAL_IMPORTANCE_VALUES,
        ExplainParams.PER_CLASS_RANK,
        ExplainParams.PER_CLASS_VALUES
    ]
    local_params = [
        ExplainParams.LOCAL_IMPORTANCE_VALUES
    ]
    with open(os.path.join(path, EXPLANATION_METADATA), 'r') as f:
        metadata = json.load(f)
        is_global = metadata[ExplainType.GLOBAL]
        is_local = metadata[ExplainType.LOCAL]

    if is_local and is_global:
        local_kwargs = _get_kwargs(path, shared_params + local_params)
        local_explanation = _create_local_explanation(**local_kwargs)
        global_kwargs = _get_kwargs(path, shared_params + global_params, local_explanation=local_explanation)
        return _create_global_explanation(**global_kwargs)
    elif is_local:
        local_kwargs = _get_kwargs(path, shared_params + local_params)
        return _create_local_explanation(**local_kwargs)
    elif is_global:
        global_kwargs = _get_kwargs(path, shared_params + global_params)
        return _create_global_explanation(**global_kwargs)


# Monkey patch the save and load explanation methods in explanation.py for backwards compat
patched_explanation.save_explanation = save_explanation
patched_explanation.load_explanation = load_explanation


def _get_explanation_metadata(explanation):
    classification = ClassesMixin._does_quack(explanation)
    is_raw = False if not FeatureImportanceExplanation._does_quack(explanation) else explanation.is_raw
    is_eng = False if not FeatureImportanceExplanation._does_quack(explanation) else explanation.is_engineered
    num_features = 0 if not FeatureImportanceExplanation._does_quack(explanation) else explanation.num_features
    prop_dict = {
        ExplainParams.EXPLANATION_ID: explanation.id,
        ExplainType.MODEL: ExplainType.CLASSIFICATION if classification else ExplainType.REGRESSION,
        ExplainType.DATA: ExplainType.TABULAR,
        ExplainType.MODEL_TASK: explanation.model_task,
        ExplainType.METHOD: explanation.method,
        ExplainType.MODEL_CLASS: explanation.model_type,
        ExplainType.IS_RAW: is_raw,
        ExplainType.IS_ENG: is_eng,
        ExplainType.GLOBAL: GlobalExplanation._does_quack(explanation),
        ExplainType.LOCAL: LocalExplanation._does_quack(explanation),
        ExplainParams.NUM_CLASSES: 1 if not ClassesMixin._does_quack(explanation) else explanation.num_classes,
        ExplainParams.NUM_EXAMPLES: 0 if not LocalExplanation._does_quack(explanation) else explanation.num_examples,
        ExplainParams.NUM_FEATURES: num_features,
        ExplainParams.IS_LOCAL_SPARSE: LocalExplanation._does_quack(explanation) and explanation.is_local_sparse,
        SERIALIZATION_VERSION: 1,
    }
    return prop_dict


def _get_value_from_file(file_var):
    json_input = json.load(file_var)
    meta = json_input['metadata']
    data = json_input['data']
    if meta == 'DataFrame':
        return pd.DataFrame(data)
    elif meta == 'DatasetWrapper':
        if isinstance(data, list):
            data = np.array(data)
        return DatasetWrapper(data)
    elif meta == 'DatasetWrapperSparse':
        data = _convert_artifact_to_sparse_local(data)
        return DatasetWrapper(data)
    elif meta == 'ndarray':
        return np.array(data)
    elif meta == 'list':
        return data
    elif meta == 'sparse':
        return _convert_artifact_to_sparse_local(data)
    else:
        raise Exception('Unrecognized data type in deserialization: {}'.format(meta))


def _get_kwargs(path, params, local_explanation=None):
    with open(os.path.join(path, EXPLANATION_METADATA), 'r') as f:
        metadata = json.load(f)
    numpy_params = {ExplainParams.EVAL_DATA, ExplainParams.INIT_DATA}
    is_local_sparse = False
    if ExplainParams.IS_LOCAL_SPARSE in metadata:
        is_local_sparse = metadata[ExplainParams.IS_LOCAL_SPARSE]
    if not is_local_sparse:
        numpy_params.add(ExplainParams.LOCAL_IMPORTANCE_VALUES)

    kwargs = {}
    for param in params:
        if os.path.exists(os.path.join(path, param + '.json')):
            with open(os.path.join(path, param + '.json'), 'r') as f:
                kwargs[param] = _get_value_from_file(f)
                if param in numpy_params:
                    kwargs[param] = np.array(kwargs[param])

    param_list = [
        ExplainParams.METHOD,
        ExplainParams.MODEL_TASK,
        ExplainParams.NUM_FEATURES,
        ExplainParams.IS_RAW,
        ExplainParams.IS_ENG,
        ExplainParams.EXPLANATION_ID
    ]
    for param in param_list:
        kwargs[param] = metadata[param]
    kwargs[ExplainParams.MODEL_TYPE] = metadata[ExplainType.MODEL_CLASS]
    classification = metadata[ExplainType.MODEL] == ExplainType.CLASSIFICATION
    kwargs[ExplainParams.CLASSIFICATION] = classification
    if classification:
        kwargs[ExplainParams.NUM_CLASSES] = metadata[ExplainParams.NUM_CLASSES]
    if local_explanation is not None:
        kwargs[ExplainParams.LOCAL_EXPLANATION] = local_explanation
    return kwargs


def _convert_sparse_data(value, explanation):
    """Converts the given sparse local importance values to list format.

    :param value: The local importance values to convert.
    :type value: list[scipy.sparse.csr_matrix] or scipy.sparse.csr_matrix
    :return: The local importance values as a list representation.
    :rtype: list[list[int | float]]
    """
    if issparse(value):
        return _convert_sparse_to_list(value)
    else:
        sparse_data = []
        for sparse_local_values in value:
            sparse_data.append(_convert_sparse_to_list(sparse_local_values))
        return sparse_data


def _convert_sparse_to_list(sparse_local_values):
    """Converts a sparse explanation to a list.

    :param sparse_local_values: The sparse matrix to be converted to a list representation that includes
        the data, indices, indptr and shape.
    :type sparse_local_values: scipy.sparse.csr_matrix
    :return: The local importance values as a list representation.
    :rtype: list[list[int | float]]
    """
    sparse_data = []
    sparse_data.append(sparse_local_values.data.tolist())
    sparse_data.append(sparse_local_values.indices.tolist())
    sparse_data.append(sparse_local_values.indptr.tolist())
    sparse_data.append(list(sparse_local_values.shape))
    return sparse_data


def _convert_artifact_to_sparse_local(local_vals_sparse):
    """Converts a saved representation of an explanation to sparse local importance values.

    :param local_vals_sparse: The list representation that includes
        the data, indices, indptr and shape to be converted to a sparse matrix.
        In multiclass case this includes the per class local importance values
        as well, so it will be a 3 dimensional list of values.
    :type local_vals_sparse: list[list[int | float]] or list[list[list[int | float]]]
    :return: The local importance values as a sparse matrix.
    :rtype: list[scipy.sparse.csr_matrix] or scipy.sparse.csr_matrix
    """
    if isinstance(local_vals_sparse[0][0], list):
        local_importance_vals = []
        for class_importance_values in local_vals_sparse:
            local_importance_vals.append(_convert_list_to_sparse(class_importance_values))
    else:
        local_importance_vals = _convert_list_to_sparse(local_vals_sparse)
    return local_importance_vals


def _convert_list_to_sparse(sparse_local_values_list):
    """Converts a list representation of an explanation to a sparse matrix.

    :param sparse_local_values_list: The list representation that includes
        the data, indices, indptr and shape to be converted to a sparse matrix.
    :type sparse_local_values_list: list[list[int | float]]
    :return: The local importance values as a sparse matrix.
    :rtype: scipy.sparse.csr_matrix
    """
    data = sparse_local_values_list[0]
    indices = sparse_local_values_list[1]
    indptr = sparse_local_values_list[2]
    shape = tuple(sparse_local_values_list[3])
    return csr_matrix((data, indices, indptr), shape)
