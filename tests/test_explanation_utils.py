# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import logging

import numpy as np
import pytest
from constants import owner_email_tools_and_ux
from interpret_community.common.constants import Scipy
from interpret_community.common.explanation_utils import (
    _FEATURES, _RANKING, _VALUES, _convert_to_list, _generate_augmented_data,
    _get_feature_map_from_list_of_indexes, _get_raw_feature_importances,
    _is_identity, _is_one_to_many, _should_compress_sparse_matrix,
    _sort_feature_list_multiclass, _sort_feature_list_single, _sort_values,
    _sparse_order_imp, _two_dimensional_slice)
from raw_explain.utils import _get_feature_map_from_indices_list
from scipy.sparse import csr_matrix, eye

test_logger = logging.getLogger(__name__)


@pytest.mark.owner(email=owner_email_tools_and_ux)
@pytest.mark.usefixtures('_clean_dir')
class TestExplanationUtils(object):

    def test_working(self):
        assert True

    def test_convert_to_list_1d(self):
        numpy_1d = np.ones(4)
        list_1d = [1] * 4
        assert _convert_to_list(numpy_1d) == list_1d

    def test_convert_to_list_2d_full_numpy(self):
        numpy_2d = np.ones((3, 4))
        list_2d = [[1] * 4] * 3
        assert _convert_to_list(numpy_2d) == list_2d

    def test_convert_to_list_2d_list_of_numpy(self):
        numpy_2d = np.ones(4)
        numpy_list = [numpy_2d] * 3
        list_2d = [[1] * 4] * 3
        assert _convert_to_list(numpy_list) == list_2d

    def test_sort_values(self):
        feature_list = ['feature0', 'feature1', 'feature2', 'feature3']
        order = [2, 3, 0, 1]
        assert np.array_equal(_sort_values(feature_list, order),
                              np.array(['feature2', 'feature3', 'feature0', 'feature1']))

    def test_sort_feature_list_single(self):
        feature_list = ['feature0', 'feature1', 'feature2', 'feature3']
        order = [2, 3, 0, 1]
        assert _sort_feature_list_single(feature_list, order) == ['feature2', 'feature3', 'feature0', 'feature1']

    def test_sort_feature_list_multiclass(self):
        feature_list = ['feature0', 'feature1', 'feature2', 'feature3']
        order = [
            [2, 3, 0, 1],
            [1, 3, 2, 0]
        ]
        output = [
            ['feature2', 'feature3', 'feature0', 'feature1'],
            ['feature1', 'feature3', 'feature2', 'feature0']
        ]
        assert _sort_feature_list_multiclass(feature_list, order) == output

    def test_two_dimensional_slice(self):
        big_list = [
            ['feature2', 'feature3', 'feature0', 'feature1'],
            ['feature1', 'feature3', 'feature2', 'feature0']
        ]
        output = [
            ['feature2', 'feature3'],
            ['feature1', 'feature3']
        ]
        assert _two_dimensional_slice(big_list, 2) == output

    def test_generate_augmented_data_ndarray(self):
        x = np.ones((3, 6))
        x_augmented = _generate_augmented_data(x)
        assert x_augmented.shape[0] == 6
        assert x_augmented.shape[1] == 6

    def test_generate_augmented_data_sparse(self):
        x = csr_matrix(np.zeros((3, 6)))
        x_augmented = _generate_augmented_data(x)
        assert x_augmented.shape[0] == 6
        assert x_augmented.shape[1] == 6

    def test_get_raw_feats_regression(self):
        feat_imps = np.ones((2, 5))
        feat_imps[1] = 2 * np.ones(5)
        raw_feat_indices = [[0, 1, 2], [3, 4]]
        feature_map = _get_feature_map_from_indices_list(raw_feat_indices, 2, 5)
        raw_imps = _get_raw_feature_importances(feat_imps, [feature_map])
        assert np.all(raw_imps == [[3, 2], [6, 4]])

        raw_imps = _get_raw_feature_importances(feat_imps, [csr_matrix(feature_map)])
        assert np.all(raw_imps == [[3, 2], [6, 4]])

    def test_get_raw_feats_classification(self):
        feat_imps = np.ones((2, 3, 5))
        feat_imps[1] = 2 * np.ones((3, 5))
        raw_feat_indices = [[0, 1, 2], [3, 4]]
        feature_map = _get_feature_map_from_indices_list(raw_feat_indices, num_raw_cols=2, num_generated_cols=5)
        raw_imps = _get_raw_feature_importances(feat_imps, [feature_map])
        raw_feat_imps_truth = \
            [
                [
                    [3, 2],
                    [3, 2],
                    [3, 2]
                ],
                [
                    [6, 4],
                    [6, 4],
                    [6, 4]
                ],
            ]
        assert np.all(raw_imps == raw_feat_imps_truth)

    def test_get_raw_feats_regression_many_to_many(self):
        feat_imps = np.ones((2, 5))
        feat_imps[1] = 2 * np.ones(5)
        raw_feat_indices = [[0, 1, 2], [3, 4, 1]]
        feature_map = _get_feature_map_from_indices_list(raw_feat_indices, 2, 5)
        feature_map[0, 1] = 0.5
        feature_map[1, 1] = 0.5
        raw_imps = _get_raw_feature_importances(feat_imps, [feature_map])
        assert np.all(raw_imps == [[2.5, 2.5], [5, 5]])

        raw_imps = _get_raw_feature_importances(feat_imps, [csr_matrix(feature_map)])
        assert np.all(raw_imps == [[2.5, 2.5], [5, 5]])

    def test_get_raw_feats_classification_many_to_many(self):
        feat_imps = np.ones((2, 3, 5))
        feat_imps[1] = 2 * np.ones((3, 5))
        raw_feat_indices = [[0, 1, 2], [3, 4, 1]]
        feature_map = _get_feature_map_from_indices_list(raw_feat_indices, num_raw_cols=2, num_generated_cols=5)
        feature_map[0, 1] = 0.5
        feature_map[1, 1] = 0.5
        raw_imps = _get_raw_feature_importances(feat_imps, [feature_map])
        raw_feat_imps_truth = \
            [
                [
                    [2.5, 2.5],
                    [2.5, 2.5],
                    [2.5, 2.5]
                ],
                [
                    [5, 5],
                    [5, 5],
                    [5, 5]
                ],
            ]
        assert np.all(raw_imps == raw_feat_imps_truth)

        # check for sparse feature map
        raw_imps = _get_raw_feature_importances(feat_imps, [csr_matrix(feature_map)])
        assert np.all(raw_imps == raw_feat_imps_truth)

        # check for un-normalized many to many weights
        feature_map[0, 1] = 1
        feature_map[1, 1] = 1
        raw_imps = _get_raw_feature_importances(feat_imps, [feature_map])
        assert np.all(raw_imps == raw_feat_imps_truth)

    def test_get_feature_map_from_list_of_indexes(self):
        feature_map_as_adjacency_list = [[0, 1, 2], [2, 3]]

        feature_map = _get_feature_map_from_list_of_indexes(feature_map_as_adjacency_list)
        actual_feature_map = np.zeros((2, 4))
        actual_feature_map[0, [0, 1]] = 1
        actual_feature_map[0, 2] = 0.5
        actual_feature_map[1, 2] = 0.5
        actual_feature_map[1, 3] = 1

        assert np.all(feature_map == actual_feature_map)

    def test_is_one_to_many(self):
        one_to_many = np.eye(5, 6)
        many_to_one = np.zeros((3, 4))
        many_to_one[0, 1] = 1
        many_to_one[1, 1] = 1
        many_to_many = np.zeros((3, 4))
        many_to_many[0, 1] = 1
        many_to_many[1, 1] = 1
        many_to_many[0, 2] = 0.2

        assert _is_one_to_many(one_to_many)
        assert not _is_one_to_many(many_to_one)
        assert not _is_one_to_many(many_to_many)

    def test_is_identity(self):
        identity = eye(10, format=Scipy.CSR_FORMAT)
        assert _is_identity(identity)
        dense_not_identity = np.ones((10, 20))
        assert not _is_identity(dense_not_identity)
        sparse_not_identity = csr_matrix(dense_not_identity)
        assert not _is_identity(sparse_not_identity)

    def test_sparse_order_imp(self):
        x = csr_matrix(np.array([[1, 2, 3, 0, 0, 0], [5, 6, 7, 0, 0, 0], [8, 9, 10, 0, 0, 0]]))
        assert np.array_equal(_sparse_order_imp(x, values_type=_RANKING), [[2, 1, 0], [2, 1, 0], [2, 1, 0]])
        assert np.array_equal(_sparse_order_imp(x, values_type=_RANKING, top_k=2), [[2, 1], [2, 1], [2, 1]])
        assert np.array_equal(_sparse_order_imp(x, values_type=_VALUES), [[3, 2, 1], [7, 6, 5], [10, 9, 8]])
        assert np.array_equal(_sparse_order_imp(x, values_type=_VALUES, top_k=2), [[3, 2], [7, 6], [10, 9]])
        assert np.array_equal(_sparse_order_imp(x, values_type=_FEATURES), [[2, 1, 0], [2, 1, 0], [2, 1, 0]])
        feats = ["one", "two", "three", "four", "five", "six"]
        assert np.array_equal(_sparse_order_imp(x, values_type=_FEATURES, features=feats),
                              [['three', 'two', 'one'], ['three', 'two', 'one'], ['three', 'two', 'one']])
        assert np.array_equal(_sparse_order_imp(x, values_type=_FEATURES, features=feats, top_k=2),
                              [['three', 'two'], ['three', 'two'], ['three', 'two']])

    def test_should_compress_sparse_matrix(self):
        # Create a single sparse matrix that is sparse, validate we keep it as sparse
        sparse_matrix = csr_matrix((3, 4))
        assert not _should_compress_sparse_matrix(sparse_matrix)
        # Create a single sparse matrix that is mostly dense, validate we make it dense
        dense_matrix = csr_matrix(np.array([[1, 2, 3, 0], [5, 6, 7, 0], [8, 9, 10, 0]]))
        assert _should_compress_sparse_matrix(dense_matrix)
        # Create a (multiclass) list of sparse matrices that are mostly sparse, validate we keep as sparse
        sparse_matrix2 = sparse_matrix.copy()
        # Set some random values
        sparse_matrix2[:, 1] = 5.7
        sparse_matrices = [sparse_matrix, sparse_matrix, sparse_matrix2]
        assert not _should_compress_sparse_matrix(sparse_matrices)
        # Create a (multiclass) list of sparse matrices that are mostly dense, validate we keep as dense
        dense_matrix2 = csr_matrix(np.array([[4, 3, 2, 1], [8, 7, 6, 5], [12, 11, 10, 9]]))
        dense_matrices = [dense_matrix, dense_matrix, dense_matrix2]
        assert _should_compress_sparse_matrix(dense_matrices)
        # Create a (multiclass) list of sparse and dense matrices, more dense than sparse
        mixed_more_dense_matrices = [sparse_matrix, sparse_matrix2, dense_matrix, dense_matrix2, dense_matrix]
        assert _should_compress_sparse_matrix(mixed_more_dense_matrices)
        # Create a (multiclass) list of sparse and dense matrices, more sparse than dense
        mixed_more_sparse_matrices = [sparse_matrix, sparse_matrix2, sparse_matrix, sparse_matrix2, dense_matrix]
        assert not _should_compress_sparse_matrix(mixed_more_sparse_matrices)
