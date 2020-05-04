# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Defines a helpful dataset wrapper to allow operations such as summarizing data, taking the subset or sampling."""

import pandas as pd
from scipy.sparse import issparse
import numpy as np

from ..common.explanation_utils import _summarize_data, _generate_augmented_data
from ..common.explanation_utils import module_logger
from ..common.constants import Defaults
from sklearn.base import TransformerMixin, BaseEstimator
from pandas.api.types import is_datetime64_any_dtype as is_datetime

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', 'Starting from version 2.2.1', UserWarning)
    from shap.common import DenseData


class CustomTimestampFeaturizer(BaseEstimator, TransformerMixin):
    """An estimator for featurizing timestamp columns to numeric data.

    :param features: Feature column names.
    :type features: list[str]
    """

    def __init__(self, features):
        """Initialize the CustomTimestampFeaturizer.

        :param features: Feature column names.
        :type features: list[str]
        """
        self._features = features
        self._time_col_names = []
        return

    def fit(self, X):
        """Fits the CustomTimestampFeaturizer.

        :param X: The dataset containing timestamp columns to featurize.
        :type X: numpy.array or pandas.DataFrame or iml.datatypes.DenseData or
            scipy.sparse.csr_matrix
        """
        # If the data was previously successfully summarized, then there are no
        # timestamp columns as it must be numeric.
        # Also, if the dataset is sparse, we can assume there are no timestamps
        if isinstance(X, DenseData) or issparse(X):
            return self
        tmp_dataset = X
        # If numpy array, temporarily convert to pandas for easier and uniform timestamp handling
        if isinstance(X, np.ndarray):
            tmp_dataset = pd.DataFrame(X, columns=self._features)
        self._time_col_names = [column for column in tmp_dataset.columns if is_datetime(tmp_dataset[column])]
        # Calculate the min date for each column
        self._min = []
        for time_col_name in self._time_col_names:
            self._min.append(tmp_dataset[time_col_name].map(lambda x: x.timestamp()).min())
        return self

    def transform(self, X):
        """Transforms the timestamp columns to numeric type in the given dataset.

        Specifically, extracts the year, month, day, hour, minute, second and time
        since min timestamp in the training dataset.

        :param X: The dataset containing timestamp columns to featurize.
        :type X: numpy.array or pandas.DataFrame or iml.datatypes.DenseData or
            scipy.sparse.csr_matrix
        :return: The transformed dataset.
        :rtype: numpy.array or iml.datatypes.DenseData or scipy.sparse.csr_matrix
        """
        tmp_dataset = X
        if len(self._time_col_names) > 0:
            # Temporarily convert to pandas for easier and uniform timestamp handling
            if isinstance(X, np.ndarray):
                tmp_dataset = pd.DataFrame(X, columns=self._features)
            # Get the year, day, month, hour, minute, second
            for idx, time_col_name in enumerate(self._time_col_names):
                tmp_dataset[time_col_name + '_year'] = tmp_dataset[time_col_name].map(lambda x: x.year)
                tmp_dataset[time_col_name + '_month'] = tmp_dataset[time_col_name].map(lambda x: x.month)
                tmp_dataset[time_col_name + '_day'] = tmp_dataset[time_col_name].map(lambda x: x.day)
                tmp_dataset[time_col_name + '_hour'] = tmp_dataset[time_col_name].map(lambda x: x.hour)
                tmp_dataset[time_col_name + '_minute'] = tmp_dataset[time_col_name].map(lambda x: x.minute)
                tmp_dataset[time_col_name + '_second'] = tmp_dataset[time_col_name].map(lambda x: x.second)
                # Replace column itself with difference from min value, leave as same name
                # to keep index so order of other columns remains the same for other transformations
                tmp_dataset[time_col_name] = tmp_dataset[time_col_name].map(lambda x: x.timestamp() - self._min[idx])
            X = tmp_dataset.values
        return X


class DatasetWrapper(object):
    """A wrapper around a dataset to make dataset operations more uniform across explainers.

    :param dataset: A matrix of feature vector examples (# examples x # features) for
        initializing the explainer.
    :type dataset: numpy.array or pandas.DataFrame or iml.datatypes.DenseData or
        scipy.sparse.csr_matrix
    """

    def __init__(self, dataset):
        """Initialize the dataset wrapper.

        :param dataset: A matrix of feature vector examples (# examples x # features) for
            initializing the explainer.
        :type dataset: numpy.array or pandas.DataFrame or iml.datatypes.DenseData or
            scipy.sparse.csr_matrix
        """
        self._features = None
        self._original_dataset_with_type = dataset
        self._dataset_is_df = isinstance(dataset, pd.DataFrame)
        self._dataset_is_series = isinstance(dataset, pd.Series)
        self._default_index_cols = ['index']
        self._default_index = True
        if self._dataset_is_df:
            self._features = dataset.columns.values.tolist()
        if self._dataset_is_df or self._dataset_is_series:
            dataset = dataset.values
        self._dataset = dataset
        self._original_dataset = dataset
        self._summary_dataset = None
        self._column_indexer = None
        self._subset_taken = False
        self._summary_computed = False
        self._string_indexed = False
        self._one_hot_encoded = False
        self._one_hot_encoder = None
        self._timestamp_featurized = False
        self._timestamp_featurizer = None

    @property
    def dataset(self):
        """Get the dataset.

        :return: The underlying dataset.
        :rtype: numpy.array or iml.datatypes.DenseData or scipy.sparse.csr_matrix
        """
        return self._dataset

    @property
    def typed_dataset(self):
        """Get the dataset in the original type, pandas DataFrame or Series.

        :return: The underlying dataset.
        :rtype: numpy.array or pandas.DataFrame or pandas.Series or iml.datatypes.DenseData or scipy.sparse matrix
        """
        wrapper_func = self.typed_wrapper_func
        return wrapper_func(self._dataset)

    def typed_wrapper_func(self, dataset, keep_index_as_feature=False):
        """Get a wrapper function to convert the dataset to the original type, pandas DataFrame or Series.

        :param dataset: The dataset to convert to original type.
        :type dataset: numpy.array or scipy.sparse.csr_matrix
        :param keep_index_as_feature: Whether to keep the index as a feature when converting back.
            Off by default to convert it back to index.
        :type keep_index_as_feature: bool
        :return: A wrapper function for a given dataset to convert to original type.
        :rtype: function that outputs the original type
        """
        if self._dataset_is_df:
            if len(dataset.shape) == 1:
                dataset = dataset.reshape(1, dataset.shape[0])
            original_dtypes = self._original_dataset_with_type.dtypes
            output_types = dict(original_dtypes)
            dataframe = pd.DataFrame(dataset, columns=self._features)
            if not self._default_index:
                if keep_index_as_feature:
                    # Add the index name to type as feature dtype
                    for idx, name in enumerate(self._original_dataset_with_type.index.names):
                        level_values_dtype = self._original_dataset_with_type.index.get_level_values(idx).dtype
                        output_types.update({name: level_values_dtype})
                else:
                    dataframe = dataframe.set_index(self._default_index_cols)
            return dataframe.astype(output_types)
        elif self._dataset_is_series:
            return pd.Series(dataset)
        else:
            return dataset

    @property
    def original_dataset(self):
        """Get the original dataset prior to performing any operations.

        Note: if the original dataset was a pandas dataframe, this will return the numpy version.

        :return: The original dataset.
        :rtype: numpy.array or iml.datatypes.DenseData or scipy.sparse matrix
        """
        return self._original_dataset

    @property
    def original_dataset_with_type(self):
        """Get the original typed dataset which could be a numpy array or pandas DataFrame or pandas Series.

        :return: The original dataset.
        :rtype: numpy.array or pandas.DataFrame or pandas.Series or iml.datatypes.DenseData or scipy.sparse matrix
        """
        return self._original_dataset_with_type

    @property
    def num_features(self):
        """Get the number of features (columns) on the dataset.

        :return: The number of features (columns) in the dataset.
        :rtype: int
        """
        evaluation_examples_temp = self._dataset
        if isinstance(evaluation_examples_temp, pd.DataFrame):
            evaluation_examples_temp = evaluation_examples_temp.values
        if len(evaluation_examples_temp.shape) == 1:
            return len(evaluation_examples_temp)
        elif issparse(evaluation_examples_temp):
            return evaluation_examples_temp.shape[1]
        else:
            return len(evaluation_examples_temp[0])

    @property
    def summary_dataset(self):
        """Get the summary dataset without any subsetting.

        :return: The original dataset or None if summary was not computed.
        :rtype: numpy.array or iml.datatypes.DenseData or scipy.sparse.csr_matrix
        """
        return self._summary_dataset

    def _set_default_index_cols(self, dataset):
        if dataset.index.names is not None:
            self._default_index_cols = dataset.index.names

    def set_index(self):
        """Undo reset_index.  Set index as feature on internal dataset to be an index again.
        """
        if self._dataset_is_df:
            dataset = self.typed_dataset
            self._features = dataset.columns.values.tolist()
            self._dataset = dataset.values
        self._default_index = True

    def reset_index(self):
        """Reset index to be part of the features on the dataset.
        """
        dataset = self._original_dataset_with_type
        if self._dataset_is_df:
            self._default_index = pd.Index(np.arange(0, len(dataset))).equals(dataset.index)
            reset_dataset = dataset
            if not self._default_index:
                self._set_default_index_cols(dataset)
                reset_dataset = dataset.reset_index()
                # Move index columns to the end of the dataframe to ensure
                # index arguments are still valid to original dataset
                dcols = reset_dataset.columns.tolist()
                for default_index_col in self._default_index_cols:
                    dcols.insert(len(dcols), dcols.pop(dcols.index(default_index_col)))
                reset_dataset = reset_dataset.reindex(columns=dcols)
            self._features = reset_dataset.columns.values.tolist()
            self._dataset = reset_dataset.values

    def get_features(self, features=None, explain_subset=None, **kwargs):
        """Get the features of the dataset if None on current kwargs.

        :return: The features of the dataset if currently None on kwargs.
        :rtype: list
        """
        if features is not None:
            if explain_subset is not None:
                return np.array(features)[explain_subset].tolist()
            return features
        if explain_subset is not None and self._features is not None:
            return np.array(self._features)[explain_subset].tolist()
        if self._features is None:
            return list(range(self._dataset.shape[1]))
        return self._features

    def get_column_indexes(self, features, categorical_features):
        """Get the column indexes for the given column names.

        :param features: The full list of existing column names.
        :type features: list[str]
        :param categorical_features: The list of categorical feature names to get indexes for.
        :type categorical_features: list[str]
        :return: The list of column indexes.
        :rtype: list[int]
        """
        return [features.index(categorical_feature) for categorical_feature in categorical_features]

    def string_index(self, columns=None):
        """Indexes categorical string features on the dataset.

        :param columns: Optional parameter specifying the subset of columns that may need to be string indexed.
        :type columns: list
        :return: The transformation steps to index the given dataset.
        :rtype: ColumnTransformer
        """
        if self._string_indexed:
            return self._column_indexer
        # Optimization so we don't redo this operation multiple times on the same dataset
        self._string_indexed = True
        # If the data was previously successfully summarized, then there are no
        # categorical columns as it must be numeric.
        # Also, if the dataset is sparse, we can assume there are no categorical strings
        if isinstance(self._dataset, DenseData) or issparse(self._dataset):
            return None
        # If the user doesn't have a newer version of scikit-learn with OrdinalEncoder, don't do encoding
        try:
            from sklearn.compose import ColumnTransformer
            from sklearn.preprocessing import OrdinalEncoder
        except ImportError:
            return None
        tmp_dataset = self._dataset
        # Temporarily convert to pandas for easier and uniform string handling
        if isinstance(self._dataset, np.ndarray):
            tmp_dataset = pd.DataFrame(self._dataset, dtype=self._dataset.dtype)
        categorical_col_names = list(np.array(list(tmp_dataset))[(tmp_dataset.applymap(type) == str).all(0)])
        if categorical_col_names:
            all_columns = tmp_dataset.columns
            if columns is not None:
                categorical_col_indices = \
                    [all_columns.get_loc(col_name) for col_name in categorical_col_names if col_name in columns]
            else:
                categorical_col_indices = [all_columns.get_loc(col_name) for col_name in categorical_col_names]
            ordinal_enc = OrdinalEncoder()
            ct = ColumnTransformer([('ord', ordinal_enc, categorical_col_indices)], remainder='drop')
            string_indexes_dataset = ct.fit_transform(tmp_dataset)
            # Inplace replacement of columns
            # (danger: using remainder=passthrough with ColumnTransformer will change column order!)
            for idx, categorical_col_index in enumerate(categorical_col_indices):
                self._dataset[:, categorical_col_index] = string_indexes_dataset[:, idx]
            self._column_indexer = ct
        return self._column_indexer

    def one_hot_encode(self, columns):
        """Indexes categorical string features on the dataset.

        :param columns: Parameter specifying the subset of column indexes that may need to be one-hot-encoded.
        :type columns: list[int]
        :return: The transformation steps to one-hot-encode the given dataset.
        :rtype: OneHotEncoder
        """
        if self._one_hot_encoded:
            return self._one_hot_encoder
        # Optimization so we don't redo this operation multiple times on the same dataset
        self._one_hot_encoded = True
        # If the data was previously successfully summarized, then there are no
        # categorical columns as it must be numeric.
        # Also, if the dataset is sparse, we can assume there are no categorical strings
        if not columns or isinstance(self._dataset, DenseData) or issparse(self._dataset):
            return None
        # If the user doesn't have a newer version of scikit-learn with OneHotEncoder, don't do encoding
        try:
            from sklearn.compose import ColumnTransformer
            from sklearn.preprocessing import OneHotEncoder
        except ImportError:
            return None
        one_hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
        self._one_hot_encoder = ColumnTransformer([('ord', one_hot_encoder, columns)], remainder='passthrough')
        # Note this will change column order, the one hot encoded columns will be at the start and the
        # rest of the columns at the end
        self._dataset = self._one_hot_encoder.fit_transform(self._dataset).astype(float)
        return self._one_hot_encoder

    def timestamp_featurizer(self):
        """Featurizes the timestamp columns.

        :return: The transformation steps to featurize the timestamp columns.
        :rtype: DatasetWrapper
        """
        if self._timestamp_featurized:
            return self._timestamp_featurizer
        # Optimization so we don't redo this operation multiple times on the same dataset
        self._timestamp_featurized = True
        # If the data was previously successfully summarized, then there are no
        # categorical columns as it must be numeric.
        # Also, if the dataset is sparse, we can assume there are no categorical strings
        if isinstance(self._dataset, DenseData) or issparse(self._dataset):
            return None
        typed_dataset_without_index = self.typed_wrapper_func(self._dataset, keep_index_as_feature=True)
        self._timestamp_featurizer = CustomTimestampFeaturizer(self._features).fit(typed_dataset_without_index)
        self._dataset = self._timestamp_featurizer.transform(self._dataset)
        return self._timestamp_featurizer

    def apply_indexer(self, column_indexer, bucket_unknown=False):
        """Indexes categorical string features on the dataset.

        :param column_indexer: The transformation steps to index the given dataset.
        :type column_indexer: ColumnTransformer
        :param bucket_unknown: If true, buckets unknown values to separate categorical level.
        :type bucket_unknown: bool
        """
        if self._string_indexed or issparse(self._dataset):
            return
        name, ordinal_encoder, cols = column_indexer.transformers_[0]
        all_categories = ordinal_encoder.categories_

        def convert_cols(category_to_index, value, unknown):
            if value in category_to_index:
                index = category_to_index[value]
            elif not bucket_unknown:
                # Add new index on the fly - note the background data does NOT need to
                # contain all possible category levels!
                index = len(category_to_index) + 1
                category_to_index[value] = index
            else:
                # Put all unknown indexes into a separate 'unknown' bucket
                index = unknown
                category_to_index[value] = index
            return index

        for idx, i in enumerate(cols):
            categories_for_col = all_categories[idx]
            category_to_index = dict(zip(categories_for_col, range(len(categories_for_col))))
            unknown = len(category_to_index) + 1
            self._dataset[:, i] = list(map(lambda x: convert_cols(category_to_index, x, unknown), self._dataset[:, i]))
        # Ensure element types are float and not object
        self._dataset = self._dataset.astype(float)
        self._string_indexed = True

    def apply_one_hot_encoder(self, one_hot_encoder):
        """One-hot-encode categorical string features on the dataset.

        :param one_hot_encoder: The transformation steps to one-hot-encode the given dataset.
        :type one_hot_encoder: OneHotEncoder
        """
        if self._one_hot_encoded or issparse(self._dataset):
            return
        self._dataset = one_hot_encoder.transform(self._dataset).astype(float)
        self._one_hot_encoded = True

    def apply_timestamp_featurizer(self, timestamp_featurizer):
        """Apply timestamp featurization on the dataset.

        :param timestamp_featurizer: The transformation steps to featurize timestamps in the given dataset.
        :type timestamp_featurizer: CustomTimestampFeaturizer
        """
        if self._timestamp_featurized or issparse(self._dataset):
            return
        self._dataset = timestamp_featurizer.transform(self._dataset)
        self._timestamp_featurized = True

    def compute_summary(self, nclusters=10, **kwargs):
        """Summarizes the dataset if it hasn't been summarized yet."""
        if self._summary_computed:
            return
        self._summary_dataset = _summarize_data(self._dataset, nclusters)
        self._dataset = self._summary_dataset
        self._summary_computed = True

    def augment_data(self, max_num_of_augmentations=np.inf):
        """Augment the current dataset.

        :param max_augment_data_size: number of times we stack permuted x to augment.
        :type max_augment_data_size: int
        """
        self._dataset = _generate_augmented_data(self._dataset, max_num_of_augmentations=max_num_of_augmentations)

    def take_subset(self, explain_subset):
        """Take a subset of the dataset if not done before.

        :param explain_subset: A list of column indexes to take from the original dataset.
        :type explain_subset: list
        """
        if self._subset_taken:
            return
        # Edge case: Take the subset of the summary in this case,
        # more optimal than recomputing the summary!
        explain_subset = np.array(explain_subset)
        if isinstance(self._dataset, DenseData):
            group_names = np.array(self._dataset.group_names)[explain_subset].tolist()
            self._dataset = DenseData(self._dataset.data[:, explain_subset], group_names)
        else:
            self._dataset = self._dataset[:, explain_subset]
        self._subset_taken = True

    def _reduce_examples(self, max_dim_clustering=Defaults.MAX_DIM):
        """Reduces the dimensionality of the examples if dimensionality is higher than max_dim_clustering.

        If the dataset is sparse, we mean-scale the data and then run
        truncated SVD to reduce the number of features to max_dim_clustering.  For dense
        dataset, we also scale the data and then run PCA to reduce the number of features to
        max_dim_clustering.
        This is used to get better clustering results in _find_k.

        :param max_dim_clustering: Dimensionality threshold for performing reduction.
        :type max_dim_clustering: int
        """
        from sklearn.decomposition import TruncatedSVD, PCA
        from sklearn.preprocessing import StandardScaler
        num_cols = self._dataset.shape[1]
        # Run PCA or SVD on input data and reduce to about MAX_DIM features prior to clustering
        components = min(max_dim_clustering, num_cols)
        reduced_examples = self._dataset
        if components != num_cols:
            if issparse(self._dataset):
                module_logger.debug('Reducing sparse data with StandardScaler and TruncatedSVD')
                normalized_examples = StandardScaler(with_mean=False).fit_transform(self._dataset)
                reducer = TruncatedSVD(n_components=components)
            else:
                module_logger.debug('Reducing normal data with StandardScaler and PCA')
                normalized_examples = StandardScaler().fit_transform(self._dataset)
                reducer = PCA(n_components=components)
            module_logger.info('reducing dimensionality to {} components for clustering'.format(str(components)))
            reduced_examples = reducer.fit_transform(normalized_examples)
        return reduced_examples

    def _find_k_kmeans(self, max_dim_clustering=Defaults.MAX_DIM):
        """Use k-means to downsample the examples.

        Starting from k_upper_bound, cuts k in half each time and run k-means
        clustering on the examples.  After each run, computes the
        silhouette score and stores k with highest silhouette score.
        We use optimal k to determine how much to downsample the examples.

        :param max_dim_clustering: Dimensionality threshold for performing reduction.
        :type max_dim_clustering: int
        """
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        from math import log, isnan, ceil
        reduced_examples = self._reduce_examples(max_dim_clustering)
        num_rows = self._dataset.shape[0]
        k_upper_bound = 2000
        k_list = []
        k = min(num_rows / 2, k_upper_bound)
        for i in range(int(ceil(log(num_rows, 2) - 7))):
            k_list.append(int(k))
            k /= 2
        prev_highest_score = -1
        prev_highest_index = 0
        opt_k = int(k)
        for k_index, k in enumerate(k_list):
            module_logger.info('running KMeans with k: {}'.format(str(k)))
            km = KMeans(n_clusters=k).fit(reduced_examples)
            clusters = km.labels_
            num_clusters = len(set(clusters))
            k_too_big = num_clusters <= 1
            if k_too_big or num_clusters == reduced_examples.shape[0]:
                score = -1
            else:
                score = silhouette_score(reduced_examples, clusters)
            if isnan(score):
                score = -1
            module_logger.info('KMeans silhouette score: {}'.format(str(score)))
            # Find k with highest silhouette score for optimal clustering
            if score >= prev_highest_score and not k_too_big:
                prev_highest_score = score
                prev_highest_index = k_index
        opt_k = k_list[prev_highest_index]
        module_logger.info('best silhouette score: {}'.format(str(prev_highest_score)))
        module_logger.info('found optimal k for KMeans: {}'.format(str(opt_k)))
        return opt_k

    def _find_k_hdbscan(self, max_dim_clustering=Defaults.MAX_DIM):
        """Use hdbscan to downsample the examples.

        We use optimal k to determine how much to downsample the examples.

        :param max_dim_clustering: Dimensionality threshold for performing reduction.
        :type max_dim_clustering: int
        """
        import hdbscan
        num_rows = self._dataset.shape[0]
        reduced_examples = self._reduce_examples(max_dim_clustering)
        hdbs = hdbscan.HDBSCAN(min_cluster_size=2).fit(reduced_examples)
        clusters = hdbs.labels_
        opt_k = len(set(clusters))
        clustering_threshold = 5
        samples = opt_k * clustering_threshold
        module_logger.info(('found optimal k for hdbscan: {},'
                            ' will use clustering_threshold * k for sampling: {}').format(str(opt_k), str(samples)))
        return min(samples, num_rows)

    def sample(self, max_dim_clustering=Defaults.MAX_DIM, sampling_method=Defaults.HDBSCAN):
        """Sample the examples.

        First does random downsampling to upper_bound rows,
        then tries to find the optimal downsample based on how many clusters can be constructed
        from the data.  If sampling_method is hdbscan, uses hdbscan to cluster the
        data and then downsamples to that number of clusters.  If sampling_method is k-means,
        uses different values of k, cutting in half each time, and chooses the k with highest
        silhouette score to determine how much to downsample the data.
        The danger of using only random downsampling is that we might downsample too much
        or too little, so the clustering approach is a heuristic to give us some idea of
        how much we should downsample to.

        :param max_dim_clustering: Dimensionality threshold for performing reduction.
        :type max_dim_clustering: int
        :param sampling_method: Method to use for sampling, can be 'hdbscan' or 'kmeans'.
        :type sampling_method: str
        """
        from sklearn.utils import resample
        # bounds are rough estimates that came from manual investigation
        lower_bound = 200
        upper_bound = 10000
        num_rows = self._dataset.shape[0]
        module_logger.info('sampling examples')
        # If less than lower_bound rows, just return the full dataset
        if num_rows < lower_bound:
            return self._dataset
        # If more than upper_bound rows, sample randomly
        elif num_rows > upper_bound:
            module_logger.info('randomly sampling to 10k rows')
            self._dataset = resample(self._dataset, n_samples=upper_bound, random_state=7)
            num_rows = upper_bound
        if sampling_method == Defaults.HDBSCAN:
            try:
                opt_k = self._find_k_hdbscan(max_dim_clustering)
            except Exception as ex:
                module_logger.warning(('Failed to use hdbscan due to error: {}'
                                      '\nEnsure hdbscan is installed with: pip install hdbscan').format(str(ex)))
                opt_k = self._find_k_kmeans(max_dim_clustering)
        else:
            opt_k = self._find_k_kmeans(max_dim_clustering)
        # Resample based on optimal number of clusters
        if (opt_k < num_rows):
            self._dataset = resample(self._dataset, n_samples=opt_k, random_state=7)
        return self._dataset
