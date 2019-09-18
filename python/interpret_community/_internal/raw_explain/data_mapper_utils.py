# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Contains functions useful for data mapper."""

import pandas


def _get_list_from_slice(s):
    """Get a list from a slice object.

    :param s: slice object representing list of columns
    :type s: slice
    :return: list of elements that were in the slice object
    :rtype: list[int]
    """
    args = [s.start, s.stop]
    if s.step is not None:
        args.append(s.step)
    return list(range(*args))


def extract_column(x, transformer_config):
    """Extract column from DataFrame/numpy array x and reshape if column_name is list.

    :param x: input raw data
    :type x: numpy array or pandas.DataFrame
    :param transformer_config: TransformerConfig used to reshape input data
    :type transformer_config: TransformerConfig
    :return: column or columns from input data
    :rtype: numpy.array
    """
    columns = transformer_config.columns

    if isinstance(x, pandas.DataFrame):
        x_column = x[columns].values
    else:
        x_column = x[:, columns]
    if len(x.shape) == 1 and not transformer_config.one_dimensional:
        x_column = x_column.reshape(-1, 1)

    return x_column


def get_transformer_config_tuples_from_transformations_list(transformations):
    """Get transformer and transformer config tuples from list of transformations in sklearn-pandas format.

    :param transformations: list of transformations to be applied to input data in sklearn-pandas format
    :type transformations: list[tuple]
    :return: list of two element tuples. First element is a transformer object and second element is a
    TransformerConfig object
    :rtype: list[tuple]
    """
    tuples = []
    for columns, transformer in transformations:
        one_dimensional = False
        if isinstance(columns, str) or isinstance(columns, int):
            one_dimensional = True
            columns = [columns]
        tuples.append((transformer, TransformerConfig(columns, one_dimensional)))

    return tuples


def get_transformer_config_tuples_from_column_transformer(column_transformer):
    """Get tuples of transformer, transformer config from sklearn.compose.ColumnTransformer

    :param column_transformer: ColumnTransformer that transforms the input data
    :type column_transformer: sklearn.compose.ColumnTransformer
    :return: list of tuples of the form of (transformer, TransformerConfig)
    :rtype: list[tuple]
    """
    tuples = []
    for _, transformer, columns in column_transformer.transformers_:
        if transformer == "drop":
            continue
        elif transformer == "passthrough":
            transformer = None

        if isinstance(columns, slice):
            columns = _get_list_from_slice(columns)

        one_dimensional = False
        if isinstance(columns, str) or isinstance(columns, int):
            one_dimensional = True
            columns = [columns]

        tuples.append((transformer, TransformerConfig(columns, one_dimensional)))

    return tuples


class TransformerConfig:
    """Class that stores information about the columns on which a transformation operates."""

    def __init__(self, columns=None, one_dimensional=False):
        """Initialize the TransformerConfig object.

        :param one_dimensional: boolean flag to indicate whether the data is one dimensional.
        :type one_dimensional: bool
        :param columns: list of input columns
        :type columns: [str]
        """
        self._one_dimensional = one_dimensional
        self._columns = columns

    @property
    def one_dimensional(self):
        """Bool that indicates whether the given transformer expects one dimensional input."""
        return self._one_dimensional

    @property
    def columns(self):
        """List of columns which the transformer takes as input."""
        return self._columns
