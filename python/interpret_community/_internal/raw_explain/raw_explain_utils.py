# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Contains functions useful for generating raw explanation."""

from interpret_community.dataset.dataset_wrapper import DatasetWrapper
from .data_mapper import DataMapper


def transform_with_datamapper(x, datamapper):
    """Transform the input using the _datamapper field in obj.

    :param x: input data
    :type x: numpy array or pandas DataFrame or DatasetWrapper
    :param datamapper: datamapper object
    :type datamapper: DataMapper
    :return: transformed data
    :rtype: numpy.array or scipy.sparse matrix or DatasetWrapper
    """
    x_is_dataset_wrapper = isinstance(x, DatasetWrapper)
    input_data = x
    if x_is_dataset_wrapper:
        input_data = x.original_dataset_with_type
    transformed_x = datamapper.transform(input_data)

    if x_is_dataset_wrapper:
        return DatasetWrapper(transformed_x)
    else:
        return transformed_x


def get_datamapper_and_transformed_data(examples=None, transformations=None, allow_all_transformations=False):
    """Get data mapper as well as transformed examples.

    :param examples: input data
    :type examples: numpy array or pandas DataFrame or DatasetWrapper
    :param transformations: transformations passed from any of DeepExplainer, KernelExplainer, MimicExplainer,
    TreeExplainer and TabularExplainer
    :type transformations: documented in constructor params of any of DeepExplainer, KernelExplainer, MimicExplainer,
    TreeExplainer and TabularExplainer
    :param allow_all_transformations: whether to allow transformations other than one to many.
    :type allow_all_transformations: bool
    :return: tuple of data mapper and transformed data
    :rtype: tuple[DataMapper, <data_type>] where <data_type> is one of numpy array or pandas DataFrame or
    DatasetWrapper
    """
    data_mapper = DataMapper(transformations, allow_all_transformations=allow_all_transformations)
    if examples is not None:
        examples = transform_with_datamapper(examples, data_mapper)

    return data_mapper, examples
