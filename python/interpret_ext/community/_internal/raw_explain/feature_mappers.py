# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

from abc import ABCMeta, abstractmethod
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Binarizer, KernelCenterer, LabelEncoder, MaxAbsScaler, MinMaxScaler, Normalizer, \
    OneHotEncoder, QuantileTransformer, RobustScaler, StandardScaler


def get_feature_mapper_for_pipeline(pipeline_obj):
    """Get FeatMapper object from a sklearn.pipeline.Pipeline object.

    :param pipeline_obj: pipeline object
    :type pipeline_obj: sklearn.pipeline.Pipeline
    :return: feat mapper for pipeline
    :rtype: PipelineFeatureMapper
    """
    """Get feat mapper for a pipeline, iterating over the transformers."""

    steps = []
    count = 0
    for _, transformer in pipeline_obj.steps:
        transformer_type = type(transformer)
        if transformer_type in encoders_to_mappers_dict:
            steps.append((str(count), encoders_to_mappers_dict[transformer_type](transformer)))
            count += 1
        elif transformer_type == Pipeline:
            steps.append((str(count), get_feature_mapper_for_pipeline(transformer)))
        else:
            steps.append((str(count), ManytoManyMapper(transformer)))

    return PipelineFeatureMapper(Pipeline(steps))


class FeatureMapper(object):
    """A class that supports both feature map from raw to engineered as well as a transform method."""

    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, transformer):
        """

        :param transformer: object that has a .transform method
        :type transformer: class that has a .transform method
        """
        self._transformer = transformer
        self._feature_map = None

    @property
    def transformer(self):
        return self._transformer

    @abstractmethod
    def transform(self, x):
        """

        :param x: input data
        :type x: pandas.DataFrame or numpy.array
        :rtype: transformed data
        :rtype: numpy.array or scipy.sparse matrix
        """
        pass

    @property
    def feature_map(self):
        """

        :return: feature map from raw to generated
        :rtype: numpy.array
        """
        if self._feature_map is None:
            raise ValueError("transform not called")
        return self._feature_map

    def set_feature_map(self, feature_map):
        """

        :param feature_map: feature map associated with the FeatureMapper
        :type feature_map: numpy.array
        """
        self._feature_map = feature_map

    def fit(self, x):
        """Dummy fit so that this can go in an sklearn.pipeline.Pipeline.

        :param x: input data
        :type x: numpy.array or pandas.DataFrame
        :return: self
        :rtype: FeatureMapper
        """
        return self


class IdentityMapper(FeatureMapper):
    """FeatMapper for one to one mappings."""

    def __init__(self, transformer):
        """

        :param transformer: object that has a .transform method
        :type transformer: class that has a .transform method
        """
        super(IdentityMapper, self).__init__(transformer)

    def _build_feature_map(self, num_cols):
        """Build a feature map for one to one mappings from raw to generated.

        :param num_cols: number of columns in input.
        :type num_cols: int
        """
        self.set_feature_map(np.eye(num_cols))

    def transform(self, x):
        """Transform input data.

        :param x: input data
        :type x: numpy.array or pandas.DataFrame
        :return: transformed data
        :rtype: numpy.array
        """
        result = self.transformer.transform(x)
        if self._feature_map is None:
            self._build_feature_map(result.shape[1])

        return result


class PassThroughMapper(FeatureMapper):
    """FeatureMapper to use when only one column is the input."""

    def __init__(self, transformer):
        """

        :param transformer: object that has a .transform method
        :type transformer: class that has a .transform method
        """
        super(PassThroughMapper, self).__init__(transformer)

    def _build_feature_map(self, num_output_cols):
        """Build a feature map for mappings from raw to generated when the input column is just one.

        :param num_output_cols: number of output columns
        :type num_output_cols: [int]
        """
        self.set_feature_map(np.ones((1, num_output_cols)))

    def transform(self, x):
        """

        :param x: input data
        :type x: numpy.array or pandas.DataFrame
        :return: transformed data
        :rytpe: numpy.array or scipy.sparse matrix
        """
        x_transformed = self.transformer.transform(x)
        if self._feature_map is None:
            self._build_feature_map(x_transformed.shape[1])

        return x_transformed


class PipelineFeatureMapper(FeatureMapper):
    """FeatureMapper for a sklearn pipeline of feat mappers"""

    def __init__(self, transformer):
        """

        :param transformer: object that has a .transform method
        :type transformer: class that has a .transform method
        """
        super(PipelineFeatureMapper, self).__init__(transformer)

    def _build_feature_map(self, feature_mappers):
        """Build a feature map for mappings from raw to generated for a pipeline of FeatMapper's.

        :param feature_mappers: list of feat mappers
        :type feature_mappers: [FeatureMapper]
        """

        curr_map = feature_mappers[0].feature_map
        for next_map in feature_mappers[1:]:
            curr_map = curr_map.dot(next_map.feature_map)

        self.set_feature_map(curr_map)

    def transform(self, x):
        """

        :param x: input data
        :type x: numpy.array or pandas.DataFrame
        :return: transformed data
        :rtype: numpy.array or scipy.sparse matrix
        """
        ret = self.transformer.transform(x)
        if self._feature_map is None:
            self._build_feature_map([s[1] for s in self._transformer.steps])

        return ret


class OneHotEncoderMapper(FeatureMapper):
    """OneHotEncoder FeatureMapper"""
    def __init__(self, transformer):
        """Build a feature map for OneHotEncoder.

        :param transformer: object of type onehotencoder
        :type transformer: sklearn.preprocessing.OneHotEncoder
        """
        super(OneHotEncoderMapper, self).__init__(transformer)

    def _build_feature_map(self):
        """Build feature map when transformer is a one hot encoder."""
        cat_lens = [len(cat) for cat in self.transformer.categories_]
        output_cols = int(sum(cat_lens))
        input_cols = len(self.transformer.categories_)
        feature_map = np.zeros((input_cols, output_cols))
        last_num_cols = 0
        for i, cat_len in enumerate(cat_lens):
            feature_map[i, last_num_cols:(last_num_cols + cat_len)] = np.ones((cat_len))
            last_num_cols += cat_len

        self.set_feature_map(feature_map)

    def transform(self, x):
        """

        :param x: input data
        :type x: numpy.array
        :return: transformed data
        :rtype x: numpy.array
        """
        ret = self.transformer.transform(x)
        if self._feature_map is None:
            self._build_feature_map()

        return ret


class ManytoManyMapper(FeatureMapper):
    def __init__(self, transformer):
        """Build a feature map for a many to many transformer.

        :param transformer: object that has a .transform method
        :type transformer: class that has a .transform method
        """
        super(ManytoManyMapper, self).__init__(transformer)
        self._transformer = transformer

    def _build_feature_map(self, in_feature_len=None, out_feature_len=None):
        """Generate a feature map so that weights of generated features are equally divided between parents and every
        parent feature map has every generated feature.

        :param in_feature_len: number of input features
        :type in_feature_len: int
        :param out_feature_len: number of output features
        :type out_feature_len: int
        """
        self.set_feature_map(np.ones((in_feature_len, out_feature_len)) * 1.0 / in_feature_len)

    def transform(self, x):
        """Get transformed data from input.

        :param x: input data
        :type x: numpy.array or pandas.DataFrame
        :return: transformed data
        :rtype: numpy.array or scipy.sparse matrix
        """
        x_transformed = self._transformer.transform(x)
        if self._feature_map is None:
            self._build_feature_map(in_feature_len=x.shape[1], out_feature_len=x_transformed.shape[1])

        return x_transformed


class FuncTransformer:
    def __init__(self, func):
        """

        :param func: function that transforms the data
        :type func: function that takes in numpy.array/pandas.Dataframe and outputs numpy.array or scipy.sparse matrix
        """
        self._func = func

    def transform(self, x):
        """

        :param x: input data
        :type x: numpy.array or pandas.DataFrame
        :return: transformed data
        :rtype: numpy.array or scipy.sparse matrix
        """
        return self._func(x)


# dictionary containing currently identified preprocessors/transformers that result in one to many maps.
encoders_to_mappers_dict = {
    Binarizer: IdentityMapper,
    KernelCenterer: IdentityMapper,
    LabelEncoder: IdentityMapper,
    MaxAbsScaler: IdentityMapper,
    MinMaxScaler: IdentityMapper,
    Normalizer: IdentityMapper,
    QuantileTransformer: IdentityMapper,
    RobustScaler: IdentityMapper,
    StandardScaler: IdentityMapper,
    OneHotEncoder: OneHotEncoderMapper,
}

try:
    from sklearn.impute import MissingIndicator, SimpleImputer
    from sklearn.preprocessing import KBinsDiscretizer, OrdinalEncoder, PowerTransformer

    encoders_to_mappers_dict.update([
        (SimpleImputer, IdentityMapper),
        (KBinsDiscretizer, IdentityMapper),
        (MissingIndicator, IdentityMapper),
        (OrdinalEncoder, IdentityMapper),
        (PowerTransformer, IdentityMapper)
    ])
except ImportError:
    # sklearn version earlier than 0.20.0
    pass
