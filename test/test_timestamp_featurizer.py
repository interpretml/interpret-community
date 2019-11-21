# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import pytest
import logging

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype as is_datetime

from interpret_community.dataset.dataset_wrapper import CustomTimestampFeaturizer
from constants import DatasetConstants, owner_email_tools_and_ux
from common_utils import create_timeseries_data

test_logger = logging.getLogger(__name__)


@pytest.mark.owner(email=owner_email_tools_and_ux)
@pytest.mark.usefixtures('clean_dir')
class TestTimestampFeaturizer(object):

    def test_working(self):
        assert True

    def test_no_timestamps(self, iris):
        # create pandas dataframes without any timestamps
        x_train = pd.DataFrame(data=iris[DatasetConstants.X_TRAIN], columns=iris[DatasetConstants.FEATURES])
        x_test = pd.DataFrame(data=iris[DatasetConstants.X_TEST], columns=iris[DatasetConstants.FEATURES])
        featurizer = CustomTimestampFeaturizer(iris[DatasetConstants.FEATURES]).fit(x_train)
        result = featurizer.transform(x_test)
        # Assert result is same as before, pandas dataframe
        assert(isinstance(result, pd.DataFrame))
        # Assert the result is the same as the original passed in data (no featurization was done)
        assert(result.equals(x_test))

    @pytest.mark.parametrize("sample_cnt_per_grain,grains_dict", [
        (240, {}),
        (20, {'fruit': ['apple', 'grape'], 'store': [100, 200, 50]})])
    def test_timestamp_featurization(self, sample_cnt_per_grain, grains_dict):
        # create timeseries data
        X, _ = create_timeseries_data(sample_cnt_per_grain, 'time', 'y', grains_dict)
        original_cols = list(X.columns.values)
        # featurize and validate the timestamp column
        featurizer = CustomTimestampFeaturizer(original_cols).fit(X)
        result = featurizer.transform(X)
        # Form a temporary dataframe for validation
        tmp_result = pd.DataFrame(result)
        # Assert there are no timestamp columns
        assert([column for column in tmp_result.columns if is_datetime(tmp_result[column])] == [])
        # Assert we have the expected number of columns - 1 time columns * 6 featurized plus original
        assert(result.shape[1] == len(original_cols) + 6)
