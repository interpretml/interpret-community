# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import pytest
import datetime
import numpy as np
import pandas as pd

from interpret_community.common.serialization_utils import _serialize_json_safe

from constants import owner_email_tools_and_ux


@pytest.mark.owner(email=owner_email_tools_and_ux)
@pytest.mark.usefixtures('clean_dir')
class TestSerializationUtils(object):

    def test_serialize_json_safe_basic(self):
        values = [0, 1, 2, 3, 4, 5]
        result = _serialize_json_safe(values)
        assert result == [0, 1, 2, 3, 4, 5]

        values = ['a', 'b', 'a', 'c', 'a', 'b']
        result = _serialize_json_safe(values)
        assert result == ['a', 'b', 'a', 'c', 'a', 'b']

    def test_serialize_json_safe_missing(self):
        values = [0, np.nan, 2, 3, 4, 5]
        result = _serialize_json_safe(values)
        assert result == [0, 0, 2, 3, 4, 5]

        values = [0, np.inf, 2, 3, 4, 5]
        result = _serialize_json_safe(values)
        assert result == [0, 0, 2, 3, 4, 5]

        values = ['a', 'b', 'a', np.nan, 'a', 'b']
        result = _serialize_json_safe(values)
        assert result == ['a', 'b', 'a', 0, 'a', 'b']

    def test_serialize_json_safe_aggregate_types(self):
        o = {
            'a': [1, 2, 3],
            'c': 'b'
        }
        result = _serialize_json_safe(o)
        assert result == o

        o = ('a', [1, 2, 3])
        result = _serialize_json_safe(o)
        assert result == o

        values = np.array([[1, 2, 3], [4, 5, 6]])
        result = _serialize_json_safe(values)
        assert result == values.tolist()

    def test_serialize_timestamp(self):
        datetime_str = "2020-10-10"
        datetime_object = datetime.datetime.strptime(datetime_str, "%Y-%m-%d")
        result = _serialize_json_safe(datetime_object)
        assert datetime_str in result

    def test_serialize_via_json_timestamp(self):
        timestamp_obj = pd.Timestamp(2020, 1, 1)
        assert isinstance(timestamp_obj, pd._libs.tslibs.timestamps.Timestamp)
        result = _serialize_json_safe(timestamp_obj)
        assert result is not None
        assert "2020" in result
