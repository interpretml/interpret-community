# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import pytest
from constants import owner_email_tools_and_ux
from interpret_community.common.error_handling import _format_exception


@pytest.mark.owner(email=owner_email_tools_and_ux)
@pytest.mark.usefixtures('_clean_dir')
class TestErrorHandling(object):
    def test_format_exception(self):
        error_str = "Some error"
        exception_type = "Exception"
        exception_str = _format_exception(Exception(error_str))
        assert error_str in exception_str
        assert exception_type in exception_str
