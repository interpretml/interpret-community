import pytest


@pytest.fixture(scope='session')
def tracking_uri():
    return 'sqlite:///mlflowtest.db'
