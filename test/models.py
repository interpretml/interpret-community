# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import os
import pandas as pd
import numpy as np


def retrieve_model(model, **kwargs):
    # if data not extracted, download zip and extract
    outdirname = 'models.5.15.2019'
    if not os.path.exists(outdirname):
        try:
            from urllib import urlretrieve
        except ImportError:
            from urllib.request import urlretrieve
        import zipfile
        zipfilename = outdirname + '.zip'
        urlretrieve('https://publictestdatasets.blob.core.windows.net/data/' + zipfilename, zipfilename)
        with zipfile.ZipFile(zipfilename, 'r') as unzip:
            unzip.extractall('.')
    extension = os.path.splitext(model)[1]
    filepath = os.path.join(outdirname, model)
    if extension == '.pkl':
        from joblib import load
        return load(filepath, **kwargs)
    else:
        raise Exception('Unrecognized file extension: ' + extension)


class DataFrameTestModel(object):
    def __init__(self, sample_df):
        self._sample_df = sample_df
    
    def fit(self, X):
        self._assert_df_index_and_columns(X)

    def predict(self, X_pred):
        self._assert_df_index_and_columns(X_pred)
        return np.repeat(0, X_pred.shape[0])

    def _assert_df_index_and_columns(self, X):
        assert isinstance(X, pd.DataFrame)
        assert list(X.index.names) == list(self._sample_df.index.names)
        assert list(X.columns) == list(self._sample_df.columns)
