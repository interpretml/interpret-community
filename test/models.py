# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import os


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
