# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import os


def extract_images(**kwargs):
    """Extracts images locally for testing.

    :return: The directory location of the extracted images.
    :rtype: str
    """
    # if data not extracted, download zip and extract
    outdirname = 'images.5.30.2019'
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
    return outdirname
