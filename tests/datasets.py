import os


def retrieve_dataset(dataset, **kwargs):
    # if data not extracted, download zip and extract
    outdirname = 'datasets.4.27.2021'
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
    extension = os.path.splitext(dataset)[1]
    filepath = os.path.join(outdirname, dataset)
    if extension == '.npz':
        # sparse format file
        from scipy.sparse import load_npz
        return load_npz(filepath)
    elif extension == '.svmlight':
        from sklearn import datasets
        return datasets.load_svmlight_file(filepath)
    elif extension == '.json':
        import json
        with open(filepath, encoding='utf-8') as f:
            dataset = json.load(f)
        return dataset
    elif extension == '.csv':
        import pandas as pd
        return pd.read_csv(filepath, **kwargs)
    else:
        raise Exception('Unrecognized file extension: ' + extension)
