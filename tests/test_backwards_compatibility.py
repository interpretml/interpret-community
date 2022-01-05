# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Tests loading older models for backwards compatibility"""

import json
from os import path

import numpy as np
import pandas as pd
from datasets import retrieve_dataset


def test_model_backcompat_global(mimic_explainer):
    class DummyModel:
        def predict(self):
            return

        def predict_proba(self):
            return

    dummy_model = DummyModel()
    model_file = 'old_mimic_model.json'
    if not path.exists(model_file):
        model_file = path.join('tests', model_file)
    with open(model_file, 'r') as file:
        data = file.read()
    properties = json.loads(data)
    explainer = mimic_explainer._load(dummy_model, properties)
    # Use the surrogate model as the original model
    explainer.model = explainer.surrogate_model
    global_explanation = explainer.explain_global()
    assert len(global_explanation.global_importance_values) == explainer.surrogate_model.model._n_features


def test_model_backcompat_local(mimic_explainer):
    class DummyModel:
        def predict(self, X):
            return X['TotalBalance']

    dummy_model = DummyModel()
    model_file = 'old_mimic_model2.json'
    if not path.exists(model_file):
        model_file = path.join('tests', model_file)
    with open(model_file, 'r') as file:
        data = file.read()
    properties = json.loads(data)
    explainer = mimic_explainer._load(dummy_model, properties)
    eval_data = retrieve_dataset('backcompat_data.csv')
    df = pd.DataFrame(np.random.randint(0, eval_data.shape[0], size=(eval_data.shape[0], 674 - 5)))
    eval_data = eval_data[eval_data.columns[-5:]]
    eval_data = pd.concat([df, eval_data], axis=1)
    local_explanation = explainer.explain_local(eval_data)
    assert local_explanation._local_importance_values.shape[1] == explainer.surrogate_model.model._n_features
    global_explanation = explainer.explain_global(eval_data)
    assert len(global_explanation.global_importance_values) == explainer.surrogate_model.model._n_features
