# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Tests loading older models for backwards compatibility"""

import json
from os import path


def test_model_backcompat(mimic_explainer):
    class DummyModel:
        def predict(self):
            return

        def predict_proba(self):
            return

    dummy_model = DummyModel()
    model_file = 'old_mimic_model.json'
    if not path.exists(model_file):
        model_file = path.join('test', model_file)
    with open(model_file, 'r') as file:
        data = file.read()
    properties = json.loads(data)
    explainer = mimic_explainer._load(dummy_model, properties)
    # Use the surrogate model as the original model
    explainer.model = explainer.surrogate_model
    global_explanation = explainer.explain_global()
    assert len(global_explanation.global_importance_values) == explainer.surrogate_model.model._n_features
