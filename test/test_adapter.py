# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import pytest

# Tests for adapter
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from interpret.blackbox import ShapKernel

import pandas as pd
import numpy as np

from interpret_community.explanation.adapter import data_to_explanation

from constants import owner_email_tools_and_ux


@pytest.mark.owner(email=owner_email_tools_and_ux)
@pytest.mark.usefixtures('clean_dir')
class TestAdapter(object):
    def test_working(self):
        assert True
    def test_data_to_explanation(self, verify_tabular):
        interpret_explanation = self.create_kernel_explanation()
        interpret_explanation_data = interpret_explanation.data()
        explanation = data_to_explanation(interpret_explanation_data)

    def create_kernel_explanation(self):
        boston = load_boston()
        feature_names = list(boston.feature_names)
        df = pd.DataFrame(boston.data, columns=feature_names)
        df["target"] = boston.target
        # df = df.sample(frac=0.1, random_state=1)
        train_cols = df.columns[0:-1]
        label = df.columns[-1]
        X = df[train_cols]
        y = df[label]
        seed = 1
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=seed)
        pca = PCA()
        rf = RandomForestRegressor(n_estimators=100, n_jobs=-1)
        blackbox_model = Pipeline([('pca', pca), ('rf', rf)])
        blackbox_model.fit(X_train, y_train)
        background_val = np.median(X_train, axis=0).reshape(1, -1)
        shap = ShapKernel(predict_fn=blackbox_model.predict, data=background_val, feature_names=feature_names)
        shap_local = shap.explain_local(X_test[:5], y_test[:5], name='SHAP')
        return shap_local