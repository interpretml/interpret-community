# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""End-to-end public API tests for supported Python versions."""

from inspect import signature

import numpy as np
import pandas as pd
import pytest
from interpret_community import TabularExplainer
from interpret_community.explanation.serialization import (load_explanation,
                                                           save_explanation)
from interpret_community.lime.lime_explainer import LIMEExplainer
from interpret_community.mimic.mimic_explainer import MimicExplainer
from interpret_community.mimic.models.linear_model import \
    LinearExplainableModel
from interpret_community.permutation.permutation_importance import PFIExplainer
from interpret_community.shap.kernel_explainer import KernelExplainer
from interpret_community.shap.linear_explainer import LinearExplainer
from interpret_community.shap.tree_explainer import TreeExplainer
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_diabetes, load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FEATURES = ['feature_0', 'feature_1', 'feature_2', 'feature_3']


@pytest.fixture(scope='module')
def iris_data():
    """Return deterministic multiclass train and evaluation data."""
    data = load_iris()
    return train_test_split(
        data.data, data.target, test_size=12, random_state=7,
        stratify=data.target)


def _assert_local_explanation(explanation, method, rows, features=4):
    assert explanation.method == method
    values = np.asarray(explanation.local_importance_values)
    assert values.shape[-2:] == (rows, features)
    assert np.isfinite(values).all()


def _assert_global_explanation(explanation, method, features=4):
    assert explanation.method == method
    values = np.asarray(explanation.global_importance_values)
    assert values.shape[-1] == features
    assert np.isfinite(values).all()


def test_tabular_dispatches_to_linear_and_tree_explainers(iris_data):
    """Exercise automatic linear and tree explainer selection."""
    x_train, x_test, y_train, _ = iris_data

    linear_model = LogisticRegression(max_iter=300).fit(x_train, y_train)
    linear = TabularExplainer(
        linear_model, x_train, features=FEATURES,
        classes=['setosa', 'versicolor', 'virginica'])
    assert isinstance(linear.explainer, LinearExplainer)
    _assert_local_explanation(
        linear.explain_local(x_test[:5]), 'shap_linear', 5)
    _assert_global_explanation(
        linear.explain_global(x_test[:5]), 'shap_linear')

    tree_model = RandomForestClassifier(
        n_estimators=8, max_depth=3, n_jobs=1,
        random_state=7).fit(x_train, y_train)
    tree = TabularExplainer(
        tree_model, x_train, features=FEATURES,
        classes=['setosa', 'versicolor', 'virginica'])
    assert isinstance(tree.explainer, TreeExplainer)
    _assert_local_explanation(
        tree.explain_local(x_test[:5]), 'shap_tree', 5)
    _assert_global_explanation(
        tree.explain_global(x_test[:5]), 'shap_tree')


def test_direct_linear_tree_and_kernel_apis(iris_data):
    """Exercise direct grey-box and black-box explainer APIs."""
    x_train, x_test, y_train, _ = iris_data

    linear_model = LogisticRegression(max_iter=300).fit(x_train, y_train)
    linear = LinearExplainer(
        linear_model, x_train, features=FEATURES,
        classes=['setosa', 'versicolor', 'virginica'])
    _assert_local_explanation(
        linear.explain_local(x_test[:3]), 'shap_linear', 3)

    tree_model = RandomForestClassifier(
        n_estimators=8, max_depth=3, n_jobs=1,
        random_state=7).fit(x_train, y_train)
    tree = TreeExplainer(
        tree_model, features=FEATURES,
        classes=['setosa', 'versicolor', 'virginica'])
    _assert_local_explanation(
        tree.explain_local(x_test[:3]), 'shap_tree', 3)

    kernel_model = KNeighborsClassifier(n_neighbors=3).fit(x_train, y_train)
    kernel = KernelExplainer(
        kernel_model, x_train[:20], features=FEATURES,
        classes=['setosa', 'versicolor', 'virginica'],
        nsamples=32, nclusters=2, show_progress=False)
    _assert_local_explanation(
        kernel.explain_local(x_test[:2]), 'shap_kernel', 2)


def test_mimic_multiclass_and_sparse_streaming(iris_data):
    """Exercise multiclass surrogate fitting and sparse aggregation."""
    x_train, x_test, y_train, _ = iris_data
    teacher = LogisticRegression(max_iter=300).fit(x_train, y_train)

    mimic = MimicExplainer(
        teacher, x_train, LinearExplainableModel,
        explainable_model_args={'solver': 'liblinear'},
        augment_data=False, features=FEATURES,
        classes=['setosa', 'versicolor', 'virginica'])
    if 'multi_class' not in signature(LogisticRegression).parameters:
        assert mimic.surrogate_model.model.solver == 'lbfgs'
    _assert_local_explanation(
        mimic.explain_local(x_test[:5]), 'mimic.linear', 5)
    _assert_global_explanation(
        mimic.explain_global(x_test[:5]), 'mimic.linear')

    sparse_train = sparse.csr_matrix(x_train)
    sparse_test = sparse.csr_matrix(x_test[:5])
    sparse_teacher = LogisticRegression(max_iter=300).fit(
        sparse_train, y_train)
    sparse_mimic = MimicExplainer(
        sparse_teacher, sparse_train, LinearExplainableModel,
        explainable_model_args={'sparse_data': True},
        augment_data=False, features=FEATURES,
        classes=['setosa', 'versicolor', 'virginica'])
    full = sparse_mimic.explain_global(sparse_test)
    streamed = sparse_mimic.explain_global(
        sparse_test, include_local=False, batch_size=2)
    np.testing.assert_allclose(
        full.global_importance_values,
        streamed.global_importance_values)


def test_pfi_and_lime_public_apis(iris_data):
    """Exercise permutation and LIME public APIs."""
    x_train, x_test, y_train, y_test = iris_data
    classifier = LogisticRegression(max_iter=300).fit(x_train, y_train)

    pfi = PFIExplainer(
        classifier, metric='f1_score',
        metric_args={'average': 'micro'}, features=FEATURES)
    _assert_global_explanation(
        pfi.explain_global(x_test, y_test), 'pfi')
    with pytest.raises(Exception, match='not in supported list'):
        PFIExplainer(
            classifier, metric='not-a-metric',
            metric_args={'average': 'micro'}, features=FEATURES)

    lime = LIMEExplainer(
        classifier, x_train, features=FEATURES,
        classes=['setosa', 'versicolor', 'virginica'],
        nclusters=2, show_progress=False)
    _assert_local_explanation(
        lime.explain_local(x_test[:2]), 'lime', 2)


def test_raw_feature_pipeline_and_serialization(tmp_path):
    """Exercise raw feature mapping and explanation serialization."""
    data = pd.DataFrame({
        'category_1': ['a', 'b', 'a', 'c'] * 12,
        'category_2': ['x', 'y', 'x', 'z'] * 12,
        'numeric': np.arange(48, dtype=float)
    })
    labels = np.array([0, 1, 0, 1] * 12)
    transformations = ColumnTransformer([
        ('categories', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), ['category_1', 'category_2']),
        ('numeric', Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), ['numeric'])
    ])
    transformed = transformations.fit_transform(data)
    model = LogisticRegression(max_iter=300).fit(transformed, labels)
    explainer = TabularExplainer(
        model, data, features=list(data.columns),
        classes=['zero', 'one'], transformations=transformations)
    assert isinstance(explainer.explainer, LinearExplainer)
    explanation = explainer.explain_local(data.iloc[:3])
    _assert_local_explanation(
        explanation, 'shap_linear', 3, features=3)
    assert explanation.features == list(data.columns)

    path = tmp_path / 'explanation'
    save_explanation(explanation, str(path))
    loaded = load_explanation(str(path))
    np.testing.assert_allclose(
        loaded.local_importance_values,
        explanation.local_importance_values)
    assert loaded.features == explanation.features


def test_regression_global_api():
    """Exercise regression global explanations with a linear model."""
    data = load_diabetes()
    x_train, x_test, y_train, _ = train_test_split(
        data.data[:, :4], data.target, test_size=8, random_state=7)
    model = LinearRegression().fit(x_train, y_train)
    explainer = TabularExplainer(
        model, x_train, features=FEATURES)
    _assert_global_explanation(
        explainer.explain_global(x_test), 'shap_linear')
