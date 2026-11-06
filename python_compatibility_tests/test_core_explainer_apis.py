# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""End-to-end public API tests for supported Python versions."""

import pickle
from inspect import signature

import numpy as np
import pandas as pd
import pytest
import shap
from interpret_community import TabularExplainer
from interpret_community.common.base_explainer import BaseExplainer
from interpret_community.common.constants import ModelTask
from interpret_community.common.explanation_utils import _summarize_data
from interpret_community.dataset.decorator import (init_tabular_decorator,
                                                   tabular_decorator)
from interpret_community.explanation.serialization import (load_explanation,
                                                           save_explanation)
from interpret_community.lime.lime_explainer import LIMEExplainer
from interpret_community.mimic.mimic_explainer import MimicExplainer
from interpret_community.mimic.models.linear_model import \
    LinearExplainableModel
from interpret_community.permutation.permutation_importance import PFIExplainer
from interpret_community.shap.deep_explainer import DeepExplainer
from interpret_community.shap.gpu_kernel_explainer import GPUKernelExplainer
from interpret_community.shap.kernel_explainer import KernelExplainer
from interpret_community.shap.linear_explainer import LinearExplainer
from interpret_community.shap.tree_explainer import TreeExplainer
from ml_wrappers import DatasetWrapper
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


@pytest.mark.parametrize(('explainer_class', 'method'), [
    (TabularExplainer, 'shap_linear'),
    (LinearExplainer, 'shap_linear'),
    (KernelExplainer, 'shap_kernel'),
    (MimicExplainer, 'mimic.linear'),
    (LIMEExplainer, 'lime')
])
@pytest.mark.parametrize('infer_features', [False, True])
def test_raw_feature_pipeline_and_serialization(
        tmp_path, explainer_class, method, infer_features):
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
    kwargs = {}
    if explainer_class in (KernelExplainer, LIMEExplainer):
        kwargs.update(nclusters=2, show_progress=False)
    if explainer_class is MimicExplainer:
        kwargs.update(explainable_model=LinearExplainableModel, augment_data=False)
    explainer = explainer_class(
        model, data, features=None if infer_features else list(data.columns),
        classes=['zero', 'one'], transformations=transformations, **kwargs)
    if explainer_class is TabularExplainer:
        assert isinstance(explainer.explainer, LinearExplainer)
    explanation = explainer.explain_local(data.iloc[:3])
    _assert_local_explanation(
        explanation, method, 3, features=3)
    expected_features = list(data.columns)
    if infer_features and explainer_class not in (LIMEExplainer, MimicExplainer):
        expected_features = None
    assert explanation.features == expected_features
    repeated = explainer.explain_local(data.iloc[:3])
    _assert_local_explanation(repeated, method, 3, features=3)
    streamed = explainer.explain_global(
        data.iloc[:3], include_local=False, batch_size=2)
    _assert_global_explanation(streamed, method, features=3)
    assert streamed.features == expected_features
    with pytest.raises(ValueError, match='Number of features: 3 feature names: 7'):
        explainer_class(
            model, data, features=['engineered'] * transformed.shape[1],
            classes=['zero', 'one'], transformations=transformations, **kwargs)

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


@pytest.mark.parametrize('explainer_class', [
    TabularExplainer, LinearExplainer, KernelExplainer, DeepExplainer,
    GPUKernelExplainer, LIMEExplainer, MimicExplainer
])
@pytest.mark.parametrize('feature_count', [0, 3, 5])
@pytest.mark.parametrize('data_type', ['numpy', 'numpy_matrix', 'dataframe', 'sparse', 'wrapped'])
def test_initialization_rejects_mismatched_features(
        explainer_class, feature_count, data_type):
    data = np.ones((5, 4))
    if data_type == 'numpy_matrix':
        data = np.asmatrix(data)
    elif data_type == 'dataframe':
        data = pd.DataFrame(data, columns=FEATURES)
    elif data_type == 'sparse':
        data = sparse.csr_matrix(data)
    elif data_type == 'wrapped':
        data = DatasetWrapper(data)
    kwargs = {'features': ['feature'] * feature_count}
    if explainer_class is MimicExplainer:
        kwargs['explainable_model'] = LinearExplainableModel
    with pytest.raises(ValueError, match=(
            'Number of features: 4 feature names: {}'.format(feature_count))):
        explainer_class(object(), data, **kwargs)


def test_initialization_validates_positional_features():
    class Explainer:
        @init_tabular_decorator
        def __init__(self, model, initialization_examples, features=None):
            pytest.fail('Invalid features must be rejected before initialization')

    with pytest.raises(ValueError, match='Number of features: 4 feature names: 1'):
        Explainer(object(), np.ones((5, 4)), ['only_one'])


def test_tabular_decorator_without_feature_attributes():
    class CustomExplainer(BaseExplainer):
        def explain_global(self, evaluation_examples):
            return self.explain_local(evaluation_examples)

        @tabular_decorator
        def explain_local(self, evaluation_examples):
            return evaluation_examples.dataset.shape

    explainer = CustomExplainer()
    assert explainer.explain_local(np.ones((3, 4))) == (3, 4)


@pytest.mark.parametrize('explainer_class', [TreeExplainer, PFIExplainer])
@pytest.mark.parametrize('feature_count', [0, 3, 5])
@pytest.mark.parametrize('include_local', [True, False])
def test_evaluation_rejects_mismatched_features(
        iris_data, explainer_class, feature_count, include_local):
    x_train, x_test, y_train, y_test = iris_data
    model = RandomForestClassifier(
        n_estimators=2, random_state=7).fit(x_train, y_train)
    explainer = explainer_class(model, features=['feature'] * feature_count)
    message = 'Number of features: 4 feature names: {}'.format(feature_count)
    if explainer_class is PFIExplainer:
        args, kwargs = (x_test, y_test), {}
    else:
        args, kwargs = (x_test,), {'include_local': include_local}
    with pytest.raises(ValueError, match=message):
        explainer.explain_global(*args, **kwargs)
    if explainer_class is TreeExplainer:
        with pytest.raises(ValueError, match=message):
            explainer.explain_local(x_test)


@pytest.mark.parametrize('features', [None, FEATURES, np.array(FEATURES), list(range(4))])
def test_matching_and_inferred_features(iris_data, features):
    x_train, x_test, y_train, _ = iris_data
    model = LogisticRegression(max_iter=300).fit(x_train, y_train)
    explainer = TabularExplainer(model, x_train, features=features)
    explanation = explainer.explain_local(x_test[:2])
    _assert_local_explanation(explanation, 'shap_linear', 2)
    assert explanation.num_features == 4
    if features is not None:
        assert explanation.features == list(features)


def test_tree_initialization_reads_dataframe_metadata_only(iris_data):
    class MetadataOnlyFrame(pd.DataFrame):
        @property
        def values(self):
            pytest.fail('Feature-width validation must not materialize DataFrame values')

    x_train, _, y_train, _ = iris_data
    model = RandomForestClassifier(n_estimators=2, random_state=7).fit(x_train, y_train)
    frame = MetadataOnlyFrame({
        FEATURES[0]: x_train[:, 0] > 5,
        FEATURES[1]: x_train[:, 1] > 3,
        FEATURES[2]: x_train[:, 2],
        FEATURES[3]: x_train[:, 3]
    })
    explainer = TabularExplainer(model, frame, features=FEATURES)
    assert isinstance(explainer.explainer, TreeExplainer)
    with pytest.raises(ValueError, match='Number of features: 4 feature names: 3'):
        TabularExplainer(model, frame, features=FEATURES[:-1])


def test_subset_validates_full_feature_names(iris_data):
    x_train, x_test, y_train, y_test = iris_data
    model = LinearRegression().fit(x_train, y_train)
    explainer = PFIExplainer(
        model, features=FEATURES, explain_subset=[0, 2], show_progress=False)
    explanation = explainer.explain_global(x_test, y_test)
    _assert_global_explanation(explanation, 'pfi', features=2)
    assert explanation.features == [FEATURES[0], FEATURES[2]]


def test_reusing_wrapped_subset_data(iris_data):
    x_train, x_test, y_train, _ = iris_data
    model = LogisticRegression(max_iter=300).fit(x_train, y_train)
    explainer = LIMEExplainer(
        model, x_train[:20], features=FEATURES, classes=['zero', 'one', 'two'],
        explain_subset=[0, 2], nclusters=2, show_progress=False)
    wrapped = DatasetWrapper(x_test[:2].copy())
    for _ in range(2):
        explanation = explainer.explain_local(wrapped)
        _assert_local_explanation(explanation, 'lime', 2, features=2)
        assert explanation.features == [FEATURES[0], FEATURES[2]]


def test_mimic_inferred_names_survive_featurization():
    data = np.array([[i % 3, i % 2, i] for i in range(36)], dtype=float)
    model = LinearRegression().fit(data, data.sum(axis=1))
    explainer = MimicExplainer(
        model, data, LinearExplainableModel, augment_data=False,
        categorical_features=[0, 1])
    full = explainer.explain_global(data[:4])
    streamed = explainer.explain_global(data[:4], include_local=False, batch_size=2)
    np.testing.assert_allclose(
        full.global_importance_values, streamed.global_importance_values)
    _assert_local_explanation(explainer.explain_local(data[:2]), 'mimic.linear', 2, features=6)


@pytest.mark.parametrize('reset_index', ['reset', True, 'reset_teacher'])
@pytest.mark.parametrize('index_type', ['named', 'multi', 'default'])
def test_mimic_index_feature_names(reset_index, index_type):
    data = pd.DataFrame({
        'a': np.arange(20, dtype=float),
        'b': np.arange(20, dtype=float) ** 2
    })
    if index_type == 'named':
        data.index = pd.Index(np.arange(20) + 100, name='row_id')
    elif index_type == 'multi':
        data.index = pd.MultiIndex.from_arrays(
            [np.arange(20) + 100, np.arange(20) + 200], names=['row_id', 'group_id'])
    features = list(data.columns)
    if index_type != 'default' and reset_index != 'reset_teacher':
        features += list(data.index.names)
    model = LinearRegression().fit(data, data['a'] * 2)
    kwargs = {'reset_index': reset_index, 'augment_data': False}
    explainer = MimicExplainer(
        model, data, LinearExplainableModel, features=features, **kwargs)
    for _ in range(2):
        explanation = explainer.explain_local(data.iloc[:2])
        _assert_local_explanation(
            explanation, 'mimic.linear', 2, features=len(features))
        assert explanation.features == features
    for bad_features in (features[:-1], features + ['extra']):
        with pytest.raises(ValueError, match='number of feature names'):
            MimicExplainer(
                model, data, LinearExplainableModel, features=bad_features, **kwargs)


@pytest.mark.parametrize('explainer_class', [TabularExplainer, KernelExplainer, LIMEExplainer])
def test_summarized_initialization_validates_features(iris_data, explainer_class):
    x_train, x_test, y_train, _ = iris_data
    model = LogisticRegression(max_iter=300).fit(x_train, y_train)
    summary = _summarize_data(x_train, 2)
    kwargs = {}
    if explainer_class in (KernelExplainer, LIMEExplainer):
        kwargs.update(nclusters=2, show_progress=False)
    explainer = explainer_class(
        model, summary, features=FEATURES, classes=['zero', 'one', 'two'], **kwargs)
    explanation = explainer.explain_local(x_test[:2])
    assert np.asarray(explanation.local_importance_values).shape == (3, 2, 4)
    assert explanation.features == FEATURES
    with pytest.raises(ValueError, match='Number of features: 4 feature names: 3'):
        explainer_class(
            model, summary, features=FEATURES[:-1], classes=['zero', 'one', 'two'], **kwargs)


@pytest.mark.parametrize('explainer_class', [TabularExplainer, LinearExplainer])
@pytest.mark.parametrize('background_type', [
    'numpy_matrix', 'tuple', 'statistics', 'independent', 'partition', 'impute',
    'independent_statistics', 'impute_statistics',
    'mean_only_statistics', 'independent_mean_only_statistics',
    'none_covariance_statistics', 'tuple_none_covariance'
])
def test_linear_background_validates_features(iris_data, explainer_class, background_type):
    x_train, x_test, y_train, _ = iris_data
    model = LinearRegression().fit(x_train, y_train)
    statistics = {'mean': x_train.mean(axis=0), 'cov': np.cov(x_train, rowvar=False)}
    if 'mean_only' in background_type:
        statistics.pop('cov')
    elif 'none_covariance' in background_type:
        statistics['cov'] = None
    if background_type == 'numpy_matrix':
        background = np.asmatrix(x_train)
    elif background_type.startswith('tuple'):
        background = (statistics['mean'], statistics['cov'])
    elif background_type in ('statistics', 'mean_only_statistics', 'none_covariance_statistics'):
        background = statistics
    else:
        masker_class = {
            'independent': shap.maskers.Independent,
            'partition': shap.maskers.Partition,
            'impute': shap.maskers.Impute
        }[background_type.split('_')[0]]
        background = masker_class(
            statistics if background_type.endswith('_statistics') else x_train)
    if isinstance(background, dict):
        try:
            shap.LinearExplainer(model, background)
        except NotImplementedError:
            pytest.skip('Installed SHAP does not support bare dictionary backgrounds')
    explainer = explainer_class(model, background, features=FEATURES)
    _assert_local_explanation(explainer.explain_local(x_test[:2]), 'shap_linear', 2)
    for bad_features in ([], ['only_one'], FEATURES[:-1], FEATURES + ['extra']):
        with pytest.raises(ValueError, match='Number of features: 4 feature names:'):
            explainer_class(model, background, features=bad_features)


def _sum_predictions(data):
    return np.asarray(data).sum(axis=1)


def test_legacy_mimic_pickle_without_input_features():
    data = np.array([[i % 3, i % 2, i] for i in range(36)], dtype=float)
    explainer = MimicExplainer(
        _sum_predictions, data, LinearExplainableModel, is_function=True,
        model_task=ModelTask.Regression, augment_data=False,
        categorical_features=[0, 1])
    explainer.explain_local(data[:2])
    del explainer._input_features
    restored = pickle.loads(pickle.dumps(explainer))
    for _ in range(2):
        explanation = restored.explain_local(data[:2])
        _assert_local_explanation(explanation, 'mimic.linear', 2, features=6)
    streamed = restored.explain_global(data[:4], include_local=False, batch_size=2)
    _assert_global_explanation(streamed, 'mimic.linear', features=6)
