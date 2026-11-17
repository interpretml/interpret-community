# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Held-out surrogate selection without changing MimicExplainer."""

import pickle
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from interpret_community.common.constants import ModelTask, ResetIndex
from interpret_community.common.exception import ScenarioNotSupportedException
from interpret_community.mimic import MimicExplainer
from interpret_community.mimic import model_selection as selection
from interpret_community.mimic import select_mimic_explainer
from interpret_community.mimic.models import (DecisionTreeExplainableModel,
                                              LGBMExplainableModel,
                                              LinearExplainableModel,
                                              SGDExplainableModel)
from ml_wrappers import DatasetWrapper
from scipy import sparse
from scipy.special import expit
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_classification
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

CANDIDATE_ARGS = {LGBMExplainableModel: {'n_estimators': 8, 'n_jobs': 1, 'verbosity': -1}}


@pytest.fixture
def regression_data():
    data = np.random.RandomState(7).normal(size=(100, 4))
    teacher = LinearRegression().fit(data[:80], data[:80] @ np.array([3, -2, 1, 4]))
    return teacher, data[:80], data[80:]


def _record_explainers(monkeypatch):
    instances = {}

    def construct(model, data, candidate, **kwargs):
        explainer = MimicExplainer(model, data, candidate, **kwargs)
        instances[candidate] = explainer
        return explainer

    monkeypatch.setattr(selection, 'MimicExplainer', construct)
    return instances


@pytest.mark.parametrize('use_sparse', [False, True])
def test_default_candidates_regression(monkeypatch, regression_data, use_sparse):
    teacher, train, evaluation = regression_data
    if use_sparse:
        train, evaluation = sparse.csr_matrix(train), sparse.csr_matrix(evaluation)
    instances = _record_explainers(monkeypatch)
    arguments = dict(CANDIDATE_ARGS)
    arguments[LinearExplainableModel] = {'sparse_data': use_sparse}
    best, scores = select_mimic_explainer(
        teacher, train, evaluation, augment_data=False, candidate_args=arguments)
    assert scores[LGBMExplainableModel] is not None
    assert scores[SGDExplainableModel] is not None
    for candidate, score in scores.items():
        if score is not None:
            expected = r2_score(
                teacher.predict(evaluation), instances[candidate].surrogate_model.predict(evaluation))
            assert score == pytest.approx(expected)
    winner = max((cls for cls in scores if scores[cls] is not None), key=scores.get)
    assert best is instances[winner]
    assert 'classification' not in arguments[LinearExplainableModel]
    assert CANDIDATE_ARGS[LGBMExplainableModel] == {
        'n_estimators': 8, 'n_jobs': 1, 'verbosity': -1}


@pytest.mark.parametrize(('num_classes', 'display_classes', 'teacher_type'), [
    (2, False, 'model'), (2, True, 'model'), (3, False, 'model'),
    (3, True, 'model'), (2, False, 'function'), (3, False, 'function'),
    (2, False, 'single_column')
])
def test_classification_scores_distillation_targets(
        monkeypatch, num_classes, display_classes, teacher_type):
    data, labels = make_classification(
        n_samples=120, n_features=4, n_informative=3, n_redundant=0,
        n_classes=num_classes, n_clusters_per_class=1, random_state=17)
    labels = np.array(['label-{}'.format(label + 10) for label in labels])
    train, evaluation, y_train, _ = train_test_split(
        data, labels, test_size=24, random_state=17, stratify=labels)
    teacher = LogisticRegression(max_iter=300).fit(train, y_train)
    if num_classes == 2:
        monkeypatch.setattr(
            teacher, 'predict', lambda examples: np.repeat(teacher.classes_[0], examples.shape[0]))
    instances = _record_explainers(monkeypatch)
    classes = ['display-{}'.format(i) for i in range(num_classes)] if display_classes else None
    model = teacher
    if teacher_type == 'function':
        model = teacher.predict_proba
    elif teacher_type == 'single_column':
        def model(examples):
            return teacher.predict_proba(examples)[:, 1:2]
    best, scores = select_mimic_explainer(
        model, train, evaluation,
        classes=classes, is_function=teacher_type != 'model', model_task=ModelTask.Classification,
        augment_data=False, candidate_args=CANDIDATE_ARGS)
    expected_labels = np.argmax(teacher.predict_proba(evaluation), axis=1)
    assert all(score is not None for score in scores.values())
    for candidate, score in scores.items():
        predictions = instances[candidate].surrogate_model.predict(evaluation)
        if num_classes == 2:
            predictions = (expit(predictions) > 0.5).astype(int)
        assert score == pytest.approx(accuracy_score(expected_labels, predictions))
    winner = max(scores, key=scores.get)
    assert best is instances[winner]
    assert 0.0 <= best.get_surrogate_model_replication_measure(evaluation[:1]) <= 1.0


@pytest.mark.parametrize('input_type', ['array', 'dataframe', 'wrapper'])
def test_categorical_preprocessing_preserves_inputs(input_type):
    data = np.empty((100, 2), dtype=object)
    data[:, 0] = np.tile(['a', 'b'], 50)
    data[:, 1] = np.random.RandomState(8).normal(size=100)
    if input_type != 'array':
        data = pd.DataFrame(data, columns=['category', 'value'])
    train, evaluation = data[:80].copy(), data[80:].copy()
    original_train, original_evaluation = np.asarray(train).copy(), np.asarray(evaluation).copy()

    def teacher(examples):
        examples = np.asarray(examples)
        margin = examples[:, 1].astype(float) + np.where(examples[:, 0] == 'a', 1, -1)
        probabilities = expit(margin)
        return np.column_stack((1 - probabilities, probabilities))

    if input_type == 'wrapper':
        train, evaluation = DatasetWrapper(train), DatasetWrapper(evaluation)
    best, scores = select_mimic_explainer(
        teacher, train, evaluation, is_function=True, model_task=ModelTask.Classification,
        categorical_features=[0], augment_data=False, candidate_args=CANDIDATE_ARGS)
    assert isinstance(best, MimicExplainer)
    assert scores[LGBMExplainableModel] is not None
    assert scores[SGDExplainableModel] is not None
    assert best.get_surrogate_model_replication_measure(evaluation) == pytest.approx(
        max(score for score in scores.values() if score is not None))
    if input_type == 'wrapper':
        train, evaluation = train.original_dataset_with_type, evaluation.original_dataset_with_type
    np.testing.assert_array_equal(np.asarray(train), original_train)
    np.testing.assert_array_equal(np.asarray(evaluation), original_evaluation)


@pytest.mark.parametrize('legacy', [False, True])
@pytest.mark.parametrize('reduce_features', [False, True])
def test_explicit_transformations_score_engineered_teacher_inputs(regression_data, legacy, reduce_features):
    _, train, evaluation = regression_data
    train = train * 10 + 50
    evaluation = evaluation * 10 + 50
    if reduce_features:
        transformation = ColumnTransformer([('scale', StandardScaler(), [0, 1])]).fit(train)
    else:
        transformation = Pipeline([('scale', StandardScaler())]).fit(train)
    engineered = transformation.transform(train)
    teacher = LinearRegression().fit(engineered, engineered @ np.arange(1, engineered.shape[1] + 1))
    best, scores = select_mimic_explainer(
        teacher, train, evaluation, candidates=[LinearExplainableModel],
        transformations=transformation if reduce_features else [(list(range(train.shape[1])), transformation)],
        reset_index=ResetIndex.Reset, augment_data=False)
    assert scores[LinearExplainableModel] == pytest.approx(1.0)
    expected = teacher.predict(transformation.transform(evaluation))
    assert r2_score(expected, best.surrogate_model.predict(transformation.transform(evaluation))) == pytest.approx(1.0)
    if legacy:
        del best._index_feature_count
        best = pickle.loads(pickle.dumps(best))
    assert best.get_surrogate_model_replication_measure(evaluation) == pytest.approx(1.0)
    assert best.explain_local(evaluation[:3]).num_examples == 3


def test_timestamp_preprocessing():
    data = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=100),
        'value': np.random.RandomState(7).normal(size=100)
    })

    def teacher(examples):
        return examples['date'].dt.day.to_numpy() + examples['value'].to_numpy()

    _, scores = select_mimic_explainer(
        teacher, data[:80], data[80:], is_function=True, model_task=ModelTask.Regression,
        candidates=[LGBMExplainableModel],
        candidate_args=CANDIDATE_ARGS, augment_data=False)
    assert all(score is not None for score in scores.values())


@pytest.mark.parametrize('reset_index', [
    ResetIndex.Ignore, ResetIndex.Reset, ResetIndex.ResetTeacher
])
def test_index_preprocessing(reset_index):
    data = pd.DataFrame(
        np.random.RandomState(9).normal(size=(100, 2)), columns=['a', 'b'],
        index=pd.Index(np.arange(100) + 10, name='row'))
    teacher = LinearRegression().fit(data[:80], 2 * data['a'][:80] - data['b'][:80])
    _, scores = select_mimic_explainer(
        teacher, data[:80], data[80:], candidates=[LinearExplainableModel],
        reset_index=reset_index, augment_data=False)
    assert scores[LinearExplainableModel] == pytest.approx(1.0)


def test_negative_scores_ties_failures_and_no_refit(monkeypatch, regression_data):
    teacher, train, evaluation = regression_data
    original = train.copy()
    values = {LGBMExplainableModel: -2.0, LinearExplainableModel: None,
              SGDExplainableModel: -1.0, DecisionTreeExplainableModel: -1.0}
    instances = []

    def construct(model, data, candidate, **kwargs):
        np.testing.assert_array_equal(data, original)
        data[:] = 99
        explainer = SimpleNamespace(
            candidate=candidate, get_surrogate_model_replication_measure=lambda data: score(candidate, data))
        instances.append(explainer)
        return explainer

    def score(candidate, data):
        if values[candidate] is None:
            raise RuntimeError('candidate scoring failed')
        return values[candidate]

    monkeypatch.setattr(selection, 'MimicExplainer', construct)
    best, scores = select_mimic_explainer(teacher, train, evaluation)
    assert best is instances[2]
    assert len(instances) == 4
    assert scores == values
    np.testing.assert_array_equal(train, original)


@pytest.mark.parametrize('reset_index', [ResetIndex.Ignore, ResetIndex.Reset, ResetIndex.ResetTeacher])
@pytest.mark.parametrize('index_offset', [0, 10])
def test_index_dependent_teacher_uses_original_validation_index(reset_index, index_offset):
    data = pd.DataFrame({'value': np.arange(100, dtype=float)},
                        index=pd.Index(np.arange(100) + index_offset, name='row'))

    def teacher(examples):
        return examples.index.to_numpy(dtype=float)

    best, scores = select_mimic_explainer(
        teacher, data[:80], data[80:], candidates=[LinearExplainableModel, DecisionTreeExplainableModel],
        is_function=True, model_task=ModelTask.Regression, augment_data=False, reset_index=reset_index)
    assert scores[LinearExplainableModel] == pytest.approx(1.0)
    surrogate_data = data[80:].to_numpy()
    if index_offset and reset_index == ResetIndex.Reset:
        surrogate_data = np.column_stack((surrogate_data, data[80:].index.to_numpy()))
    expected = r2_score(teacher(data[80:]), best.surrogate_model.predict(surrogate_data))
    assert best.get_surrogate_model_replication_measure(data[80:]) == pytest.approx(expected)


@pytest.mark.parametrize('training_has_index', [False, True])
@pytest.mark.parametrize('reset_index', [ResetIndex.Reset, True, ResetIndex.ResetTeacher])
def test_index_layout_is_fitted_not_reinferred(training_has_index, reset_index):
    data = pd.DataFrame(
        np.random.RandomState(9).normal(size=(100, 2)), columns=['a', 'b'])
    data.index.name = 'row'
    train, evaluation = data[:80].copy(), data[80:].copy()
    if training_has_index:
        train.index += 10
        evaluation.reset_index(drop=True, inplace=True)
    teacher = LinearRegression().fit(train, 2 * train['a'] - train['b'])
    _, scores = select_mimic_explainer(
        teacher, train, evaluation, candidates=[LinearExplainableModel],
        reset_index=reset_index, augment_data=False)
    assert scores[LinearExplainableModel] == pytest.approx(1.0)


@pytest.mark.parametrize('legacy', [False, True])
@pytest.mark.parametrize('serialization', ['pickle', 'json'])
@pytest.mark.parametrize('column_target', [False, True])
def test_serialized_metric_preserves_fitted_index_layout(legacy, serialization, column_target):
    data = pd.DataFrame(
        np.random.RandomState(9).normal(size=(100, 2)), columns=['a', 'b'],
        index=pd.Index(np.arange(100) + 10, name='row'))
    train, evaluation = data[:80], data[80:].reset_index(drop=True)
    target = 2 * train['a'] - train['b']
    teacher = LinearRegression().fit(train, target.to_frame() if column_target else target)
    explainer = MimicExplainer(
        teacher, train, LGBMExplainableModel, reset_index=ResetIndex.Reset,
        explainable_model_args=CANDIDATE_ARGS[LGBMExplainableModel].copy(),
        model_task=ModelTask.Regression, augment_data=False)
    if legacy:
        del explainer._index_feature_count
    if serialization == 'pickle':
        restored = pickle.loads(pickle.dumps(explainer))
    else:
        restored = MimicExplainer._load(teacher, explainer._save())
    assert restored.get_surrogate_model_replication_measure(evaluation) == pytest.approx(
        explainer.get_surrogate_model_replication_measure(evaluation))
    with pytest.raises(ScenarioNotSupportedException, match='single instance'):
        restored.get_surrogate_model_replication_measure(evaluation[:1])


@pytest.mark.parametrize('index_type', ['multi', 'datetime'])
def test_fitted_index_levels_use_saved_preprocessing(index_type):
    data = pd.DataFrame(np.random.RandomState(9).normal(size=(100, 2)), columns=['a', 'b'])
    if index_type == 'multi':
        data.index = pd.MultiIndex.from_arrays(
            [np.arange(100) + 10, np.arange(100) + 20], names=['row', 'group'])
    else:
        data.index = pd.date_range('2020-01-01', periods=100, name='date')
    teacher = LinearRegression().fit(data[:80], 2 * data['a'][:80] - data['b'][:80])
    explainer = MimicExplainer(
        teacher, data[:80].copy(), LGBMExplainableModel, reset_index=ResetIndex.Reset,
        explainable_model_args=CANDIDATE_ARGS[LGBMExplainableModel].copy(), augment_data=False)
    assert np.isfinite(explainer.get_surrogate_model_replication_measure(data[80:]))


@pytest.mark.parametrize('fallback_fails', [False, True])
def test_all_failures_use_only_configured_fallback(
        monkeypatch, regression_data, fallback_fails):
    teacher, train, evaluation = regression_data
    calls = []
    fallback_args = {'fit_intercept': False}
    log_warning = Mock()
    monkeypatch.setattr(selection._logger, 'warning', log_warning)

    def construct(model, data, candidate, explainable_model_args, **kwargs):
        calls.append((candidate, explainable_model_args.copy()))
        explainable_model_args['mutated'] = True
        if candidate is LGBMExplainableModel:
            raise RuntimeError('candidate fit failed')
        if fallback_fails:
            raise ValueError('fallback fit failed')
        return SimpleNamespace(candidate=candidate)

    monkeypatch.setattr(selection, 'MimicExplainer', construct)
    kwargs = dict(candidates=[LGBMExplainableModel], explainable_model=LinearExplainableModel,
                  explainable_model_args=fallback_args)
    if fallback_fails:
        with pytest.raises(ValueError, match='fallback fit failed'):
            select_mimic_explainer(teacher, train, evaluation, **kwargs)
    else:
        best, scores = select_mimic_explainer(teacher, train, evaluation, **kwargs)
        assert best.candidate is LinearExplainableModel
        assert scores == {LGBMExplainableModel: None}
    assert calls == [(LGBMExplainableModel, {}), (LinearExplainableModel, fallback_args)]
    assert fallback_args == {'fit_intercept': False}
    log_warning.assert_called_once()
    assert log_warning.call_args.kwargs['exc_info'] is True


def test_nonfinite_score_is_unavailable(monkeypatch, regression_data):
    teacher, train, evaluation = regression_data
    monkeypatch.setattr('interpret_community.mimic.mimic_explainer.r2_score', lambda *args: np.nan)
    _, scores = select_mimic_explainer(
        teacher, train, evaluation, candidates=[LinearExplainableModel],
        explainable_model=LinearExplainableModel, augment_data=False)
    assert scores == {LinearExplainableModel: None}


def test_public_metric_preserves_regression_contract_and_wrapped_inputs(regression_data):
    teacher, train, evaluation = regression_data
    explainer = MimicExplainer(teacher, train, LinearExplainableModel, augment_data=False)
    expected = r2_score(teacher.predict(evaluation), explainer.surrogate_model.predict(evaluation))
    wrapped = DatasetWrapper(evaluation.copy())
    score = explainer.get_surrogate_model_replication_measure(wrapped)
    assert isinstance(score, float)
    assert score == pytest.approx(expected)
    np.testing.assert_array_equal(wrapped.dataset, evaluation)
    with pytest.raises(ScenarioNotSupportedException, match='single instance'):
        explainer.get_surrogate_model_replication_measure(evaluation[:1])


@pytest.mark.parametrize('invalid', ['teacher_nan', 'surrogate_nan', 'surrogate_inf'])
def test_public_metric_rejects_nonfinite_outputs_before_decoding(monkeypatch, regression_data, invalid):
    _, train, evaluation = regression_data

    def teacher(examples):
        probabilities = expit(np.asarray(examples)[:, 0])
        return np.column_stack((1 - probabilities, probabilities))

    explainer = MimicExplainer(
        teacher, train, LinearExplainableModel, is_function=True,
        model_task=ModelTask.Classification, augment_data=False)
    if invalid == 'teacher_nan':
        monkeypatch.setattr(explainer, 'function', lambda examples: np.full((examples.shape[0], 2), np.nan))
    else:
        value = np.nan if invalid == 'surrogate_nan' else np.inf
        monkeypatch.setattr(
            explainer.surrogate_model, 'predict', lambda examples: np.full(examples.shape[0], value))
    with pytest.raises(ValueError, match='predictions must be finite'):
        explainer.get_surrogate_model_replication_measure(evaluation)


@pytest.mark.parametrize('invalid', [
    'empty_training', 'short_evaluation', 'wrong_width', 'wrong_columns',
    'no_candidates', 'duplicates', 'bad_candidate', 'bad_fallback', 'unknown_args'
])
def test_invalid_selection_inputs_fail_before_training(monkeypatch, regression_data, invalid):
    teacher, train, evaluation = regression_data
    kwargs = {}
    if invalid == 'empty_training':
        train = train[:0]
    elif invalid == 'short_evaluation':
        evaluation = evaluation[:1]
    elif invalid == 'wrong_width':
        evaluation = evaluation[:, :2]
    elif invalid == 'wrong_columns':
        train = pd.DataFrame(train, columns=list('abcd'))
        evaluation = pd.DataFrame(evaluation, columns=list('abdc'))
    elif invalid == 'no_candidates':
        kwargs['candidates'] = []
    elif invalid == 'duplicates':
        kwargs['candidates'] = [LinearExplainableModel, LinearExplainableModel]
    elif invalid == 'bad_candidate':
        kwargs['candidates'] = [LinearRegression]
    elif invalid == 'bad_fallback':
        kwargs['explainable_model'] = LinearRegression
    else:
        kwargs.update(candidates=[LinearExplainableModel], candidate_args=CANDIDATE_ARGS)

    def unexpected_construction(*args, **kwargs):
        pytest.fail('Invalid input must fail before constructing an explainer')

    monkeypatch.setattr(selection, 'MimicExplainer', unexpected_construction)
    with pytest.raises(ValueError, match='data|columns|Candidates|Candidate|fallback|candidate_args'):
        select_mimic_explainer(teacher, train, evaluation, **kwargs)
