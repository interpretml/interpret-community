# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import pytest
import logging

import numpy as np

from interpret_community.explanation.explanation import BaseExplanation, FeatureImportanceExplanation, \
    LocalExplanation, GlobalExplanation, ExpectedValuesMixin, ClassesMixin, PerClassMixin, _DatasetsMixin, \
    _ModelIdMixin
from constants import owner_email_tools_and_ux

test_logger = logging.getLogger(__name__)


class BaseValid(object):
    @property
    def method(self):
        return 'some_string'

    @property
    def id(self):
        return 'some_other_string'

    @property
    def model_task(self):
        return 'some_third_string'

    @property
    def model_type(self):
        return None


class FeatureImportanceValid(object):
    @property
    def features(self):
        return None

    @property
    def num_features(self):
        return None

    @property
    def is_raw(self):
        return False

    @property
    def is_engineered(self):
        return False


class LocalValuesValid(object):
    @property
    def local_importance_values(self):
        return [[.2, .4, .01], [.3, .2, 0]]

    @property
    def num_examples(self):
        return None


class GlobalValid(object):
    @property
    def global_importance_rank(self):
        return [0, 1]

    @property
    def global_importance_values(self):
        return [.2, .4, .01]


class ExpectedValuesValid(object):
    @property
    def expected_values(self):
        return .37


class ClassesValid(object):
    @property
    def classes(self):
        return None

    @property
    def num_classes(self):
        return None


class PerClassValid(object):
    @property
    def per_class_rank(self):
        return [0, 1]

    @property
    def per_class_values(self):
        return [.2, .4, .03]


class _DatasetsValid(object):
    @property
    def init_data(self):
        return 'a_dataset_id'

    @property
    def eval_data(self):
        return[[.2, .4, .01], [.3, .2, 0]]

    @property
    def eval_y_predicted(self):
        return None

    @property
    def eval_y_predicted_proba(self):
        return None


class _ModelIdValid(object):
    @property
    def model_id(self):
        return None


@pytest.mark.owner(email=owner_email_tools_and_ux)
@pytest.mark.usefixtures('clean_dir')
class TestDoesQuack(object):

    def test_working(self):
        assert True

    def test_does_quack_base_explanation(self):
        assert BaseExplanation._does_quack(BaseValid())

    def test_does_quack_base_explanation_negative(self):
        class NoMethod(object):
            @property
            def id(self):
                return 'some_other_string'

            @property
            def model_task(self):
                return 'model_task_string'

            @property
            def model_type(self):
                return 'model_type_string'
        assert not BaseExplanation._does_quack(NoMethod())

        class NoModelTask(object):
            @property
            def method(self):
                return 'some_string'

            @property
            def id(self):
                return 'some_other_string'

            @property
            def model_type(self):
                return 'model_type_string'
        assert not BaseExplanation._does_quack(NoModelTask())

        class NoModelType(object):
            @property
            def method(self):
                return 'some_string'

            @property
            def id(self):
                return 'some_other_string'

            @property
            def model_task(self):
                return 'model_task_string'
        assert not BaseExplanation._does_quack(NoModelType())

        class NoId(object):
            @property
            def model_task(self):
                return 'model_task_string'

            @property
            def model_type(self):
                return 'model_type_string'

            @property
            def method(self):
                return 'some_string'
        assert not BaseExplanation._does_quack(NoId())

        class MethodInt(object):
            @property
            def model_task(self):
                return 'model_task_string'

            @property
            def model_type(self):
                return 'model_type_string'

            @property
            def method(self):
                return 5

            @property
            def id(self):
                return 'some_other_string'
        assert not BaseExplanation._does_quack(MethodInt())

        class ModelTaskInt(object):
            @property
            def model_task(self):
                return 12

            @property
            def model_type(self):
                return 'model_type_string'

            @property
            def method(self):
                return 5

            @property
            def id(self):
                return 'some_other_string'
        assert not BaseExplanation._does_quack(MethodInt())

        class IdInt(object):
            @property
            def model_task(self):
                return 'model_task_string'

            @property
            def model_type(self):
                return 'model_type_string'

            @property
            def method(self):
                return 'some_string'

            @property
            def id(self):
                return 5
        assert not BaseExplanation._does_quack(IdInt())

    def test_does_quack_base_explanation_non_property(self):
        class FeatureNonProp(object):
            def features(self):
                return None
        assert not BaseExplanation._does_quack(FeatureNonProp())

    def test_does_quack_feature_importance_explanation(self):
        ValidFeatureImportanceExplanation = type('ValidFeatureImportanceExplanation',
                                                 (BaseValid, FeatureImportanceValid),
                                                 {})
        assert FeatureImportanceExplanation._does_quack(ValidFeatureImportanceExplanation())

    def test_does_quack_feature_importance_explanation_negative(self):
        NoFeatureExp = type('InvalidFeatureImportanceExplanation', (BaseValid,), {})
        assert not FeatureImportanceExplanation._does_quack(NoFeatureExp())

        class FeatImpNoFeatures(object):
            @property
            def num_features(self):
                return None

            @property
            def is_raw(self):
                return True

            @property
            def is_engineered(self):
                return False

        FeatImpNoFeaturesExp = type('InvalidFeatureImportanceExplanation', (FeatImpNoFeatures, BaseValid), {})
        assert not FeatureImportanceExplanation._does_quack(FeatImpNoFeaturesExp())

        class FeatImpNoRawTag(object):
            @property
            def num_features(self):
                return None

            @property
            def features(self):
                return None

            @property
            def is_engineered(self):
                return None

        FeatImpNoRawTagExp = type('InvalidFeatureImportanceExplanation', (FeatImpNoRawTag, BaseValid), {})
        assert not FeatureImportanceExplanation._does_quack(FeatImpNoRawTagExp())

        class FeatImpIsRawNonBool(object):
            @property
            def num_features(self):
                return None

            @property
            def features(self):
                return None

            @property
            def is_raw(self):
                return [1, 2, 3]

            @property
            def is_engineered(self):
                return False

        FeatImpIsRawNonBoolExp = type('InvalidFeatureImportanceExplanation', (FeatImpIsRawNonBool, BaseValid), {})
        assert not FeatureImportanceExplanation._does_quack(FeatImpIsRawNonBoolExp())

        class FeatImpNoNumFeatures(object):
            @property
            def features(self):
                return None

            @property
            def is_raw(self):
                return True

            @property
            def is_engineered(self):
                return False

        FeatImpNoNumFeatsExp = type('InvalidFeatureImportanceExplanation', (FeatImpNoNumFeatures, BaseValid), {})
        assert not FeatureImportanceExplanation._does_quack(FeatImpNoNumFeatsExp())

    def test_does_quack_local_explanation(self):
        ValidLocalExp = type('ValidLocalExplanation', (BaseValid, FeatureImportanceValid, LocalValuesValid), {})
        assert LocalExplanation._does_quack(ValidLocalExp())

    def test_does_quack_local_explanation_negative(self):
        NoFeatureLocalExp = type('InvalidLocalExplanation', (LocalValuesValid,), {})
        assert not LocalExplanation._does_quack(NoFeatureLocalExp())

        NoLocalLocalExp = type('InvalidLocalExplanation', (BaseValid,), {})
        assert not LocalExplanation._does_quack(NoLocalLocalExp())

        class LocalExplanationNone(object):
            @property
            def local_importance_values(self):
                return None

            @property
            def num_examples(self):
                return None
        LocalNoneLocalExp = type('InvalidLocalExplanation',
                                 (LocalExplanationNone, FeatureImportanceValid, BaseValid),
                                 {})
        assert not LocalExplanation._does_quack(LocalNoneLocalExp())

        class LocalExplanationNonList(object):
            @property
            def local_importance_values(self):
                return 5

            @property
            def num_examples(self):
                return None
        LocalNonListLocalExp = type('InvalidLocalExplanation',
                                    (LocalExplanationNonList, FeatureImportanceValid, BaseValid),
                                    {})
        assert not LocalExplanation._does_quack(LocalNonListLocalExp())

        class LocalExplanationNumpy(object):
            @property
            def local_importance_values(self):
                return np.ones((5, 3))

            @property
            def num_examples(self):
                return None
        LocalNumpyLocalExp = type('InvalidLocalExplanation',
                                  (LocalExplanationNumpy, FeatureImportanceValid, BaseValid),
                                  {})
        assert not LocalExplanation._does_quack(LocalNumpyLocalExp())

        class LocalNoNumExamples(object):
            @property
            def local_importance_values(self):
                return [[.2, .4, .01], [.3, .2, 0]]
        LocalNoNumExamplesExp = type('InvalidLocalExplanation',
                                     (LocalNoNumExamples, FeatureImportanceValid, BaseValid),
                                     {})
        assert not LocalExplanation._does_quack(LocalNoNumExamplesExp())

    def test_does_quack_global_explanation(self):
        ValidGlobalLocalExp = type('ValidGlobalExplanation',
                                   (BaseValid, FeatureImportanceValid, LocalValuesValid, GlobalValid),
                                   {})
        assert GlobalExplanation._does_quack(ValidGlobalLocalExp())

        ValidGlobalOnlyExp = type('ValidGlobalExplanation', (BaseValid, FeatureImportanceValid, GlobalValid), {})
        assert GlobalExplanation._does_quack(ValidGlobalOnlyExp())

    def test_does_quack_global_explanation_negative(self):
        NoFeatureGlobalExp = type('InvalidGlobalExplanation', (BaseValid, GlobalValid), {})
        assert not GlobalExplanation._does_quack(NoFeatureGlobalExp())

        class NoGlobalRank(object):
            @property
            def global_importance_values(self):
                return [.2, .4, .01]
        NoRankGlobalExp = type('InvalidGlobalExplanation', (NoGlobalRank, FeatureImportanceValid, BaseValid), {})
        assert not GlobalExplanation._does_quack(NoRankGlobalExp())

        class NoGlobalValues(object):
            @property
            def global_importance_rank(self):
                return [0, 1]
        NoValuesGlobalExp = type('InvalidGlobalExplanation', (NoGlobalValues, FeatureImportanceValid, BaseValid), {})
        assert not GlobalExplanation._does_quack(NoValuesGlobalExp())

        class GlobalRankNone(object):
            @property
            def global_importance_rank(self):
                return None

            @property
            def global_importance_values(self):
                return [.2, .4, .01]
        RankNoneGlobalExp = type('InvalidGlobalExplanation', (GlobalRankNone, FeatureImportanceValid, BaseValid), {})
        assert not GlobalExplanation._does_quack(RankNoneGlobalExp())

        class GlobalValuesNone(object):
            @property
            def global_importance_rank(self):
                return [0, 1]

            @property
            def global_importance_values(self):
                return None
        ValuesNoneGlobalExp = type('InvalidGlobalExplanation',
                                   (GlobalValuesNone, FeatureImportanceValid, BaseValid),
                                   {})
        assert not GlobalExplanation._does_quack(ValuesNoneGlobalExp())

    def test_does_quack_expected_values_mixin(self):
        ValidExpectedValues = type('ValidExpectedValues', (ExpectedValuesValid,), {})
        assert ExpectedValuesMixin._does_quack(ValidExpectedValues())

    def test_does_quack_expected_values_mixin_negative(self):
        NoExpectedValues = type('InvalidExpectedValues', (BaseValid, FeatureImportanceValid, GlobalValid), {})
        assert not ExpectedValuesMixin._does_quack(NoExpectedValues())

        class ExpectedValuesNone(object):
            @property
            def expected_values(self):
                return None
        ExpectedValuesNoneMixin = type('InvalidExpectedValues', (ExpectedValuesNone,), {})
        assert not ExpectedValuesMixin._does_quack(ExpectedValuesNoneMixin())

    def test_does_quack_classes_mixin(self):
        ValidClasses = type('ValidClasses', (ClassesValid,), {})
        assert ClassesMixin._does_quack(ValidClasses())

    def test_does_quack_classes_mixin_negative(self):
        class NoClasses(object):
            @property
            def num_classes(self):
                return None
        assert not ClassesMixin._does_quack(NoClasses())

        class NoNumClasses(object):
            @property
            def classes(self):
                return None
        NoNumClasses = type('InvalidClasses', (PerClassValid, NoNumClasses), {})
        assert not ClassesMixin._does_quack(NoNumClasses())

    def test_does_quack_per_class(self):
        ValidPerClass = type('ValidPerClass', (ClassesValid, PerClassValid), {})
        assert PerClassMixin._does_quack(ValidPerClass())

    def test_does_quack_per_class_negative(self):
        NoClasses = type('InvalidPerClass', (PerClassValid,), {})
        assert not PerClassMixin._does_quack(NoClasses())

        class NoPerClassRank(object):
            @property
            def per_class_values(self):
                return [.2, .4, .01]
        NoRankPerClassExp = type('InvalidPerClassExplanation', (NoPerClassRank, BaseValid), {})
        assert not PerClassMixin._does_quack(NoRankPerClassExp())

        class NoPerClassValues(object):
            @property
            def per_class_rank(self):
                return [0, 1]
        NoValuesPerClassExp = type('InvalidPerClassExplanation', (NoPerClassValues, BaseValid), {})
        assert not PerClassMixin._does_quack(NoValuesPerClassExp())

        class PerClassRankNone(object):
            @property
            def per_class_rank(self):
                return None

            @property
            def per_class_values(self):
                return [.2, .4, .01]
        RankNonePerClassExp = type('InvalidPerClassExplanation', (PerClassRankNone, BaseValid), {})
        assert not PerClassMixin._does_quack(RankNonePerClassExp())

        class PerClassValuesNone(object):
            @property
            def per_class_rank(self):
                return [0, 1]

            @property
            def per_class_values(self):
                return None
        ValuesNonePerClassExp = type('InvalidPerClassExplanation', (PerClassValuesNone, BaseValid), {})
        assert not PerClassMixin._does_quack(ValuesNonePerClassExp())

    def test_does_quack_datasets_mixin(self):
        ValidDatasets = type('ValidDatasets', (_DatasetsValid,), {})
        assert _DatasetsMixin._does_quack(ValidDatasets())

    def test_does_quack_datasets_negative(self):
        NoDatasets = type('InvalidDatasets', (BaseValid,), {})
        assert not _DatasetsMixin._does_quack(NoDatasets())

        class NoTrainData(object):
            @property
            def eval_data(self):
                return [[.2, .4, .01], [.3, .2, 0]]

            @property
            def eval_y_predicted(self):
                return None

            @property
            def eval_y_predicted_proba(self):
                return None
        NoTrainExp = type('InvalidDatasets', (NoTrainData, BaseValid), {})
        assert not _DatasetsMixin._does_quack(NoTrainExp())

        class NoTestData(object):
            @property
            def init_data(self):
                return 'a_dataset_id'

            @property
            def eval_y_predicted(self):
                return None

            @property
            def eval_y_predicted_proba(self):
                return None
        NoTestExp = type('InvalidDatasets', (NoTestData, BaseValid), {})
        assert not _DatasetsMixin._does_quack(NoTestExp())

        class NoEvalYPredicted(object):
            @property
            def init_data(self):
                return 'a_dataset_id'

            @property
            def eval_data(self):
                return[[.2, .4, .01], [.3, .2, 0]]

            @property
            def eval_y_predicted_proba(self):
                return None
        NoEvalYPExp = type('InvalidDatasets', (NoEvalYPredicted, BaseValid), {})
        assert not _DatasetsMixin._does_quack(NoEvalYPExp())

        class NoEvalYPredictedProba(object):
            @property
            def init_data(self):
                return 'a_dataset_id'

            @property
            def eval_data(self):
                return[[.2, .4, .01], [.3, .2, 0]]

            @property
            def eval_y_predicted(self):
                return None
        NoEvalYPPExp = type('InvalidDatasets', (NoEvalYPredictedProba, BaseValid), {})
        assert not _DatasetsMixin._does_quack(NoEvalYPPExp())

    def test_does_quack_model_id_mixin(self):
        ValidModelId = type('ValidModelId', (_ModelIdValid,), {})
        assert _ModelIdMixin._does_quack(ValidModelId())

    def test_does_quack_model_id_negative(self):
        NoModelId = type('InvalidModelId', (BaseValid,), {})
        assert not _ModelIdMixin._does_quack(NoModelId())
