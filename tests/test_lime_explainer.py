# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

# Tests for LIME Explainer
import logging

import pytest
from common_tabular_tests import VerifyTabularTests
from constants import owner_email_tools_and_ux
from interpret_community.lime.lime_explainer import LIMEExplainer

test_logger = logging.getLogger(__name__)
test_logger.setLevel(logging.DEBUG)


@pytest.mark.owner(email=owner_email_tools_and_ux)
@pytest.mark.usefixtures('_clean_dir')
class TestLIMEExplainer(object):
    def setup_class(self):
        def create_explainer(model, x_train, is_function=False, **kwargs):
            return LIMEExplainer(model, x_train, is_function=is_function, **kwargs)

        self.verify_tabular = VerifyTabularTests(test_logger, create_explainer)

    def test_working(self):
        assert True

    def test_explain_model_local(self):
        iris_overall_expected_features = self.verify_tabular.iris_overall_expected_features
        iris_per_class_expected_features = self.iris_per_class_expected_features
        self.verify_tabular.verify_explain_model_local(iris_overall_expected_features,
                                                       iris_per_class_expected_features)

    def test_explain_model_local_dnn(self):
        self.verify_tabular.verify_explain_model_local_dnn()

    def test_explain_model_local_without_include_local(self):
        iris_overall_expected_features = self.verify_tabular.iris_overall_expected_features
        iris_per_class_expected_features = self.iris_per_class_expected_features
        self.verify_tabular.verify_explain_model_local(iris_overall_expected_features,
                                                       iris_per_class_expected_features,
                                                       include_local=False)

    def test_explain_model_local_regression_without_include_local(self):
        self.verify_tabular.verify_explain_model_local_regression(include_local=False)

    def test_explain_model_local_regression_dnn(self):
        self.verify_tabular.verify_explain_model_local_regression_dnn()

    def test_explain_model_pandas_input(self):
        self.verify_tabular.verify_explain_model_pandas_input()

    def test_explain_model_pandas_input_without_include_local(self):
        self.verify_tabular.verify_explain_model_pandas_input(include_local=False)

    def test_explain_model_npz_linear(self):
        self.verify_tabular.verify_explain_model_npz_linear()

    def test_explain_model_sparse(self):
        self.verify_tabular.verify_explain_model_sparse()

    def test_explain_model_hashing(self):
        self.verify_tabular.verify_explain_model_hashing()

    def test_explain_model_with_summarize_data(self):
        iris_overall_expected_features = self.verify_tabular.iris_overall_expected_features
        iris_per_class_expected_features = self.iris_per_class_expected_features
        self.verify_tabular.verify_explain_model_with_summarize_data(iris_overall_expected_features,
                                                                     iris_per_class_expected_features)

    def test_explain_model_subset_classification_dense(self):
        self.verify_tabular.verify_explain_model_subset_classification_dense()

    def test_explain_model_subset_regression_sparse(self):
        self.verify_tabular.verify_explain_model_subset_regression_sparse()

    def test_explain_model_subset_classification_sparse(self):
        self.verify_tabular.verify_explain_model_subset_classification_sparse()

    def test_explain_model_scoring_with_sampling_regression_sparse(self):
        self.verify_tabular.verify_explain_model_with_sampling_regression_sparse()

    def test_explain_model_throws_on_bad_classifier_and_classes(self):
        self.verify_tabular.verify_explain_model_throws_on_bad_classifier_and_classes()

    def test_explain_model_throws_on_bad_pipeline_and_classes(self):
        self.verify_tabular.verify_explain_model_throws_on_bad_pipeline_and_classes()

    def test_explain_model_throws_on_classifier_and_no_classes(self):
        self.verify_tabular.verify_explain_model_throws_on_classifier_and_no_classes()

    def test_explain_model_local_single(self):
        self.verify_tabular.verify_explain_model_local_single()

    def test_explain_model_categorical(self):
        self.verify_tabular.verify_explain_model_categorical(pass_categoricals=True,
                                                             verify_same_shape=True)

    def test_explain_with_transformations_list_classification(self):
        self.verify_tabular.verify_explain_model_transformations_list_classification()

    def test_explain_with_transformations_column_transformer_classification(self):
        self.verify_tabular.verify_explain_model_transformations_column_transformer_classification()

    def test_explain_with_transformations_list_regression(self):
        self.verify_tabular.verify_explain_model_transformations_list_regression()

    def test_explain_with_transformations_column_transformer_regression(self):
        self.verify_tabular.verify_explain_model_transformations_column_transformer_regression()

    @property
    def iris_per_class_expected_features(self):
        return [['petal length', 'petal width', 'sepal length', 'sepal width'],
                ['petal length', 'petal width', 'sepal length', 'sepal width'],
                ['petal length', 'petal width', 'sepal width', 'sepal length']]
