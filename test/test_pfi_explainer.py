# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import pytest

# Tests for Permutation Feature Importance Explainer
import logging

from interpret.community.permutation.permutation_importance import PFIExplainer
from constants import owner_email_tools_and_ux

from common_tabular_tests import VerifyTabularTests

test_logger = logging.getLogger(__name__)
test_logger.setLevel(logging.DEBUG)


@pytest.mark.owner(email=owner_email_tools_and_ux)
@pytest.mark.usefixtures('clean_dir')
class TestPFIExplainer(object):
    def setup_class(self):
        def create_explainer(model, x_train, is_function=False, **kwargs):
            # Note: we ignore x_train here!
            return PFIExplainer(model, is_function=is_function, **kwargs)

        self.verify_tabular = VerifyTabularTests(test_logger,
                                                 create_explainer, specify_policy=False)

    def test_working(self):
        assert True

    def test_explain_model_local(self):
        iris_overall_expected_features = self.iris_overall_expected_features
        self.verify_tabular.verify_explain_model_local(iris_overall_expected_features,
                                                       is_per_class=False,
                                                       has_explain_local=False,
                                                       true_labels_required=True)

    def test_explain_model_local_dnn(self):
        self.verify_tabular.verify_explain_model_local_dnn(is_per_class=False,
                                                           has_explain_local=False,
                                                           true_labels_required=True)

    def test_explain_model_local_regression(self):
        self.verify_tabular.verify_explain_model_local_regression(has_explain_local=False,
                                                                  true_labels_required=True)

    def test_explain_model_local_regression_dnn(self):
        self.verify_tabular.verify_explain_model_local_regression_dnn(has_explain_local=False,
                                                                      true_labels_required=True)

    def test_explain_model_pandas_input(self):
        self.verify_tabular.verify_explain_model_pandas_input(has_explain_local=False,
                                                              true_labels_required=True)

    def test_explain_model_npz_linear(self):
        self.verify_tabular.verify_explain_model_npz_linear(true_labels_required=True)

    def test_explain_model_sparse(self):
        self.verify_tabular.verify_explain_model_sparse(true_labels_required=True)

    def test_explain_model_hashing(self):
        self.verify_tabular.verify_explain_model_hashing(true_labels_required=True)

    def test_explain_model_subset_classification_dense(self):
        self.verify_tabular.verify_explain_model_subset_classification_dense(true_labels_required=True,
                                                                             is_local=False)

    def test_explain_model_subset_regression_sparse(self):
        self.verify_tabular.verify_explain_model_subset_regression_sparse(true_labels_required=True,
                                                                          is_local=False)

    def test_explain_model_subset_classification_sparse(self):
        self.verify_tabular \
            .verify_explain_model_subset_classification_sparse(true_labels_required=True,
                                                               is_local=False)

    def test_explain_model_scoring_with_sampling_regression_sparse(self):
        self.verify_tabular \
            .verify_explain_model_scoring_with_sampling_regression_sparse(true_labels_required=True)

    def test_explain_model_throws_on_bad_classifier_and_classes(self):
        self.verify_tabular.verify_explain_model_throws_on_bad_classifier_and_classes()

    def test_explain_model_throws_on_bad_pipeline_and_classes(self):
        self.verify_tabular.verify_explain_model_throws_on_bad_pipeline_and_classes()

    def test_explain_with_transformations_list_classification(self):
        self.verify_tabular.verify_explain_model_transformations_list_classification(true_labels_required=True)

    def test_explain_with_transformations_column_transformer_classification(self):
        self.verify_tabular \
            .verify_explain_model_transformations_column_transformer_classification(true_labels_required=True)

    def test_explain_with_transformations_list_regression(self):
        self.verify_tabular.verify_explain_model_transformations_list_regression(true_labels_required=True)

    def test_explain_with_transformations_column_transformer_regression(self):
        self.verify_tabular \
            .verify_explain_model_transformations_column_transformer_regression(true_labels_required=True)

    @property
    def iris_overall_expected_features(self):
        return ['petal length', 'petal width', 'sepal width', 'sepal length']
