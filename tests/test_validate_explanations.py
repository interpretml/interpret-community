# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import logging

# Tests for model explainability SDK
import numpy as np
import pytest
import shap
from scipy import stats
from sklearn.pipeline import Pipeline

try:
    import cuml
    if cuml.__version__ == '0.18.0':
        from cuml.experimental.explainer import KernelExplainer
    elif cuml.__version__ == '0.19.0':
        from cuml.explainer import KernelExplainer
    rapids_installed = True
except ImportError:
    rapids_installed = False

from common_utils import (create_cancer_data, create_cuml_svm_classifier,
                          create_sklearn_linear_regressor,
                          create_sklearn_logistic_regressor,
                          create_sklearn_random_forest_classifier,
                          create_sklearn_random_forest_regressor)
from constants import owner_email_tools_and_ux
from interpret_community.common.constants import ExplainParams
from interpret_community.common.metrics import ndcg
from interpret_community.common.policy import SamplingPolicy
from interpret_community.tabular_explainer import TabularExplainer
from sklearn.model_selection import train_test_split

test_logger = logging.getLogger(__name__)
test_logger.setLevel(logging.INFO)


@pytest.mark.owner(email=owner_email_tools_and_ux)
@pytest.mark.usefixtures("_clean_dir")
class TestExplainerValidity(object):
    def test_working(self):
        assert True

    def test_verify_pipeline_model_coefficient_explanation(self):
        # Validate our explainer against an explainable linear model
        X, y = shap.datasets.adult()
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
        # Note: in pipeline case, we use KernelExplainer;
        # in linear case we use LinearExplainer which is much faster
        pipeline = [True, False]
        threshold = [0.85, 0.76]
        for idx, is_pipeline in enumerate(pipeline):
            # Fit a logistic regression classifier
            model = create_sklearn_logistic_regressor(x_train, y_train, pipeline=is_pipeline)

            # Create tabular explainer
            exp = TabularExplainer(model, x_train, features=list(range(x_train.shape[1])))
            test_logger.info("Running explain model for test_verify_linear_model_coefficient_explanation")
            # Validate evaluation sampling
            policy = {ExplainParams.SAMPLING_POLICY: SamplingPolicy(allow_eval_sampling=True)}
            explanation = exp.explain_global(x_test, **policy)
            mean_train = np.mean(x_train.values, axis=0)
            # Retrieve the model coefficients
            if isinstance(model, Pipeline):
                model = model.steps[0][1]
            coefficients = model.coef_[0]
            # Normalize the coefficients by mean for a rough ground-truth of importance
            norm_coeff = mean_train * coefficients
            # order coefficients by importance
            norm_coeff_imp = np.abs(norm_coeff).argsort()[..., ::-1]
            # Calculate the correlation
            validate_correlation(explanation.global_importance_rank, norm_coeff_imp, threshold[idx])

    def test_verify_linear_model_coefficient_explanation(self):
        # Validate our explainer against an explainable linear model
        X, y = shap.datasets.adult()
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
        # Fit a logistic regression classifier
        model = create_sklearn_logistic_regressor(x_train, y_train)

        # Create tabular explainer
        exp = TabularExplainer(model, x_train, features=list(range(x_train.shape[1])))
        test_logger.info("Running explain model for test_verify_linear_model_coefficient_explanation")
        # Validate evaluation sampling
        policy = {ExplainParams.SAMPLING_POLICY: SamplingPolicy(allow_eval_sampling=True)}
        explanation = exp.explain_global(x_test, **policy)
        mean_train = np.mean(x_train.values, axis=0)
        # Retrieve the model coefficients
        coefficients = model.coef_[0]
        # Normalize the coefficients by mean for a rough ground-truth of importance
        norm_coeff = mean_train * coefficients
        # order coefficients by importance
        norm_coeff_imp = np.abs(norm_coeff).argsort()[..., ::-1]
        # Calculate the correlation
        validate_correlation(explanation.global_importance_rank, norm_coeff_imp, 0.76)

    def test_validate_against_shap(self):
        # Validate our explainer against shap library directly
        X, y = shap.datasets.adult()
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.02, random_state=7)
        # Fit several classifiers
        tree_classifiers = [create_sklearn_random_forest_classifier(x_train, y_train)]
        non_tree_classifiers = [create_sklearn_logistic_regressor(x_train, y_train)]
        tree_regressors = [create_sklearn_random_forest_regressor(x_train, y_train)]
        non_tree_regressors = [create_sklearn_linear_regressor(x_train, y_train)]
        # For each model, validate we get the same results as calling shap directly
        test_logger.info("Running tree classifiers in test_validate_against_shap")
        for model in tree_classifiers:
            # Run shap directly for comparison
            exp = shap.TreeExplainer(model)
            explanation = exp.shap_values(x_test)
            shap_overall_imp = get_shap_imp_classification(explanation)
            overall_imp = tabular_explainer_imp(model, x_train, x_test)
            validate_correlation(overall_imp, shap_overall_imp, 0.95)

        test_logger.info("Running non tree classifiers in test_validate_against_shap")
        for model in non_tree_classifiers:
            # Run shap directly for comparison
            clustered = shap.kmeans(x_train, 10)
            exp = shap.KernelExplainer(model.predict_proba, clustered)
            explanation = exp.shap_values(x_test)
            shap_overall_imp = get_shap_imp_classification(explanation)
            overall_imp = tabular_explainer_imp(model, x_train, x_test)
            validate_correlation(overall_imp, shap_overall_imp, 0.82)

        test_logger.info("Running tree regressors in test_validate_against_shap")
        for model in tree_regressors:
            # Run shap directly for comparison
            exp = shap.TreeExplainer(model)
            explanation = exp.shap_values(x_test)
            shap_overall_imp = get_shap_imp_regression(explanation)
            overall_imp = tabular_explainer_imp(model, x_train, x_test)
            validate_correlation(overall_imp, shap_overall_imp, 0.95)

        test_logger.info("Running non tree regressors in test_validate_against_shap")
        for model in non_tree_regressors:
            # Run shap directly for comparison
            clustered = shap.kmeans(x_train, 10)
            exp = shap.KernelExplainer(model.predict, clustered)
            explanation = exp.shap_values(x_test)
            shap_overall_imp = get_shap_imp_regression(explanation)
            overall_imp = tabular_explainer_imp(model, x_train, x_test)
            validate_correlation(overall_imp, shap_overall_imp, 0.95)

        if not rapids_installed:
            pytest.skip("cuML not installed; will skip testing GPU Explainer")
        else:
            test_logger.info("Running GPU non tree classifiers in test_validate_against_shap")
            x_train, x_test, y_train, y_validation, _, _ = create_cancer_data()
            gpu_non_tree_classifiers = [create_cuml_svm_classifier(x_train.astype(np.float32),
                                                                   y_train.astype(np.float32))]
            for model in gpu_non_tree_classifiers:
                exp = KernelExplainer(model=model.predict_proba,
                                      data=x_train.astype(np.float32))
                explanation = exp.shap_values(x_test.astype(np.float32))
                shap_overall_imp = get_shap_imp_classification(explanation)
                overall_imp = tabular_explainer_imp(model,
                                                    x_train.astype(np.float32),
                                                    x_test.astype(np.float32),
                                                    use_gpu=True)
                validate_correlation(overall_imp, shap_overall_imp, 0.95)


def tabular_explainer_imp(model, x_train, x_test, allow_eval_sampling=True, use_gpu=False):
    # Create tabular explainer
    exp = TabularExplainer(model, x_train, features=list(range(x_train.shape[1])), use_gpu=use_gpu)
    # Validate evaluation sampling
    policy = {ExplainParams.SAMPLING_POLICY: SamplingPolicy(allow_eval_sampling=allow_eval_sampling)}
    explanation = exp.explain_global(x_test, **policy)
    return explanation.global_importance_rank


def validate_correlation(true_order, validate_order, threshold, top_values=10):
    computed_ndcg = ndcg(true_order, validate_order, top_values)
    ndcg_info_message = "ndcg: " + str(computed_ndcg)
    test_logger.info(ndcg_info_message)
    assert computed_ndcg > threshold


def validate_spearman_correlation(overall_imp, shap_overall_imp, threshold):
    # Calculate the spearman rank-order correlation
    rho, p_val = stats.spearmanr(overall_imp, shap_overall_imp)
    # Validate that the coefficients from the linear model are highly correlated with the results from shap
    spearman_info_message = ("Calculated spearman correlation coefficient rho: {} and p_val: {}"
                             .format(str(rho), str(p_val)))
    test_logger.info(spearman_info_message)
    assert rho > threshold


def get_shap_imp_classification(explanation):
    global_importance_values = np.mean(np.mean(np.absolute(explanation), axis=1), axis=0)
    return global_importance_values.argsort()[..., ::-1]


def get_shap_imp_regression(explanation):
    global_importance_values = np.mean(np.absolute(explanation), axis=0)
    return global_importance_values.argsort()[..., ::-1]
