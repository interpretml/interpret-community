# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Pin public parameter order, methods, properties, and class attributes."""

from inspect import Parameter, getattr_static, signature

import pytest
from interpret_community import TabularExplainer
from interpret_community.explanation.explanation import (
    BaseExplanation, ClassesMixin, ExpectedValuesMixin,
    FeatureImportanceExplanation, GlobalExplanation, LocalExplanation,
    PerClassMixin)
from interpret_community.lime.lime_explainer import LIMEExplainer
from interpret_community.mimic.mimic_explainer import MimicExplainer
from interpret_community.permutation.permutation_importance import PFIExplainer
from interpret_community.shap.deep_explainer import DeepExplainer
from interpret_community.shap.gpu_kernel_explainer import GPUKernelExplainer
from interpret_community.shap.kernel_explainer import KernelExplainer
from interpret_community.shap.linear_explainer import LinearExplainer
from interpret_community.shap.tree_explainer import TreeExplainer

_EXPLAINER_METHODS = {
    'explain_global': 'evaluation_examples sampling_policy include_local batch_size',
    'explain_local': 'evaluation_examples'
}
_BASE_METHODS = {'data': 'key', 'visualize': 'key'}
_BASE_PROPERTIES = {'id', 'method', 'model_task', 'model_type', 'name', 'selector'}
_FEATURE_PROPERTIES = _BASE_PROPERTIES | {
    'features', 'num_features', 'is_raw', 'is_engineered'
}


@pytest.mark.parametrize(('cls', 'parameters', 'methods'), [
    (TabularExplainer,
     'model initialization_examples explain_subset features classes transformations '
     'allow_all_transformations model_task use_gpu **kwargs', _EXPLAINER_METHODS),
    (TreeExplainer,
     'model explain_subset features classes shap_values_output transformations '
     'allow_all_transformations **kwargs', _EXPLAINER_METHODS),
    (LinearExplainer,
     'model initialization_examples explain_subset features classes transformations '
     'allow_all_transformations **kwargs', _EXPLAINER_METHODS),
    (KernelExplainer,
     'model initialization_examples is_function explain_subset nsamples features classes '
     'nclusters show_progress transformations allow_all_transformations model_task **kwargs',
     _EXPLAINER_METHODS),
    (DeepExplainer,
     'model initialization_examples explain_subset nclusters features classes transformations '
     'allow_all_transformations model_task is_classifier check_additivity **kwargs',
     _EXPLAINER_METHODS),
    (GPUKernelExplainer,
     'model initialization_examples explain_subset is_function nsamples features classes '
     'nclusters show_progress transformations allow_all_transformations model_task **kwargs',
     _EXPLAINER_METHODS),
    (LIMEExplainer,
     'model initialization_examples is_function explain_subset nclusters features classes '
     'verbose categorical_features show_progress transformations allow_all_transformations '
     'model_task **kwargs', _EXPLAINER_METHODS),
    (MimicExplainer,
     'model initialization_examples explainable_model explainable_model_args is_function '
     'augment_data max_num_of_augmentations explain_subset features classes transformations '
     'allow_all_transformations shap_values_output categorical_features model_task reset_index **kwargs',
     {'explain_global': 'evaluation_examples include_local batch_size',
      'explain_local': 'evaluation_examples',
      'get_surrogate_model_replication_measure': 'training_data'}),
    (PFIExplainer,
     'model is_function metric metric_args is_error_metric explain_subset features classes '
     'transformations allow_all_transformations seed for_classifier_use_predict_proba '
     'show_progress model_task **kwargs',
     {'explain_global': 'evaluation_examples true_labels'})
], ids=lambda value: value.__name__ if isinstance(value, type) else None)
def test_explainer_api_contract(cls, parameters, methods):
    _assert_parameters(cls.__init__, parameters)
    _assert_public_members(
        cls, methods, set(), {'available_explanations', 'explainer_type'})


@pytest.mark.parametrize(('cls', 'parameters', 'methods', 'properties'), [
    (BaseExplanation, 'method model_task model_type explanation_id **kwargs',
     _BASE_METHODS, _BASE_PROPERTIES),
    (FeatureImportanceExplanation,
     'features num_features is_raw is_engineered **kwargs',
     _BASE_METHODS, _FEATURE_PROPERTIES),
    (LocalExplanation, 'local_importance_values **kwargs',
     {**_BASE_METHODS,
      'get_local_importance_rank': '',
      'get_ranked_local_names': 'top_k',
      'get_ranked_local_values': 'top_k',
      'get_raw_explanation': 'feature_maps raw_feature_names eval_data',
      'get_raw_feature_importances': 'raw_to_output_maps'},
     _FEATURE_PROPERTIES | {
         'local_importance_values', 'num_examples', 'is_local_sparse'}),
    (GlobalExplanation,
     'global_importance_values global_importance_rank ranked_global_names ranked_global_values **kwargs',
     {**_BASE_METHODS,
      'get_feature_importance_dict': 'top_k',
      'get_ranked_global_names': 'top_k',
      'get_ranked_global_values': 'top_k',
      'get_raw_explanation': 'feature_maps raw_feature_names eval_data',
      'get_raw_feature_importances': 'feature_maps'},
     _FEATURE_PROPERTIES | {'global_importance_rank', 'global_importance_values'}),
    (ExpectedValuesMixin, 'expected_values **kwargs',
     _BASE_METHODS, _BASE_PROPERTIES | {'expected_values'}),
    (ClassesMixin, 'classes num_classes **kwargs',
     _BASE_METHODS, _BASE_PROPERTIES | {'classes', 'num_classes'}),
    (PerClassMixin,
     'per_class_values per_class_rank ranked_per_class_names ranked_per_class_values **kwargs',
     {**_BASE_METHODS,
      'get_ranked_per_class_names': 'top_k',
      'get_ranked_per_class_values': 'top_k'},
     _BASE_PROPERTIES | {
         'classes', 'num_classes', 'per_class_rank', 'per_class_values'})
], ids=lambda value: value.__name__ if isinstance(value, type) else None)
def test_explanation_api_contract(cls, parameters, methods, properties):
    _assert_parameters(cls.__init__, parameters)
    _assert_public_members(cls, methods, properties, set())


def _assert_parameters(function, parameters):
    expected = [
        (name.lstrip('*'),
         Parameter.VAR_KEYWORD if name.startswith('**')
         else Parameter.POSITIONAL_OR_KEYWORD)
        for name in ('self ' + parameters).split()
    ]
    assert [(param.name, param.kind)
            for param in signature(function).parameters.values()] == expected


def _assert_public_members(cls, methods, properties, attributes):
    actual_methods = {}
    actual_properties = set()
    actual_attributes = set()
    for name in dir(cls):
        if name.startswith('_'):
            continue
        member = getattr_static(cls, name)
        if isinstance(member, property):
            actual_properties.add(name)
        elif callable(member):
            actual_methods[name] = member
        else:
            actual_attributes.add(name)
    assert set(actual_methods) == set(methods)
    assert actual_properties == properties
    assert actual_attributes == attributes
    for name, parameters in methods.items():
        _assert_parameters(actual_methods[name], parameters)
