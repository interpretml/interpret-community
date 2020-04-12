from interpret_community.common.explanation_utils import _get_raw_feature_importances
from interpret_community.explanation.explanation import _create_raw_feats_global_explanation
from interpret.api.base import ExplanationMixin
from interpret_community._internal.raw_explain.raw_explain_utils import transform_with_datamapper

from interpret_community._internal.raw_explain.data_mapper import DataMapper
from interpret_community.explanation.explanation import _create_raw_feats_local_explanation, _get_raw_explainer_create_explanation_kwargs


def _data_to_explanation(data):
    """Create an explanation from raw dictionary data.
    :param data: the get_data() form of an interpret Explanation
    :type explanation: dict
    :return: an Explanation object
    :rtype: KernelExplantion
    """
    kwargs = {ExplainParams.METHOD: ExplainType.SHAP_KERNEL}
    local_importance_values = data['mli'][0]['value']['scores']
    expected_values = data['mli'][0]['value']['intercept']
    classification = len(local_importance_values.shape) == 3
    kwargs[ExplainParams.MODEL_TYPE] = ExplainType.FUNCTION
    if classification:
        kwargs[ExplainParams.MODEL_TASK] = ExplainType.CLASSIFICATION
    else:
        kwargs[ExplainParams.MODEL_TASK] = ExplainType.REGRESSION
    kwargs[ExplainParams.LOCAL_IMPORTANCE_VALUES] = local_importance_values
    kwargs[ExplainParams.EXPECTED_VALUES] = expected_values
    return _create_local_explanation(**kwargs)


def _create_interpret_community_explanation(explanation):
    return _data_to_explanation(explanation.data(-1))

def keep_raw_features(explanation):
    if not isinstance(explanation, BaseExplanation):
        explanation = _create_interpret_community_explanation(explanation)
    kwargs = _get_raw_explainer_create_explanation_kwargs(explanation=explanation)
    return _create_raw_feats_local_explanation(explanation,
                                               feature_maps=[data_mapper.feature_map],
                                               **kwargs)
