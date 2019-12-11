from interpret_community.common.constants import ExplainParams, ExplainType, ModelTask
from interpret_community.explanation.explanation import _create_local_explanation


def data_to_explanation(data):
    """Create an explanation from raw dictionary data.

    :param data: the get_data() form of an interpret Explanation
    :type explanation: dict
    :return: an Explanation object
    :rtype: KernelExplantion
    """
    print(data)
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
