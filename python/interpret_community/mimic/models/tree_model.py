# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Defines an explainable tree model."""

from ...common.constants import ShapValuesOutput, ExplainableModelType, Extension
from .explainable_model import BaseExplainableModel, _get_initializer_args, _clean_doc
from .tree_model_utils import _explain_local_tree_surrogate, \
    _expected_values_tree_surrogate
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from ...common.explanation_utils import _get_dense_examples

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', 'Starting from version 2.2.1', UserWarning)
    import shap

DEFAULT_RANDOM_STATE = 123


class DecisionTreeExplainableModel(BaseExplainableModel):
    available_explanations = [Extension.GLOBAL, Extension.LOCAL]
    explainer_type = Extension.GLASSBOX

    """Decision Tree explainable model.

    :param multiclass: Set to true to generate a multiclass model.
    :type multiclass: bool
    :param random_state: Int to seed the model.
    :type random_state: int
    :param shap_values_output: The type of the output from explain_local when using TreeExplainer.
        Currently only types 'default', 'probability' and 'teacher_probability' are supported.  If
        'probability' is specified, then we approximately scale the raw log-odds values from the
        TreeExplainer to probabilities.
    :type shap_values_output: interpret_community.common.constants.ShapValuesOutput
    :param classification: Indicates if this is a classification or regression explanation.
    :type classification: bool
    """

    def __init__(self, multiclass=False, random_state=DEFAULT_RANDOM_STATE,
                 shap_values_output=ShapValuesOutput.DEFAULT, classification=True, **kwargs):
        """Initialize the DecisionTreeExplainableModel.

        :param multiclass: Set to true to generate a multiclass model.
        :type multiclass: bool
        :param random_state: Int to seed the model.
        :type random_state: int
        :param shap_values_output: The type of the output from explain_local when using TreeExplainer.
            Currently only types 'default', 'probability' and 'teacher_probability' are supported.  If
            'probability' is specified, then we approximately scale the raw log-odds values from the
            TreeExplainer to probabilities.
        :type shap_values_output: interpret_community.common.constants.ShapValuesOutput
        :param classification: Indicates if this is a classification or regression explanation.
        :type classification: bool
        """
        self.multiclass = multiclass
        self.random_state = random_state
        kwargs['random_state'] = random_state
        if self.multiclass:
            initializer = DecisionTreeClassifier
        else:
            initializer = DecisionTreeRegressor
        initializer_args = _get_initializer_args(kwargs)
        self._tree = initializer(**initializer_args)
        super(DecisionTreeExplainableModel, self).__init__(**kwargs)
        self._logger.debug('Initializing DecisionTreeExplainableModel')
        self._method = 'tree'
        self._tree_explainer = None
        self._shap_values_output = shap_values_output
        self._classification = classification

    __init__.__doc__ = (__init__.__doc__ +
                        '\nIf multiclass=True, uses the parameters for DecisionTreeClassifier:\n' +
                        _clean_doc(DecisionTreeClassifier.__doc__) +
                        '\nOtherwise, if multiclass=False, uses the parameters for DecisionTreeRegressor:\n' +
                        _clean_doc(DecisionTreeRegressor.__doc__))

    def fit(self, dataset, labels, **kwargs):
        """Call tree fit to fit the explainable model.

        :param dataset: The dataset to train the model on.
        :type dataset: numpy or scipy array
        :param labels: The labels to train the model on.
        :type labels: numpy or scipy array
        """
        self._tree.fit(dataset, labels, **kwargs)

    fit.__doc__ = (fit.__doc__ +
                   '\nIf multiclass=True, uses the parameters for DecisionTreeClassifier:\n' +
                   _clean_doc(DecisionTreeClassifier.fit.__doc__) +
                   '\nOtherwise, if multiclass=False, uses the parameters for DecisionTreeRegressor:\n' +
                   _clean_doc(DecisionTreeRegressor.fit.__doc__))

    def predict(self, dataset, **kwargs):
        """Call tree predict to predict labels using the explainable model.

        :param dataset: The dataset to predict on.
        :type dataset: numpy or scipy array
        :return: The predictions of the model.
        :rtype: list
        """
        return self._tree.predict(dataset)

    predict.__doc__ = (predict.__doc__ +
                       '\nIf multiclass=True, uses the parameters for DecisionTreeClassifier:\n' +
                       _clean_doc(DecisionTreeClassifier.predict.__doc__) +
                       '\nOtherwise, if multiclass=False, uses the parameters for DecisionTreeRegressor:\n' +
                       _clean_doc(DecisionTreeRegressor.predict.__doc__))

    def predict_proba(self, dataset, **kwargs):
        """Call tree predict_proba to predict probabilities using the explainable model.

        :param dataset: The dataset to predict probabilities on.
        :type dataset: numpy or scipy array
        :return: The predictions of the model.
        :rtype: list
        """
        if self.multiclass:
            return self._tree.predict_proba(dataset)
        else:
            raise Exception('predict_proba not supported for regression or binary classification dataset')

    predict_proba.__doc__ = (predict_proba.__doc__ +
                             '\nIf multiclass=True, uses the parameters for DecisionTreeClassifier:\n' +
                             _clean_doc(DecisionTreeClassifier.predict_proba.__doc__) +
                             '\nOtherwise predict_proba is not supported for regression or binary classification.\n')

    def explain_global(self, **kwargs):
        """Call tree model feature importances to get the global feature importances from the tree surrogate model.

        :return: The global explanation of feature importances.
        :rtype: list
        """
        return self._tree.feature_importances_

    def explain_local(self, evaluation_examples, probabilities=None, **kwargs):
        """Use TreeExplainer to get the local feature importances from the trained explainable model.

        :param evaluation_examples: The evaluation examples to compute local feature importances for.
        :type evaluation_examples: numpy or scipy array
        :param probabilities: If output_type is probability, can specify the teacher model's
            probability for scaling the shap values.
        :type probabilities: numpy.ndarray
        :return: The local explanation of feature importances.
        :rtype: Union[list, numpy.ndarray]
        """
        if self._tree_explainer is None:
            self._tree_explainer = shap.TreeExplainer(self._tree)
        evaluation_examples = _get_dense_examples(evaluation_examples)
        return _explain_local_tree_surrogate(self._tree, evaluation_examples, self._tree_explainer,
                                             self._shap_values_output, self._classification,
                                             probabilities, self.multiclass)

    @property
    def expected_values(self):
        """Use TreeExplainer to get the expected values.

        :return: The expected values of the decision tree tree model.
        :rtype: list
        """
        if self._tree_explainer is None:
            self._tree_explainer = shap.TreeExplainer(self._tree)
        return _expected_values_tree_surrogate(self._tree, self._tree_explainer, self._shap_values_output,
                                               self._classification, self.multiclass)

    @property
    def model(self):
        """Retrieve the underlying model.

        :return: The decision tree model, either classifier or regressor.
        :rtype: Union[DecisionTreeClassifier, DecisionTreeRegressor]
        """
        return self._tree

    @staticmethod
    def explainable_model_type(self):
        """Retrieve the model type.

        :return: Tree explainable model type.
        :rtype: ExplainableModelType
        """
        return ExplainableModelType.TREE_EXPLAINABLE_MODEL_TYPE
