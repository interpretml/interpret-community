# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Defines an explainable linear model."""
import numpy as np
import scipy as sp

from .explainable_model import BaseExplainableModel, BaseGlassboxModel, BaseGlassboxRegressor, \
    BaseGlassboxClassifier, _get_initializer_args
from sklearn.linear_model import LinearRegression as SKLinearRegression, \
    LogisticRegression as SKLogisticRegression, SGDClassifier, SGDRegressor
from ...common.constants import ExplainableModelType, Extension, SHAPDefaults, \
    ExplainParams, ExplainType, GlassboxModels, Defaults
from ...explanation.explanation import _create_global_explanation
from ...shap.linear_explainer import LinearExplainer
from ...common.explanation_utils import _order_imp
from ...common.aggregate import _get_explain_global_agg_kwargs

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', 'Starting from version 2.2.1', UserWarning)
    import shap

DEFAULT_RANDOM_STATE = 123
FEATURE_DEPENDENCE = 'interventional'


def _create_linear_explainer(model, multiclass, mean, covariance, seed):
    """Create the linear explainer or, in multiclass case, list of explainers.

    :param model: The linear model to compute the shap values for.
    :type model: linear model that implements sklearn.predict or sklearn.predict_proba
    :param multiclass: True if this is a multiclass model.
    :type multiclass: bool
    :param mean: The mean of the dataset by columns.
    :type mean: numpy.array
    :param covariance: The covariance matrix of the dataset.
    :type covariance: numpy.array
    :param seed: Random number seed.
    :type seed: int
    """
    np.random.seed(seed)
    if multiclass:
        explainers = []
        coefs = model.coef_
        intercepts = model.intercept_
        if isinstance(intercepts, np.ndarray):
            intercepts = intercepts.tolist()
        if isinstance(intercepts, list):
            coef_intercept_list = zip(coefs, intercepts)
        else:
            coef_intercept_list = [(coef, intercepts) for coef in coefs]
        for class_coef, intercept in coef_intercept_list:
            linear_explainer = shap.LinearExplainer((class_coef, intercept), (mean, covariance),
                                                    feature_dependence=SHAPDefaults.INDEPENDENT)
            explainers.append(linear_explainer)
        return explainers
    else:
        model_coef = model.coef_
        model_intercept = model.intercept_
        return shap.LinearExplainer((model_coef, model_intercept), (mean, covariance),
                                    feature_dependence=SHAPDefaults.INDEPENDENT)


def _compute_local_shap_values(linear_explainer, evaluation_examples, classification):
    """Compute the local shap values.

    :param linear_explainer: The linear explainer or list of linear explainers in multiclass case.
    :type linear_explainer: Union[LinearExplainer, list[LinearExplainer]]
    :param evaluation_examples: The evaluation examples.
    :type evaluation_examples: numpy or scipy array
    """
    # Multiclass case
    if isinstance(linear_explainer, list):
        shap_values = []
        for explainer in linear_explainer:
            explainer_shap_values = explainer.shap_values(evaluation_examples)
            if isinstance(explainer_shap_values, list):
                explainer_shap_values = explainer_shap_values[0]
            shap_values.append(explainer_shap_values)
        return shap_values
    shap_values = linear_explainer.shap_values(evaluation_examples)
    if not classification and isinstance(shap_values, list):
        shap_values = shap_values[0]
    return shap_values


class LinearBase(BaseGlassboxModel):
    def __init__(self, initializer, **kwargs):
        initializer_args = _get_initializer_args(kwargs)
        self._linear = initializer(**initializer_args)
        super(LinearBase, self).__init__(**kwargs)
        self._logger.debug('Initializing LinearBase')
        self._method = None
        self._linear_explainer = None

    def fit(self, dataset, labels, **kwargs):
        """Call linear fit to fit the explainable model.

        Store the mean and covariance of the background data for local explanation.

        :param dataset: The dataset to train the model on.
        :type dataset: numpy or scipy array
        :param labels: The labels to train the model on.
        :type labels: numpy or scipy array
        """
        self._linear.fit(dataset, labels, **kwargs)
        original_mean = np.asarray(dataset.mean(0))
        if len(original_mean.shape) == 2:
            mean_shape = original_mean.shape[1]
            self.mean = original_mean.reshape((mean_shape,))
        else:
            self.mean = original_mean
        if not sp.sparse.issparse(dataset):
            self.covariance = np.cov(dataset, rowvar=False)
        else:
            # Not needed for sparse case
            self.covariance = None

    def predict(self, dataset, **kwargs):
        """Call linear predict to predict labels using the explainable model.

        :param dataset: The dataset to predict on.
        :type dataset: numpy or scipy array
        :return: The predictions of the model.
        :rtype: list
        """
        return self._linear.predict(dataset)

    def explain_global(self, **kwargs):
        """Explain the model globally by either using the coef or aggregating local explanations to global.

        :return: The global explanation of feature importances.
        :rtype: list
        """
        coef = self._linear.coef_
        if (len(coef.shape) == 2):
            return np.mean(coef, axis=0)
        return coef

    @property
    def model(self):
        """Retrieve the underlying model.

        :return: The linear model, either classifier or regressor.
        :rtype: Union[LogisticRegression, LinearRegression]
        """
        return self._linear

    @staticmethod
    def explainable_model_type(self):
        """Retrieve the model type.

        :return: Linear explainable model type.
        :rtype: ExplainableModelType
        """
        return ExplainableModelType.LINEAR_EXPLAINABLE_MODEL_TYPE


class LinearClassifierMixin(LinearBase):
    def __init__(self, initializer, **kwargs):
        super(LinearClassifierMixin, self).__init__(initializer, **kwargs)
        self._logger.debug('Initializing LinearClassifierMixin')

    def predict_proba(self, dataset, **kwargs):
        """Call linear predict_proba to predict probabilities using the explainable model.

        :param dataset: The dataset to predict probabilities on.
        :type dataset: numpy or scipy array
        :return: The predictions of the model.
        :rtype: list
        """
        return self._linear.predict_proba(dataset, **kwargs)


class LinearRegression(LinearBase, BaseGlassboxRegressor):
    available_explanations = [Extension.GLOBAL, Extension.LOCAL]
    explainer_type = Extension.GLASSBOX

    def __init__(self, **kwargs):
        """Initialize the LinearRegression glassbox model."""
        initializor = SKLinearRegression
        super(LinearRegression, self).__init__(initializor, **kwargs)
        self._logger.debug('Initializing LinearRegression')

    def fit(self, dataset, labels, **kwargs):
        """Call linear fit to fit the glassbox model.

        :param dataset: The dataset to train the model on.
        :type dataset: numpy or scipy array or pandas dataframe
        :param labels: The labels to train the model on.
        :type labels: numpy or scipy array
        """
        super(LinearRegression, self).fit(dataset, labels, **kwargs)

    def explain_global(self, evaluation_examples=None, include_local=True,
                       batch_size=Defaults.DEFAULT_BATCH_SIZE, **kwargs):
        """Explain the model globally by either using the coef or aggregating local explanations to global.

        :param evaluation_examples: A matrix of feature vector examples (# examples x # features) on which to
            explain the model's output.  If specified, computes feature importances through aggregation.
        :type evaluation_examples: numpy.array or pandas.DataFrame or scipy.sparse.csr_matrix
        :param include_local: Include the local explanations in the returned global explanation.
            If evaluation examples are specified and include_local is False, will stream the local
            explanations to aggregate to global.
        :type include_local: bool
        :param batch_size: If include_local is False, specifies the batch size for aggregating
            local explanations to global.
        :type batch_size: int
        :return: A model explanation object. It is guaranteed to be a GlobalExplanation. If evaluation_examples are
            passed in, it will also have the properties of a LocalExplanation. If the model is a classifier (has
            predict_proba), it will have the properties of ClassesMixin, and if evaluation_examples were passed in it
            will also have the properties of PerClassMixin.
        :rtype: DynamicGlobalExplanation
        """
        coef = super(LinearRegression, self).explain_global()
        kwargs = {ExplainParams.METHOD: GlassboxModels.LINEAR_REGRESSION}
        kwargs = _get_explain_global_agg_kwargs(self, coef, False, model=self,
                                                evaluation_examples=evaluation_examples, include_local=include_local,
                                                batch_size=batch_size, **kwargs)
        return _create_global_explanation(**kwargs)

    def explain_local(self, evaluation_examples, **kwargs):
        """Use LinearExplainer to get the local feature importances from the trained glassbox model.

        :param evaluation_examples: The evaluation examples to compute local feature importances for.
        :type evaluation_examples: numpy or scipy array or pandas dataframe
        :return: The local explanation of feature importances.
        :rtype: Union[list, numpy.ndarray]
        """
        if self._linear_explainer is None:
            self._linear_explainer = LinearExplainer(self._linear, (self.mean, self.covariance))
        return self._linear_explainer.explain_local(evaluation_examples)

    def predict(self, dataset, **kwargs):
        """Call linear predict to predict labels using the glassbox model.

        :param dataset: The dataset to predict on.
        :type dataset: numpy or scipy array or pandas dataframe
        :return: The predictions of the model.
        :rtype: list
        """
        return super(LinearRegression, self).predict(dataset, **kwargs)

    __init__.__doc__ = (__init__.__doc__ +
                        '\nUses the parameters for LinearRegression:\n' +
                        SKLinearRegression.__doc__.replace('-', ''))

    fit.__doc__ = (fit.__doc__ +
                   '\nUses the parameters for LinearRegression:\n' +
                   SKLinearRegression.fit.__doc__.replace('-', ''))

    predict.__doc__ = (predict.__doc__ +
                       '\nUses the parameters for LinearRegression:\n' +
                       SKLinearRegression.predict.__doc__.replace('-', ''))


class LogisticRegression(LinearClassifierMixin, BaseGlassboxClassifier):
    available_explanations = [Extension.GLOBAL, Extension.LOCAL]
    explainer_type = Extension.GLASSBOX

    def __init__(self, classes=None, **kwargs):
        """Initialize the LogisticRegression glassbox model.

        :param classes: Class names as a list of strings. The order of the class names should match
            that of the model output.  Only required if explaining classifier.
        :type classes: list[str]
        """
        initializor = SKLogisticRegression
        super(LogisticRegression, self).__init__(initializor, **kwargs)
        self._logger.debug('Initializing LogisticRegression')
        self._classes = classes

    def fit(self, dataset, labels, **kwargs):
        """Call linear fit to fit the glassbox model.

        :param dataset: The dataset to train the model on.
        :type dataset: numpy or scipy array or pandas dataframe
        :param labels: The labels to train the model on.
        :type labels: numpy or scipy array
        """
        super(LogisticRegression, self).fit(dataset, labels, **kwargs)

    def explain_global(self, evaluation_examples=None, include_local=True,
                       batch_size=Defaults.DEFAULT_BATCH_SIZE, **kwargs):
        """Explain the model globally by either using the coef or aggregating local explanations to global.

        :param evaluation_examples: A matrix of feature vector examples (# examples x # features) on which to
            explain the model's output.  If specified, computes feature importances through aggregation.
        :type evaluation_examples: numpy.array or pandas.DataFrame or scipy.sparse.csr_matrix
        :param include_local: Include the local explanations in the returned global explanation.
            If evaluation examples are specified and include_local is False, will stream the local
            explanations to aggregate to global.
        :type include_local: bool
        :param batch_size: If include_local is False, specifies the batch size for aggregating
            local explanations to global.
        :type batch_size: int
        :return: A model explanation object. It is guaranteed to be a GlobalExplanation. If evaluation_examples are
            passed in, it will also have the properties of a LocalExplanation. If the model is a classifier (has
            predict_proba), it will have the properties of ClassesMixin, and if evaluation_examples were passed in it
            will also have the properties of PerClassMixin.
        :rtype: DynamicGlobalExplanation
        """
        coef = super(LogisticRegression, self).explain_global()
        kwargs = {ExplainParams.METHOD: GlassboxModels.LOGISTIC_REGRESSION}
        kwargs = _get_explain_global_agg_kwargs(self, coef, True, model=self,
                                                evaluation_examples=evaluation_examples, include_local=include_local,
                                                batch_size=batch_size, classes=self._classes, **kwargs)
        return _create_global_explanation(**kwargs)

    def explain_local(self, evaluation_examples, **kwargs):
        """Use LinearExplainer to get the local feature importances from the trained explainable model.

        :param evaluation_examples: The evaluation examples to compute local feature importances for.
        :type evaluation_examples: numpy or scipy array or pandas dataframe
        :return: The local explanation of feature importances.
        :rtype: Union[list, numpy.ndarray]
        """
        if self._linear_explainer is None:
            self._linear_explainer = LinearExplainer(self._linear, (self.mean, self.covariance))
        return self._linear_explainer.explain_local(evaluation_examples)

    def predict(self, dataset, **kwargs):
        """Call linear predict to predict labels using the glassbox model.

        :param dataset: The dataset to predict on.
        :type dataset: numpy or scipy array or pandas dataframe
        :return: The predictions of the model.
        :rtype: list
        """
        return super(LogisticRegression, self).predict(dataset, **kwargs)

    def predict_proba(self, dataset, **kwargs):
        """Call linear predict_proba to predict probabilities using the glassbox model.

        :param dataset: The dataset to predict probabilities on.
        :type dataset: numpy or scipy array or pandas dataframe
        :return: The predictions of the model.
        :rtype: list
        """
        return super(LogisticRegression, self).predict_proba(dataset, **kwargs)

    __init__.__doc__ = (__init__.__doc__ +
                        '\nUses the parameters for LogisticRegression:\n' +
                        SKLogisticRegression.__doc__.replace('-', ''))

    fit.__doc__ = (fit.__doc__ +
                   '\nUses the parameters for LogisticRegression:\n' +
                   SKLogisticRegression.fit.__doc__.replace('-', ''))

    predict.__doc__ = (predict.__doc__ +
                       '\nUses the parameters for LogisticRegression:\n' +
                       SKLogisticRegression.predict.__doc__.replace('-', ''))

    predict_proba.__doc__ = (predict_proba.__doc__ +
                             '\nUses the parameters for LogisticRegression:\n' +
                             SKLogisticRegression.predict_proba.__doc__.replace('-', ''))


class LinearExplainableModel(LinearClassifierMixin, LinearBase, BaseExplainableModel):
    available_explanations = [Extension.GLOBAL, Extension.LOCAL]
    explainer_type = Extension.GLASSBOX

    """Linear explainable model.

    :param multiclass: Set to true to generate a multiclass model.
    :type multiclass: bool
    :param random_state: Int to seed the model.
    :type random_state: int
    """

    def __init__(self, multiclass=False, random_state=DEFAULT_RANDOM_STATE, classification=True, **kwargs):
        """Initialize the LinearExplainableModel.

        :param multiclass: Set to true to generate a multiclass model.
        :type multiclass: bool
        :param random_state: Int to seed the model.
        :type random_state: int
        """
        self.multiclass = multiclass
        self.random_state = random_state
        if self.multiclass:
            initializer = SKLogisticRegression
            kwargs['random_state'] = random_state
        else:
            initializer = SKLinearRegression
        super(LinearExplainableModel, self).__init__(initializer, **kwargs)
        self._logger.debug('Initializing LinearExplainableModel')
        self._method = 'mimic.linear'
        self._linear_explainer = None
        self._classification = classification

    def fit(self, dataset, labels, **kwargs):
        """Call linear fit to fit the explainable model.

        :param dataset: The dataset to train the model on.
        :type dataset: numpy or scipy array
        :param labels: The labels to train the model on.
        :type labels: numpy or scipy array
        """
        super(LinearExplainableModel, self).fit(dataset, labels, **kwargs)

    def explain_local(self, evaluation_examples, **kwargs):
        """Use LinearExplainer to get the local feature importances from the trained explainable model.

        :param evaluation_examples: The evaluation examples to compute local feature importances for.
        :type evaluation_examples: numpy or scipy array
        :return: The local explanation of feature importances.
        :rtype: Union[list, numpy.ndarray]
        """
        if self._linear_explainer is None:
            self._linear_explainer = _create_linear_explainer(self._linear, self.multiclass, self.mean,
                                                              self.covariance, self.random_state)
        return _compute_local_shap_values(self._linear_explainer, evaluation_examples, self._classification)

    @property
    def expected_values(self):
        """Use LinearExplainer to get the expected values.

        :return: The expected values of the linear model.
        :rtype: list
        """
        if self._linear_explainer is None:
            self._linear_explainer = _create_linear_explainer(self._linear, self.multiclass, self.mean,
                                                              self.covariance, self.random_state)
        if isinstance(self._linear_explainer, list):
            expected_values = []
            for explainer in self._linear_explainer:
                expected_values.append(explainer.expected_value)
            return expected_values
        else:
            expected_values = self._linear_explainer.expected_value
            if self._classification and not self.multiclass:
                expected_values = [-expected_values, expected_values]
            return expected_values

    def predict(self, dataset, **kwargs):
        """Call linear predict to predict labels using the explainable model.

        :param dataset: The dataset to predict on.
        :type dataset: numpy or scipy array
        :return: The predictions of the model.
        :rtype: list
        """
        return super(LinearExplainableModel, self).predict(dataset, **kwargs)

    def predict_proba(self, dataset, **kwargs):
        """Call linear predict_proba to predict probabilities using the explainable model.

        :param dataset: The dataset to predict probabilities on.
        :type dataset: numpy or scipy array
        :return: The predictions of the model.
        :rtype: list
        """
        if self.multiclass:
            return self._linear.predict_proba(dataset, **kwargs)
        else:
            raise Exception('predict_proba not supported for regression or binary classification dataset')

    __init__.__doc__ = (__init__.__doc__ +
                        '\nIf multiclass=True, uses the parameters for LogisticRegression:\n' +
                        SKLogisticRegression.__doc__.replace('-', '') +
                        '\nOtherwise, if multiclass=False, uses the parameters for LinearRegression:\n' +
                        SKLinearRegression.__doc__.replace('-', ''))

    fit.__doc__ = (fit.__doc__ +
                   '\nIf multiclass=True, uses the parameters for LogisticRegression:\n' +
                   SKLogisticRegression.fit.__doc__.replace('-', '') +
                   '\nOtherwise, if multiclass=False, uses the parameters for LinearRegression:\n' +
                   SKLinearRegression.fit.__doc__.replace('-', ''))

    predict.__doc__ = (predict.__doc__ +
                       '\nIf multiclass=True, uses the parameters for LogisticRegression:\n' +
                       SKLogisticRegression.predict.__doc__.replace('-', '') +
                       '\nOtherwise, if multiclass=False, uses the parameters for LinearRegression:\n' +
                       SKLinearRegression.predict.__doc__.replace('-', ''))

    predict_proba.__doc__ = (predict_proba.__doc__ +
                             '\nIf multiclass=True, uses the parameters for LogisticRegression:\n' +
                             SKLogisticRegression.predict_proba.__doc__.replace('-', '') +
                             '\nOtherwise predict_proba is not supported for regression or binary classification.\n')


class SGDExplainableModel(BaseExplainableModel):
    available_explanations = [Extension.GLOBAL, Extension.LOCAL]
    explainer_type = Extension.GLASSBOX

    """Stochastic Gradient Descent explainable model.

    :param multiclass: Set to true to generate a multiclass model.
    :type multiclass: bool
    :param random_state: Int to seed the model.
    :type random_state: int
    """

    def __init__(self, multiclass=False, random_state=DEFAULT_RANDOM_STATE, classification=True, **kwargs):
        """Initialize the SGDExplainableModel.

        :param multiclass: Set to true to generate a multiclass model.
        :type multiclass: bool
        :param random_state: Int to seed the model.
        :type random_state: int
        """
        self.multiclass = multiclass
        self.random_state = random_state
        if self.multiclass:
            initializer = SGDClassifier
        else:
            initializer = SGDRegressor
        initializer_args = _get_initializer_args(kwargs)
        self._sgd = initializer(random_state=random_state, **initializer_args)
        super(SGDExplainableModel, self).__init__(**kwargs)
        self._logger.debug('Initializing SGDExplainableModel')
        self._method = 'mimic.sgd'
        self._sgd_explainer = None
        self._classification = classification

    __init__.__doc__ = (__init__.__doc__ +
                        '\nIf multiclass=True, uses the parameters for SGDClassifier:\n' +
                        SGDClassifier.__doc__.replace('-', '') +
                        '\nOtherwise, if multiclass=False, uses the parameters for SGDRegressor:\n' +
                        SGDRegressor.__doc__.replace('-', ''))

    def fit(self, dataset, labels, **kwargs):
        """Call linear fit to fit the explainable model.

        Store the mean and covariance of the background data for local explanation.

        :param dataset: The dataset to train the model on.
        :type dataset: numpy or scipy array
        :param labels: The labels to train the model on.
        :type labels: numpy or scipy array
        """
        self._sgd.fit(dataset, labels, **kwargs)
        original_mean = np.asarray(dataset.mean(0))
        if len(original_mean.shape) == 2:
            mean_shape = original_mean.shape[1]
            self.mean = original_mean.reshape((mean_shape,))
        else:
            self.mean = original_mean
        if not sp.sparse.issparse(dataset):
            self.covariance = np.cov(dataset, rowvar=False)
        else:
            # Not needed for sparse case
            self.covariance = None

    fit.__doc__ = (fit.__doc__ +
                   '\nIf multiclass=True, uses the parameters for SGDClassifier:\n' +
                   SGDClassifier.fit.__doc__.replace('-', '') +
                   '\nOtherwise, if multiclass=False, uses the parameters for SGDRegressor:\n' +
                   SGDRegressor.fit.__doc__.replace('-', ''))

    def predict(self, dataset, **kwargs):
        """Call SGD predict to predict labels using the explainable model.

        :param dataset: The dataset to predict on.
        :type dataset: numpy or scipy array
        :return: The predictions of the model.
        :rtype: list
        """
        return self._sgd.predict(dataset)

    predict.__doc__ = (predict.__doc__ +
                       '\nIf multiclass=True, uses the parameters for SGDClassifier:\n' +
                       SGDClassifier.predict.__doc__.replace('-', '') +
                       '\nOtherwise, if multiclass=False, uses the parameters for SGDRegressor:\n' +
                       SGDRegressor.predict.__doc__.replace('-', ''))

    def predict_proba(self, dataset, **kwargs):
        """Call SGD predict_proba to predict probabilities using the explainable model.

        :param dataset: The dataset to predict probabilities on.
        :type dataset: numpy or scipy array
        :return: The predictions of the model.
        :rtype: list
        """
        if self.multiclass:
            return self._sgd.predict_proba(dataset)
        else:
            raise Exception('predict_proba not supported for regression or binary classification dataset')

    predict_proba.__doc__ = (predict_proba.__doc__ +
                             '\nIf multiclass=True, uses the parameters for SGDClassifier:\n' +
                             SGDClassifier.predict_proba.__doc__.replace('-', '')
                             .replace(':class:`sklearn.calibration.CalibratedClassifierCV`',
                                      'CalibratedClassifierCV') +
                             '\nOtherwise predict_proba is not supported for regression or binary classification.\n')

    def explain_global(self, **kwargs):
        """Call coef to get the global feature importances from the SGD surrogate model.

        :return: The global explanation of feature importances.
        :rtype: list
        """
        coef = self._sgd.coef_
        if (len(coef.shape) == 2):
            return np.mean(coef, axis=0)
        return coef

    def explain_local(self, evaluation_examples, **kwargs):
        """Use LinearExplainer to get the local feature importances from the trained explainable model.

        :param evaluation_examples: The evaluation examples to compute local feature importances for.
        :type evaluation_examples: numpy or scipy array
        :return: The local explanation of feature importances.
        :rtype: Union[list, numpy.ndarray]
        """
        if self._sgd_explainer is None:
            self._sgd_explainer = _create_linear_explainer(self._sgd, self.multiclass, self.mean,
                                                           self.covariance, self.random_state)
        return _compute_local_shap_values(self._sgd_explainer, evaluation_examples, self._classification)

    @property
    def expected_values(self):
        """Use LinearExplainer to get the expected values.

        :return: The expected values of the linear model.
        :rtype: list
        """
        if self._sgd_explainer is None:
            self._sgd_explainer = _create_linear_explainer(self._sgd, self.multiclass, self.mean,
                                                           self.covariance, self.random_state)
        if isinstance(self._sgd_explainer, list):
            expected_values = []
            for explainer in self._sgd_explainer:
                expected_values.append(explainer.expected_value)
            return expected_values
        else:
            expected_values = self._sgd_explainer.expected_value
            if self._classification and not self.multiclass:
                expected_values = [-expected_values, expected_values]
            return expected_values

    @property
    def model(self):
        """Retrieve the underlying model.

        :return: The SGD model, either classifier or regressor.
        :rtype: Union[SGDClassifier, SGDRegressor]
        """
        return self._sgd
