# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Defines an explainable linear model."""
import numpy as np
from scipy.sparse import issparse, csr_matrix

from .explainable_model import BaseExplainableModel, _get_initializer_args, _clean_doc
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier, SGDRegressor, Lasso
from ...common.constants import ExplainableModelType, Extension, SHAPDefaults
from ...common.explanation_utils import _summarize_data
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', 'Starting from version 2.2.1', UserWarning)
    import shap
    from shap.common import DenseData

DEFAULT_RANDOM_STATE = 123
FEATURE_DEPENDENCE = 'interventional'
LINEAR_PENALTY = 'penalty'
LINEAR_L2 = 'l2'
LINEAR_SOLVER = 'solver'
LINEAR_LBFGS = 'lbfgs'
LINEAR_MULTICLASS = 'multi_class'
LINEAR_MULTINOMIAL = 'multinomial'
LINEAR_RANDOM_STATE = 'random_state'


class LinearExplainer(shap.LinearExplainer):
    """Linear explainer with support for sparse data and sparse output."""

    def __init__(self, model, data, feature_dependence=FEATURE_DEPENDENCE):
        """Initialize the LinearExplainer.

        :param model: The linear model to compute the shap values for as a (coef, intercept) tuple.
        :type model: (numpy.matrix, double)
        :param data: The mean and covariance of the dataset.
        :type data: (scipy.csr_matrix, None)
        :param feature_dependence: Indicates the type of feature dependency, interventional by default.
        :type feature_dependence: str
        """
        self.is_sparse = data[1] is None
        if self.is_sparse:
            # Sparse case
            self.coef = model[0]
            if not issparse(self.coef):
                self.coef = np.asmatrix(self.coef)
            self.intercept = model[1]
            self._background = data[0]
            self.expected_value = np.array(self._background.dot(self.coef.T) + self.intercept).flatten()
        else:
            # Dense case
            super(LinearExplainer, self).__init__(model, data, feature_dependence=feature_dependence)

    def shap_values(self, evaluation_examples):
        """Estimate the SHAP values for a set of samples.

        :param evaluation_examples: The evaluation examples.
        :type evaluation_examples: numpy or scipy array
        :return: For models with a single output this returns a matrix of SHAP values
            (# samples x # features). Each row sums to the difference between the model output for that
            sample and the expected value of the model output (which is stored as expected_value
            attribute of the explainer).
        :rtype: Union[list, numpy.ndarray]
        """
        if self.is_sparse:
            assert len(evaluation_examples.shape) == 2, "Sparse instance must have 2 dimensions!"
            assert self.coef.shape[0] == 1, "Multiclass coefficients need to be evaluated separately"
            mean_multiplier = csr_matrix(np.ones((evaluation_examples.shape[0], 1)))
            return (evaluation_examples - mean_multiplier * self._background).multiply(self.coef[0]).tocsr()
        else:
            return super().shap_values(evaluation_examples)


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
            linear_explainer = LinearExplainer((class_coef, intercept), (mean, covariance),
                                               feature_dependence=SHAPDefaults.INDEPENDENT)
            explainers.append(linear_explainer)
        return explainers
    else:
        model_coef = model.coef_
        model_intercept = model.intercept_
        return LinearExplainer((model_coef, model_intercept), (mean, covariance),
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


def _compute_background_data(dataset):
    """Compute the background data for linear explainer.

    :param dataset: The input dataset to compute background for.
    :type dataset: numpy or scipy array
    """
    background = _summarize_data(dataset)
    if isinstance(background, DenseData):
        background = background.data
    if not issparse(background) and len(background.shape) == 2:
        mean_shape = background.shape[1]
        # Take mean of clusters to get better representation of background
        if background.shape[0] > 1:
            background = background.mean(axis=0)
        # Check again prior to reshape
        if len(background.shape) == 2:
            background = background.reshape((mean_shape,))
    return background


class LinearExplainableModel(BaseExplainableModel):
    available_explanations = [Extension.GLOBAL, Extension.LOCAL]
    explainer_type = Extension.GLASSBOX

    """Linear explainable model.

    :param multiclass: Set to true to generate a multiclass model.
    :type multiclass: bool
    :param random_state: Int to seed the model.
    :type random_state: int
    :param classification: Indicates whether the model is used for classification or regression scenario.
    :type classification: bool
    :param sparse_data: Indicates whether the training data will be sparse.
    :type sparse_data: bool
    """

    def __init__(self, multiclass=False, random_state=DEFAULT_RANDOM_STATE, classification=True,
                 sparse_data=False, **kwargs):
        """Initialize the LinearExplainableModel.

        :param multiclass: Set to true to generate a multiclass model.
        :type multiclass: bool
        :param random_state: Int to seed the model.
        :type random_state: int
        :param classification: Indicates whether the model is used for classification or regression scenario.
        :type classification: bool
        :param sparse_data: Indicates whether the training data will be sparse.
        :type sparse_data: bool
        """
        self.multiclass = multiclass
        self.random_state = random_state
        self._sparse_data = sparse_data
        if self.multiclass:
            initializer = LogisticRegression
            if self._sparse_data:
                kwargs[LINEAR_PENALTY] = LINEAR_L2
            if LINEAR_SOLVER not in kwargs:
                kwargs[LINEAR_SOLVER] = LINEAR_LBFGS
                kwargs[LINEAR_MULTICLASS] = LINEAR_MULTINOMIAL
            kwargs[LINEAR_RANDOM_STATE] = random_state
        else:
            if self._sparse_data:
                initializer = Lasso
            else:
                initializer = LinearRegression
        initializer_args = _get_initializer_args(kwargs)
        self._linear = initializer(**initializer_args)
        super(LinearExplainableModel, self).__init__(**kwargs)
        self._logger.debug('Initializing LinearExplainableModel')
        self._method = 'linear'
        self._linear_explainer = None
        self._classification = classification

    __init__.__doc__ = (__init__.__doc__ +
                        '\nIf multiclass=True, uses the parameters for LogisticRegression:\n' +
                        _clean_doc(LogisticRegression.__doc__) +
                        '\nOtherwise, if multiclass=False, uses the parameters for LinearRegression:\n' +
                        _clean_doc(LinearRegression.__doc__))

    def fit(self, dataset, labels, **kwargs):
        """Call linear fit to fit the explainable model.

        Store the mean and covariance of the background data for local explanation.

        :param dataset: The dataset to train the model on.
        :type dataset: numpy or scipy array
        :param labels: The labels to train the model on.
        :type labels: numpy or scipy array
        """
        self._linear.fit(dataset, labels, **kwargs)
        self._background = _compute_background_data(dataset)
        if not issparse(dataset):
            self.covariance = np.cov(dataset, rowvar=False)
        else:
            # Not needed for sparse case
            self.covariance = None

    fit.__doc__ = (fit.__doc__ +
                   '\nIf multiclass=True, uses the parameters for LogisticRegression:\n' +
                   _clean_doc(LogisticRegression.fit.__doc__) +
                   '\nOtherwise, if multiclass=False, uses the parameters for LinearRegression:\n' +
                   _clean_doc(LinearRegression.fit.__doc__))

    def predict(self, dataset, **kwargs):
        """Call linear predict to predict labels using the explainable model.

        :param dataset: The dataset to predict on.
        :type dataset: numpy or scipy array
        :return: The predictions of the model.
        :rtype: list
        """
        return self._linear.predict(dataset)

    predict.__doc__ = (predict.__doc__ +
                       '\nIf multiclass=True, uses the parameters for LogisticRegression:\n' +
                       _clean_doc(LogisticRegression.predict.__doc__) +
                       '\nOtherwise, if multiclass=False, uses the parameters for LinearRegression:\n' +
                       _clean_doc(LinearRegression.predict.__doc__))

    def predict_proba(self, dataset, **kwargs):
        """Call linear predict_proba to predict probabilities using the explainable model.

        :param dataset: The dataset to predict probabilities on.
        :type dataset: numpy or scipy array
        :return: The predictions of the model.
        :rtype: list
        """
        if self.multiclass:
            return self._linear.predict_proba(dataset)
        else:
            raise Exception('predict_proba not supported for regression or binary classification dataset')

    predict_proba.__doc__ = (predict_proba.__doc__ +
                             '\nIf multiclass=True, uses the parameters for LogisticRegression:\n' +
                             _clean_doc(LogisticRegression.predict_proba.__doc__) +
                             '\nOtherwise predict_proba is not supported for regression or binary classification.\n')

    def explain_global(self, **kwargs):
        """Call coef to get the global feature importances from the linear surrogate model.

        :return: The global explanation of feature importances.
        :rtype: list
        """
        coef = self._linear.coef_
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
        if self._linear_explainer is None:
            self._linear_explainer = _create_linear_explainer(self._linear, self.multiclass, self._background,
                                                              self.covariance, self.random_state)
        return _compute_local_shap_values(self._linear_explainer, evaluation_examples, self._classification)

    @property
    def expected_values(self):
        """Use LinearExplainer to get the expected values.

        :return: The expected values of the linear model.
        :rtype: list
        """
        if self._linear_explainer is None:
            self._linear_explainer = _create_linear_explainer(self._linear, self.multiclass, self._background,
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
        self._method = 'sgd'
        self._sgd_explainer = None
        self._classification = classification

    __init__.__doc__ = (__init__.__doc__ +
                        '\nIf multiclass=True, uses the parameters for SGDClassifier:\n' +
                        _clean_doc(SGDClassifier.__doc__) +
                        '\nOtherwise, if multiclass=False, uses the parameters for SGDRegressor:\n' +
                        _clean_doc(SGDRegressor.__doc__))

    def fit(self, dataset, labels, **kwargs):
        """Call linear fit to fit the explainable model.

        Store the mean and covariance of the background data for local explanation.

        :param dataset: The dataset to train the model on.
        :type dataset: numpy or scipy array
        :param labels: The labels to train the model on.
        :type labels: numpy or scipy array
        """
        self._sgd.fit(dataset, labels, **kwargs)
        self._background = _compute_background_data(dataset)
        if not issparse(dataset):
            self.covariance = np.cov(dataset, rowvar=False)
        else:
            # Not needed for sparse case
            self.covariance = None

    fit.__doc__ = (fit.__doc__ +
                   '\nIf multiclass=True, uses the parameters for SGDClassifier:\n' +
                   _clean_doc(SGDClassifier.fit.__doc__) +
                   '\nOtherwise, if multiclass=False, uses the parameters for SGDRegressor:\n' +
                   _clean_doc(SGDRegressor.fit.__doc__))

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
                       _clean_doc(SGDClassifier.predict.__doc__) +
                       '\nOtherwise, if multiclass=False, uses the parameters for SGDRegressor:\n' +
                       _clean_doc(SGDRegressor.predict.__doc__))

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
                             _clean_doc(SGDClassifier.predict_proba.__doc__)
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
            self._sgd_explainer = _create_linear_explainer(self._sgd, self.multiclass, self._background,
                                                           self.covariance, self.random_state)
        return _compute_local_shap_values(self._sgd_explainer, evaluation_examples, self._classification)

    @property
    def expected_values(self):
        """Use LinearExplainer to get the expected values.

        :return: The expected values of the linear model.
        :rtype: list
        """
        if self._sgd_explainer is None:
            self._sgd_explainer = _create_linear_explainer(self._sgd, self.multiclass, self._background,
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
