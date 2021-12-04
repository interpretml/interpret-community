# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Defines an explainable lightgbm model."""

import inspect
import json
import logging
import warnings

from packaging import version
from scipy.sparse import issparse

from ...common.constants import (ExplainableModelType, Extension,
                                 LightGBMSerializationConstants,
                                 ShapValuesOutput)
from .explainable_model import (BaseExplainableModel, _clean_doc,
                                _get_initializer_args)
from .tree_model_utils import (_expected_values_tree_surrogate,
                               _explain_local_tree_surrogate)

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', 'Starting from version 2.2.1', UserWarning)
    import shap
    try:
        import lightgbm
        from lightgbm import Booster, LGBMClassifier, LGBMRegressor
        if (version.parse(lightgbm.__version__) <= version.parse('2.2.1')):
            print("Using older than supported version of lightgbm, please upgrade to version greater than 2.2.1")
    except ImportError:
        print("Could not import lightgbm, required if using LGBMExplainableModel")

DEFAULT_RANDOM_STATE = 123
_N_FEATURES = '_n_features'
_N_CLASSES = '_n_classes'
NUM_ITERATIONS = 'num_iterations'
_FITTED = 'fitted_'


class _LGBMFunctionWrapper(object):
    """Decorate the predict method, temporary workaround for sparse case until TreeExplainer support is added.

    :param function: The prediction function to wrap.
    :type function: function
    """

    def __init__(self, function):
        """Wraps a function to reshape the input data.

        :param function: The prediction function to wrap.
        :type function: function
        """
        self._function = function

    def predict_wrapper(self, X, *args, **kwargs):
        """Wraps a prediction function from lightgbm learner.

        If version is ==3.0.0, densifies the input dataset.

        :param X: The model evaluation examples.
        :type X: numpy.array
        :return: Prediction result.
        :rtype: numpy.array
        """
        if issparse(X):
            X = X.toarray()
        return self._function(X, *args, **kwargs)


class _SparseTreeExplainer(object):

    """Wraps the lightgbm model to enable sparse feature contributions.

    If version is >=3.1.0, runs on sparse input data by calling predict function directly.

    :param lgbm: The lightgbm model to wrap.
    :type lgbm: LGBMModel
    :param tree_explainer: The tree_explainer used for dense data.
    :type tree_explainer: shap.TreeExplainer
    """

    def __init__(self, lgbm, tree_explainer):
        """Wraps the lightgbm model to enable sparse feature contributions.

        :param lgbm: The lightgbm model to wrap.
        :type lgbm: LGBMModel
        :param tree_explainer: The tree_explainer used for dense data.
        :type tree_explainer: shap.TreeExplainer
        """
        self._lgbm = lgbm
        self._tree_explainer = tree_explainer
        self._num_iters = -1
        # Get the number of iterations trained for from the booster
        if hasattr(self._lgbm._Booster, 'params'):
            if NUM_ITERATIONS in self._lgbm._Booster.params:
                self._num_iters = self._lgbm._Booster.params[NUM_ITERATIONS]
        # If best iteration specified, use that
        if self._lgbm._best_iteration is not None:
            self._num_iters = self._lgbm._best_iteration
        self.expected_value = None

    def shap_values(self, X):
        """Calls lightgbm predict directly for sparse case.

        If lightgbm version is >=3.1.0, runs on sparse input data
        by calling predict function directly with pred_contrib=True.
        Uses tree explainer for dense input data.

        :param X: The model evaluation examples.
        :type X: numpy.array or scipy.sparse.csr_matrix
        :return: The feature importance values.
        :rtype: numpy.array, scipy.sparse or list of scipy.sparse
        """
        if issparse(X):
            shap_values = self._lgbm.predict(X,
                                             num_iteration=self._num_iters,
                                             pred_contrib=True)
            if isinstance(shap_values, list):
                shape = shap_values[0].shape
                self.expected_value = shap_values[0][0, shape[1] - 1]
                for idx, class_values in enumerate(shap_values):
                    shap_values[idx] = class_values[:, :shape[1] - 1]
            else:
                shape = shap_values.shape
                self.expected_value = shap_values[0, shape[1] - 1]
                shap_values = shap_values[:, :shape[1] - 1]
        else:
            shap_values = self._tree_explainer.shap_values(X)
            self.expected_value = self._tree_explainer.expected_value
        return shap_values


class LGBMExplainableModel(BaseExplainableModel):
    available_explanations = [Extension.GLOBAL, Extension.LOCAL]
    explainer_type = Extension.GLASSBOX

    """LightGBM (fast, high performance framework based on decision tree) explainable model.

    Please see documentation for more details: https://github.com/Microsoft/LightGBM

    Additional arguments to LightGBMClassifier and LightGBMRegressor can be passed through kwargs.

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
        """Initialize the LightGBM Model.

        Additional arguments to LightGBMClassifier and LightGBMRegressor can be passed through kwargs.

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
        initializer_args = _get_initializer_args(kwargs)
        if self.multiclass:
            initializer = LGBMClassifier
        else:
            initializer = LGBMRegressor
        self._lgbm = initializer(random_state=random_state, **initializer_args)
        super(LGBMExplainableModel, self).__init__(**kwargs)
        self._logger.debug('Initializing LGBMExplainableModel')
        self._method = 'lightgbm'
        self._tree_explainer = None
        self._shap_values_output = shap_values_output
        self._classification = classification

    try:
        __init__.__doc__ = (__init__.__doc__ +
                            '\nIf multiclass=True, uses the parameters for LGBMClassifier:\n' +
                            _clean_doc(LGBMClassifier.__init__.__doc__) +
                            '\nOtherwise, if multiclass=False, uses the parameters for LGBMRegressor:\n' +
                            _clean_doc(LGBMRegressor.__init__.__doc__))
    except Exception:
        pass

    def fit(self, dataset, labels, **kwargs):
        """Call lightgbm fit to fit the explainable model.

        :param dataset: The dataset to train the model on.
        :type dataset: numpy.array or pandas.DataFrame or scipy.sparse.csr_matrix
        :param labels: The labels to train the model on.
        :type labels: numpy.array
        """
        self._lgbm.fit(dataset, labels, **kwargs)

    try:
        fit.__doc__ = (fit.__doc__ +
                       '\nIf multiclass=True, uses the parameters for LGBMClassifier:\n' +
                       _clean_doc(LGBMClassifier.fit.__doc__) +
                       '\nOtherwise, if multiclass=False, uses the parameters for LGBMRegressor:\n' +
                       _clean_doc(LGBMRegressor.fit.__doc__))
    except Exception:
        pass

    def predict(self, dataset, **kwargs):
        """Call lightgbm predict to predict labels using the explainable model.

        :param dataset: The dataset to predict on.
        :type dataset: numpy.array or pandas.DataFrame or scipy.sparse.csr_matrix
        :return: The predictions of the model.
        :rtype: list
        """
        return self._lgbm.predict(dataset, **kwargs)

    try:
        predict.__doc__ = (predict.__doc__ +
                           '\nIf multiclass=True, uses the parameters for LGBMClassifier:\n' +
                           _clean_doc(LGBMClassifier.predict.__doc__) +
                           '\nOtherwise, if multiclass=False, uses the parameters for LGBMRegressor:\n' +
                           _clean_doc(LGBMRegressor.predict.__doc__))
    except Exception:
        pass

    def predict_proba(self, dataset, **kwargs):
        """Call lightgbm predict_proba to predict probabilities using the explainable model.

        :param dataset: The dataset to predict probabilities on.
        :type dataset: numpy.array or pandas.DataFrame or scipy.sparse.csr_matrix
        :return: The predictions of the model.
        :rtype: list
        """
        if self.multiclass:
            return self._lgbm.predict_proba(dataset, **kwargs)
        else:
            raise Exception("predict_proba not supported for regression or binary classification dataset")

    try:
        predict_proba.__doc__ = (predict_proba.__doc__ +
                                 '\nIf multiclass=True, uses the parameters for LGBMClassifier:\n' +
                                 _clean_doc(LGBMClassifier.predict_proba.__doc__) +
                                 '\nOtherwise predict_proba is not supported for ' +
                                 'regression or binary classification.\n')
    except Exception:
        pass

    def explain_global(self, **kwargs):
        """Call lightgbm feature importances to get the global feature importances from the explainable model.

        :return: The global explanation of feature importances.
        :rtype: numpy.ndarray
        """
        return self._lgbm.feature_importances_

    def _init_tree_explainer(self):
        """Creates the TreeExplainer.

        Includes a temporary fix for lightgbm 3.0 by wrapping predict method
        for sparse case to output dense data.
        Includes another temporary fix for lightgbm >= 3.1 to call predict
        function directly for sparse input data until shap TreeExplainer
        support is added.
        """
        if self._tree_explainer is None:
            self._tree_explainer = shap.TreeExplainer(self._lgbm)
            if version.parse('3.1.0') <= version.parse(lightgbm.__version__):
                self._tree_explainer = _SparseTreeExplainer(self._lgbm, self._tree_explainer)
            elif version.parse('3.0.0') == version.parse(lightgbm.__version__):
                wrapper = _LGBMFunctionWrapper(self._tree_explainer.model.original_model.predict)
                self._tree_explainer.model.original_model.predict = wrapper.predict_wrapper

    def explain_local(self, evaluation_examples, probabilities=None, **kwargs):
        """Use TreeExplainer to get the local feature importances from the trained explainable model.

        :param evaluation_examples: The evaluation examples to compute local feature importances for.
        :type evaluation_examples: numpy.array or pandas.DataFrame or scipy.sparse.csr_matrix
        :param probabilities: If output_type is probability, can specify the teacher model's
            probability for scaling the shap values.
        :type probabilities: numpy.ndarray
        :return: The local explanation of feature importances.
        :rtype: Union[list, numpy.ndarray]
        """
        self._init_tree_explainer()
        return _explain_local_tree_surrogate(self._lgbm, evaluation_examples, self._tree_explainer,
                                             self._shap_values_output, self._classification,
                                             probabilities, self.multiclass)

    @property
    def expected_values(self):
        """Use TreeExplainer to get the expected values.

        :return: The expected values of the LightGBM tree model.
        :rtype: list
        """
        self._init_tree_explainer()
        return _expected_values_tree_surrogate(self._lgbm, self._tree_explainer, self._shap_values_output,
                                               self._classification, self.multiclass)

    @property
    def model(self):
        """Retrieve the underlying model.

        :return: The lightgbm model, either classifier or regressor.
        :rtype: Union[LGBMClassifier, LGBMRegressor]
        """
        return self._lgbm

    @staticmethod
    def explainable_model_type():
        """Retrieve the model type.

        :return: Tree explainable model type.
        :rtype: ExplainableModelType
        """
        return ExplainableModelType.TREE_EXPLAINABLE_MODEL_TYPE

    def _save(self):
        """Return a string dictionary representation of the LGBMExplainableModel.

        :return: A serialized dictionary representation of the LGBMExplainableModel.
        :rtype: dict
        """
        properties = {}
        # Save all of the properties
        for key, value in self.__dict__.items():
            if key in LightGBMSerializationConstants.nonify_properties:
                properties[key] = None
            elif key in LightGBMSerializationConstants.save_properties:
                # Save booster model to string representation
                # This is not recommended but can be necessary to get around pickle being not secure
                # See here for more info:
                # https://github.com/Microsoft/LightGBM/issues/1942
                # https://github.com/Microsoft/LightGBM/issues/1217
                properties[key] = value.booster_.model_to_string()
            else:
                properties[key] = json.dumps(value)
        # Need to add _n_features
        properties[_N_FEATURES] = self._lgbm._n_features
        # And if classification case need to add _n_classes
        if self.multiclass:
            properties[_N_CLASSES] = self._lgbm._n_classes
        if hasattr(self._lgbm, _FITTED):
            properties[_FITTED] = json.dumps(getattr(self._lgbm, _FITTED))
        return properties

    @staticmethod
    def _load(properties):
        """Load a LGBMExplainableModel from the given properties.

        :param properties: A serialized dictionary representation of the LGBMExplainableModel.
        :type properties: dict
        :return: The deserialized LGBMExplainableModel.
        :rtype: interpret_community.mimic.models.LGBMExplainableModel
        """
        # create the LGBMExplainableModel without any properties using the __new__ function, similar to pickle
        lgbm_model = LGBMExplainableModel.__new__(LGBMExplainableModel)
        # Get _n_features
        _n_features = properties.pop(_N_FEATURES)
        # If classification case get _n_classes
        if json.loads(properties[LightGBMSerializationConstants.MULTICLASS]):
            _n_classes = properties.pop(_N_CLASSES)
        fitted_ = None
        if _FITTED in properties:
            fitted_ = json.loads(properties[_FITTED])
        elif version.parse('3.3.1') <= version.parse(lightgbm.__version__):
            # If deserializing older model in newer version set this to true to prevent errors on calls
            fitted_ = True
        # load all of the properties
        for key, value in properties.items():
            # Regenerate the properties on the fly
            if key in LightGBMSerializationConstants.nonify_properties:
                if key == LightGBMSerializationConstants.LOGGER:
                    parent = logging.getLogger(__name__)
                    lightgbm_identity = json.loads(properties[LightGBMSerializationConstants.IDENTITY])
                    lgbm_model.__dict__[key] = parent.getChild(lightgbm_identity)
                elif key == LightGBMSerializationConstants.TREE_EXPLAINER:
                    lgbm_model.__dict__[key] = None
                else:
                    raise Exception("Unknown nonify key on deserialize in LightGBMExplainableModel: {}".format(key))
            elif key in LightGBMSerializationConstants.save_properties:
                # Load the booster from file and re-create the LGBMClassifier or LGBMRegressor
                # This is not recommended but can be necessary to get around pickle being not secure
                # See here for more info:
                # https://github.com/Microsoft/LightGBM/issues/1942
                # https://github.com/Microsoft/LightGBM/issues/1217
                booster_args = {LightGBMSerializationConstants.MODEL_STR: value}
                is_multiclass = json.loads(properties[LightGBMSerializationConstants.MULTICLASS])
                if is_multiclass:
                    objective = LightGBMSerializationConstants.MULTICLASS
                else:
                    objective = LightGBMSerializationConstants.REGRESSION
                if LightGBMSerializationConstants.MODEL_STR in inspect.getargspec(Booster).args:
                    extras = {LightGBMSerializationConstants.OBJECTIVE: objective}
                    lgbm_booster = Booster(**booster_args, params=extras)
                else:
                    # For backwards compatibility with older versions of lightgbm
                    booster_args[LightGBMSerializationConstants.OBJECTIVE] = objective
                    lgbm_booster = Booster(params=booster_args)
                if is_multiclass:
                    new_lgbm = LGBMClassifier()
                    new_lgbm._Booster = lgbm_booster
                    new_lgbm._n_classes = _n_classes
                else:
                    new_lgbm = LGBMRegressor()
                    new_lgbm._Booster = lgbm_booster
                # Specify fitted_ for newer versions of lightgbm on deserialize
                if fitted_ is not None:
                    new_lgbm.fitted_ = fitted_
                new_lgbm._n_features = _n_features
                lgbm_model.__dict__[key] = new_lgbm
            elif key in LightGBMSerializationConstants.enum_properties:
                # NOTE: If more enums added in future, will need to handle this differently
                lgbm_model.__dict__[key] = ShapValuesOutput(json.loads(value))
            else:
                lgbm_model.__dict__[key] = json.loads(value)
        return lgbm_model
