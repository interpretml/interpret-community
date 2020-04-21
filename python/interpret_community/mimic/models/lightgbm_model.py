# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Defines an explainable lightgbm model."""

from .explainable_model import BaseExplainableModel, _get_initializer_args, _clean_doc
from .tree_model_utils import _explain_local_tree_surrogate, _expected_values_tree_surrogate
from ...common.constants import ShapValuesOutput, LightGBMSerializationConstants, \
    ExplainableModelType, Extension
import json
import warnings
import logging
import inspect

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', 'Starting from version 2.2.1', UserWarning)
    import shap
    try:
        from lightgbm import LGBMRegressor, LGBMClassifier, Booster
        import lightgbm
        from packaging import version
        if (version.parse(lightgbm.__version__) <= version.parse('2.2.1')):
            print("Using older than supported version of lightgbm, please upgrade to version greater than 2.2.1")
    except ImportError:
        print("Could not import lightgbm, required if using LGBMExplainableModel")

DEFAULT_RANDOM_STATE = 123
_N_FEATURES = '_n_features'
_N_CLASSES = '_n_classes'


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
        :type dataset: numpy or scipy array
        :param labels: The labels to train the model on.
        :type labels: numpy or scipy array
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
        :type dataset: numpy or scipy array
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
        :type dataset: numpy or scipy array
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
            self._tree_explainer = shap.TreeExplainer(self._lgbm)
        return _explain_local_tree_surrogate(self._lgbm, evaluation_examples, self._tree_explainer,
                                             self._shap_values_output, self._classification,
                                             probabilities, self.multiclass)

    @property
    def expected_values(self):
        """Use TreeExplainer to get the expected values.

        :return: The expected values of the LightGBM tree model.
        :rtype: list
        """
        if self._tree_explainer is None:
            self._tree_explainer = shap.TreeExplainer(self._lgbm)
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
    def explainable_model_type(self):
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
        lightgbm = LGBMExplainableModel.__new__(LGBMExplainableModel)
        # Get _n_features
        _n_features = properties.pop(_N_FEATURES)
        # If classification case get _n_classes
        if json.loads(properties[LightGBMSerializationConstants.MULTICLASS]):
            _n_classes = properties.pop(_N_CLASSES)
        # load all of the properties
        for key, value in properties.items():
            # Regenerate the properties on the fly
            if key in LightGBMSerializationConstants.nonify_properties:
                if key == LightGBMSerializationConstants.LOGGER:
                    parent = logging.getLogger(__name__)
                    lightgbm_identity = json.loads(properties[LightGBMSerializationConstants.IDENTITY])
                    lightgbm.__dict__[key] = parent.getChild(lightgbm_identity)
                elif key == LightGBMSerializationConstants.TREE_EXPLAINER:
                    lightgbm.__dict__[key] = None
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
                new_lgbm._n_features = _n_features
                lightgbm.__dict__[key] = new_lgbm
            elif key in LightGBMSerializationConstants.enum_properties:
                # NOTE: If more enums added in future, will need to handle this differently
                lightgbm.__dict__[key] = ShapValuesOutput(json.loads(value))
            else:
                lightgbm.__dict__[key] = json.loads(value)
        return lightgbm
