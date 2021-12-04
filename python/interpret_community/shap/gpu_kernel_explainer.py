# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


"""Defines the GPUKernelExplainer for computing explanations on black box models or functions."""

import numpy as np

from .._internal.raw_explain.raw_explain_utils import (
    get_datamapper_and_transformed_data, transform_with_datamapper)
from ..common.aggregate import add_explain_global_method
from ..common.blackbox_explainer import (
    BlackBoxExplainer, add_prepare_function_and_summary_method,
    init_blackbox_decorator)
from ..common.constants import (Attributes, Defaults, ExplainParams,
                                ExplainType, Extension, ModelTask)
from ..common.explanation_utils import (_append_shap_values_instance,
                                        _convert_single_instance_to_multi,
                                        _convert_to_list)
from ..common.model_wrapper import _wrap_model
from ..dataset.dataset_wrapper import DatasetWrapper
from ..dataset.decorator import init_tabular_decorator, tabular_decorator
from ..explanation.explanation import (
    _create_local_explanation, _create_raw_feats_local_explanation,
    _get_raw_explainer_create_explanation_kwargs)
from .kwargs_utils import _get_explain_global_kwargs

try:
    import cuml
    if cuml.__version__ == '0.18.0':
        from cuml.experimental.explainer import \
            KernelExplainer as cuml_Kernel_SHAP
    elif cuml.__version__ > '0.18.0':
        from cuml.explainer import KernelExplainer as cuml_Kernel_SHAP
    rapids_installed = True
except ImportError:
    rapids_installed = False


@add_prepare_function_and_summary_method
@add_explain_global_method
class GPUKernelExplainer(BlackBoxExplainer):
    available_explanations = [Extension.GLOBAL, Extension.LOCAL]
    explainer_type = Extension.BLACKBOX

    """
    GPU version of the Kernel Explainer for explaining black box models or functions.

    Uses cuml's GPU Kernel SHAP.
    https://docs.rapids.ai/api/cuml/stable/api.html#shap-kernel-explainer

    Characteristics of the GPU version:
     * Unlike the SHAP package, ``nsamples`` is a parameter at the
       initialization of the explainer and there is a small initialization
       time.
     * Only tabular data is supported for now, via passing the background
       dataset explicitly.
     * Sparse data support is planned for the near future.
     * Further optimizations are in progress. For example, if the background
       dataset has constant value columns and the observation has the same
       value in some entries, the number of evaluations of the function can
       be reduced.

    :param model: Function that takes a matrix of samples (n_samples, n_features) and
        computes the output for those samples with shape (n_samples).
    :type model: object
    :param initialization_examples: A matrix of feature vector examples (# examples x # features) for
        initializing the explainer.
    :type initialization_examples: numpy.array or pandas.DataFrame
    :param explain_subset: List of feature indices. If specified, only selects a subset of the
        features in the evaluation dataset for explanation, which will speed up the explanation
        process when number of features is large and the user already knows the set of interested
        features. The subset can be the top-k features from the model summary.
    :type explain_subset: list[int]
    :param nsamples: int (default = 2 * data.shape[1] + 2048)
        Number of times to re-evaluate the model when explaining each
        prediction. More samples lead to lower variance estimates of the SHAP
        values. The "auto" setting uses nsamples = 2 * X.shape[1] + 2048.
    :type nsamples: 'auto' or int
    :param features: A list of feature names.
    :type features: list[str]
    :param classes: Class names as a list of strings. The order of the class names should match
        that of the model output. Only required if explaining classifier.
    :type classes: list[str]
    :param nclusters: Number of means to use for approximation. A dataset is summarized with nclusters mean
        samples weighted by the number of data points they each represent. When the number of initialization
        examples is larger than (10 x nclusters), those examples will be summarized with k-means where
        k = nclusters.
    :type nclusters: int
    :param show_progress: Default to 'False'. Determines whether to display the explanation status bar
        when using shap_values from the cuML KernelExplainer.
    :type show_progress: bool
    :param transformations: sklearn.compose.ColumnTransformer or a list of tuples describing the column name and
        transformer. When transformations are provided, explanations are of the features before the transformation.
        The format for a list of transformations is same as the one here:
        https://github.com/scikit-learn-contrib/sklearn-pandas.
        If you are using a transformation that is not in the list of sklearn.preprocessing transformations that
        are supported by the `interpret-community <https://github.com/interpretml/interpret-community>`_
        package, then this parameter cannot take a list of more than one column as input for the transformation.
        You can use the following sklearn.preprocessing  transformations with a list of columns since these are
        already one to many or one to one: Binarizer, KBinsDiscretizer, KernelCenterer, LabelEncoder, MaxAbsScaler,
        MinMaxScaler, Normalizer, OneHotEncoder, OrdinalEncoder, PowerTransformer, QuantileTransformer,
        RobustScaler, StandardScaler.
        Examples for transformations that work::
            [
                (["col1", "col2"], sklearn_one_hot_encoder),
                (["col3"], None) #col3 passes as is
            ]
            [
                (["col1"], my_own_transformer),
                (["col2"], my_own_transformer),
            ]
        An example of a transformation that would raise an error since it cannot be interpreted as one to many::
            [
                (["col1", "col2"], my_own_transformer)
            ]
        The last example would not work since the interpret-community package can't determine whether
        my_own_transformer gives a many to many or one to many mapping when taking a sequence of columns.
    :type transformations: sklearn.compose.ColumnTransformer or list[tuple]
    :param allow_all_transformations: Allow many to many and many to one transformations.
    :type allow_all_transformations: bool
    :param model_task: Optional parameter to specify whether the model is a classification or regression model.
        In most cases, the type of the model can be inferred based on the shape of the output, where a classifier
        has a predict_proba method and outputs a 2 dimensional array, while a regressor has a predict method and
        outputs a 1 dimensional array.
    :type model_task: str
    """
    @init_tabular_decorator
    @init_blackbox_decorator
    def __init__(self, model, initialization_examples, explain_subset=None,
                 is_function=False, nsamples=Defaults.AUTO, features=None,
                 classes=None, nclusters=10,
                 show_progress=False, transformations=None, allow_all_transformations=False,
                 model_task=ModelTask.Unknown, **kwargs):
        """
        Initialize GPU Kernel Explainer.
        """
        if not rapids_installed:
            raise RuntimeError(
                "cuML is required to use GPU explainers. Check https://rapids.ai/start.html for more \
                information on how to install it.")
        self._datamapper = None
        if transformations is not None:
            self._datamapper, initialization_examples = get_datamapper_and_transformed_data(
                examples=initialization_examples, transformations=transformations,
                allow_all_transformations=allow_all_transformations)
        # string-index the initialization examples
        self._column_indexer = initialization_examples.string_index()
        wrapped_model, eval_ml_domain = _wrap_model(model, initialization_examples, model_task, is_function)
        super(GPUKernelExplainer, self).__init__(wrapped_model, is_function=is_function,
                                                 model_task=eval_ml_domain, **kwargs)
        self._logger.debug('Initializing GPUKernelExplainer')
        self._method = 'cuml.explainer.kernel'
        self.initialization_examples = initialization_examples
        self.features = features
        self.classes = classes
        self.nclusters = nclusters
        self.show_progress = show_progress
        self.explain_subset = explain_subset
        self.nsamples = nsamples
        self.is_function = is_function
        self.transformations = transformations
        self._allow_all_transformations = allow_all_transformations
        if self.show_progress:
            raise Warning("show_progress=True is not supported in cuML 0.19. Future release\
                           will support it.")
        self._reset_evaluation_background(self.function, **kwargs)

    def _reset_evaluation_background(self, function, **kwargs):
        """Modify the explainer to use the new evaluation example for background data.

        Note: when constructing an explainer, an evaluation example is not available and hence the initialization
        data is used.

        :param function: Function.
        :type function: Function that accepts a 2d ndarray
        """
        function, summary = self._prepare_function_and_summary(function, self.original_data_ref,
                                                               self.current_index_list,
                                                               explain_subset=self.explain_subset,
                                                               use_gpu=True,
                                                               **kwargs)
        # RAPIDS 0.18 expects an int
        if self.nsamples == 'auto':
            self.nsamples = 2 * summary.data.shape[1] + 2048
        self.explainer = cuml_Kernel_SHAP(model=function, data=summary.data, nsamples=self.nsamples)

    def _reset_wrapper(self):
        self.explainer = None
        self.current_index_list = [0]
        self.original_data_ref = [None]
        self.initialization_examples = DatasetWrapper(self.initialization_examples.original_dataset)

    @tabular_decorator
    def explain_global(self, evaluation_examples, sampling_policy=None,
                       include_local=True, batch_size=Defaults.DEFAULT_BATCH_SIZE):
        """Explain the model globally by aggregating local explanations to global.
        :param evaluation_examples: A matrix of feature vector examples (# examples x # features) on which
            to explain the model's output.
        :type evaluation_examples: numpy.array or pandas.DataFrame or scipy.sparse.csr_matrix
        :param sampling_policy: Optional policy for sampling the evaluation examples.  See documentation on
            SamplingPolicy for more information.
        :type sampling_policy: interpret_community.common.policy.SamplingPolicy
        :param include_local: Include the local explanations in the returned global explanation.
            If include_local is False, will stream the local explanations to aggregate to global.
        :type include_local: bool
        :param batch_size: If include_local is False, specifies the batch size for aggregating
            local explanations to global.
        :type batch_size: int
        :return: A model explanation object. It is guaranteed to be a GlobalExplanation which also has the properties
            of LocalExplanation and ExpectedValuesMixin. If the model is a classifier, it will have the properties of
            PerClassMixin.
        :rtype: DynamicGlobalExplanation
        """
        kwargs = _get_explain_global_kwargs(sampling_policy, ExplainType.SHAP_KERNEL, include_local, batch_size)
        kwargs[ExplainParams.INIT_DATA] = self.initialization_examples
        original_evaluation_examples = evaluation_examples.typed_dataset
        kwargs[ExplainParams.EVAL_DATA] = original_evaluation_examples
        ys_dict = self._get_ys_dict(original_evaluation_examples,
                                    transformations=self.transformations,
                                    allow_all_transformations=self._allow_all_transformations)
        kwargs.update(ys_dict)
        kwargs[ExplainParams.NUM_FEATURES] = evaluation_examples.num_features

        return self._explain_global(evaluation_examples, **kwargs)

    def _get_explain_local_kwargs(self, evaluation_examples):
        original_evaluation_examples = evaluation_examples.typed_dataset
        if self._datamapper is not None:
            evaluation_examples = transform_with_datamapper(evaluation_examples, self._datamapper)

        if self._column_indexer:
            evaluation_examples.apply_indexer(self._column_indexer)
        # Compute subset info prior
        if self.explain_subset:
            evaluation_examples.take_subset(self.explain_subset)

        # sample the evaluation examples
        if self.sampling_policy is not None and self.sampling_policy.allow_eval_sampling:
            sampling_method = self.sampling_policy.sampling_method
            max_dim_clustering = self.sampling_policy.max_dim_clustering
            evaluation_examples.sample(max_dim_clustering, sampling_method=sampling_method)
        kwargs = {ExplainParams.METHOD: ExplainType.SHAP_KERNEL}
        if self.classes is not None:
            kwargs[ExplainParams.CLASSES] = self.classes
        kwargs[ExplainParams.FEATURES] = evaluation_examples.get_features(features=self.features,
                                                                          explain_subset=self.explain_subset)
        original_evaluation = evaluation_examples.original_dataset
        kwargs[ExplainParams.NUM_FEATURES] = evaluation_examples.num_features
        evaluation_examples = evaluation_examples.dataset

        self._logger.debug('Running GPUKernelExplainer')

        if self.explain_subset:
            # Need to reset state before and after explaining a subset of data with wrapper function
            self._reset_wrapper()
            self.original_data_ref[0] = original_evaluation
            self.current_index_list.append(0)
            output_shap_values = None
            for ex_idx, example in enumerate(evaluation_examples):
                self.current_index_list[0] = ex_idx
                # Note: when subsetting with KernelExplainer, for correct results we need to
                # set the background to be the evaluation data for columns not specified in subset
                self._reset_evaluation_background(self.function, nclusters=self.nclusters)
                tmp_shap_values = self.explainer.shap_values(example)
                if output_shap_values is None:
                    output_shap_values = _convert_single_instance_to_multi(tmp_shap_values)
                else:
                    output_shap_values = _append_shap_values_instance(output_shap_values, tmp_shap_values)
            # Need to reset state before and after explaining a subset of data with wrapper function
            self._reset_wrapper()
            shap_values = output_shap_values
        else:
            shap_values = self.explainer.shap_values(evaluation_examples)

        classification = isinstance(shap_values, list)
        expected_values = None
        if hasattr(self.explainer, Attributes.EXPECTED_VALUE):
            expected_values = np.array(self.explainer.expected_value)
        local_importance_values = _convert_to_list(shap_values)
        if classification:
            kwargs[ExplainParams.MODEL_TASK] = ExplainType.CLASSIFICATION
        else:
            kwargs[ExplainParams.MODEL_TASK] = ExplainType.REGRESSION
        if self.model is not None:
            kwargs[ExplainParams.MODEL_TYPE] = str(type(self.model))
        else:
            kwargs[ExplainParams.MODEL_TYPE] = ExplainType.FUNCTION
        kwargs[ExplainParams.LOCAL_IMPORTANCE_VALUES] = np.array(local_importance_values)
        kwargs[ExplainParams.EXPECTED_VALUES] = expected_values
        kwargs[ExplainParams.CLASSIFICATION] = classification
        kwargs[ExplainParams.INIT_DATA] = self.initialization_examples
        kwargs[ExplainParams.EVAL_DATA] = original_evaluation_examples
        ys_dict = self._get_ys_dict(original_evaluation_examples,
                                    transformations=self.transformations,
                                    allow_all_transformations=self._allow_all_transformations)
        kwargs.update(ys_dict)
        return kwargs

    @tabular_decorator
    def explain_local(self, evaluation_examples):
        """Explain the function locally by using SHAP's KernelExplainer.
        :param evaluation_examples: A matrix of feature vector examples (# examples x # features) on which
            to explain the model's output.
        :type evaluation_examples: DatasetWrapper
        :return: A model explanation object. It is guaranteed to be a LocalExplanation which also has the properties
            of ExpectedValuesMixin. If the model is a classifier, it will have the properties of the ClassesMixin.
        :rtype: DynamicLocalExplanation
        """
        kwargs = self._get_explain_local_kwargs(evaluation_examples)
        explanation = _create_local_explanation(**kwargs)

        if self._datamapper is None:
            return explanation
        else:
            # if transformations have been passed, then return raw features explanation
            raw_kwargs = _get_raw_explainer_create_explanation_kwargs(kwargs=kwargs)
            return _create_raw_feats_local_explanation(explanation, feature_maps=[self._datamapper.feature_map],
                                                       features=self.features, **raw_kwargs)
