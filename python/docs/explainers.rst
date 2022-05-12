.. _explainers:

Supported Models
================

This API supports models that are trained on datasets in Python `numpy.ndarray`, `pandas.DataFrame`, or `scipy.sparse.csr_matrix` format.


The explanation functions accept both models and pipelines as input as long as the model or pipeline implements a `predict` or `predict_proba` function that conforms to the Scikit convention. If not compatible, you can wrap your model's prediction function into a wrapper function that transforms the output into the format that is supported (predict or predict_proba of Scikit), and pass that wrapper function to your selected interpretability techniques.  

If a pipeline script is provided, the explanation function assumes that the running pipeline script returns a prediction. The repository also supports models trained via **PyTorch**, **TensorFlow**, and **Keras** deep learning frameworks.


Supported Explainers
====================

The following are a list of the explainers available in the community repository:

.. list-table:: Explainers
   :widths: 15 70 15
   :header-rows: 1

   * - Interpretability Technique
     - Description
     - Type
   * - SHAP Kernel Explainer
     - `SHAP <https://github.com/slundberg/shap>`_'s Kernel explainer uses a specially weighted local linear regression to estimate SHAP values for **any model**.
     - Model-agnostic
   * - GPU SHAP Kernel Explainer
     - GPU Kernel explainer uses `cuML <https://docs.rapids.ai/api/cuml/stable/index.html>`_'s GPU accelerated version of SHAP's Kernel Explainer to estimate SHAP values for **any model**. It's main advantage is to provide acceleration to fast GPU models, like those in cuML. But it can also be used with CPU-based models, where speedups can still be achieved but they might be limited due to data transfers and speed of models themselves.
     - Model-agnostic
   * - SHAP Tree Explainer
     - `SHAP <https://github.com/slundberg/shap>`_'s Tree explainer, which focuses on the polynomial time fast SHAP value estimation algorithm specific to **trees and ensembles of trees**.
     - Model-specific
   * - SHAP Deep Explainer
     - Based on the explanation from `SHAP <https://github.com/slundberg/shap>`_, Deep Explainer "is a high-speed approximation algorithm for SHAP values in deep learning models that builds on a connection with DeepLIFT described in the `SHAP NIPS paper <https://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions>`_. **TensorFlow** models and **Keras** models using the TensorFlow backend are supported (there is also preliminary support for PyTorch)".
     - Model-specific
   * - SHAP Linear Explainer
     - `SHAP <https://github.com/slundberg/shap>`_'s Linear explainer computes SHAP values for a **linear model**, optionally accounting for inter-feature correlations.
     - Model-specific
   * - Mimic Explainer (Global Surrogate)
     - Mimic explainer is based on the idea of training `global surrogate models <https://christophm.github.io/interpretable-ml-book/global.html>`_ to mimic blackbox models. A global surrogate model is an intrinsically interpretable model that is trained to approximate the predictions of **any black box model** as accurately as possible. Data scientists can interpret the surrogate model to draw conclusions about the black box model. You can use one of the following interpretable models as your surrogate model: LightGBM (LGBMExplainableModel), Linear Regression (LinearExplainableModel), Stochastic Gradient Descent explainable model (SGDExplainableModel), and Decision Tree (DecisionTreeExplainableModel).
     - Model-agnostic
   * - Permutation Feature Importance Explainer (PFI)
     - Permutation Feature Importance is a technique used to explain classification and regression models that is inspired by `Breiman's Random Forests paper <https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf>`_ (see section 10). At a high level, the way it works is by randomly shuffling data one feature at a time for the entire dataset and calculating how much the performance metric of interest changes. The larger the change, the more important that feature is. PFI can explain the overall behavior of **any underlying model** but does not explain individual predictions.
     - Model-agnostic
   * - LIME Explainer
     - Local Interpretable Model-agnostic Explanations (LIME) is a local linear approximation of the model's behavior. The explainer wraps the `LIME tabular explainer <https://github.com/marcotcr/lime>`_ with a uniform API and additional functionality.
     - Model-agnostic


Besides the interpretability techniques described above, Interpret-Community supports another `SHAP-based explainer <https://github.com/slundberg/shap>`_, called `TabularExplainer`. Depending on the model, `TabularExplainer` uses one of the supported SHAP explainers:

.. list-table:: TabularExplainer
   :widths: 25 50
   :header-rows: 1

   * - Original Model
     - Invoked Explainer
   * - Tree-based models
     - SHAP TreeExplainer
   * - Deep Neural Network models
     - SHAP DeepExplainer
   * - Linear models
     - SHAP LinearExplainer
   * - None of the above
     - SHAP KernelExplainer or GPUKernelExplainer
