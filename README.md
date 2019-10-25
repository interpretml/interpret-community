

[![Build Status](https://dev.azure.com/responsibleai/interpret-extensions/_apis/build/status/interpretml.interpret-community?branchName=master)](https://dev.azure.com/responsibleai/interpret-extensions/_build/latest?definitionId=5&branchName=master)
![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)
![versions](https://img.shields.io/badge/python-2.7%20%7C%203.6-blue)

Interpret Community SDK
=============================================================


The Interpret Community builds on [Interpret](https://github.com/Microsoft/interpret), an open source python package from Microsoft Research for training interpretable models and helping to explain blackbox systems, by adding additional extensions from the community to interpret ML models.

This repository contains an SDK and Jupyter notebooks with examples to showcase its use.

# Contents

- [Overview of Interpret-Community](#intro)
- [Interpret vs. Interpret-Community](#comparison)
- [Target Audience](#target)
- [Try our notebooks in your favorite cloud](#try)
- [Getting Started](#getting-started)
- [Models](#models)
- [Example](#Example)
- [Contributing](#Contributing)
- [Code of Conduct](#code)
- [Build Status](#BuildStatus)
- [Additional References](#Refs)

# <a name="intro"></a>

# Overview of Interpret-Community
Interpret-Community is an experimental repository that hosts a wide range of community developed machine learning interpretability techniques. This repository makes it easy for anyone involved in the development of a machine learning system to improve transparency around their machine learning models.


This repository incorporates community developed interpretability techniques under one roof with a unified set of data structures and visualization. Users could experiment with different interpretability techniques, and/or add their custom-made interpretability techniques and more easily perform comparative analysis to evaluate their brand new explainers. Using these tools, one can explain machine learning models globally on all data, or locally on a specific data point using the state-of-art technologies in an easy-to-use and scalable fashion. In particular, this released open source toolkit:
1. Actively incorporates innovative interpretability techniques, and allows for further expansion by researchers and data scientists
2. Creates a common API across the integrated libraries
3. Applies optimizations to make it possible to run on real-world datasets at scale
4. Provides improvements such as the capability to "reverse the feature engineering pipeline" to provide users with feature importance values and model interpretability insights in terms of the original raw features rather than engineered features
5. Provides interactive and exploratory visualization to empower data scientists to gain significant insight into their data
# <a name="comparison"></a>

# Interpret vs. Interpret-Community


Interpret-Community and its peer repository, Interpret, both serve as a tool for researchers, machine learning engineers, software developers, data scientists, and business executives to get insights on machine learning models. The peer repository, Interpret, hosts a core set of interpretability techniques from the research community. Interpret-Community extends Interpret with additional interpretability techniques and additional utility functions to handle real-world datasets and workflows.



  # <a name="target"></a>

 # Target Audience
1. Machine Learning Interpretability Researchers: Interpret's extension hooks make it easy to extend and thus, interpretability researchers who are interested in adding their own techniques, can easily add them to the community repository and compare it to state-of-the-art and proven interpretability techniques and/or other community techniques.

2. Developers/Data Scientists: Having all of the interpretability techniques in one place makes it easy for data scientists to experiment with different interpretability techniques, and explain their model in a scalable and seamless manner. The set of rich interactive visualizations allow developers and data scientists to train and deploy more transparent machine learning models instead of wasting time and effort on generating customized visualizations, addressing scalability issues by optimizing third-party interpretability techniques, and adopting/operationalizing interpretability techniques.
3. Business Executives: The core logic and visualizations are beneficial for raising awareness among those involved in developing AI applications, allow them to audit model predictions for potential bias and use this insight to help establish stronger governance capabilities., and establish a strong governance framework around the use of AI applications.


# <a name="try"></a>

# Try our notebooks in your favorite cloud

[![Azure Notebooks](https://notebooks.azure.com/launch.png)](https://notebooks.azure.com/import/gh/microsoft/interpret-community)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/interpretml/interpret-community)

<a name="getting started"></a>

## Getting Started

This repository uses Anaconda to simplify package and environment management.

To setup on your local machine:

<details><summary><strong><em>1. Set up Environment</em></strong></summary>

    a. Install Anaconda with Python >= 3.6
       [Miniconda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) is a quick way to get started.


    b. Create conda environment named interp and install packages

```
    conda create --name interp python=3.6 anaconda

```

    Optional, additional reading:

    - [conda cheat sheet](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf)
    - [jupyter](https://pypi.org/project/jupyter/)
    - [nb_conda](https://github.com/Anaconda-Platform/nb_conda)

<details><summary><strong><em>On Windows: c. Activate conda environment</strong></em></summary>

```
    activate interp
```
</details>

<details><summary><strong><em>On Linux and Mac:</em> c. Activate conda environment</em></strong></summary>

```
    source activate interp
```
</details>
<br></br>
</details>

<details>

<summary><strong><em>2. Clone the Interpret-Community repository</em></strong></summary>

Clone and cd into the repository
```
git clone https://github.com/Microsoft/Interpret-community
cd interpret-community
```
</details>

<details>
<summary><strong><em>3. Install Python module, packages and necessary distributions</em></strong></summary>


```
pip install -e ./python
```
If you intend to run repository tests:
```
pip install -r requirements.txt
```


<details><summary><strong><em>On Windows: </strong></em></summary>

Pytorch installation if desired:
```
    pip install https://download.pytorch.org/whl/cpu/torch-1.3.0%2Bcpu-cp36-cp36m-win_amd64.whl
    pip install torchvision==0.4.1
```

lightgbm installation if desired:
```
    conda install --yes -c conda-forge lightgbm
```

</details>


<details><summary><strong><em>On MacOS: </strong></em></summary>

Pytorch installation if desired:
```
    pip install torch==1.3.0
    pip install torchvision==0.4.1
```

lightgbm installation if desired (requires Homebrew):
```
    brew install libomp
    conda install --yes -c conda-forge lightgbm
```

If installing the package generally gives an error about the `certifi` package, run this first:
```
    pip install --upgrade certifi --ignore-installed
```

</details>

<details>
<summary><strong><em>4. Set up and run Jupyter Notebook server </em></strong></summary>

Install and run Jupyter Notebook
```
if needed:
          pip install jupyter
          conda install nb_conda
then:
jupyter notebook
```
</details>

<!---{% from interpret.ext.blackbox import TabularExplainer %}
--->

# <a name="models"></a>

# Models

[//]: #  (Mehrnoosh todo: this section requires rewording )

[//]: #  (Add ref to
https://docs.microsoft.com/en-us/python/api/azureml-explain-model/azureml.explain.model?view=azure-ml-py)



Any models that are trained on datasets in Python numpy.array, pandas.DataFrame, iml.datatypes.DenseData, or scipy.sparse.csr_matrix format are supported by this API.
The explanation functions accept both models and pipelines as input. If a model is provided, the model must implement the prediction function predict or predict_proba that conforms to the Scikit convention.  If a pipeline script is provided, the explanation function assumes that the running pipeline script returns a prediction. The repository also supports models trained via PyTorch, TensorFlow, and Keras deep learning frameworks. The following are a list of the experimental explainers available in the community repository:




* [SHAP](https://github.com/slundberg/shap) Tree Explainer: SHAP’s tree explainer, which focuses on polynomial time fast SHAP value estimation algorithm specific to trees and ensembles of trees.
* [SHAP](https://github.com/slundberg/shap) Deep Explainer: Based on the explanation from SHAP, Deep Explainer "is a high-speed approximation algorithm for SHAP values in deep learning models that builds on a connection with DeepLIFT described in the SHAP NIPS paper. TensorFlow models and Keras models using the TensorFlow backend are supported (there is also preliminary support for PyTorch)".
* [SHAP](https://github.com/slundberg/shap) Kernel Explainer: SHAP's Kernel explainer uses a specially weighted local linear regression to estimate SHAP values for any model.
* [SHAP](https://github.com/slundberg/shap): SHAP's Linear Explainer computes SHAP values for a linear model, optionally accounting for inter-feature correlations.

* Mimic Explainer: Mimic explainer is based on the idea of [global surrogate models](https://christophm.github.io/interpretable-ml-book/global.html)'s. A global surrogate model is an intrinsically interpretable model that is trained to approximate the predictions of a black box model as accurately as possible. Data scientist can interpret the surrogate model to draw conclusions about the black box model. This repository supporots the following interpretable models as surrogate model: LightGBM (LGBMExplainableModel), Linear/Logistic Regression (LinearExplainableModel), Stochastic Gradient Descent explainable model (SGDExplainableModel), and Decision Tree (DecisionTreeExplainableModel).
* Permutation Feature Importance Explainer: Permutation Feature Importance is a technique used to explain classification and regression models that is inspired by [Breiman's Random Forests paper](https://www.stat.berkeley.edu/%7Ebreiman/randomforest2001.pdf) (section 10). At a high level, it works by randomly shuffling data one feature at a time for the entire dataset and calculating how much the performance metric of interest decreases. The larger the change, the more important that feature is.
* LIME Explainer: LIME Explainer uses the state-of-the-art Local interpretable model-agnostic explanations [(LIME)](https://github.com/marcotcr/lime) algorithm to create local surrogate models. Unlike the global surrogate models, LIME focuses on training local surrogate models to explain individual predictions.

Tabular Explainer: Used with tabular datasets, it currently employs the following logic to invoke the direct SHAP explainers:


| Original Model   | Invoked Explainer  |
|-----|-----|
| Tree-based models | SHAP TreeExplainer|
| Deep Neural Network models| SHAP DeepExplainer|
| Linear models | SHAP LinearExplainer |
| None of the above  | SHAP KernelExplainer |

<a name=Example></a>

# Example

<a name=Contributing></a>

# Contributing
[//]: #  (Vincent: is CLA required when we go public? )

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repositories using our CLA.

<a name=Code></a>

# Code of Conduct

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Reporting Security Issues

[//]: # ( Vincent: can we delete this section? )

Security issues and bugs should be reported privately, via email, to the Microsoft Security
Response Center (MSRC) at [secure@microsoft.com](mailto:secure@microsoft.com). You should
receive a response within 24 hours. If for some reason you do not, please follow up via
email to ensure we received your original message. Further information, including the
[MSRC PGP](https://technet.microsoft.com/en-us/security/dn606155) key, can be found in
the [Security TechCenter](https://technet.microsoft.com/en-us/security/default).

<a name=BuildStatus></a>

# Build Status

[![Build Status](https://dev.azure.com/responsibleai/interpret-extensions/_apis/build/status/microsoft.interpret-community?branchName=master)](https://dev.azure.com/responsibleai/interpret-extensions/_build/latest?definitionId=5&branchName=master)

<a name=Refs></a>

# Additional References
