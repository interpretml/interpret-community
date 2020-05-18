

[![Build Status](https://dev.azure.com/responsibleai/interpret-community/_apis/build/status/Nightly?branchName=master)](https://dev.azure.com/responsibleai/interpret-community/_build/latest?definitionId=41&branchName=master)
![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)
![versions](https://img.shields.io/badge/python-3.6-blue?link=https://www.python.org/downloads/release/python-360)
![PyPI](https://img.shields.io/pypi/v/interpret-community?color=blue?link=http://pypi.org/project/interpret-community/)

Interpret Community SDK
=============================================================


Interpret-Community is an experimental repository extending [Interpret](https://github.com/interpretml/interpret), with additional interpretability techniques and utility functions to handle real-world datasets and workflows for explaining models trained on **tabular data**. This repository contains the Interpret-Community SDK and Jupyter notebooks with examples to showcase its use.

# Contents

- [Overview of Interpret-Community](#intro)
- [Getting Started](#getting-started)
- [Supported Models](#models)
- [Supported Interpretability Techniques](#explainers)
- [Use Interpret-Community](#Example)
- [Contributing](#Contributing)
- [Code of Conduct](#code)

# <a name="intro"></a>

# Overview of Interpret-Community
Interpret-Community extends the [Interpret](https://github.com/interpretml/interpret) repository and incorporates further community developed and experimental interpretability techniques and functionalities that are designed to enable interpretability for real world scenarios. Interpret-Community enables adding new experimental techniques (or functionalities) and performing comparative analysis to evaluate them.

Interpret-Community 

1. Actively incorporates innovative experimental interpretability techniques and allows for further expansion by researchers and data scientists
2. Applies optimizations to make it possible to run interpretability techniques on real-world datasets at scale
3. Provides improvements such as the capability to "reverse the feature engineering pipeline" to provide model insights in terms of the original raw features rather than engineered features
4. Provides interactive and exploratory visualizations to empower data scientists to gain meaningful insight into their data

 
# <a name="try"></a>

<a name="getting started"></a>

# Getting Started

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

[conda cheat sheet](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf)
[jupyter](https://pypi.org/project/jupyter/)
[nb_conda](https://github.com/Anaconda-Platform/nb_conda)

<details><summary><strong><em>On Linux and Windows: c. Activate conda environment</strong></em></summary>

```
    activate interp
```
</details>

<details><summary><strong><em>On Mac:</em> c. Activate conda environment</em></strong></summary>

```
    source activate interp
```
</details>

</details>

<details>

<summary><strong><em>2. Clone the Interpret-Community repository</em></strong></summary>

Clone and cd into the repository
```
git clone https://github.com/interpretml/interpret-community
cd interpret-community
```
</details>

<details>
<summary><strong><em>3. Install Python module, packages and necessary distributions</em></strong></summary>


```
pip install interpret-community
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
<details><summary><strong><em>On Linux: </strong></em></summary>
Pytorch installation if desired:

```
    pip install torch==1.3.0
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
    pip install interpret-community
```

</details>

<details>
<summary><strong><em>4. Set up and run Jupyter Notebook server </em></strong></summary>

Install and run Jupyter Notebook
```
if needed:
          pip install jupyter
then:
jupyter notebook
```
</details>
</details>

<!---{% from interpret.ext.blackbox import TabularExplainer %}
--->

# <a name="models"></a>
# Supported Models

This API supports models that are trained on datasets in Python `numpy.array`, `pandas.DataFrame`, `iml.datatypes.DenseData`, or `scipy.sparse.csr_matrix` format.


The explanation functions accept both models and pipelines as input as long as the model or pipeline implements a `predict` or `predict_proba` function that conforms to the Scikit convention. If not compatible, you can wrap your model's prediction function into a wrapper function that transforms the output into the format that is supported (predict or predict_proba of Scikit), and pass that wrapper function to your selected interpretability techniques.  

If a pipeline script is provided, the explanation function assumes that the running pipeline script returns a prediction. The repository also supports models trained via **PyTorch**, **TensorFlow**, and **Keras** deep learning frameworks.


# Supported Explainers

[//]: #  (Mehrnoosh todo: this section requires rewording )

[//]: #  (Add ref to
https://docs.microsoft.com/en-us/python/api/azureml-explain-model/azureml.explain.model?view=azure-ml-py)


The following are a list of the experimental explainers available in the community repository:
Interpretability Technique|Description|Type
|--|--|--------------------|
|SHAP Kernel Explainer| [SHAP](https://github.com/slundberg/shap)'s Kernel explainer uses a specially weighted local linear regression to estimate SHAP values for **any model**.|Model-agnostic|
|SHAP Tree Explainer| [SHAP](https://github.com/slundberg/shap)’s Tree explainer, which focuses on the polynomial time fast SHAP value estimation algorithm specific to **trees and ensembles of trees**.|Model-specific|
|SHAP Deep Explainer| Based on the explanation from [SHAP](https://github.com/slundberg/shap), Deep Explainer "is a high-speed approximation algorithm for SHAP values in deep learning models that builds on a connection with DeepLIFT described in the [SHAP NIPS paper](https://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions). **TensorFlow** models and **Keras** models using the TensorFlow backend are supported (there is also preliminary support for PyTorch)".|Model-specific|
|SHAP Linear Explainer| [SHAP](https://github.com/slundberg/shap)'s Linear explainer computes SHAP values for a **linear model**, optionally accounting for inter-feature correlations.|Model-specific|
|Mimic Explainer (Global Surrogate)| Mimic explainer is based on the idea of training [global surrogate models](https://christophm.github.io/interpretable-ml-book/global.html) to mimic blackbox models. A global surrogate model is an intrinsically interpretable model that is trained to approximate the predictions of **any black box model** as accurately as possible. Data scientists can interpret the surrogate model to draw conclusions about the black box model. You can use one of the following interpretable models as your surrogate model: LightGBM (LGBMExplainableModel), Linear Regression (LinearExplainableModel), Stochastic Gradient Descent explainable model (SGDExplainableModel), and Decision Tree (DecisionTreeExplainableModel).|Model-agnostic|
|Permutation Feature Importance Explainer (PFI)| Permutation Feature Importance is a technique used to explain classification and regression models that is inspired by [Breiman's Random Forests paper](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf) (see section 10). At a high level, the way it works is by randomly shuffling data one feature at a time for the entire dataset and calculating how much the performance metric of interest changes. The larger the change, the more important that feature is. PFI can explain the overall behavior of **any underlying model** but does not explain individual predictions. |Model-agnostic|




Besides the interpretability techniques described above, Interpret-Community supports another [SHAP-based explainer](https://github.com/slundberg/shap), called `TabularExplainer`. Depending on the model, `TabularExplainer` uses one of the supported SHAP explainers:


  | Original Model   | Invoked Explainer  |
  |-----|-----|
  | Tree-based models | SHAP TreeExplainer|
  | Deep Neural Network models| SHAP DeepExplainer|
  | Linear models | SHAP LinearExplainer |
  | None of the above  | SHAP KernelExplainer |


## Example Notebooks


- [Blackbox interpretability for binary classification](https://github.com/interpretml/interpret-community/blob/master/notebooks/explain-binary-classification-local.ipynb)
- [Blackbox interpretability for multi-class classification](https://github.com/interpretml/interpret-community/blob/master/notebooks/explain-multiclass-classification-local.ipynb)
- [Blackbox interpretability for regression](https://github.com/interpretml/interpret-community/blob/master/notebooks/explain-regression-local.ipynb)

- [Blackbox interpretability with simple raw feature transformations](https://github.com/interpretml/interpret-community/blob/master/notebooks/simple-feature-transformations-explain-local.ipynb)
- [Blackbox interpretability with advanced raw feature transformations](https://github.com/interpretml/interpret-community/blob/master/notebooks/advanced-feature-transformations-explain-local.ipynb)


<a name=Example></a>

# Use Interpret-Community


## Interpretability in training


1. Train your model


    ```python
    # load breast cancer dataset, a well-known small dataset that comes with scikit-learn
    from sklearn.datasets import load_breast_cancer
    from sklearn import svm
    from sklearn.model_selection import train_test_split
    breast_cancer_data = load_breast_cancer()
    classes = breast_cancer_data.target_names.tolist()
    
    # split data into train and test
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(breast_cancer_data.data,            
                                                        breast_cancer_data.target,  
                                                        test_size=0.2,
                                                        random_state=0)
    clf = svm.SVC(gamma=0.001, C=100., probability=True)
    model = clf.fit(x_train, y_train)
    ```

2. Call the explainer: To initialize an explainer object, you need to pass your model and some training data to the explainer's constructor. You can also optionally pass in feature names and output class names (if doing classification) which will be used to make your explanations and visualizations more informative. Here is how to instantiate an explainer object using `TabularExplainer`, `MimicExplainer`, or `PFIExplainer` locally. `TabularExplainer` calls one of the four SHAP explainers underneath (`TreeExplainer`, `DeepExplainer`, `LinearExplainer`, or `KernelExplainer`), and automatically selects the most appropriate one for your use case. You can however, call each of its four underlying explainers directly.

    ```python
    from interpret.ext.blackbox import TabularExplainer

    # "features" and "classes" fields are optional
    explainer = TabularExplainer(model, 
                                 x_train, 
                                 features=breast_cancer_data.feature_names, 
                                 classes=classes)
    ```

    or

    ```python

    from interpret.ext.blackbox import MimicExplainer
    
    # you can use one of the following four interpretable models as a global surrogate to the black box model
    
    from interpret.ext.glassbox import LGBMExplainableModel
    from interpret.ext.glassbox import LinearExplainableModel
    from interpret.ext.glassbox import SGDExplainableModel
    from interpret.ext.glassbox import DecisionTreeExplainableModel

    # "features" and "classes" fields are optional
    # augment_data is optional and if true, oversamples the initialization examples to improve surrogate model accuracy to fit original model.  Useful for high-dimensional data where the number of rows is less than the number of columns. 
    # max_num_of_augmentations is optional and defines max number of times we can increase the input data size.
    # LGBMExplainableModel can be replaced with LinearExplainableModel, SGDExplainableModel, or DecisionTreeExplainableModel
    explainer = MimicExplainer(model, 
                               x_train, 
                               LGBMExplainableModel, 
                               augment_data=True, 
                               max_num_of_augmentations=10, 
                               features=breast_cancer_data.feature_names, 
                               classes=classes)
    ```
   or

    ```python
    from interpret.ext.blackbox import PFIExplainer 
    
    # "features" and "classes" fields are optional
    explainer = PFIExplainer(model, 
                             features=breast_cancer_data.feature_names, 
                             classes=classes)
    ```



The following two sections demonstrate how you can get aggregate (global) and instance-level (local) feature importance values. Instance-level feature importance measures focus on the contribution of features for a specific prediction (e.g., why did the model predict an 80% chance of breast cancer for Mary?), whereas aggregate-level feature importance takes all predictions into account (Overall, what are the top important features in predicting a high risk for breast cancer?):
## Overall (Global) feature importance values

Get the aggregate feature importance values.
    
```python

# you can use the training data or the test data here
global_explanation = explainer.explain_global(x_train)

# if you used the PFIExplainer in the previous step, use the next line of code instead
# global_explanation = explainer.explain_global(x_train, true_labels=y_test)

# sorted feature importance values and feature names
sorted_global_importance_values = global_explanation.get_ranked_global_values()
sorted_global_importance_names = global_explanation.get_ranked_global_names()


# alternatively, you can print out a dictionary that holds the top K feature names and values
global_explanation.get_feature_importance_dict()
```

## Instance-level (local) feature importance values
Get the instance-level feature importance values: use the following function calls to explain an individual instance or a group of instances. Please note that PFIExplainer does not support instance-level explanations.

```python
# explain the first data point in the test set
local_explanation = explainer.explain_local(x_test[0])

# sorted feature importance values and feature names
sorted_local_importance_names = local_explanation.get_ranked_local_names()
sorted_local_importance_values = local_explanation.get_ranked_local_values()
```

or

```python
# explain the first five data points in the test set
local_explanation = explainer.explain_local(x_test[0:5])

# sorted feature importance values and feature names
sorted_local_importance_names = local_explanation.get_ranked_local_names()
sorted_local_importance_values = local_explanation.get_ranked_local_values()
```


## Raw feature transformations

Optionally, you can pass your feature transformation pipeline to the explainer to receive explanations in terms of the raw features before the transformation (rather than engineered features). If you skip this, the explainer provides explanations in terms of engineered features.


The format of supported transformations is same as the one described in [sklearn-pandas](https://github.com/scikit-learn-contrib/sklearn-pandas). In general, any transformations are supported as long as they operate on a single column and are therefore clearly one to many. 

We can explain raw features by either using a `sklearn.compose.ColumnTransformer` or a list of fitted transformer tuples. The cell below uses `sklearn.compose.ColumnTransformer`. 

```python
from sklearn.compose import ColumnTransformer

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# append classifier to preprocessing pipeline.
# now we have a full prediction pipeline.
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', LogisticRegression(solver='lbfgs'))])


# append classifier to preprocessing pipeline.
# now we have a full prediction pipeline.
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', LogisticRegression(solver='lbfgs'))])


# clf.steps[-1][1] returns the trained classification model
# pass transformation as an input to create the explanation object
# "features" and "classes" fields are optional
tabular_explainer = TabularExplainer(clf.steps[-1][1],
                                     initialization_examples=x_train,
                                     features=dataset_feature_names,
                                     classes=dataset_classes,
                                     transformations=preprocessor)
```

In case you want to run the example with the list of fitted transformer tuples, use the following code: 
```python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn_pandas import DataFrameMapper

# assume that we have created two arrays, numerical and categorical, which holds the numerical and categorical feature names

numeric_transformations = [([f], Pipeline(steps=[('imputer', SimpleImputer(
    strategy='median')), ('scaler', StandardScaler())])) for f in numerical]

categorical_transformations = [([f], OneHotEncoder(
    handle_unknown='ignore', sparse=False)) for f in categorical]

transformations = numeric_transformations + categorical_transformations

# append model to preprocessing pipeline.
# now we have a full prediction pipeline.
clf = Pipeline(steps=[('preprocessor', DataFrameMapper(transformations)),
                      ('classifier', LogisticRegression(solver='lbfgs'))])

# clf.steps[-1][1] returns the trained classification model
# pass transformation as an input to create the explanation object
# "features" and "classes" fields are optional
tabular_explainer = TabularExplainer(clf.steps[-1][1],
                                     initialization_examples=x_train,
                                     features=dataset_feature_names,
                                     classes=dataset_classes,
                                     transformations=transformations)
```




## Visualizations

Load the visualization dashboard in your noteboook to understand and interpret your model:

### Global visualizations

The following plots provide a global view of the trained model along with its predictions and explanations.

|Plot|Description|
|----|-----------|
|Data Exploration| Displays an overview of the dataset along with prediction values.|
|Global Importance|Aggregates feature importance values of individual datapoints to show the model's overall top K (configurable K) important features. Helps understanding of underlying model's overall behavior.|
|Explanation Exploration|Demonstrates how a feature affects a change in model’s prediction values, or the probability of prediction values. Shows impact of feature interactions.|
|Summary Importance|Uses indiviual feature importance values across all data points to show the distribution of each feature's impact on the prediction value. Using this diagram, you investigate in what direction the feature values affects the prediction values.
|


![Visualization Dashboard Global](https://docs.microsoft.com/en-us/azure/machine-learning/service/media/machine-learning-interpretability-explainability/global-charts.png)



### Local visualizations

You can click on any individual data point at any time of the preceding plots to load the local feature importance plot for the given data point.

|Plot|Description|
|----|-----------|
|Local Importance|Shows the top K (configurable K) important features for an individual prediction. Helps illustrate the local behavior of the underlying model on a specific data point.|
|Perturbation Exploration (what if analysis)|Allows changes to feature values of the selected data point and observe resulting changes to prediction value.|
|Individual Conditional Expectation (ICE)| Allows feature value changes from a minimum value to a maximum value. Helps illustrate how the data point's prediction changes when a feature changes.|


![Visualization Dashboard Local Feature Importance](https://docs.microsoft.com/en-us/azure/machine-learning/service/media/machine-learning-interpretability-explainability/local-charts.png)

![Visualization Dashboard Feature Perturbation](https://docs.microsoft.com/en-us/azure/machine-learning/service/media/machine-learning-interpretability-explainability/perturbation.gif)

![Visualization Dashboard ICE Plots](https://docs.microsoft.com/en-us/azure/machine-learning/service/media/machine-learning-interpretability-explainability/ice-plot.png)


To load the visualization dashboard, use the following code:

```python
from interpret_community.widget import ExplanationDashboard

ExplanationDashboard(global_explanation, model, datasetX=x_test)
```


<a name=Contributing></a>

# Contributing
[//]: #  (Vincent: is CLA required when we go public? )

This project welcomes contributions and suggestions.  Most contributions require you to agree to the Github Developer Certificate of Origin, DCO.
For details, please visit https://probot.github.io/apps/dco/.

The Developer Certificate of Origin (DCO) is a lightweight way for contributors to certify that they wrote or otherwise have the right to submit the code they are contributing to the project. Here is the full text of the DCO, reformatted for readability:
```
By making a contribution to this project, I certify that:
(a) The contribution was created in whole or in part by me and I have the right to submit it under the open source license indicated in the file; or
(b) The contribution is based upon previous work that, to the best of my knowledge, is covered under an appropriate open source license and I have the right under that license to submit that work with modifications, whether created in whole or in part by me, under the same open source license (unless I am permitted to submit under a different license), as indicated in the file; or
(c) The contribution was provided directly to me by some other person who certified (a), (b) or (c) and I have not modified it.
(d) I understand and agree that this project and the contribution are public and that a record of the contribution (including all personal information I submit with it, including my sign-off) is maintained indefinitely and may be redistributed consistent with this project or the open source license(s) involved.
Contributors sign-off that they adhere to these requirements by adding a Signed-off-by line to commit messages.
This is my commit message

Signed-off-by: Random J Developer <random@developer.example.org>
Git even has a -s command line option to append this automatically to your commit message:
$ git commit -s -m 'This is my commit message'
```

When you submit a pull request, a DCO bot will automatically determine whether you need to certify.
Simply follow the instructions provided by the bot. 

<a name=Code></a>

# Code of Conduct

This project has adopted the his project has adopted the [GitHub Community Guidelines](https://help.github.com/en/github/site-policy/github-community-guidelines).

## Reporting Security Issues

[//]: # ( Vincent: can we delete this section? )

Security issues and bugs should be reported privately, via email, to the Microsoft Security
Response Center (MSRC) at [secure@microsoft.com](mailto:secure@microsoft.com). You should
receive a response within 24 hours. If for some reason you do not, please follow up via
email to ensure we received your original message. Further information, including the
[MSRC PGP](https://technet.microsoft.com/en-us/security/dn606155) key, can be found in
the [Security TechCenter](https://technet.microsoft.com/en-us/security/default).

