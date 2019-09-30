

[![Build Status](https://dev.azure.com/responsibleai/interpret-extensions/_apis/build/status/microsoft.interpret-community?branchName=master)](https://dev.azure.com/responsibleai/interpret-extensions/_build/latest?definitionId=5&branchName=master)
![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)
![versions](https://img.shields.io/badge/python-2.7%20%7C%203.6-blue)

Interpret Community Extensions SDK
=============================================================


The Interpret Community Extensions builds on [InterpretML](https://github.com/Microsoft/interpret), an open source python package from Microsoft Research for training interpretable models and helping to explain blackbox systems, by adding additional extensions from the community to interpret ML models.

This repository contains an SDK and Jupyter notebooks with examples to showcase its use.

# Contents

- [Try our notebooks in your favorite cloud](#try)
- [Getting Started](#gettingstarted) 
- [Models](#models)
- [Example](#Example)
- [Contributing](#Contributing)
- [Code of Conduct](#code)
- [Build Status](#BuildStatus)
- [Additional References](#Refs)

# <a name="try"></a> 

# Try our notebooks in your favorite cloud

[![Azure Notebooks](https://notebooks.azure.com/launch.png)](https://notebooks.azure.com/import/gh/microsoft/interpret-community)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/interpretml/interpret-community)

<a name="getting started"></a>

## Getting Started

This repo uses Anaconda to simplify package and environment management.

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

<details><summary><strong><em>On Linux:</em> c. Activate conda environment</em></strong></summary>

```
    source activate interp
```
</details>
<br></br>
</details>
 
<details>

<summary><strong><em>2. Clone the interpret-community repository</em></strong></summary>

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
If you intend to run repo tests:
```
pip install -r requirements.txt
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

* The API supports both dense (numpy or pandas) and sparse (scipy) datasets

* For more advanced users, individual explainers can be used

* The TabularExplainer provides local and global feature importances  
    *  The best explainer is automatically chosen for the user based on the model
        - Best implies fastest execution time and highest interpretabilty accuracy.
* Local feature importances are for each evaluation row
* Global feature importances summarize the most importance features at the model-level
 * The KernelExplainer and MimicExplainer are for BlackBox models
 * The MimicExplainer is faster but less accurate than the KernelExplainer
 * The TreeExplainer is for tree-based models
 * The DeepExplainer is for DNN tensorflow or pytorch models
[shap](https://github.com/slundberg/shap) and [lime](https://github.com/marcotcr/lime) have docs


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
provided by the bot. You will only need to do this once across all repos using our CLA.

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
