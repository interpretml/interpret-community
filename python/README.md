# Microsoft Interpret Community SDK for Python

### This package has been tested with Python 2.7 and 3.6

The Interpret Community builds on Interpret, an open source python package from Microsoft Research for training interpretable models, and helps to explain blackbox systems by adding additional extensions from the community to interpret ML models.

Interpret-Community is an experimental repository that hosts a wide range of community developed machine learning interpretability techniques. This repository makes it easy for anyone involved in the development of a machine learning system to improve transparency around their machine learning models. Data scientists, machine learning engineers, and researchers can easily add their own interpretability techniques via the set of extension hooks built into the peer repository, Interpret, and expand this repository to include their custom-made interpretability techniques.

Highlights of the package include:

- The TabularExplainer can be used to give local and global feature importances
- The best explainer is automatically chosen for the user based on the model
- Local feature importances are for each evaluation row
- Global feature importances summarize the most importance features at the model-level
- The API supports both dense (numpy or pandas) and sparse (scipy) datasets
- For more advanced users, individual explainers can be used
- The KernelExplainer, PFIExplainer and MimicExplainer are for BlackBox models
- The MimicExplainer is faster but less accurate than the KernelExplainer
- The TreeExplainer is for tree-based models
- The DeepExplainer is for DNN tensorflow or pytorch models

Please see the github website for the documentation and sample notebooks:
https://github.com/interpretml/interpret-community
