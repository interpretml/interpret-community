Microsoft Interpret Extensions SDK for Python
=============================================================

This package has been tested with Python 2.7 and 3.6.
=====================================================

This is the initial extensions SDK release.

Machine learning (ML) explain model package is used to interpret black box ML models.

 * The TabularExplainer can be used to give local and global feature importances
 * The best explainer is automatically chosen for the user based on the model
 * Local feature importances are for each evaluation row
 * Global feature importances summarize the most importance features at the model-level
 * The API supports both dense (numpy or pandas) and sparse (scipy) datasets
 * For more advanced users, individual explainers can be used
 * The KernelExplainer and MimicExplainer are for BlackBox models
 * The MimicExplainer is faster but less accurate than the KernelExplainer
 * The TreeExplainer is for tree-based models
 * The DeepExplainer is for DNN tensorflow or pytorch models
