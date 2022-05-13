.. _usage:

Use Interpret-Community
=========================


Interpretability in training
----------------------------


1. Train your model

   .. code-block:: python

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
    
      # alternatively, a cuML estimator can be trained here for GPU model
      # ensure RAPIDS is installed - refer to https://rapids.ai/ for more information
      import cuml
      from cuml.model_selection import train_test_split
      x_train, x_test, y_train, y_test = train_test_split(breast_cancer_data.data,            
                                                          breast_cancer_data.target,  
                                                          test_size=0.2,
                                                          random_state=0)
      clf = cuml.svm.SVC(gamma=0.001, C=100., probability=True)
      model = clf.fit(x_train, y_train)


2. Call the explainer: To initialize an explainer object, you need to pass your model and some training data to the explainer's constructor. You can also optionally pass in feature names and output class names (if doing classification) which will be used to make your explanations and visualizations more informative. Here is how to instantiate an explainer object using `TabularExplainer`, `MimicExplainer`, or `PFIExplainer` locally. `TabularExplainer` calls one of the five SHAP explainers underneath (`TreeExplainer`, `DeepExplainer`, `LinearExplainer`, `KernelExplainer`, or `GPUKernelExplainer`), and automatically selects the most appropriate one for your use case. You can also call any of its four underlying explainers directly.

   .. code-block:: python

      from interpret.ext.blackbox import TabularExplainer

      # "features" and "classes" fields are optional
      explainer = TabularExplainer(model,
                                   x_train,
                                   features=breast_cancer_data.feature_names,
                                   classes=classes)
      # to utilise the GPU KernelExplainer, set parameter `use_gpu=True`

    or

   .. code-block:: python

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

   or

   .. code-block:: python

      from interpret.ext.blackbox import PFIExplainer 
    
      # "features" and "classes" fields are optional
      explainer = PFIExplainer(model, 
                               features=breast_cancer_data.feature_names, 
                               classes=classes)

After instantiating an explainer object, you can call the `explain_local` and `explain_global` methods to get local and global explanations.

For information on how to compute the explanation and view the feature importance values, please see the next section on `importances`_.
