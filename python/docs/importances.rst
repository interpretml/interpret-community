.. _importances:

Importance Values
=================

The feature importance values are used to rank the features in the model from most to least important.

Broadly, we can think of the importance values at a global and local level.

Instance-level feature importance measures focus on the contribution of features for a specific prediction (e.g., why did the model predict an 80% chance of breast cancer for Mary?), whereas aggregate-level feature importance takes all predictions into account (Overall, what are the top important features in predicting a high risk for breast cancer?):

Some feature importance values also have useful properties.

For example, for shap values, which come from game theory, the output score of the model is the sum of the feature importance values for each feature.

The following two sections demonstrate how you can get aggregate (global) and instance-level (local) feature importance values from an interpret-community style explanation.

Overall (Global) feature importance values
------------------------------------------

Get the aggregate feature importance values.
    
   .. code-block:: python

      # you can use the training data or the test data here
      global_explanation = explainer.explain_global(x_train)

      # if you used the PFIExplainer in the previous step, use the next line of code instead
      # global_explanation = explainer.explain_global(x_train, true_labels=y_test)

      # sorted feature importance values and feature names
      sorted_global_importance_values = global_explanation.get_ranked_global_values()
      sorted_global_importance_names = global_explanation.get_ranked_global_names()


      # alternatively, you can print out a dictionary that holds the top K feature names and values
      global_explanation.get_feature_importance_dict()


Instance-level (Local) feature importance values
------------------------------------------------

Get the instance-level feature importance values: use the following function calls to explain an individual instance or a group of instances. Please note that PFIExplainer does not support instance-level explanations.

   .. code-block:: python

      # explain the first data point in the test set
      local_explanation = explainer.explain_local(x_test[0])

      # sorted feature importance values and feature names
      sorted_local_importance_names = local_explanation.get_ranked_local_names()
      sorted_local_importance_values = local_explanation.get_ranked_local_values()

or

   .. code-block:: python

      # explain the first five data points in the test set
      local_explanation = explainer.explain_local(x_test[0:5])

      # sorted feature importance values and feature names
      sorted_local_importance_names = local_explanation.get_ranked_local_names()
      sorted_local_importance_values = local_explanation.get_ranked_local_values()
