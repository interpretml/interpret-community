# Proposed interpret + interpret-community in interpret(.experimental)

All imports would be from interpret.experimental.*. Shap functions would be dropped when SHAP is fully supported except get_shap_explainer(TabularExplainer)

interpret.experimental.shap (should live in shap longer term)

``` code-block:: python
interpret.experimental.shap.get_shap_explainer(model, data, allow_kernel=False)
# Progress: TabularExplainer constructor

interpret.experimental.get_summary
# Progress: Exists in interpret-community

interpret.experimental.shap_local_explanation(shap_explainer, params)
# Progress: Exists in interpret

interpret.experimental.shap_global_explanation
# Progress: Partially exists in interpret-community
```

## SHAP Experimental Explanations while waiting for SHAP interpret explanation outputs

###  Proposed API: Use helper functions in creating SHAP explainers instead of wrapper objects

``` code-block:: python
deep_explaienr = DeepExplainer(model, interpret.experimental.shap.get_summary(..))
shap_explainer = interpret.experimental.shap.get_shap_explainer(model, initialization_data, ..., allow_kernel=False)

# Replacement for TabularExplainer

interpret_local_explanation = interpret.experimental.shap.shap_local_explanation(shap_explainer, ...)

interpret_global_explanation = interpret.experimental.shap.shap_global_explanation(shap_explainer, ...)

direct_shap_explainer = shap.LinearExplainer(...)

direct_interpret_local_explanation = interpret.experimental.shap.shap_local_explanation(direct_shap_explainer, ...)
```


### Future API:
``` code-block:: python
shap_explainer = shap.LinearExplainer(...)
explanation = shap_explainer.explain_local(...)
```


#### Existing: All explainers know about raw feature transformations. And explain_* needs to know
#### All of the below classes internally call shap_explainer.shap_values
``` code-block:: python
explanation = SHAPKernel(...).explain_local
explanation = TabularExplainer(...).explain_global
explanation = DeepExplainer(...).explain_local
explanation = TreeExplainer(...).explain_global
explanation = KernelExplainer(...).explain_global
```


## Data features for adding subsetting, sampling, and raw feature support

``` code-block:: python
interpret.experimental.data.unify_data # Progress: Exists in interpret
interpret.experimental.data.add_subsetting # Progress: Exists in interpret-community but split over many classes
interpret.experimental.data.add_sampling # Progress: Exists in interpret-community, 2-3 separate functions
interpret.experimental.data.DataMapper # Progress: Exists in interpret-community


# Simple examples
dm = DataMapper(data, transformations)
feat_data = dm.transform(data)
explanation = Explainer(predict_fn, feat_data).explain_local
raw_feat_explanation = dm.keep_raw_features(explanation)
```


### Raw Feature Explanations

#### API Overview:
``` code-block:: python
dm = DataMapper(data, transformations))
feat_data = dm.tranform(data)

explanation = explainer().explain_local(feat_data)
raw_feat_explanation = keep_raw_features(dm, explanation)  # This needs a bit of a refactor

# Existing: All explainers know about raw feature transformations. And explain_* needs to know
explanation = explainer(tranformations).explain_local(feat_data)
```



### Helpers around predictions and surrogate models
``` code-block::python
interpret.experimental.predict.unify_predict_fn # Exists in interpret, interpret-community too, serialization wrapper layer?
interpret.experimental.get_predict_fn # Exists in interpret-community

predict_fn = unify_predict(...)
predict_fn.is_classification
```

### Non experimental changes:
#### MimicExplainer:
not added as is yet -> glass_box_model.fit(x_train, predict_fn(x_train))

#### PermutationImportance # Exists in interpret-community
Uses sklearn classifier/regression paradigm

The idea is to keep interpret-community as is in the short term. Migrate to the experimental functions, and longer term more of the code will be added into modular components within interpret directly(not experimental) with little to no need for interpret-community. azureml-interpret will serve the AzureML specific use cases.
