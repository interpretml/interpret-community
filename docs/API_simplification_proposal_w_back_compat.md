

TabularExplainer(model,
                 initialization_examples,
                 explain_subset=None,
                 features=None, classes=None,
                 transformations=None,
                 allow_all_transformations=False,
                 model_task=ModelTask.Unknown,
                 **kwargs)  #  internals for logging

LinearExplainableModel(multiclass=False,
                       random_state=DEFAULT_RANDOM_STATE,
                       classification=True,
                       sparse_data=False)

LinearExplainer(model, data, feature_dependence=FEATURE_DEPENDENCE)

SGDExplainableModel(model, data, feature_dependence=FEATURE_DEPENDENCE)



linear_model = interpret.experimental.get_linear_model(multiclass=False,
                                                       random_state=DEFAULT_RANDOM_STATE,
                                                       classification=True,
                                                       sparse_data=False)
linear_explainer = interpret.experimental.get_explainer(linear_model, allow_kernel=False)
ExplainableModel(linear_model, linear_explainer)


sgd_model = interpret.experimental.get_sgd_model(multiclass=False,
                                                 random_state=DEFAULT_RANDOM_STATE,
                                                 classification=True,
                                                 sparse_data=False)
sgd_explainer = interpret.experimental.get_explainer(sgd_model)
ExplainableModel(sgd_model, sgd_explainer)
exp_model = ExplainableModel(sgd_model, get_scoring_explainer(sgd_model.explainer))  # ?

interpret.mlflow.log_model("outputs/model/", exp_model)  # proposal for mlflow.interpret, mlflow showed interest in showcasing this sort of extension

TabularExplainer # -> get_explainer function + construct without initialize + calls fit before explaining

explainer = interpret.experimental.get_explainer(model)
explainable_model = ExplainableModel(model, explainer, feature_names=None, class_names=None)
# What is model task for?
explainable_model.fit(x_train, y)
# calls explainer.initialize(x_train)
explainable_model.explain_global = explainer.explain_global
explainable_model.explain_local = explainer.explain_local

# Transofrmations + subset?
# interpret.experimental.preprocess
x_train = preprocess.subset(x_train, *args, **kwargs)  # explanations after would be subsetted
x_train = preprocess.remove_subset(x_train)  # explanations after would not be subsetted
# if not an object, 
x_train = preprocess.sampler(x_train, *args, **kwargs)
x_train = preprocess.remove_sampler(x_train)  # explanations after would not be subsetted

unify_data

PFIExplainer(model,
             is_function=False,
             metric=None,
             metric_args=None,
             is_error_metric=False,
             explain_subset=None,
             features=None,
             classes=None,
             transformations=None,
             allow_all_transformations=False,
             seed=0,
             for_classifier_use_predict_proba=False,
             show_progress=True,
             model_task=ModelTask.Unknown)

def get_predict_function(model, dataset):  # We already implement this internally
    if model is not None:
        wrapped_model, _ = _wrap_model(model, evaluation_examples, model_task, False)
        if self.for_classifier_use_predict_proba:
            def model_predict_proba_func(dataset):
                return wrapped_model.predict_proba(typed_wrapper_func(dataset))
            return model_predict_proba_func
        else:
            def model_predict_func(dataset):
                return wrapped_model.predict(typed_wrapper_func(dataset))
            return model_predict_func
    else:
        wrapped_function, _ = _wrap_model(self.function, evaluation_examples, model_task, True) # check if function is on class or not

        def user_defined_or_default_predict_func(dataset):
            return wrapped_function(typed_wrapper_func(dataset))
        return user_defined_or_default_predict_func

interpret.experimental.get_pfi_explainer(model,
                                         is_function=False,
                                         metric=None,
                                         metric_args=None,
                                         is_error_metric=False,
                                         explain_subset=None,
                                         features=None,
                                         classes=None,
                                         transformations=None,
                                         allow_all_transformations=False,
                                         seed=0,
                                         for_classifier_use_predict_proba=False,
                                         show_progress=True,
                                         model_task=ModelTask.Unknown)

x_train = preprocess.add_feature_names(x_train, feature_names)  # stores in feature_names[x_train] = feature_names, other features can do the same
x_train = preprocess.add_class_names(x_train, class_names)
def get_pfi_explainer(*args):
    predict_fn = get_predict_fn(model) if not is_function else model
    metric_obj = PFIMetric(metric, metric_args)
    if predict_fn.is_classification:
        return PFIExplainerC(predict_fn or predict_proba, metric, show_progress=None) # SKlearn convention
    else:
        return PFIExplainerR(predict_fn, metric, show_progress=None)

x_train = explain_subset(data, indices)
pfi_explainer = get_pfi_explainer(x_train, *args)
# Allow all transformations? Seed? Model task?  All can be kept, just need to understand what they are
# for metric is a name or a class instance, constructed from name, has args in it



# Sparse support Add it to unify_data in interpret, call on any input data within the system
# Can we join the 8 references into something like unify data?
# https://github.com/interpretml/interpret-community/search?q=%22sp.sparse.issparse%22&unscoped_q=%22sp.sparse.issparse%22


# Tabular explainer init -> get_explainer, glassbox models return self.explainer
# https://github.com/interpretml/interpret-community/blob/b5f8a401f2c39afbb944d463fb6684094f1c6ab4/python/interpret_community/tabular_explainer.p



x_train = explain_augment(x_train)
x_train = explain_set_features(x_train, feature_names)
x_train = explain_set_classes(x_train, class_names)
x_train = remove_explain_augment(x_train)

explainer = explain_raw(explainer, transformations)  # adds pre and post processing for raw explanations
explainer = remove_explain_raw(explainer)

def get_mimic_explainer(*args):
    predict_fn = get_predict_fn(model, initialization_examples) if not is_function else model
    x_train = explain_augment(x_train, max_augs=10)
    x_train = explain_subset(x_train, indices)
    x_train = explain_subset(x_train, indices)
    if predict_fn.is_classification:
        return MimicExplainerC(predict_fn, mimic_model)
    else:
        return MimicExplainerR(predict_fn, mimic_model)

interpret.experimental.get_mimic_explainer(model, initialization_examples, explainable_model, explainable_model_args=None,
                                           is_function=False, augment_data=True, max_num_of_augmentations=10, explain_subset=None,
                                           features=None, classes=None, transformations=None, allow_all_transformations=False,
                                           shap_values_output=ShapValuesOutput.DEFAULT, categorical_features=None,
                                           model_task=ModelTask.Unknown, reset_index=ResetIndex.Ignore)



"""
Open questions:
- What is allow_all_transformations?
- What do we pass the seed to when a param?
- Is shap_values_output a configuration for explanation aggregation or does it affect computed info internally? Is it post processing?
- categorical_features, classes, features, are these for the UX? Should they be
  added to the dataset?
"""
