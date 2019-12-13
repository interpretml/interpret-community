from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC
import pandas as pd
import numpy as np

from interpret.blackbox import ShapKernel

# get the IBM employee attrition dataset
outdirname = 'dataset.6.21.19'
try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve
import zipfile
zipfilename = outdirname + '.zip'
urlretrieve('https://publictestdatasets.blob.core.windows.net/data/' + zipfilename, zipfilename)
with zipfile.ZipFile(zipfilename, 'r') as unzip:
    unzip.extractall('.')
attritionData = pd.read_csv('./WA_Fn-UseC_-HR-Employee-Attrition.csv')

# Dropping Employee count as all values are 1 and hence attrition is independent of this feature
attritionData = attritionData.drop(['EmployeeCount'], axis=1)
# Dropping Employee Number since it is merely an identifier
attritionData = attritionData.drop(['EmployeeNumber'], axis=1)

attritionData = attritionData.drop(['Over18'], axis=1)

# Since all values are 80
attritionData = attritionData.drop(['StandardHours'], axis=1)

# Converting target variables from string to numerical values
target_map = {'Yes': 1, 'No': 0}
attritionData["Attrition_numerical"] = attritionData["Attrition"].apply(lambda x: target_map[x])
target = attritionData["Attrition_numerical"]

attritionXData = attritionData.drop(['Attrition_numerical', 'Attrition'], axis=1)

# Split data into train and test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(attritionXData,
                                                    target,
                                                    test_size = 0.2,
                                                    random_state=0,
                                                    stratify=target)
# Creating dummy columns for each categorical feature
categorical = []
for col, value in attritionXData.iteritems():
    if value.dtype == 'object':
        categorical.append(col)

# Store the numerical columns in a list numerical
numerical = attritionXData.columns.difference(categorical)
from sklearn.compose import ColumnTransformer

# We create the preprocessing pipelines for both numeric and categorical data.
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

transformations = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical),
        ('cat', categorical_transformer, categorical)])

# Append classifier to preprocessing pipeline.
# Now we have a full prediction pipeline.
clf = Pipeline(steps=[('preprocessor', transformations),
                      ('classifier', SVC(C = 1.0, probability=True, gamma='auto'))])
model = clf.fit(x_train, y_train)

# 1. Using SHAP TabularExplainer
# clf.steps[-1][1] returns the trained classification model
from interpret_community import add_preprocessing, add_postprocessing
from interpret_community._internal.raw_explain.raw_explain_utils import transform_with_datamapper

from interpret_community._internal.raw_explain.data_mapper import DataMapper
from interpret_community.explanation.explanation import _create_raw_feats_local_explanation, _get_raw_explainer_create_explanation_kwargs

data_mapper = DataMapper(transformations, allow_all_transformations=True)
def postprocessor(explanation):
    kwargs = _get_raw_explainer_create_explanation_kwargs(kwargs={"method":"SHAP"})
    import pdb;pdb.set_trace()
    return _create_raw_feats_local_explanation(explanation,
                                               feature_maps=[data_mapper.feature_map],
                                               **kwargs)
explainer = add_preprocessing(ShapKernel)(clf.steps[-1][1].predict_proba,
                                          x_train,
                                          preprocessors=[data_mapper.transform])

import pdb;pdb.set_trace()
from interpret_community.common.constants import ExplainParams, ExplainType, ModelTask
from interpret_community.explanation.explanation import _create_local_explanation


def data_to_explanation(data):
    """Create an explanation from raw dictionary data.
    :param data: the get_data() form of an interpret Explanation
    :type explanation: dict
    :return: an Explanation object
    :rtype: KernelExplantion
    """
    kwargs = {ExplainParams.METHOD: ExplainType.SHAP_KERNEL}
    local_importance_values = data['mli'][0]['value']['scores']
    expected_values = data['mli'][0]['value']['intercept']
    classification = len(local_importance_values.shape) == 3
    kwargs[ExplainParams.MODEL_TYPE] = ExplainType.FUNCTION
    if classification:
        kwargs[ExplainParams.MODEL_TASK] = ExplainType.CLASSIFICATION
    else:
        kwargs[ExplainParams.MODEL_TASK] = ExplainType.REGRESSION
    kwargs[ExplainParams.LOCAL_IMPORTANCE_VALUES] = local_importance_values
    kwargs[ExplainParams.EXPECTED_VALUES] = expected_values
    return _create_local_explanation(**kwargs)


def create_interpret_community_explanation(explanation):
    return data_to_explanation(explanation.data(-1))

explainer = add_postprocessing(ShapKernel)(clf.steps[-1][1].predict_proba,
                                           x_train,
                                           preprocessors=[data_mapper.transform],
                                           postprocessors=[create_interpret_community_explanation, postprocessor])


explanation = explainer.explain_local(x_train[:1], y_train[:1])
# classes=["Not leaving", "leaving"], 
# transformations=transformations)
