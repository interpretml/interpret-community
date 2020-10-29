import pytest

# Tests for explanation dashboard
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from datasets import retrieve_dataset
from interpret_community.mimic.models.lightgbm_model import LGBMExplainableModel
from interpret_community.common.constants import ModelTask
from interpret_community.widget import ExplanationDashboard
from common_utils import create_lightgbm_classifier

from constants import owner_email_tools_and_ux


@pytest.mark.owner(email=owner_email_tools_and_ux)
@pytest.mark.usefixtures('clean_dir')
class TestExplanationDashboard:
    def test_raw_timestamp_explanation(self, mimic_explainer):
        df = retrieve_dataset('insurance_claims.csv', na_values='?',
                              parse_dates=['policy_bind_date', 'incident_date'])
        label = 'fraud_reported'
        df_y = df[label]
        df_X = df.drop(columns=label)
        x_train, x_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.2, random_state=7)
        str_cols = df_X.select_dtypes(exclude=[np.number, np.datetime64]).columns.tolist()
        dt_cols = df_X.select_dtypes(include=[np.datetime64]).columns.tolist()
        numeric_cols = df_X.select_dtypes(include=[np.number]).columns.tolist()
        transforms_list = []
        for str_col in str_cols:
            transforms_list.append((str_col, Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('ohe', OneHotEncoder(sparse=False))
                ]), [str_col]
            ))
        for numeric_col in numeric_cols:
            transforms_list.append((numeric_col, Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
                ]), [numeric_col]
            ))
        for dt_col in dt_cols:
            transforms_list.append((dt_col, Pipeline(steps=[
                ('scaler', StandardScaler())
                ]), [dt_col]
            ))
        transformations = ColumnTransformer(transforms_list)
        x_train_transformed = transformations.fit_transform(x_train)
        model = create_lightgbm_classifier(x_train_transformed, y_train)
        model_task = ModelTask.Classification
        features = df_X.columns.tolist()
        explainer = mimic_explainer(model, x_train, LGBMExplainableModel, transformations=transformations,
                                    features=features, model_task=model_task)
        explanation = explainer.explain_global(x_train)
        dashboard_pipeline = Pipeline(steps=[('preprocess', transformations), ('model', model)])
        ExplanationDashboard(explanation, dashboard_pipeline, datasetX=x_train, trueY=y_train)
