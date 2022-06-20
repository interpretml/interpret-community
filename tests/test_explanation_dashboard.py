# Tests for explanation dashboard
import numpy as np
import pytest
from common_utils import (create_cancer_data, create_cancer_data_booleans,
                          create_lightgbm_classifier,
                          create_sklearn_svm_classifier)
from constants import owner_email_tools_and_ux
from datasets import retrieve_dataset
from interpret import show
from interpret_community.common.constants import ModelTask
from interpret_community.mimic.models.lightgbm_model import \
    LGBMExplainableModel
from interpret_community.widget import \
    ExplanationDashboard as OldExplanationDashboard
from plotly.graph_objs import Figure
from raiwidgets import ExplanationDashboard
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@pytest.mark.owner(email=owner_email_tools_and_ux)
@pytest.mark.usefixtures('_clean_dir')
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
        ExplanationDashboard(explanation, dashboard_pipeline, dataset=x_train, true_y=y_train)

    def test_local_explanation(self, mimic_explainer):
        # Validate visualizing ExplanationDashboard with a local explanation
        x_train, x_test, y_train, y_test, feature_names, target_names = create_cancer_data()
        # Fit an SVM model
        model = create_sklearn_svm_classifier(x_train, y_train)
        explainer = mimic_explainer(model, x_train, LGBMExplainableModel,
                                    features=feature_names, classes=target_names)
        explanation = explainer.explain_local(x_test)
        ExplanationDashboard(explanation, model, dataset=x_test, true_y=y_test)

    def test_boolean_labels(self, mimic_explainer):
        # Validate visualizing ExplanationDashboard with a local explanation
        x_train, x_test, y_train, y_test, feature_names, target_names = create_cancer_data_booleans()
        # Fit an SVM model
        model = create_sklearn_svm_classifier(x_train, y_train)
        explainer = mimic_explainer(model, x_train, LGBMExplainableModel,
                                    features=feature_names, classes=target_names)
        explanation = explainer.explain_local(x_test)
        ExplanationDashboard(explanation, model, dataset=x_test, true_y=y_test)

    def test_old_explanation_dashboard(self, mimic_explainer):
        # Validate old explanation dashboard namespace works but only prints a warning
        x_train, x_test, y_train, y_test, feature_names, target_names = create_cancer_data()
        # Fit an SVM model
        model = create_sklearn_svm_classifier(x_train, y_train)
        explainer = mimic_explainer(model, x_train, LGBMExplainableModel,
                                    features=feature_names, classes=target_names)
        explanation = explainer.explain_local(x_test)
        err = ("ExplanationDashboard in interpret-community package is deprecated and removed."
               "Please use the ExplanationDashboard from raiwidgets package instead.")
        with pytest.warns(DeprecationWarning, match=err):
            OldExplanationDashboard(explanation, model, dataset=x_test, true_y=y_test)

    def test_interpret_dashboard(self, mimic_explainer):
        # Validate our explanation works with the interpret dashboard
        x_train, x_test, y_train, y_test, feature_names, target_names = create_cancer_data()
        # Fit an SVM model
        model = create_sklearn_svm_classifier(x_train, y_train)
        explainer = mimic_explainer(model, x_train, LGBMExplainableModel,
                                    features=feature_names, classes=target_names)
        explanation = explainer.explain_global(x_test)
        show(explanation)

    def test_visualize_explanation(self, mimic_explainer):
        # Validate we can call the visualize method on the explanation
        x_train, x_test, y_train, y_test, feature_names, target_names = create_cancer_data()
        # Fit an SVM model
        model = create_sklearn_svm_classifier(x_train, y_train)
        explainer = mimic_explainer(model, x_train, LGBMExplainableModel,
                                    features=feature_names, classes=target_names)
        global_explanation = explainer.explain_global(x_test)
        plot = global_explanation.visualize()
        assert isinstance(plot, Figure)
        plot = global_explanation.visualize(key=0)
        assert isinstance(plot, Figure)
        local_explanation = explainer.explain_local(x_test)
        with pytest.raises(ValueError, match="Only global explanation can be visualized with key=None."):
            local_explanation.visualize()
        plot = local_explanation.visualize(key=0)
        assert isinstance(plot, Figure)
