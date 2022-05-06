.. _transformations:

Raw feature transformations
===========================

Optionally, you can pass your feature transformation pipeline to the explainer to receive explanations in terms of the raw features before the transformation (rather than engineered features). If you skip this, the explainer provides explanations in terms of engineered features.


The format of supported transformations is same as the one described in `sklearn-pandas <https://github.com/scikit-learn-contrib/sklearn-pandas>`_. In general, any transformations are supported as long as they operate on a single column and are therefore clearly one to many. 

We can explain raw features by either using a `sklearn.compose.ColumnTransformer` or a list of fitted transformer tuples. The cell below uses `sklearn.compose.ColumnTransformer`. 

   .. code-block:: python

      from sklearn.compose import ColumnTransformer

      numeric_transformer = Pipeline(steps=[
          ('imputer', SimpleImputer(strategy='median')),
          ('scaler', StandardScaler())])

      categorical_transformer = Pipeline(steps=[
          ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
          ('onehot', OneHotEncoder(handle_unknown='ignore'))])

      preprocessor = ColumnTransformer(
          transformers=[
              ('num', numeric_transformer, numeric_features),
              ('cat', categorical_transformer, categorical_features)])

      # append classifier to preprocessing pipeline.
      # now we have a full prediction pipeline.
      clf = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', LogisticRegression(solver='lbfgs'))])


      # append classifier to preprocessing pipeline.
      # now we have a full prediction pipeline.
      clf = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', LogisticRegression(solver='lbfgs'))])


      # clf.steps[-1][1] returns the trained classification model
      # pass transformation as an input to create the explanation object
      # "features" and "classes" fields are optional
      tabular_explainer = TabularExplainer(clf.steps[-1][1],
                                           initialization_examples=x_train,
                                           features=dataset_feature_names,
                                           classes=dataset_classes,
                                           transformations=preprocessor)

In case you want to run the example with the list of fitted transformer tuples, use the following code: 

   .. code-block:: python

      from sklearn.pipeline import Pipeline
      from sklearn.impute import SimpleImputer
      from sklearn.preprocessing import StandardScaler, OneHotEncoder
      from sklearn.linear_model import LogisticRegression
      from sklearn_pandas import DataFrameMapper

      # assume that we have created two arrays, numerical and categorical, which holds the numerical and categorical feature names

      numeric_transformations = [([f], Pipeline(steps=[('imputer', SimpleImputer(
          strategy='median')), ('scaler', StandardScaler())])) for f in numerical]

      categorical_transformations = [([f], OneHotEncoder(
          handle_unknown='ignore', sparse=False)) for f in categorical]

      transformations = numeric_transformations + categorical_transformations

      # append model to preprocessing pipeline.
      # now we have a full prediction pipeline.
      clf = Pipeline(steps=[('preprocessor', DataFrameMapper(transformations)),
                            ('classifier', LogisticRegression(solver='lbfgs'))])

      # clf.steps[-1][1] returns the trained classification model
      # pass transformation as an input to create the explanation object
      # "features" and "classes" fields are optional
      tabular_explainer = TabularExplainer(clf.steps[-1][1],
                                           initialization_examples=x_train,
                                           features=dataset_feature_names,
                                           classes=dataset_classes,
                                           transformations=transformations)
