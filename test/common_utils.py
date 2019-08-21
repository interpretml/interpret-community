# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

# Defines common utilities for explanations
import numpy as np
import pandas as pd
from sklearn import svm, ensemble, linear_model
from sklearn.datasets import load_iris, load_boston
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.base import TransformerMixin
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation

import torch
import torch.nn as nn
import torch.nn.functional as F

from pandas import read_csv

from datasets import retrieve_dataset


def create_binary_newsgroups_data():
    categories = ['alt.atheism', 'soc.religion.christian']
    newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
    newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)
    class_names = ['atheism', 'christian']
    return newsgroups_train, newsgroups_test, class_names


def create_random_forest_tfidf():
    vectorizer = TfidfVectorizer(lowercase=False)
    rf = RandomForestClassifier(n_estimators=500, random_state=777)
    return Pipeline([('vectorizer', vectorizer), ('rf', rf)])


def create_random_forest_vectorizer():
    vectorizer = CountVectorizer(lowercase=False, min_df=0.0, binary=True)
    rf = RandomForestClassifier(n_estimators=500, random_state=777)
    return Pipeline([('vectorizer', vectorizer), ('rf', rf)])


def create_logistic_vectorizer():
    vectorizer = CountVectorizer(lowercase=False, min_df=0.0, binary=True)
    lr = LogisticRegression(random_state=777)
    return Pipeline([('vectorizer', vectorizer), ('lr', lr)])


def create_linear_vectorizer():
    vectorizer = CountVectorizer(lowercase=False, min_df=0.0, binary=True)
    lr = LinearRegression()
    return Pipeline([('vectorizer', vectorizer), ('lr', lr)])


def create_sklearn_random_forest_classifier(X, y):
    rfc = ensemble.RandomForestClassifier(max_depth=4, random_state=777)
    model = rfc.fit(X, y)
    return model


def create_lightgbm_classifier(X, y):
    lgbm = LGBMClassifier(boosting_type='gbdt', learning_rate=0.1,
                          max_depth=5, n_estimators=200, n_jobs=1, random_state=777)
    model = lgbm.fit(X, y)
    return model


def create_xgboost_classifier(X, y):
    xgb = XGBClassifier(learning_rate=0.1, max_depth=3, n_estimators=100,
                        n_jobs=1, random_state=777)
    model = xgb.fit(X, y)
    return model


def create_sklearn_svm_classifier(X, y, probability=True):
    clf = svm.SVC(gamma=0.001, C=100., probability=probability, random_state=777)
    model = clf.fit(X, y)
    return model


def create_pandas_only_svm_classifier(X, y, probability=True):
    class PandasOnlyEstimator(TransformerMixin):
        def fit(self, X, y=None, **fitparams):
            return self

        def transform(self, X, **transformparams):
            dataset_is_df = isinstance(X, pd.DataFrame)
            if not dataset_is_df:
                raise Exception("Dataset must be a pandas dataframe!")
            return X

    pandas_only = PandasOnlyEstimator()

    clf = svm.SVC(gamma=0.001, C=100., probability=probability, random_state=777)
    pipeline = Pipeline([('pandas_only', pandas_only), ('clf', clf)])
    return pipeline.fit(X, y)


def create_sklearn_random_forest_regressor(X, y):
    rfr = ensemble.RandomForestRegressor(max_depth=4, random_state=777)
    model = rfr.fit(X, y)
    return model


def create_sklearn_linear_regressor(X, y, pipeline=False):
    lin = linear_model.LinearRegression(normalize=True)
    if pipeline:
        lin = Pipeline([('lin', lin)])
    model = lin.fit(X, y)
    return model


def create_sklearn_logistic_regressor(X, y, pipeline=False):
    lin = linear_model.LogisticRegression()
    if pipeline:
        lin = Pipeline([('lin', lin)])
    model = lin.fit(X, y)
    return model


def create_keras_regressor(X, y):
    # create simple (dummy) Keras DNN model for regression
    batch_size = 128
    epochs = 12
    model = _common_model_generator(X.shape[1])
    model.add(Activation('linear'))
    model.compile(loss=keras.losses.mean_squared_error,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    model.fit(X, y,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(X, y))
    return model


def _common_pytorch_generator(numCols, numClasses=None):
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            # Apply layer normalization for stability and perf on wide variety of datasets
            # https://arxiv.org/pdf/1607.06450.pdf
            self.norm = nn.LayerNorm(numCols)
            self.fc1 = nn.Linear(numCols, 100)
            self.fc2 = nn.Dropout(p=0.2)
            if numClasses is None:
                self.fc3 = nn.Linear(100, 3)
                self.output = nn.Linear(3, 1)
            elif numClasses == 2:
                self.fc3 = nn.Linear(100, 2)
                self.output = nn.Sigmoid()
            else:
                self.fc3 = nn.Linear(100, numClasses)
                self.output = nn.Softmax()

        def forward(self, X):
            X = self.norm(X)
            X = F.relu(self.fc1(X))
            X = self.fc2(X)
            X = self.fc3(X)
            X = self.output(X)

            return X
    return Net()


def _train_pytorch_model(epochs, criterion, optimizer, net, torch_X, torch_y):
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = net(torch_X)
        loss = criterion(out, torch_y)
        loss.backward()
        optimizer.step()
        print('epoch: ', epoch, ' loss: ', loss.data.item())
    return net


def create_pytorch_regressor(X, y):
    # create simple (dummy) Pytorch DNN model for regression
    epochs = 12
    if isinstance(X, pd.DataFrame):
        X = X.values
    torch_X = torch.Tensor(X).float()
    torch_y = torch.Tensor(y).float()
    # Create network structure
    net = _common_pytorch_generator(X.shape[1])
    # Train the model
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.0001)
    return _train_pytorch_model(epochs, criterion, optimizer, net, torch_X, torch_y)


def create_keras_classifier(X, y):
    # create simple (dummy) Keras DNN model for binary classification
    batch_size = 128
    epochs = 12
    model = _common_model_generator(X.shape[1])
    model.add(Activation('sigmoid'))
    model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    model.fit(X, y,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(X, y))
    return model


def create_pytorch_classifier(X, y):
    # create simple (dummy) Pytorch DNN model for binary classification
    epochs = 12
    torch_X = torch.Tensor(X).float()
    torch_y = torch.Tensor(y).long()
    # Create network structure
    net = _common_pytorch_generator(X.shape[1], numClasses=2)
    # Train the model
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    return _train_pytorch_model(epochs, criterion, optimizer, net, torch_X, torch_y)


def create_keras_multiclass_classifier(X, y):
    batch_size = 128
    epochs = 12
    num_classes = len(np.unique(y))
    model = _common_model_generator(X.shape[1], num_classes)
    model.add(Dense(units=num_classes, activation=Activation('softmax')))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    y_train = keras.utils.to_categorical(y, num_classes)
    model.fit(X, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(X, y_train))
    return model


def create_pytorch_multiclass_classifier(X, y):
    # Get unique number of classes
    numClasses = np.unique(y).shape[0]
    # create simple (dummy) Pytorch DNN model for multiclass classification
    epochs = 12
    torch_X = torch.Tensor(X).float()
    torch_y = torch.Tensor(y).long()
    # Create network structure
    net = _common_pytorch_generator(X.shape[1], numClasses=numClasses)
    # Train the model
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    return _train_pytorch_model(epochs, criterion, optimizer, net, torch_X, torch_y)


def create_dnn_classifier_unfit(feature_number):
    # create simple (dummy) Keras DNN model for binary classification
    model = _common_model_generator(feature_number)
    model.add(Activation('sigmoid'))
    model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    return model


def create_iris_data():
    # Import Iris dataset
    iris = load_iris()
    # Split data into train and test
    x_train, x_test, y_train, y_validation = train_test_split(iris.data, iris.target,
                                                              test_size=0.2, random_state=0)
    feature_names = [name.replace(' (cm)', '') for name in iris.feature_names]
    return x_train, x_test, y_train, y_validation, feature_names, iris.target_names


def create_energy_data():
    # Import energy data
    energy_data = retrieve_dataset('energyefficiency2012_data.train.csv')
    # Get the Y1 column
    target = energy_data.iloc[:, len(energy_data.columns) - 2]
    energy_data = energy_data.iloc[:, :len(energy_data.columns) - 3]
    feature_names = energy_data.columns.values
    # Split data into train and test
    x_train, x_test, y_train, y_validation = train_test_split(energy_data, target,
                                                              test_size=0.2, random_state=0)
    return x_train, x_test, y_train, y_validation, feature_names


def create_boston_data():
    # Import Boston housing dataset
    boston = load_boston()
    # Split data into train and test
    x_train, x_test, y_train, y_validation = train_test_split(boston.data, boston.target,
                                                              test_size=0.2, random_state=7)
    return x_train, x_test, y_train, y_validation, boston.feature_names


def create_cancer_data():
    # Import cancer dataset
    cancer = retrieve_dataset('breast-cancer.train.csv', na_values='?').interpolate().astype('int64')
    cancer_target = cancer.iloc[:, 0]
    cancer_data = cancer.iloc[:, 1:]
    feature_names = cancer_data.columns.values
    target_names = ['no_cancer', 'cancer']
    # Split data into train and test
    x_train, x_test, y_train, y_validation = train_test_split(cancer_data, cancer_target,
                                                              test_size=0.2, random_state=0)
    return x_train, x_test, y_train, y_validation, feature_names, target_names


def create_reviews_data(test_size):
    reviews_data = retrieve_dataset('reviews.json')
    papers = reviews_data['paper']
    reviews = []
    evaluation = []
    for paper in papers:
        if paper['review'] is None or not paper['review']:
            continue
        reviews.append(paper['review'][0]['text'])
        evaluation.append(paper['review'][0]['evaluation'])
    return train_test_split(reviews, evaluation, test_size=test_size, random_state=7)


def create_simple_titanic_data():
    titanic_url = ('https://raw.githubusercontent.com/amueller/'
                   'scipy-2017-sklearn/091d371/notebooks/datasets/titanic3.csv')
    data = read_csv(titanic_url)
    # fill missing values
    data = data.fillna(method="ffill")
    data = data.fillna(method="bfill")
    numeric_features = ['age', 'fare']
    categorical_features = ['embarked', 'sex', 'pclass']

    y = data['survived'].values
    X = data[categorical_features + numeric_features]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, numeric_features, categorical_features


def create_complex_titanic_data():
    titanic_url = ('https://raw.githubusercontent.com/amueller/'
                   'scipy-2017-sklearn/091d371/notebooks/datasets/titanic3.csv')
    data = read_csv(titanic_url)
    X = data.drop('survived', axis=1)
    y = data['survived']

    return train_test_split(X, y, test_size=0.2, random_state=42)


def _common_model_generator(feature_number, output_length=1):
    model = Sequential()
    model.add(Dense(32, activation='relu', input_shape=(feature_number,)))
    model.add(Dropout(0.25))
    model.add(Dense(output_length, activation='relu', input_shape=(32,)))
    model.add(Dropout(0.5))
    return model
