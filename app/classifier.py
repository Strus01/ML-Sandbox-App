import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, SGDClassifier

from params import get_params

# get path to file
def get_path(file):
    dir = os.path.dirname(__file__)
    path = os.path.join(dir, '..', 'datasets', file)
    return path

# return data after train test split
def get_data(data_name):
    global X, y

    if data_name == 'Customer churn':
        X = pd.read_csv(get_path('customer_X.csv'))
        y = pd.read_csv(get_path('customer_y.csv'))

    elif data_name == 'Titanic':
        X = pd.read_csv(get_path('titanic_X.csv'))
        y = pd.read_csv(get_path('titanic_y.csv'))

    elif data_name == 'Diabetes':
        X = pd.read_csv(get_path('diabetes_X.csv'))
        y = pd.read_csv(get_path('diabetes_y.csv'))

    y.to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    return X_train, X_test, y_train, y_test


# return classifier object
def get_clf(clf_name):
    global clf
    params = get_params(clf_name)

    if clf_name == 'SVM':
        clf = SVC(**params)

    elif clf_name == 'KNN':
        clf = KNeighborsClassifier(**params)

    elif clf_name == 'Logistic Regression':
        clf = LogisticRegression(**params)

    elif clf_name == 'SGD':
        clf = SGDClassifier(**params)

    elif clf_name == 'XGBoost':
        clf = GradientBoostingClassifier(**params)

    return clf


# fit data and return y_preds
def fit_data(clf, X_train, y_train, X_test):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    return y_pred


# print classification report
def get_report(y_test, y_pred):
    st.write('**Classification report**')
    st.write('**accuracy**', round(accuracy_score(y_test, y_pred), 2))
    report = classification_report(y_test, y_pred, output_dict=True)
    df = pd.DataFrame(report).transpose()
    df.drop(['accuracy', 'macro avg', 'weighted avg'], axis='rows', inplace=True)
    df.support = df.support.astype(int)
    st.dataframe(df)


# plot confusion matrix
def conf_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    fig = plt.figure(figsize=(5, 3))
    sns.heatmap(cm, annot=True, fmt='d')
    st.pyplot(fig)
