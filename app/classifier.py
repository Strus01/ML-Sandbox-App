import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

from params import get_params


# return data after train test split
def get_data(data_name):
    if data_name == 'Customer churn':
        X = pd.read_csv(r'..\datasets\customer_X.csv')
        y = pd.read_csv(r'..\datasets\customer_y.csv')

    if data_name == 'Titanic':
        X = pd.read_csv(r'..\datasets\titanic_X.csv')
        y = pd.read_csv(r'..\datasets\titanic_y.csv')

    if data_name == 'Diabetes':
        X = pd.read_csv(r'..\datasets\diabetes_X.csv')
        y = pd.read_csv(r'..\datasets\diabetes_y.csv')

    X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    return X_train, y_train, X_test, y_test


# return classifier object
def get_clf(clf_name):
    pass


# fit data and return y_preds
def fit(clf_name):
    pass


# print classification report
def get_report(y_pred, y_test):
    pass


# plot confusion matrix
def conf_matrix(y_test, y_pred):
    pass