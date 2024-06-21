import streamlit as st
from sklearn.linear_model import LogisticRegression as LogisticRegressionClassifier 

from classifier.classifier import Classifier


class LogisticRegression(Classifier):
    def __init__(self) -> None:
        super().__init__()
        self.model = LogisticRegressionClassifier(**self.params)

    def _get_model_params(self):
        params = {
            'solver': 'liblinear',
            'max_iter': st.sidebar.slider('Max iter', 100, 300),
            'C': st.sidebar.slider('C', 0.01, 10.0),
            'penalty': st.sidebar.selectbox('Select penalty', ('l1', 'l2',))
        }
        return params
