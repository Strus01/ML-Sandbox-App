import streamlit as st
from sklearn.svm import SVC

from classifier.classifier import Classifier


class SVM(Classifier):
    def __init__(self) -> None:
        super().__init__()
        self.model = SVC(**self.params)
    
    def _get_model_params(self):
        params = {
            'C': st.sidebar.slider('C', 0.01, 10.0),
            'kernel': st.sidebar.selectbox('Select kernel', ('linear', 'rbf', 'sigmoid')),
            'gamma': st.sidebar.selectbox('Select gamma', ('scale', 'auto')),
            'decision_function_shape': st.sidebar.selectbox('Select decision function shape', ('ovo', 'ovr'))
        }
        return params
