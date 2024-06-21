import streamlit as st
from sklearn.linear_model import SGDClassifier

from classifier.classifier import Classifier


class SGD(Classifier):
    def __init__(self) -> None:
        super().__init__()
        self.model = SGDClassifier(**self.params)

    def _get_model_params(self):
        params = {
            'loss': st.sidebar.selectbox('Select loss', ('hinge', 'log', 'modified_huber', 'squared_hinge'))
        }
        return params
