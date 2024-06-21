import streamlit as st
from sklearn.ensemble import GradientBoostingClassifier

from classifier.classifier import Classifier


class GradientBoosting(Classifier):
    def __init__(self) -> None:
        super().__init__()
        self.model = GradientBoostingClassifier(**self.params)

    def _get_model_params(self):
        params = {
            'loss': st.sidebar.selectbox('Select loss', ('log_loss', 'exponential')),
            'learning_rate': st.sidebar.slider('Select learning rate', 0.01, 1.5),
            'criterion': st.sidebar.selectbox('Select criterion', ('friedman_mse', 'squared_error', 'mse'))
        }
        return params
