import streamlit as st
from sklearn.neighbors import KNeighborsClassifier

from classifier.classifier import Classifier


class KNN(Classifier):
    def __init__(self) -> None:
        super().__init__()
        self.model = KNeighborsClassifier(**self.params)

    def _get_model_params(self):
        params = {
            'n_neighbors': st.sidebar.slider('Number of neighbors', 1, 15),
            'weights': st.sidebar.selectbox('Select weight', ('uniform', 'distance')),
            'algorithm': st.sidebar.selectbox('Select algorithm', ('auto', 'ball_tree', 'kd_tree', 'brute')),
            'p': st.sidebar.slider('p', 1, 2)
        }
        return params
