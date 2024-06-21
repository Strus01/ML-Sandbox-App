from abc import ABC, abstractmethod
from typing import Dict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix


class Classifier(ABC):
    def __init__(self) -> None:
        self.params = self._get_model_params()
    
    @abstractmethod
    def _get_model_params(self) -> Dict[str, str | int | float]:
        raise NotImplementedError
    
    def fit(self, X_train: pd.DataFrame | np.ndarray, y_train: pd.DataFrame | np.ndarray) -> None:
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test: pd.DataFrame | np.ndarray, y_test: pd.DataFrame | np.ndarray) -> None:
        y_pred = self.model.predict(X_test)
        self._classification_report(y_test, y_pred)
        self._confusion_matrix(y_test, y_pred)

    def _classification_report(self, y_test: pd.DataFrame | np.ndarray, y_pred: pd.DataFrame | np.ndarray) -> None:
        st.write('**Classification report**')
        st.write('**accuracy**', round(accuracy_score(y_test, y_pred), 2))
        report = classification_report(y_test, y_pred, output_dict=True)
        df = pd.DataFrame(report).transpose()
        df.drop(['accuracy', 'macro avg', 'weighted avg'], axis='rows', inplace=True)
        df.support = df.support.astype(int)
        st.dataframe(df)
    
    def _confusion_matrix(self,  y_test: pd.DataFrame | np.ndarray, y_pred: pd.DataFrame | np.ndarray) -> None:
        cm = confusion_matrix(y_test, y_pred)
        fig = plt.figure(figsize=(5, 3))
        sns.heatmap(cm, annot=True, fmt='d')
        st.pyplot(fig)
