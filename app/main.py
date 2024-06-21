import importlib
from typing import Tuple

from classifier.classifier import Classifier
import streamlit as st

from dataset import get_dataset


def get_data_and_classifier_names() -> Tuple[str, str]:
    data_name = st.sidebar.selectbox('Select data', ('Customer churn', 'Titanic', 'Diabetes'))
    classifier_name = st.sidebar.selectbox('Select classifier', ('SVM', 'KNN', 'Logistic Regression', 'SGD', 'Gradient Boosting'))
    return data_name, classifier_name


def get_classifier(classifier_name: str) -> Classifier:
    classifier_module = importlib.import_module(f"classifier.{classifier_name.lower().replace(' ', '_')}")
    return getattr(classifier_module, classifier_name.replace(' ', ''))()


if __name__ == '__main__':
    st.title('Machine Learning Sandbox App')

    st.write('''
    #### Manually choose hyperparameters and see how it affects your model
    ''')

    data_name, classifier_name = get_data_and_classifier_names()
    classifier = get_classifier(classifier_name)

    X_train, X_test, y_train, y_test = get_dataset(data_name)

    y_pred = classifier.fit(X_train, y_train)  
    classifier.evaluate(X_test, y_test)
    