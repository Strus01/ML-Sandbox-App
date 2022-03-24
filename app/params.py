import streamlit as st


def get_params(clf_name):
    params = dict()
    if clf_name == 'SVM':
        C = st.sidebar.slider('C', 0.01, 10.0)
        kernel = st.sidebar.selectbox('Select kernel', ('linear', 'rbf', 'sigmoid'))
        gamma = st.sidebar.selectbox('Select gamma', ('scale', 'auto'))
        decision_function = st.sidebar.selectbox('Select decision function', ('ovo', 'ovr'))
        params['C'] = C
        params['kernel'] = kernel
        params['gamma'] = gamma
        params['decision_function'] = decision_function

    elif clf_name == 'KNN':
        K = st.sidebar.slider('K', 1, 15)
        weights = st.sidebar.selectbox('Select weight', ('uniform', 'distance'))
        algorithms = st.sidebar.selectbox('Select algorithm', ('auto', 'ball_tree', 'kd_tree', 'brute'))
        p = st.sidebar.slider('p', 1, 2)
        params['K'] = K
        params['weights'] = weights
        params['algorithms'] = algorithms
        params['p'] = p

    return params
