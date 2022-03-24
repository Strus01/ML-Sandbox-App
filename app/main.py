import streamlit as st
import classifier as clf


st.title('Machine Learning App')

st.write('''
**Manually chose hyperparameters and see how it affects your model**
''')

data_name = st.sidebar.selectbox('Select data', ('Customer churn', 'Titanic', 'Diabetes'))
clf_name = st.sidebar.selectbox('Select classifier', ('SVM', 'KNN'))


