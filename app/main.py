import streamlit as st
import classifier as clf


st.title('Machine Learning App')

st.write('''
#### Manually choose hyperparameters and see how it affects your model
''')

data_name = st.sidebar.selectbox('Select data', ('Customer churn', 'Titanic', 'Diabetes'))
clf_name = st.sidebar.selectbox('Select classifier', ('SVM', 'KNN', 'Logistic Regression', 'SGD', 'XGBoost'))

X_train, X_test, y_train, y_test = clf.get_data(data_name)

c = clf.get_clf(clf_name)

y_pred = clf.fit_data(c, X_train, y_train, X_test)

clf.get_report(y_test, y_pred)
clf.conf_matrix(y_test, y_pred)