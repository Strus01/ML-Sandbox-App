# ML Sandbox App
### Machine Learning app that allows you to tune hyperparameters and see it results

---------------------------------------------------------------------------------------

## How to run it

### Heroku deployment
https://ml-sandbox-heroku.herokuapp.com/

### Locally:
1. Install packages
```shell
pip install -r requirements.txt
```
2. Run script
```shell
streamlit run app/main.py
```
---------------------------------------------------------------------------------------

### You can choose between 3 datasets:
- Customer churn
- Titanic 
- Pima Diabetes

### And 5 classifiers:
- **SVM**
  - C
  - kernel
  - gamma
  - decision function shape
- **KNN**
  - Number of neighbors
  - weight
  - algorithm
  - p
- **Logistic Regression**
  - solver
  - C
  - penalty
- **SGD**
  - loss
- **XGBoost**
  - loss
  - learning rate
  - criterion

### After manually choosing hyperparameters you will see model accuracy and summary with plotted confusion matrix
