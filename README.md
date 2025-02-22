**Predicting 30-Day Survival After In-Hospital Cardiac Arrest: A Nationwide Cohort Study Using Machine Learning and SHAP Analysis**
This repository contains the code for the machine learning models and feature selection analysis used in the study titled "Predicting 30-Day Survival After In-Hospital Cardiac Arrest: A Nationwide Cohort Study Using Machine Learning and SHAP Analysis". The goal of the study is to predict 30-day survival for patients who experience in-hospital cardiac arrest (IHCA) using machine learning models, including CatBoost, XGBoost, LightGBM, and Logistic Regression.

The repository contains multiple Python programs which are as follows:

**Code Files:**
**Program_with_all_features_catboost.py:**
This program uses the CatBoost machine learning model to predict 30-day survival. The model is evaluated with 5-fold cross-validation and stratified sampling. It includes evaluations for different data imputation strategies: without imputation, with mean imputation, and with the Multiple Imputation by Chained Equations (MICE) strategy.

**Program_with_all_features_xgboost.py:**
This program utilizes the XGBoost model for predicting 30-day survival. Similar to the CatBoost model, it evaluates the performance using 5-fold cross-validation and stratified sampling with different imputation strategies (no imputation, mean imputation, and MICE).

**Program_with_all_features_lightgbm.py:**
This program applies the LightGBM algorithm to predict 30-day survival. It also uses 5-fold cross-validation with stratified sampling and tests the three imputation strategies: without imputation, with mean imputation, and with MICE.

**Program_with_all_features_logistics_regression.py:**
This program implements a Logistic Regression model for 30-day survival prediction, evaluated in the same way as the previous models with 5-fold cross-validation, stratified sampling, and the three imputation strategies.

**Feature_selection_Evaltution_with_15Features.py:**
This program performs feature selection and evaluation based on the top 15 features. The features are ranked incrementally using a 5-fold cross-validation approach with stratified sampling. The goal is to identify the most important features for predicting survival.

**Libraries Used:**
CatBoost
XGBoost
LightGBM
Logistic Regression
Pandas
Numpy
Scikit-learn
SHAP (for model explainability)
Imbalanced-learn (for dealing with class imbalance, if applicable)

**Evaluation:**
All the models are evaluated using 5-fold cross-validation, and the performance is assessed with various strategies for imputing missing data. Evaluation metrics include accuracy, precision, recall, F1-score, and ROC AUC.
