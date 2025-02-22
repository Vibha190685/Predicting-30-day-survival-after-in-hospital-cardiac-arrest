## Importing Library
import sys
print(sys.version)
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
import xgboost as xgb
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, accuracy_score, auc, roc_auc_score, precision_recall_curve, confusion_matrix, make_scorer, recall_score, roc_curve, f1_score, log_loss, classification_report
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
import multiprocessing
import optuna
import optuna.visualization
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from itables import init_notebook_mode
init_notebook_mode(all_interactive=True)


##Data Preprocessing and laoding
df=pd.read_csv('xyz.csv', index_col=0)
num_columns = df.shape[1]  # The number of columns is the second element of the shape tuple
print("Number of columns:", num_columns)
target_variable='O_death30'




###Five fold cross validation for model evalution with all features: with mean imputation
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import pickle

# Define features and target
X = df.drop(columns=['O_death30'])
y = df['O_death30']

# Identify categorical and numerical columns
categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_columns = X.select_dtypes(include=[np.number]).columns.tolist()

# Define preprocessors for numerical and categorical data
numerical_preprocessor = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # Apply mean imputation for numerical columns
    ('scaler', StandardScaler())  # Optional: Standardize the data after imputation
])

categorical_preprocessor = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Apply most frequent imputation for categorical columns
    ('encoder', OneHotEncoder(handle_unknown='ignore'))  # Encode categorical features
])

# Combine preprocessors using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_preprocessor, numerical_columns),
        ('cat', categorical_preprocessor, categorical_columns)
    ])

# Create a pipeline that combines preprocessing and the classifier
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, random_state=0))  # Logistic regression model
])

# Define the number of folds for cross-validation
n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

# Lists to store metrics and predictions
auc_scores = []
precision_scores = []
recall_scores = []
f1_scores = []
all_y_true = []
all_y_pred_prob = []

# Cross-validation loop
for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y), 1):
    # Split data into training and validation sets
    X_train_fold, X_valid_fold = X.iloc[train_idx], X.iloc[valid_idx]
    y_train_fold, y_valid_fold = y.iloc[train_idx], y.iloc[valid_idx]
    
    # Fit the model
    model.fit(X_train_fold, y_train_fold)
    
    # Save the model for later use
    with open(f'logistic_regression_fold_mean_{fold}.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Obtain predicted probabilities for the positive class
    y_pred_prob = model.predict_proba(X_valid_fold)[:, 1]
    
    # Calculate predictions using a default threshold of 0.5
    y_pred = (y_pred_prob >= 0.5).astype(int)
    
    # Calculate performance metrics
    auc = roc_auc_score(y_valid_fold, y_pred_prob)
    precision = precision_score(y_valid_fold, y_pred)
    recall = recall_score(y_valid_fold, y_pred)
    f1 = f1_score(y_valid_fold, y_pred)
    
    # Store fold results
    auc_scores.append(auc)
    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)
    
    all_y_true.extend(y_valid_fold)
    all_y_pred_prob.extend(y_pred_prob)
    
    print(f"Fold {fold}: AUC={auc:.3f}, Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")

# Report the average performance metrics across folds
average_auc = np.mean(auc_scores)
average_precision = np.mean(precision_scores)
average_recall = np.mean(recall_scores)
average_f1 = np.mean(f1_scores)

print(f"\nAverage AUC over {n_folds} folds: {average_auc:.3f}")
print(f"Average Precision: {average_precision:.3f}")
print(f"Average Recall: {average_recall:.3f}")
print(f"Average F1 Score: {average_f1:.3f}")







###With Mice impuation
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import pickle

# Define features and target
X = df.drop(columns=['O_death30'])
y = df['O_death30']

# Identify categorical and numerical columns
categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_columns = X.select_dtypes(include=[np.number]).columns.tolist()

# Define preprocessors for numerical and categorical data
numerical_preprocessor = Pipeline(steps=[
    ('imputer', IterativeImputer(random_state=0)),
    ('scaler', StandardScaler())
])

categorical_preprocessor = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessors using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_preprocessor, numerical_columns),
        ('cat', categorical_preprocessor, categorical_columns)
    ])

# Create a pipeline that combines preprocessing and the classifier
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, random_state=0))
])

# Define the number of folds for cross-validation
n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

# Lists to store metrics and predictions
auc_scores = []
precision_scores = []
recall_scores = []
f1_scores = []
all_y_true = []
all_y_pred_prob = []
logistic_predictions = []
true_labels = []
# Cross-validation loop
for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y), 1):
    # Split data into training and validation sets
    X_train_fold, X_valid_fold = X.iloc[train_idx], X.iloc[valid_idx]
    y_train_fold, y_valid_fold = y.iloc[train_idx], y.iloc[valid_idx]
    
    # Fit the model
    model.fit(X_train_fold, y_train_fold)
    
    # Save the model for later use
    with open(f'logistic_regression_fold_MICE_{fold}.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Obtain predicted probabilities for the positive class
    y_pred_prob = model.predict_proba(X_valid_fold)[:, 1]
    logistic_predictions.extend(y_pred_prob)
    # Calculate predictions using a default threshold of 0.5
    y_pred = (y_pred_prob >= 0.5).astype(int)
    true_labels.extend(y_valid_fold)
    # Calculate performance metrics
    auc = roc_auc_score(y_valid_fold, y_pred_prob)
    precision = precision_score(y_valid_fold, y_pred)
    recall = recall_score(y_valid_fold, y_pred)
    f1 = f1_score(y_valid_fold, y_pred)
    
    # Store fold results
    auc_scores.append(auc)
    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)
    
    all_y_true.extend(y_valid_fold)
    all_y_pred_prob.extend(y_pred_prob)
    
    print(f"Fold {fold}: AUC={auc:.3f}, Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")

# Report the average performance metrics across folds
average_auc = np.mean(auc_scores)
average_precision = np.mean(precision_scores)
average_recall = np.mean(recall_scores)
average_f1 = np.mean(f1_scores)

print(f"\nAverage AUC over {n_folds} folds: {average_auc:.3f}")
print(f"Average Precision: {average_precision:.3f}")
print(f"Average Recall: {average_recall:.3f}")
print(f"Average F1 Score: {average_f1:.3f}")




