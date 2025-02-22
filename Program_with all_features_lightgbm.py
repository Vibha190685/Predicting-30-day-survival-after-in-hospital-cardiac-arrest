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




###Five fold cross validation for model evalution with all features: without imputation
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import lightgbm as lgb
import pickle

# Define features and target
X = df.drop(columns=['O_death30'])
y = df['O_death30']

# Use the best parameters obtained from your Optuna study
params = best_params_lightGBM_obj1 # Replace with your actual Optuna study object

# Define the number of folds and initialize StratifiedKFold (maintains positive class proportion)
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
    # Split data into training and validation folds
    X_train_fold, X_valid_fold = X.iloc[train_idx], X.iloc[valid_idx]
    y_train_fold, y_valid_fold = y.iloc[train_idx], y.iloc[valid_idx]
    
    # Initialize LightGBM model with the best parameters
    lgb_model = LGBMClassifier(**params, random_state=42)
    
    # Train the model
    lgb_model.fit(X_train_fold, y_train_fold)
    
    # Save the model for later use
    with open(f'lightgbm_fold_WI_O1_{fold}.pkl', 'wb') as f:
        pickle.dump(lgb_model, f)
    
    # Obtain predicted probabilities for the positive class
    y_pred_prob = lgb_model.predict_proba(X_valid_fold)[:, 1]
    
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

# Report the average performance metrics across folds
average_auc = np.mean(auc_scores)
average_precision = np.mean(precision_scores)
average_recall = np.mean(recall_scores)
average_f1 = np.mean(f1_scores)

print(f"Average AUC over {n_folds} folds: {average_auc:.3f}")
print(f"Average Precision: {average_precision:.3f}")
print(f"Average Recall: {average_recall:.3f}")
print(f"Average F1 Score: {average_f1:.3f}")




###With mean impuation
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import lightgbm as lgb
import pickle

# Define features and target
X = df.drop(columns=['O_death30'])
y = df['O_death30']

categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()

# Use the best parameters obtained from your Optuna study
params = best_params_lightGBM_obj1  # Replace with your actual Optuna study object

# Define the number of folds and initialize StratifiedKFold (maintains positive class proportion)
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
    # Split data into training and validation folds (use .copy() to avoid SettingWithCopy warnings)
    X_train_fold = X.iloc[train_idx].copy()
    X_valid_fold = X.iloc[valid_idx].copy()
    y_train_fold = y.iloc[train_idx]
    y_valid_fold = y.iloc[valid_idx]
    
    # Identify numerical columns based on the defined categorical columns
    numerical_columns = [col for col in X_train_fold.columns if col not in categorical_columns]
    
    # Define a ColumnTransformer for imputation:
    # - Numerical columns: mean imputation
    # - Categorical columns: most frequent imputation
    ct = ColumnTransformer(transformers=[ 
        ('num', SimpleImputer(strategy='mean'), numerical_columns),
        ('cat', SimpleImputer(strategy='most_frequent'), categorical_columns)
    ], remainder='drop')
    
    # Fit the transformer on the training fold and transform both training and validation folds
    X_train_fold_imputed = pd.DataFrame(ct.fit_transform(X_train_fold), columns=numerical_columns + categorical_columns)
    X_valid_fold_imputed = pd.DataFrame(ct.transform(X_valid_fold), columns=numerical_columns + categorical_columns)
    
    
    # Initialize LightGBM model with the best parameters
    lgb_model = LGBMClassifier(**params, random_state=42)
    
    # Train the model
    lgb_model.fit(X_train_fold_imputed, y_train_fold, 
                  eval_set=[(X_valid_fold_imputed, y_valid_fold)])
    
    # Save the model for later use
    with open(f'lightgbm_fold_mean_O1_{fold}.pkl', 'wb') as f:
        pickle.dump(lgb_model, f)

    # Obtain predicted probabilities for the positive class
    y_pred_prob = lgb_model.predict_proba(X_valid_fold_imputed)[:, 1]
    
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
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from xgboost import XGBClassifier
import pickle

# Define features and target
X = df.drop(columns=['O_death30'])
y = df['O_death30']

# Identify categorical columns
categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
# Identify numerical columns
numerical_columns = [col for col in X.columns if col not in categorical_columns]

# Use the best parameters obtained from your Optuna study
params = best_params_lightGBM_obj1   # Replace with your actual Optuna study object

# Define the number of folds and initialize StratifiedKFold
n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

# Lists to store metrics
auc_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

# Cross-validation loop
for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y), 1):
    # Split data into training and validation sets
    X_train_fold = X.iloc[train_idx].copy()
    X_valid_fold = X.iloc[valid_idx].copy()
    y_train_fold = y.iloc[train_idx]
    y_valid_fold = y.iloc[valid_idx]
    
    # Initialize the IterativeImputer for MICE
    mice_imputer = IterativeImputer(random_state=42)
    
    # Apply MICE imputation to numerical columns
    X_train_fold[numerical_columns] = mice_imputer.fit_transform(X_train_fold[numerical_columns])
    X_valid_fold[numerical_columns] = mice_imputer.transform(X_valid_fold[numerical_columns])
    
    # For categorical columns, apply most frequent imputation
    for col in categorical_columns:
        most_frequent = X_train_fold[col].mode()[0]
        X_train_fold[col].fillna(most_frequent, inplace=True)
        X_valid_fold[col].fillna(most_frequent, inplace=True)
    
    # Initialize XGBoost model
    lgb_model = LGBMClassifier(**params, random_state=42)
    
    # Train the model
    lgb_model.fit(X_train_fold, y_train_fold, 
                  eval_set=[(X_valid_fold, y_valid_fold)])
    
    # Save the model
    with open(f'lightgbm_fold_MICE_O1_{fold}.pkl', 'wb') as f:
        pickle.dump(lgb_model, f)
    
    # Predict probabilities
    y_pred_prob = lgb_model.predict_proba(X_valid_fold)[:, 1]
    
    # Binarize predictions
    y_pred = (y_pred_prob >= 0.5).astype(int)
    
    # Calculate metrics
    auc = roc_auc_score(y_valid_fold, y_pred_prob)
    precision = precision_score(y_valid_fold, y_pred)
    recall = recall_score(y_valid_fold, y_pred)
    f1 = f1_score(y_valid_fold, y_pred)
    
    # Store metrics
    auc_scores.append(auc)
    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)
    
    print(f"Fold {fold}: AUC={auc:.3f}, Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")

# Calculate average metrics
average_auc = np.mean(auc_scores)
average_precision = np.mean(precision_scores)
average_recall = np.mean(recall_scores)
average_f1 = np.mean(f1_scores)

print(f"\nAverage AUC over {n_folds} folds: {average_auc:.3f}")
print(f"Average Precision: {average_precision:.3f}")
print(f"Average Recall: {average_recall:.3f}")
print(f"Average F1 Score: {average_f1:.3f}")


