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

categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
categorical_columns_indices = [df.columns.get_loc(col) for col in categorical_columns]

###program for feature selection abd evaltion: Catboost
import numpy as np
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import catboost as cb

# Define features and target
X = df.drop(columns=['O_death30'])
y = df['O_death30']

# Identify categorical columns (modify this based on your dataset)

cat_features = [X.columns.get_loc(col) for col in categorical_columns]  # Get the indices of categorical columns

# Use the best parameters obtained from your Optuna study for CatBoost
params_cb = best_params_catboost_obj1  # Replace with your CatBoost parameters from Optuna

# Define the number of folds
n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

# Initialize dictionary to store AUC results
auc_results = {i: [] for i in range(1, 16)}  # Store AUCs for 1 to 15 features

# Cross-validation loop
for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y), 1):
    print(f"\nProcessing Fold {fold}...")

    # Split data into training and validation folds
    X_train_fold, X_valid_fold = X.iloc[train_idx], X.iloc[valid_idx]
    y_train_fold, y_valid_fold = y.iloc[train_idx], y.iloc[valid_idx]

    # Load trained CatBoost model for this fold
    with open(f'catboost_fold_{fold}.pkl', 'rb') as f:
        cb_model = pickle.load(f)

    feature_importances = cb_model.feature_importances_

    # Get the names of the features
    feature_names = X_train_fold.columns

    # Sort feature importances in descending order
    sorted_indices = feature_importances.argsort()[::-1]

    ##features for hyper-parameter tuning
    HPT_optiuna_features=15
    HPT_optiuna_feature_indices = sorted_indices[:HPT_optiuna_features]

    
    # Get feature importance from the trained CatBoost model
    #feature_importances = cb_model.get_feature_importance()

    # Sort features by importance and select top 15
    sorted_features = sorted(enumerate(feature_importances), key=lambda x: x[1], reverse=True)

    # Top 15 features for CatBoost
    top_features = [X.columns[i] for i in [x[0] for x in sorted_features[:15]]]

    print(f"Top 15 Features for Fold {fold}: {top_features}")

    # Evaluate performance incrementally by adding top features one by one
    for num_features in range(1, 16):
        #selected_features = top_features[:num_features]  # Select top 'num_features'
        top_feature_indices = sorted_indices[:num_features]

        # Create a new DataFrame with only the selected top features
        X_train_selected = X_train_fold.iloc[:, top_feature_indices]
        X_valid_selected = X_valid_fold.iloc[:, top_feature_indices]

        # Check if the selected features are in the training data
        #X_train_selected = X_train_fold[selected_features]
        #X_valid_selected = X_valid_fold[selected_features]
        selected_categorical_columns_n = [idx for idx, col in enumerate(X_train_selected.columns) if col in categorical_columns]

        # Train a CatBoost model on the selected features, specifying categorical columns
        temp_model_cb = CatBoostClassifier(**params_cb,  # Adjust the depth of the trees
                                           cat_features=selected_categorical_columns_n,  # Specify the indices of categorical columns
                                           random_state=42,
                                           verbose=100)
        temp_model_cb.fit(X_train_selected, y_train_fold, verbose=0)

        # Get predicted probabilities for CatBoost model
        y_pred_prob_cb = temp_model_cb.predict_proba(X_valid_selected)[:, 1]  # Probabilities for class 1

        # Compute AUC score
        auc_score = roc_auc_score(y_valid_fold, y_pred_prob_cb)
        auc_results[num_features].append(auc_score)

# Compute mean AUC for each number of top features
mean_auc_results = {num_features: np.mean(auc_list) for num_features, auc_list in auc_results.items()}

# Print results
print("\nPerformance Evaluation with Incremental Feature Addition (CatBoost only):")
for num_features, auc in mean_auc_results.items():
    print(f"Top {num_features} features: Mean AUC = {auc:.4f}")

# Save results for further analysis
with open('incremental_feature_performance_catboost.pkl', 'wb') as f:
    pickle.dump(mean_auc_results, f)



###For XGBoost Model
import numpy as np
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import xgboost as xgb

# Define features and target
X = df.drop(columns=['O_death30'])
y = df['O_death30']

# Use the best parameters from Optuna study (you can adapt it to your own study if needed)
params = best_params_xgboost_obj1  # Replace this with your XGBoost parameters from Optuna

# Define the number of folds
n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

# Initialize dictionary to store AUC results
auc_results = {i: [] for i in range(1, 16)}  # Store AUCs for 1 to 15 features

# Cross-validation loop
for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y), 1):
    print(f"\nProcessing Fold {fold}...")

    # Split data into training and validation folds
    X_train_fold, X_valid_fold = X.iloc[train_idx], X.iloc[valid_idx]
    y_train_fold, y_valid_fold = y.iloc[train_idx], y.iloc[valid_idx]

    # Load trained XGBoost model for this fold
    with open(f'xgboost_fold_WI_O1_{fold}.pkl', 'rb') as f:
        xgb_model = pickle.load(f)
    
    # Get feature importance from the trained model
    feature_importances = xgb_model.get_booster().get_score(importance_type='weight')  # XGBoost feature importances

    # Sort features by importance and select top 15
    sorted_features = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
    top_features = [x[0] for x in sorted_features[:15]]  # Top 15 features
    importances = [x[1] for x in sorted_features[:15]]  # Corresponding importances

    print(f"Top 15 Features for Fold {fold}: {top_features}")

    # Evaluate performance incrementally by adding top features one by one
    for num_features in range(1, 16):
        selected_features = top_features[:num_features]  # Select top 'num_features'

        # Check if the selected features are in the training data
        X_train_selected = X_train_fold[selected_features]
        X_valid_selected = X_valid_fold[selected_features]

        # Convert to DMatrix format (XGBoost's preferred format)
        dtrain = xgb.DMatrix(X_train_selected, label=y_train_fold)
        dvalid = xgb.DMatrix(X_valid_selected, label=y_valid_fold)

        # Train an XGBoost model on the selected features
        temp_model = xgb.train(params, dtrain, num_boost_round=100)

        # Get predicted probabilities for validation set
        y_pred_prob = temp_model.predict(dvalid)

        # Compute AUC score
        auc_score = roc_auc_score(y_valid_fold, y_pred_prob)
        auc_results[num_features].append(auc_score)

# Compute mean AUC for each number of top features
mean_auc_results = {num_features: np.mean(auc_list) for num_features, auc_list in auc_results.items()}

# Print results
print("\nPerformance Evaluation with Incremental Feature Addition:")
for num_features, auc in mean_auc_results.items():
    print(f"Top {num_features} features: Mean AUC = {auc:.4f}")

# Save results for further analysis
with open('incremental_feature_performance_xgb.pkl', 'wb') as f:
    pickle.dump(mean_auc_results, f)



###For LightGBM Model
import numpy as np
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

# Define features and target
X = df.drop(columns=['O_death30'])
y = df['O_death30']

# Use the best parameters obtained from your Optuna study for LightGBM
params_lgb = best_params_lightGBM_obj1  # Replace with your LightGBM parameters from Optuna

# Define the number of folds
n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

# Initialize dictionary to store AUC results
auc_results = {i: [] for i in range(1, 16)}  # Store AUCs for 1 to 15 features

# Cross-validation loop
for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y), 1):
    print(f"\nProcessing Fold {fold}...")

    # Split data into training and validation folds
    X_train_fold, X_valid_fold = X.iloc[train_idx], X.iloc[valid_idx]
    y_train_fold, y_valid_fold = y.iloc[train_idx], y.iloc[valid_idx]

    # Load trained LightGBM model for this fold
    with open(f'lightgbm_fold_WI_UP_O1_{fold}.pkl', 'rb') as f:
        lgb_model = pickle.load(f)
    
    # Get feature importance from the trained LightGBM model
    lgb_feature_importances = lgb_model.booster_.feature_importance(importance_type='split')

    # Sort features by importance and select top 15
    lgb_sorted_features = sorted(enumerate(lgb_feature_importances), key=lambda x: x[1], reverse=True)

    # Top 15 features for LightGBM
    top_lgb_features = [x[0] for x in lgb_sorted_features[:15]]

    # Ensure feature names are selected rather than indices for LightGBM
    top_lgb_features_names = [X.columns[i] for i in top_lgb_features]

    print(f"Top 15 Features for Fold {fold}: {top_lgb_features_names}")

    # Evaluate performance incrementally by adding top features one by one
    for num_features in range(1, 16):
        selected_features = top_lgb_features_names[:num_features]  # Select top 'num_features'

        # Check if the selected features are in the training data
        X_train_selected = X_train_fold[selected_features]
        X_valid_selected = X_valid_fold[selected_features]

        # Train a LightGBM model on the selected features
        temp_model_lgb = lgb.train(params_lgb, lgb.Dataset(X_train_selected, label=y_train_fold), num_boost_round=100)

        # Get predicted probabilities for LightGBM model
        y_pred_prob_lgb = temp_model_lgb.predict(X_valid_selected)  # Use raw data for LightGBM prediction

        # Compute AUC score
        auc_score = roc_auc_score(y_valid_fold, y_pred_prob_lgb)
        auc_results[num_features].append(auc_score)

# Compute mean AUC for each number of top features
mean_auc_results = {num_features: np.mean(auc_list) for num_features, auc_list in auc_results.items()}

# Print results
print("\nPerformance Evaluation with Incremental Feature Addition (LightGBM only):")
for num_features, auc in mean_auc_results.items():
    print(f"Top {num_features} features: Mean AUC = {auc:.4f}")

# Save results for further analysis
with open('incremental_feature_performance_lgb.pkl', 'wb') as f:
    pickle.dump(mean_auc_results, f)
