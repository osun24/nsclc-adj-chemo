import os
import time
import datetime
import pandas as pd
import numpy as np
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
import joblib
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.model_selection import GridSearchCV

np.random.seed(42)  # Set random seed for reproducibility

# Determine script directory for consistent path handling in a VM
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "rsf_results_affy")
os.makedirs(output_dir, exist_ok=True)

# --- Redirect print output to both console and log file ---
class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
    def flush(self):
        for f in self.files:
            f.flush()
            
import sys
current_date = datetime.datetime.now().strftime("%Y%m%d")
log_file = open(os.path.join(output_dir, f"{current_date}-rsf_run_log-Affy.txt"), "w")
sys.stdout = Tee(sys.stdout, log_file)

def rsf_concordance_metric(y, y_pred):
    return concordance_index_censored(y['OS_STATUS'], y['OS_MONTHS'], y_pred)[0]

def create_rsf(train_df, test_df, name):
    # Create subfolder for fold-level permutation importance results
    fold_importance_dir = os.path.join(output_dir, "fold_permutation_importance")
    os.makedirs(fold_importance_dir, exist_ok=True)
    
    # Create structured arrays for survival analysis
    y_train = Surv.from_dataframe('OS_STATUS', 'OS_MONTHS', train_df)
    y_test  = Surv.from_dataframe('OS_STATUS', 'OS_MONTHS', test_df)
    
    # Define covariates: affy columns except survival columns
    binary_columns = ['Adjuvant Chemo', 'IS_MALE']
    train_df[binary_columns] = train_df[binary_columns].astype(int)
    test_df[binary_columns] = test_df[binary_columns].astype(int)
    
    for col in binary_columns:
        assert train_df[col].max() <= 1 and train_df[col].min() >= 0, f"{col} should only contain binary values (0/1)."
        
    continuous_columns = train_df.columns.difference(['OS_STATUS', 'OS_MONTHS', *binary_columns])
    features = continuous_columns.union(binary_columns)
    
    # Create feature matrices from provided train and test data
    X_train = train_df[features]
    X_test  = test_df[features]
    
    # Count number of events in training and test sets
    train_events = np.sum(y_train['OS_STATUS'])
    test_events = np.sum(y_test['OS_STATUS'])
    print(f"Number of events in training set: {train_events}")
    print(f"Number of events in test set: {test_events}")
    
    # ---------------- Nested Cross-Validation for Hyperparameter Tuning ---------------- #
    # Define custom scoring function for concordance index
    rsf_score = make_scorer(rsf_concordance_metric, greater_is_better=True)

    # Define parameter grid for grid search
    param_grid = {
        "n_estimators": [500, 750],
        "min_samples_leaf": [60, 70, 80], # removed 50 (60: ~10 groups, 70: ~9 groups, 80: ~7 groups)
        "max_features": [0.1, 0.2, 0.5], # 0.1 * 13062 = 1306
        "max_depth": [3, 5], # 4?, removed 4
    }

    # Set up outer and inner KFold CV for nested CV
    outer_cv = RepeatedKFold(n_splits=5, n_repeats=2, random_state=42) # NOTE: probably just change this random state
    #outer_cv = KFold(n_splits=5, shuffle=True, random_state=42) # NOTE: probably just change this random state
    inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)

    outer_scores = []
    outer_best_params = []
    
    # Initialize lists to collect fold-level metrics and inner CV results
    outer_fold_metrics = []
    inner_cv_results_list = []

    print("Starting Nested Cross-Validation...")
    outer_start_time = time.time()
    
    # Get total number of splits for RepeatedKFold compatibility
    total_folds = outer_cv.get_n_splits(X_train)
    
    for fold_idx, (outer_train_idx, outer_test_idx) in enumerate(outer_cv.split(X_train)):
        print(f"[Fold {fold_idx+1}/{total_folds}] Starting outer fold...")
        X_train_outer = X_train.iloc[outer_train_idx]
        y_train_outer = y_train[outer_train_idx]
        X_test_outer = X_train.iloc[outer_test_idx]
        y_test_outer = y_train[outer_test_idx]
        
        # Inner CV for hyperparameter tuning using GridSearchCV
        grid_search_inner = GridSearchCV(
            estimator=RandomSurvivalForest(random_state=42, n_jobs=-1),
            param_grid=param_grid,
            cv=inner_cv,
            scoring=rsf_score,
            n_jobs=-1,
            verbose=1
        )
        grid_search_inner.fit(X_train_outer, y_train_outer)
 
        # Save detailed inner CV results for the current outer fold
        inner_cv_results_list.append({
            'fold': fold_idx + 1,
            'cv_results': grid_search_inner.cv_results_
        })
    
        # Get best hyperparameters from inner CV
        best_params_inner = grid_search_inner.best_params_
        
        # Retrain the model with best hyperparameters on full outer fold training set
        best_model_outer = RandomSurvivalForest(random_state=42, n_jobs=-1, **best_params_inner)
        best_model_outer.fit(X_train_outer, y_train_outer)
        
        # Evaluate on outer test set
        y_pred_outer = best_model_outer.predict(X_test_outer)
        outer_c_index = concordance_index_censored(
            y_test_outer['OS_STATUS'], y_test_outer['OS_MONTHS'], y_pred_outer
        )[0]
        # Evaluate on outer training set for fold-level training performance
        y_pred_train_outer = best_model_outer.predict(X_train_outer)
        outer_train_c_index = concordance_index_censored(
            y_train_outer['OS_STATUS'], y_train_outer['OS_MONTHS'], y_pred_train_outer
        )[0]
        
        # Print selected hyperparameters and C-index results
        print(f"[Fold {fold_idx+1}] Inner Best Hyperparameters: {best_params_inner}")
        
        # Inner fold results
        print(f"[Fold {fold_idx+1}] Inner Best C-index: {grid_search_inner.best_score_:.3f}")
        
        # Outer fold results
        print(f"[Fold {fold_idx+1}] Outer Test C-index: {outer_c_index:.3f}, Outer Train C-index: {outer_train_c_index:.3f}")
        
        outer_scores.append(outer_c_index)
        outer_best_params.append(best_params_inner)
        
        # Save fold-level metrics for later post-hoc analysis
        outer_fold_metrics.append({
            'fold': fold_idx + 1,
            'train_c_index': outer_train_c_index,
            'test_c_index': outer_c_index,
            'best_params': best_params_inner,
            'train_size': len(outer_train_idx),
            'test_size': len(outer_test_idx)
        })
        
        # Save RSF model for each fold
        fold_model_path = os.path.join(output_dir, f"{name}_fold_{fold_idx+1:02d}_model.pkl")
        joblib.dump(best_model_outer, fold_model_path)
        
        # Perform permutation feature importance on the held-out test fold (to prevent overfitting)
        print(f"[Fold {fold_idx+1}] Computing permutation feature importance on held-out test fold...")
        fold_perm_start = time.time()
        fold_perm_result = permutation_importance(
            best_model_outer, X_test_outer, y_test_outer,
            scoring=lambda est, X, y: rsf_concordance_metric(y, est.predict(X)),
            n_repeats=5, random_state=42, n_jobs=-1
        )
        
        # Create importance dataframe for this fold
        fold_importance_df = pd.DataFrame({
            "Feature": X_test_outer.columns,
            "Importance": fold_perm_result.importances_mean,
            "Std": fold_perm_result.importances_std,
            "Fold": fold_idx + 1
        }).sort_values(by="Importance", ascending=False)
        
        # Save fold-level permutation importance
        fold_importance_csv = os.path.join(fold_importance_dir, f"{name}_fold_{fold_idx+1:02d}_permutation_importance.csv")
        fold_importance_df.to_csv(fold_importance_csv, index=False)
        
        # Create a bar plot for fold-level permutation importance (only for top 20 features to keep plots readable)
        top_features_plot = fold_importance_df.head(20)
        plt.figure(figsize=(12, 8))
        plt.barh(top_features_plot['Feature'][::-1], top_features_plot['Importance'][::-1],
                    xerr=top_features_plot['Std'][::-1], color='skyblue')
        plt.xlabel('Permutation Importance')
        plt.title(f'Fold {fold_idx + 1} Permutation Importance (Top 20 Features)')
        plt.tight_layout()
        plt.savefig(os.path.join(fold_importance_dir, f"{name}_fold_{fold_idx+1:02d}_permutation_importance.png"))
        plt.close()  # Close the figure to prevent memory issues
        
        fold_perm_end = time.time()
        print(f"[Fold {fold_idx+1}] Permutation importance completed in {fold_perm_end - fold_perm_start:.2f} seconds")
        print(f"[Fold {fold_idx+1}] Permutation importance saved to {fold_importance_csv}")
        
    nested_cv_end_time = time.time()
    print(f"Nested CV completed in {nested_cv_end_time - outer_start_time:.2f} seconds.")
    
    nested_cv_mean_c_index = np.mean(outer_scores)
    nested_cv_std_c_index = np.std(outer_scores)
    print(f"Nested CV Mean Test C-index: {nested_cv_mean_c_index:.3f} ± {nested_cv_std_c_index:.3f}")
    
    # ---------------- Aggregate Fold-Level Permutation Importance Results ---------------- #
    print("Aggregating fold-level permutation importance results...")
    
    # Read all fold-level importance files and combine them
    all_fold_importance = []
    for fold_idx in range(total_folds):
        fold_csv = os.path.join(fold_importance_dir, f"{name}_fold_{fold_idx+1:02d}_permutation_importance.csv")
        if os.path.exists(fold_csv):
            fold_df = pd.read_csv(fold_csv)
            all_fold_importance.append(fold_df)
    
    if all_fold_importance:
        # Combine all fold importance data
        combined_fold_importance = pd.concat(all_fold_importance, ignore_index=True)
        
        # Calculate aggregated statistics across folds for each feature
        aggregated_importance = combined_fold_importance.groupby('Feature').agg({
            'Importance': ['mean', 'std', 'min', 'max', 'count'],
            'Std': 'mean'
        }).round(6)
        
        # Flatten column names
        aggregated_importance.columns = ['_'.join(col).strip() for col in aggregated_importance.columns.values]
        aggregated_importance = aggregated_importance.reset_index()
        aggregated_importance = aggregated_importance.sort_values(by='Importance_mean', ascending=False)
        
        # Save aggregated results
        aggregated_csv = os.path.join(output_dir, f"{name}_aggregated_fold_permutation_importance.csv")
        aggregated_importance.to_csv(aggregated_csv, index=False)
        print(f"Aggregated fold-level permutation importance saved to {aggregated_csv}")
        
        # Save combined raw data
        combined_csv = os.path.join(output_dir, f"{name}_combined_fold_permutation_importance.csv")
        combined_fold_importance.to_csv(combined_csv, index=False)
        print(f"Combined fold-level permutation importance saved to {combined_csv}")
        
        # Create a summary plot of top features across folds
        top_features = aggregated_importance.head(20)
        plt.figure(figsize=(12, 8))
        plt.barh(top_features['Feature'][::-1], top_features['Importance_mean'][::-1],
                 xerr=top_features['Importance_std'][::-1], color=(0.2, 0.6, 0.8))
        plt.xlabel("Mean Permutation Importance Across Folds")
        plt.title(f"Top 20 Features - Mean Permutation Importance Across {total_folds} Folds")
        plt.tight_layout()
        aggregated_plot = os.path.join(output_dir, f"{name}_aggregated_fold_permutation_importance.png")
        plt.savefig(aggregated_plot)
        plt.close()
        print(f"Aggregated fold-level importance plot saved to {aggregated_plot}")
 
    # ---------------- Save Detailed Fold Metrics for Post-Hoc Analysis, Debug ---------------- #
    current_date = datetime.datetime.now().strftime("%Y%m%d")
    combined_rows = []
    
    # Process outer fold metrics
    for row in outer_fold_metrics:
        row_copy = row.copy()
        row_copy['fold_type'] = 'outer'
        row_copy['outer_fold'] = row_copy.pop('fold')
        combined_rows.append(row_copy)
    
    # Process inner CV results
    for item in inner_cv_results_list:
        outer_fold = item.get('fold')
        cv_results = item.get('cv_results', {})
        n_candidates = len(cv_results.get('mean_test_score', []))
        for i in range(n_candidates):
            inner_row = {
                'fold_type': 'inner',
                'outer_fold': outer_fold,
                'candidate_index': i,
                'mean_test_score': cv_results.get('mean_test_score', [None]*n_candidates)[i],
                'std_test_score': cv_results.get('std_test_score', [None]*n_candidates)[i],
                'rank_test_score': cv_results.get('rank_test_score', [None]*n_candidates)[i],
                'params': cv_results.get('params', [None]*n_candidates)[i]
            }
            combined_rows.append(inner_row)
    
    combined_df = pd.DataFrame(combined_rows)
    combined_csv_path = os.path.join(output_dir, f"{name}_rsf_all_fold_results_{current_date}.csv")
    combined_df.to_csv(combined_csv_path, index=False)
    print(f"Combined fold-level metrics (inner and outer) saved to {combined_csv_path}")

if __name__ == "__main__":
    print("Loading train data from: affyTrain.csv")
    train = pd.read_csv("affyTrain.csv")
    
    print(f"Number of events in training set: {train['OS_STATUS'].sum()} | Censored cases: {train.shape[0] - train['OS_STATUS'].sum()}")
    print("Train data shape:", train.shape)
    
    print("Loading validation data from: affyValidation.csv")
    valid = pd.read_csv("affyValidation.csv")
    
    # Combining train and validation
    train = pd.concat([train, valid], ignore_index=True)
    
    # Load test set
    print("Loading test data from: affyTest.csv")
    test = pd.read_csv("affyTest.csv")
    print(f"Number of events in test set: {test['OS_STATUS'].sum()} | Censored cases: {test.shape[0] - test['OS_STATUS'].sum()}")
    print("Test data shape:", test.shape)
    
    train['Adjuvant Chemo'] = train['Adjuvant Chemo'].replace({'OBS': 0, 'ACT': 1})
    test['Adjuvant Chemo'] = test['Adjuvant Chemo'].replace({'OBS': 0, 'ACT': 1})
    
    create_rsf(train, test, 'Affy RS') # NOTE: CHANGE THIS TO YOUR NAME