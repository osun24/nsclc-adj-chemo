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
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

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
        "min_samples_leaf": [50, 60, 70],    
        "max_features": [0.1, 0.2, 0.5], # 0.1 * 13062 = 1306
        "max_depth": [3, 4, 5],
    }

    # Set up outer and inner KFold CV for nested CV
    outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
    inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)

    outer_scores = []
    outer_best_params = []
    # Initialize lists to collect fold-level metrics and inner CV results
    outer_fold_metrics = []
    inner_cv_results_list = []

    print("Starting Nested Cross-Validation...")
    outer_start_time = time.time()
    
    for fold_idx, (outer_train_idx, outer_test_idx) in enumerate(outer_cv.split(X_train)):
        print(f"[Fold {fold_idx+1}/{outer_cv.get_n_splits()}] Starting outer fold...")
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
    
        best_model_inner = grid_search_inner.best_estimator_
        # Evaluate on outer test set
        y_pred_outer = best_model_inner.predict(X_test_outer)
        outer_c_index = concordance_index_censored(
            y_test_outer['OS_STATUS'], y_test_outer['OS_MONTHS'], y_pred_outer
        )[0]
        # Evaluate on outer training set for fold-level training performance
        y_pred_train_outer = best_model_inner.predict(X_train_outer)
        outer_train_c_index = concordance_index_censored(
            y_train_outer['OS_STATUS'], y_train_outer['OS_MONTHS'], y_pred_train_outer
        )[0]
    
        outer_scores.append(outer_c_index)
        outer_best_params.append(grid_search_inner.best_params_)
        
        # Save fold-level metrics for later post-hoc analysis
        outer_fold_metrics.append({
            'fold': fold_idx + 1,
            'train_c_index': outer_train_c_index,
            'test_c_index': outer_c_index,
            'best_params': grid_search_inner.best_params_
        })
        
        print(f"[Fold {fold_idx+1}] Completed outer fold with Test C-index: {outer_c_index:.3f}")
    
    nested_cv_end_time = time.time()
    print(f"Nested CV completed in {nested_cv_end_time - outer_start_time:.2f} seconds.")
    
    nested_cv_mean_c_index = np.mean(outer_scores)
    print(f"Nested CV Mean Test C-index: {nested_cv_mean_c_index:.3f}")
 
    # ---------------- Save Detailed Fold Metrics for Post-Hoc Analysis ---------------- #
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

    # ---------------- Final Model Training using Nested CV Results ---------------- #
    # Select best hyperparameters from nested CV outer folds (fold with highest Test C-index)
    best_fold_idx = np.argmax(outer_scores)
    best_params_nested = outer_best_params[best_fold_idx]
    print(f"Selected hyperparameters from nested CV: {best_params_nested}")
    
    # Train final model on full training set using nested CV best hyperparameters
    best_model = RandomSurvivalForest(random_state=42, n_jobs=-1, **best_params_nested)
    best_model.fit(X_train, y_train)
    
    # Save nested CV outer fold summary results to CSV
    cv_results_df = pd.DataFrame({
        "fold": list(range(1, len(outer_scores)+1)),
        "Test_C_index": outer_scores,
        "Hyperparameters": outer_best_params
    })
    results_csv = os.path.join(output_dir, f"{name}_rsf_nested_cv_results.csv")
    cv_results_df.to_csv(results_csv, index=False)
    print(f"Nested CV results saved to {results_csv}")

    # ---------------- Select Simplest Model using the 1 SE Rule from Nested CV ---------------- #
    max_score = max(outer_scores)
    std_score = np.std(outer_scores)
    one_se_threshold = max_score - std_score

    candidates = []
    for score, params in zip(outer_scores, outer_best_params):
        if score >= one_se_threshold:
            candidate = params.copy()
            candidate["score"] = score
            candidates.append(candidate)

    if candidates:
        one_se_candidates_sorted = sorted(
            candidates,
            key=lambda r: (r["n_estimators"], r["min_samples_leaf"], r["max_features"])
        )
        one_se_candidate = one_se_candidates_sorted[0]
    else:
        print("No candidate model met the 1 SE threshold. Defaulting to the best nested CV model.")
        one_se_candidate = best_params_nested
        
    # Remove "score" key from candidate dictionary
    one_se_candidate = {k: v for k, v in one_se_candidate.items() if k != "score"}
    print(f"1 SE RSF hyperparameters from nested CV: {one_se_candidate} (threshold: {one_se_threshold:.3f})")
    one_se_model = RandomSurvivalForest(random_state=42, n_jobs=-1, **one_se_candidate)
    one_se_model.fit(X_train, y_train)

    # Save best model 
    best_model_file = os.path.join(output_dir, f"{name}_final_rsf_model.pkl")
    joblib.dump(best_model, best_model_file)
    print(f"Best RSF model saved to {best_model_file}")
    
    # Save simplest 1 SE model
    one_se_model_file = os.path.join(output_dir, f"{name}_final_rsf_model_1se.pkl")
    joblib.dump(one_se_model, one_se_model_file)
    print(f"1 SE RSF model saved to {one_se_model_file}")

    # ---------------- Evaluate Best and 1 SE RSF Models ---------------- #
    best_pred_train = best_model.predict(X_train)
    best_train_c_index = concordance_index_censored(y_train['OS_STATUS'], y_train['OS_MONTHS'], best_pred_train)[0]
    best_pred_test = best_model.predict(X_test)
    best_test_c_index = concordance_index_censored(y_test['OS_STATUS'], y_test['OS_MONTHS'], best_pred_test)[0]
    print(f"Best RSF: Train C-index: {best_train_c_index:.3f}, Test C-index: {best_test_c_index:.3f}")

    one_se_pred_train = one_se_model.predict(X_train)
    one_se_train_c_index = concordance_index_censored(y_train['OS_STATUS'], y_train['OS_MONTHS'], one_se_pred_train)[0]
    one_se_pred_test = one_se_model.predict(X_test)
    one_se_test_c_index = concordance_index_censored(y_test['OS_STATUS'], y_test['OS_MONTHS'], one_se_pred_test)[0]
    print(f"1 SE RSF: Train C-index: {one_se_train_c_index:.3f}, Test C-index: {one_se_test_c_index:.3f}")
    
    print(f"Starting permutation importance for 1 SE model...")
    step1_start = time.time()
    one_se_model.fit(X_train, y_train)

    test_pred = one_se_model.predict(X_test)
    c_index = concordance_index_censored(y_test['OS_STATUS'], y_test['OS_MONTHS'], test_pred)
    print(f"Initial RSF Test C-index (1SE Model): {c_index[0]:.3f}")
    train_pred = one_se_model.predict(X_train)
    train_c_index = concordance_index_censored(y_train['OS_STATUS'], y_train['OS_MONTHS'], train_pred)
    print(f"Initial RSF Train C-index (1SE Model): {train_c_index[0]:.3f}")

    perm_result = permutation_importance(one_se_model, X_train, y_train,
                                       scoring=lambda est, X, y: rsf_concordance_metric(y, est.predict(X)),
                                       n_repeats=5, random_state=42, n_jobs=-1)
    importances = perm_result.importances_mean
    importance_df = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": importances,
        "Std": perm_result.importances_std
    }).sort_values(by="Importance", ascending=False)
    
    preselect_csv = os.path.join(output_dir, f"{name}_rsf_preselection_importances_1SE.csv")
    importance_df.to_csv(preselect_csv, index=False)
    print(f"RSF pre-selection importances saved to {preselect_csv}")
    
    top_preselect = importance_df.head(50)
    plt.figure(figsize=(12, 8))
    plt.barh(top_preselect["Feature"][::-1], top_preselect["Importance"][::-1],
             xerr=top_preselect["Std"][::-1], color=(9/255, 117/255, 181/255))
    plt.xlabel("Permutation Importance")
    plt.title("RSF Pre-Selection (Top 50 Features)")
    plt.tight_layout()
    preselect_plot = os.path.join(output_dir, f"{name}_rsf_preselection_importances_1SE.png")
    plt.savefig(preselect_plot)
    plt.close()
    print(f"RSF pre-selection plot saved to {preselect_plot}")
    
    top_n_rsf = min(1000, len(importance_df))
    selected_features_rsf = importance_df.iloc[:top_n_rsf]["Feature"].tolist()
    print(f"RSF pre-selection: selected top {len(selected_features_rsf)} features.")
    
    X_train_rsf = X_train[selected_features_rsf]
    X_test_rsf = X_test[selected_features_rsf]
    step1_end = time.time()
    print(f"Step 1 (1SE) completed in {step1_end - step1_start:.2f} seconds.")

    """# ---------------- Feature Selection on Best Model ---------------- #
    print("Starting feature selection on best model...")
    step_best_start = time.time()
    perm_result_best = permutation_importance(
        best_model, X_train, y_train,
        scoring=lambda est, X, y: rsf_concordance_metric(y, est.predict(X)),
        n_repeats=5, random_state=42, n_jobs=-1
    )
    importances_best = perm_result_best.importances_mean
    importance_df_best = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": importances_best,
        "Std": perm_result_best.importances_std
    }).sort_values(by="Importance", ascending=False)
    preselect_csv_best = os.path.join(output_dir, f"{name}_rsf_preselection_importances_best.csv")
    importance_df_best.to_csv(preselect_csv_best, index=False)
    print(f"Best model pre-selection importances saved to {preselect_csv_best}")
    
    top_preselect_best = importance_df_best.head(50)
    plt.figure(figsize=(12, 8))
    plt.barh(top_preselect_best["Feature"][::-1], top_preselect_best["Importance"][::-1],
             xerr=top_preselect_best["Std"][::-1], color=(9/255, 117/255, 181/255))
    plt.xlabel("Permutation Importance")
    plt.title("RSF Pre-Selection (Top 50 Features) - Best Model")
    plt.tight_layout()
    preselect_plot_best = os.path.join(output_dir, f"{name}_rsf_preselection_importances_best.png")
    plt.savefig(preselect_plot_best)
    plt.close()
    step_best_end = time.time()
    print(f"Feature selection on best model completed in {step_best_end - step_best_start:.2f} seconds.")
    
    top_n_best = min(1000, len(importance_df_best))
    selected_features_best = importance_df_best.iloc[:top_n_best]["Feature"].tolist()
    print(f"Best model: selected top {len(selected_features_best)} features.")"""
    
    # Return the best model, 1 SE model, and the selected features for further evaluation
    return best_model, one_se_model, selected_features_rsf

if __name__ == "__main__":
    print("Loading train data from: affyTrain.csv")
    train = pd.read_csv("affyTrain.csv")
    
    print(f"Number of events in training set: {train['OS_STATUS'].sum()} | Censored cases: {train.shape[0] - train['OS_STATUS'].sum()}")
    print("Train data shape:", train.shape)
    
    print("Loading validation data from: affyValidation.csv")
    valid = pd.read_csv("affyValidation.csv")
    
    # Handle Adjuvant Chemo column with dummies to have OBS as baseline
    train['Adjuvant Chemo'] = train['Adjuvant Chemo'].replace({'OBS': 0, 'ACT': 1})
    valid['Adjuvant Chemo'] = valid['Adjuvant Chemo'].replace({'OBS': 0, 'ACT': 1})
    
    print(f"Number of events in validation set: {valid['OS_STATUS'].sum()} | Censored cases: {valid.shape[0] - valid['OS_STATUS'].sum()}")
    print("Validation data shape:", valid.shape)
    
    create_rsf(train, valid, 'Affy RS')