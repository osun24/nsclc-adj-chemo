import os
import time
import datetime
import pandas as pd
import numpy as np
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler

# Determine script directory for consistent path handling
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "coxnet_results_affy")
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
log_file = open(os.path.join(output_dir, "coxnet_run_log-Affy.txt"), "w")
sys.stdout = Tee(sys.stdout, log_file)

def cox_concordance_metric(y, y_pred):
    return concordance_index_censored(y['OS_STATUS'], y['OS_MONTHS'], y_pred)[0]

def create_coxnet(train_df, test_df, name):
    # Create structured arrays for survival analysis
    y_train = Surv.from_dataframe('OS_STATUS', 'OS_MONTHS', train_df)
    y_test = Surv.from_dataframe('OS_STATUS', 'OS_MONTHS', test_df)
    
    # Define covariates: affy columns except survival columns
    binary_columns = ['Adjuvant Chemo', 'IS_MALE']
    train_df[binary_columns] = train_df[binary_columns].astype(int)
    test_df[binary_columns] = test_df[binary_columns].astype(int)
    
    for col in binary_columns:
        assert train_df[col].max() <= 1 and train_df[col].min() >= 0, f"{col} should only contain binary values (0/1)."
        
    continuous_columns = train_df.columns.difference(['OS_STATUS', 'OS_MONTHS', *binary_columns])
    features = continuous_columns.union(binary_columns)
    
    # Create feature matrices
    X_train = train_df[features]
    X_test = test_df[features]
    
    # Cox models require standardized features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame to keep column names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    # Count events
    train_events = np.sum(y_train['OS_STATUS'])
    test_events = np.sum(y_test['OS_STATUS'])
    print(f"Number of events in training set: {train_events}")
    print(f"Number of events in test set: {test_events}")
    
    # ---------------- Nested Cross-Validation for Hyperparameter Tuning ---------------- #
    # Define custom scoring function for concordance index
    coxnet_score = make_scorer(cox_concordance_metric, greater_is_better=True)
    
    # Define parameter grid for grid search
    # For CoxnetSurvivalAnalysis, we tune alphas (regularization strength) and l1_ratio (elastic net mixing)
    param_grid = {
        "l1_ratio": [0.1, 0.5, 0.7, 0.9, 1.0],  # 0=Ridge, 1=Lasso, between=Elastic Net
        "alphas": [[0.001, 0.01, 0.1, 1.0]],    # Regularization path
        "max_iter": [1000],
        "tol": [1e-7]
    }
    
    # Set up outer and inner KFold CV
    outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
    inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    outer_scores = []
    outer_best_params = []
    outer_fold_metrics = []
    inner_cv_results_list = []
    
    print("Starting Nested Cross-Validation...")
    outer_start_time = time.time()
    
    for fold_idx, (outer_train_idx, outer_test_idx) in enumerate(outer_cv.split(X_train_scaled)):
        print(f"[Fold {fold_idx+1}/{outer_cv.get_n_splits()}] Starting outer fold...")
        X_train_outer = X_train_scaled.iloc[outer_train_idx]
        y_train_outer = y_train[outer_train_idx]
        X_test_outer = X_train_scaled.iloc[outer_test_idx]
        y_test_outer = y_train[outer_test_idx]
        
        # Inner CV for hyperparameter tuning using GridSearchCV
        grid_search_inner = GridSearchCV(
            estimator=CoxnetSurvivalAnalysis(fit_baseline_model=True, random_state=42),
            param_grid=param_grid,
            cv=inner_cv,
            scoring=coxnet_score,
            n_jobs=-1,
            verbose=1
        )
        grid_search_inner.fit(X_train_outer, y_train_outer)
        
        # Save detailed inner CV results
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
        # Evaluate on outer training set
        y_pred_train_outer = best_model_inner.predict(X_train_outer)
        outer_train_c_index = concordance_index_censored(
            y_train_outer['OS_STATUS'], y_train_outer['OS_MONTHS'], y_pred_train_outer
        )[0]
        
        outer_scores.append(outer_c_index)
        outer_best_params.append(grid_search_inner.best_params_)
        
        # Save fold-level metrics
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
    combined_csv_path = os.path.join(output_dir, f"{name}_coxnet_all_fold_results_{current_date}.csv")
    combined_df.to_csv(combined_csv_path, index=False)
    print(f"Combined fold-level metrics (inner and outer) saved to {combined_csv_path}")
    
    # ---------------- Final Model Training using Nested CV Results ---------------- #
    # Select best hyperparameters from nested CV outer folds (fold with highest Test C-index)
    best_fold_idx = np.argmax(outer_scores)
    best_params_nested = outer_best_params[best_fold_idx]
    print(f"Selected hyperparameters from nested CV: {best_params_nested}")
    
    # Train final model on full training set using nested CV best hyperparameters
    best_model = CoxnetSurvivalAnalysis(fit_baseline_model=True, random_state=42, **best_params_nested)
    best_model.fit(X_train_scaled, y_train)
    
    # Save nested CV outer fold summary results to CSV
    cv_results_df = pd.DataFrame({
        "fold": list(range(1, len(outer_scores)+1)),
        "Test_C_index": outer_scores,
        "Hyperparameters": outer_best_params
    })
    results_csv = os.path.join(output_dir, f"{name}_coxnet_nested_cv_results.csv")
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
        # For CoxPH models, simpler models have higher l1_ratio (more sparsity) and higher alpha values
        one_se_candidates_sorted = sorted(
            candidates,
            key=lambda r: (-r["l1_ratio"][0], r["score"]),  # Higher l1_ratio first, then by score
            reverse=True
        )
        one_se_candidate = one_se_candidates_sorted[0]
    else:
        print("No candidate model met the 1 SE threshold. Defaulting to the best nested CV model.")
        one_se_candidate = best_params_nested
    
    # Remove "score" key from candidate dictionary
    one_se_candidate = {k: v for k, v in one_se_candidate.items() if k != "score"}
    print(f"1 SE CoxNet hyperparameters from nested CV: {one_se_candidate} (threshold: {one_se_threshold:.3f})")
    one_se_model = CoxnetSurvivalAnalysis(fit_baseline_model=True, random_state=42, **one_se_candidate)
    one_se_model.fit(X_train_scaled, y_train)
    
    # Save best model and scaler
    best_model_file = os.path.join(output_dir, f"{name}_final_coxnet_model.pkl")
    scaler_file = os.path.join(output_dir, f"{name}_coxnet_scaler.pkl")
    joblib.dump(best_model, best_model_file)
    joblib.dump(scaler, scaler_file)
    print(f"Best CoxNet model saved to {best_model_file}")
    print(f"Feature scaler saved to {scaler_file}")
    
    # Save simplest 1 SE model
    one_se_model_file = os.path.join(output_dir, f"{name}_final_coxnet_model_1se.pkl")
    joblib.dump(one_se_model, one_se_model_file)
    print(f"1 SE CoxNet model saved to {one_se_model_file}")
    
    # ---------------- Evaluate Best and 1 SE CoxNet Models ---------------- #
    best_pred_train = best_model.predict(X_train_scaled)
    best_train_c_index = concordance_index_censored(y_train['OS_STATUS'], y_train['OS_MONTHS'], best_pred_train)[0]
    best_pred_test = best_model.predict(X_test_scaled)
    best_test_c_index = concordance_index_censored(y_test['OS_STATUS'], y_test['OS_MONTHS'], best_pred_test)[0]
    print(f"Best CoxNet: Train C-index: {best_train_c_index:.3f}, Test C-index: {best_test_c_index:.3f}")
    
    one_se_pred_train = one_se_model.predict(X_train_scaled)
    one_se_train_c_index = concordance_index_censored(y_train['OS_STATUS'], y_train['OS_MONTHS'], one_se_pred_train)[0]
    one_se_pred_test = one_se_model.predict(X_test_scaled)
    one_se_test_c_index = concordance_index_censored(y_test['OS_STATUS'], y_test['OS_MONTHS'], one_se_pred_test)[0]
    print(f"1 SE CoxNet: Train C-index: {one_se_train_c_index:.3f}, Test C-index: {one_se_test_c_index:.3f}")
    
    # ---------------- Extract and Visualize Feature Importances ---------------- #
    print("Extracting feature importances for 1 SE model...")
    
    # Get the best alpha index for the 1 SE model
    # CoxnetSurvivalAnalysis provides multiple models along regularization path
    optimal_alpha_idx = one_se_model.alphas_.size - 1  # Default to last (most regularized)
    
    # Extract coefficients for the optimal alpha
    coefs = pd.DataFrame(
        one_se_model.coef_[:, optimal_alpha_idx],
        index=X_train.columns,
        columns=["coefficient"]
    )
    
    # Sort by absolute coefficient value
    coefs["abs_coef"] = coefs["coefficient"].abs()
    coefs_sorted = coefs.sort_values("abs_coef", ascending=False)
    
    """CHECK THIS P-VALUE CALCULATION!"""
    # Calculate approximate p-values using bootstrap resampling
    print("Calculating approximate p-values using bootstrap resampling...")
    n_bootstrap = 100
    bootstrap_coefs = np.zeros((X_train.shape[1], n_bootstrap))
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    for i in range(n_bootstrap):
        # Sample with replacement from training data
        bootstrap_indices = np.random.choice(X_train.shape[0], size=X_train.shape[0], replace=True)
        X_bootstrap = X_train_scaled.iloc[bootstrap_indices]
        y_bootstrap = y_train[bootstrap_indices]
        
        # Fit CoxNet model with same params as 1 SE model
        bootstrap_model = CoxnetSurvivalAnalysis(fit_baseline_model=True, random_state=i, **one_se_candidate)
        bootstrap_model.fit(X_bootstrap, y_bootstrap)
        
        # Extract coefficients for the same alpha index
        bootstrap_coefs[:, i] = bootstrap_model.coef_[:, optimal_alpha_idx]
    
    # Calculate p-values based on bootstrap distributions
    # For each coefficient, p-value = 2 * min(P(coef <= 0), P(coef >= 0))
    p_values = []
    for j in range(X_train.shape[1]):
        orig_coef = coefs.iloc[j, 0]
        if orig_coef == 0:  # If coefficient is zero due to regularization
            p_values.append(1.0)
        else:
            # Calculate proportion of bootstrap samples with opposite sign
            if orig_coef > 0:
                p_val = np.mean(bootstrap_coefs[j, :] <= 0)
            else:
                p_val = np.mean(bootstrap_coefs[j, :] >= 0)
            p_values.append(min(1.0, 2.0 * p_val))
    
    # Add p-values to coefficients DataFrame
    coefs["p"] = p_values
    coefs_sorted = coefs.sort_values("abs_coef", ascending=False)
    
    # Save coefficients with p-values to CSV
    coefs_csv = os.path.join(output_dir, f"{name}_coxnet_coefficients_1SE.csv")
    coefs_sorted.to_csv(coefs_csv)
    print(f"CoxNet coefficients saved to {coefs_csv}")
    
    # Plot top coefficients with p-values
    top_n = min(50, (coefs_sorted["coefficient"] != 0).sum())  # Show only non-zero coefficients, up to 50
    top_coefs = coefs_sorted.head(top_n)
    
    plt.figure(figsize=(12, 8))
    bars = plt.barh(top_coefs.index[::-1], top_coefs["coefficient"][::-1], color=(9/255, 117/255, 181/255))
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    plt.xlabel("Coefficient Value")
    plt.title(f"CoxNet Model Coefficients (Top {top_n} Features)")
    
    # Add p-value annotations to the right of the bars
    x_max = top_coefs["coefficient"].abs().max() * 1.3
    plt.xlim(right=x_max)
    
    # Add p-values on the right side of the plot
    for i, (idx, row) in enumerate(top_coefs[::-1].iterrows()):
        if row['p'] < 0.05:
            p_text = f'$\\mathbf{{p={row["p"]:.3f}}}$'
        else:
            p_text = f'p={row["p"]:.3f}'
        plt.text(x_max * 0.85, i, p_text, va='center', fontsize=8)
    
    plt.tight_layout()
    coefs_plot = os.path.join(output_dir, f"{name}_coxnet_coefficients_1SE.png")
    plt.savefig(coefs_plot)
    plt.close()
    print(f"CoxNet coefficients plot saved to {coefs_plot}")
    
    # Select features with non-zero coefficients
    selected_features = coefs_sorted[coefs_sorted["coefficient"] != 0].index.tolist()
    print(f"CoxNet selected {len(selected_features)} features with non-zero coefficients.")
    
    return best_model, one_se_model, selected_features, scaler

if __name__ == "__main__":
    print("Loading train data from: affyTrain.csv")
    train = pd.read_csv("affyTrain.csv")
    
    print(f"Number of events in training set: {train['OS_STATUS'].sum()} | Censored cases: {train.shape[0] - train['OS_STATUS'].sum()}")
    print("Train data shape:", train.shape)
    
    print("Loading validation data from: affyValidation.csv")
    valid = pd.read_csv("affyValidation.csv")
    
    print(f"Number of events in validation set: {valid['OS_STATUS'].sum()} | Censored cases: {valid.shape[0] - valid['OS_STATUS'].sum()}")
    print("Validation data shape:", valid.shape)
    
    create_coxnet(train, valid, 'Affy CPH RS')