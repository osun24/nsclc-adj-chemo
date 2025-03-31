import os
import time
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
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingRandomSearchCV
import sys

# Determine script directory for consistent path handling in a VM
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "rsf_results")
os.makedirs(output_dir, exist_ok=True)

# Redirect print output to both console and log file
class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
    def flush(self):
        for f in self.files:
            f.flush()

log_file = open(os.path.join(output_dir, "rsf_run_log-ALL.txt"), "w")
sys.stdout = Tee(sys.stdout, log_file)

def rsf_concordance_metric(y, y_pred):
    return concordance_index_censored(y['OS_STATUS'], y['OS_MONTHS'], y_pred)[0]

def create_rsf(train_df, test_df, name):
    overall_start = time.time()
    # Create structured arrays for survival analysis
    y_train = Surv.from_dataframe('OS_STATUS', 'OS_MONTHS', train_df)
    y_test  = Surv.from_dataframe('OS_STATUS', 'OS_MONTHS', test_df)
    
    # Define covariates: all columns except survival columns
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
    
    # ---------------- Internal CV for Hyperparameter Tuning ---------------- #
    rsf_score = make_scorer(rsf_concordance_metric, greater_is_better=True)
    param_distributions = {
        "n_estimators": list(range(500, 1001, 100)),
        "min_samples_leaf": list(range(50, 101, 5)),    
        "max_features": ["sqrt", "log2", 500], 
        "max_depth": [10, 20, None],
    }
    cv = KFold(n_splits=10, shuffle=True, random_state=42)
    
    print("Starting hyperparameter tuning using internal cross-validation...")
    halving_search = HalvingRandomSearchCV(
        estimator=RandomSurvivalForest(random_state=42, n_jobs=-1),
        param_distributions=param_distributions,
        factor=1.5,
        cv=cv,
        scoring=rsf_score,
        random_state=42,
        n_jobs=-1,
        verbose=1,
        min_resources=100,
    )
    halving_search.fit(X_train, y_train)
    
    best_params = halving_search.best_params_
    print(f"Best hyperparameters from internal CV: {best_params}")
    
    # Train final model on full training set using best hyperparameters
    best_model = RandomSurvivalForest(random_state=42, n_jobs=-1, **best_params)
    best_model.fit(X_train, y_train)
    
    # Save CV results to CSV
    cv_results_df = pd.DataFrame(halving_search.cv_results_)
    results_csv = os.path.join(output_dir, f"{name}_rsf_cv_results.csv")
    cv_results_df.to_csv(results_csv, index=False)
    print(f"CV results saved to {results_csv}")
    
    # ---------------- Evaluate Best RSF Model ---------------- #
    best_pred_train = best_model.predict(X_train)
    best_train_c_index = concordance_index_censored(y_train['OS_STATUS'], y_train['OS_MONTHS'], best_pred_train)[0]
    best_pred_test = best_model.predict(X_test)
    best_test_c_index = concordance_index_censored(y_test['OS_STATUS'], y_test['OS_MONTHS'], best_pred_test)[0]
    print(f"Best RSF: Train C-index: {best_train_c_index:.3f}, Test C-index: {best_test_c_index:.3f}")
    
    # ---------------- Feature Selection on Best Model ---------------- #
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
    print(f"Best model: selected top {len(selected_features_best)} features.")
    
    # Return the best model and the selected features for further evaluation
    return best_model, selected_features_best

if __name__ == "__main__":
    print("Loading train data from: allTrain.csv")
    train = pd.read_csv("allTrain.csv")
    
    print(f"Number of events in training set: {train['OS_STATUS'].sum()} | Censored cases: {train.shape[0] - train['OS_STATUS'].sum()}")
    print("Train data shape:", train.shape)
    
    print("Loading validation data from: allValidation.csv")
    valid = pd.read_csv("allValidation.csv")
    
    print(f"Number of events in validation set: {valid['OS_STATUS'].sum()} | Censored cases: {valid.shape[0] - valid['OS_STATUS'].sum()}")
    print("Validation data shape:", valid.shape)
    
    create_rsf(train, valid, 'ALL 3-23-25 RSINTERNAL')