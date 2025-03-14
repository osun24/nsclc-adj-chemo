import os
import time
import pandas as pd
import numpy as np
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
import joblib
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer

# Determine script directory for consistent path handling in a VM
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "rsf_results")
os.makedirs(output_dir, exist_ok=True)

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
    
    # ---------------- Hyperparameter search with RandomizedSearchCV ---------------- #
    # Define custom scoring function for concordance index
    rsf_score = make_scorer(rsf_concordance_metric, greater_is_better=True)

    param_distributions = {
        "n_estimators": [100, 200, 300, 400, 500],
        "min_samples_split": [5, 10, 15, 20, 25, 30, 35, 40],
        "min_samples_leaf": [5, 10, 15, 20, 25, 30],
        "max_features": ["sqrt", "log2"]
    }

    rsf = RandomSurvivalForest(random_state=42, n_jobs=-1)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    random_search = RandomizedSearchCV(estimator=rsf,
                                    param_distributions=param_distributions,
                                    n_iter=50,
                                    cv=cv,
                                    scoring=rsf_score,
                                    random_state=42,
                                    n_jobs=-1,
                                    verbose=1)
    random_search.fit(X_train, y_train)

    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    best_c_index = random_search.best_score_
    print(f"Best RSF hyperparameters: {best_params} with CV mean C-index: {best_c_index:.3f}")

    # Save all CV results to CSV
    results_df = pd.DataFrame(random_search.cv_results_)
    results_csv = os.path.join(output_dir, f"{name}_rsf_hyperparameter_results.csv")
    results_df.to_csv(results_csv, index=False)
    print(f"RSF hyperparameter results saved to {results_csv}")

    # ---------------- Select simplest model using the 1 SE rule ---------------- #
    best_index = random_search.best_index_
    best_std = random_search.cv_results_['std_test_score'][best_index]
    one_se_threshold = best_c_index - best_std

    # Select candidate parameters within 1 SE
    candidates = []
    for i, mean_score in enumerate(random_search.cv_results_['mean_test_score']):
        if mean_score >= one_se_threshold:
            params = random_search.cv_results_['params'][i]
            params['mean_test_score'] = mean_score
            params['std_test_score'] = random_search.cv_results_['std_test_score'][i]
            candidates.append(params)

    one_se_candidates_sorted = sorted(candidates, key=lambda r: (r["n_estimators"], -(r["min_samples_split"] + r["min_samples_leaf"]), r["max_features"]))
    one_se_candidate = one_se_candidates_sorted[0]
    one_se_params = {"n_estimators": one_se_candidate["n_estimators"],
                    "min_samples_split": one_se_candidate["min_samples_split"],
                    "min_samples_leaf": one_se_candidate["min_samples_leaf"],
                    "max_features": one_se_candidate["max_features"]}
    print(f"1 SE RSF hyperparameters: {one_se_params} with CV mean C-index: {one_se_candidate['mean_test_score']:.3f} (threshold: {one_se_threshold:.3f})")

    # Re-fit the 1 SE model on the full training set
    one_se_model = RandomSurvivalForest(random_state=42, n_jobs=-1, **one_se_params)
    one_se_model.fit(X_train, y_train)

    # Use the best tuned model for subsequent steps
    initial_rsf = best_model
    
    # Save best model 
    best_model_file = os.path.join(output_dir, f"{name}_final_rsf_model.pkl")
    joblib.dump(initial_rsf, best_model_file)
    print(f"Best RSF model saved to {best_model_file}")
    
    # Save simplest 1 SE model
    one_se_model_file = os.path.join(output_dir, f"{name}_final_rsf_model_1se.pkl")
    joblib.dump(one_se_model, one_se_model_file)
    print(f"1 SE RSF model saved to {one_se_model_file}")

    # ---------------- Evaluate Best and 1 SE RSF models ---------------- #
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

    # ---------------- Step 1: RSF Pre-Selection with Permutation Importance ---------------- #
    step1_start = time.time()
    initial_rsf.fit(X_train, y_train)
    
    test_pred = initial_rsf.predict(X_test)
    c_index = concordance_index_censored(y_test['OS_STATUS'], y_test['OS_MONTHS'], test_pred)
    print(f"Initial RSF Test C-index: {c_index[0]:.3f}")
    train_pred = initial_rsf.predict(X_train)
    train_c_index = concordance_index_censored(y_train['OS_STATUS'], y_train['OS_MONTHS'], train_pred)
    print(f"Initial RSF Train C-index: {train_c_index[0]:.3f}")
    
    perm_result = permutation_importance(initial_rsf, X_train, y_train,
                                           scoring=lambda est, X, y: rsf_concordance_metric(y, est.predict(X)),
                                           n_repeats=5, random_state=42, n_jobs=-1)
    importances = perm_result.importances_mean
    importance_df = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": importances,
        "Std": perm_result.importances_std
    }).sort_values(by="Importance", ascending=False)
    
    preselect_csv = os.path.join(output_dir, f"{name}_rsf_preselection_importances.csv")
    importance_df.to_csv(preselect_csv, index=False)
    print(f"RSF pre-selection importances saved to {preselect_csv}")
    
    top_preselect = importance_df.head(50)
    plt.figure(figsize=(12, 8))
    plt.barh(top_preselect["Feature"][::-1], top_preselect["Importance"][::-1],
             xerr=top_preselect["Std"][::-1], color=(9/255, 117/255, 181/255))
    plt.xlabel("Permutation Importance")
    plt.title("RSF Pre-Selection (Top 50 Features)")
    plt.tight_layout()
    preselect_plot = os.path.join(output_dir, f"{name}_rsf_preselection_importances.png")
    plt.savefig(preselect_plot)
    plt.close()
    print(f"RSF pre-selection plot saved to {preselect_plot}")
    
    top_n_rsf = min(1000, len(importance_df))
    selected_features_rsf = importance_df.iloc[:top_n_rsf]["Feature"].tolist()
    print(f"RSF pre-selection: selected top {len(selected_features_rsf)} features.")
    
    X_train_rsf = X_train[selected_features_rsf]
    X_test_rsf = X_test[selected_features_rsf]
    step1_end = time.time()
    print(f"Step 1 completed in {step1_end - step1_start:.2f} seconds.")

if __name__ == "__main__":
    # Data Loading and Preprocessing for train data
    print("Loading train data from: GPL570train.csv")
    train = pd.read_csv("GPL570train.csv")
    
    print(f"Number of events in training set: {train['OS_STATUS'].sum()} | Censored cases: {train.shape[0] - train['OS_STATUS'].sum()}")
    print("Train data shape:", train.shape)
    
    # Data Loading and Preprocessing for validation data
    print("Loading validation data from: GPL570validation.csv")
    valid = pd.read_csv("GPL570validation.csv")
    
    print(f"Number of events in validation set: {valid['OS_STATUS'].sum()} | Censored cases: {valid.shape[0] - valid['OS_STATUS'].sum()}")
    print("Validation data shape:", valid.shape)
    
    # Run the RSF pipeline with provided train and validation data
    create_rsf(train, valid, 'GPL570 3-13-25 RS')