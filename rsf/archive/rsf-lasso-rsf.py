import os
import time
import pandas as pd
import numpy as np
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
import joblib

# Determine script directory for consistent path handling in a VM
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "rsf_results")
os.makedirs(output_dir, exist_ok=True)

def rsf_concordance_score(estimator, X, y):
    preds = estimator.predict(X)
    return concordance_index_censored(y['OS_STATUS'], y['OS_MONTHS'], preds)[0]

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
    
    # ---------------- Hyperparameter search with Cross Validation ---------------- #
    best_c_index = -1
    best_cv_std = None
    best_params = None
    best_model = None
    results = []
    iteration = 0              # Counter for iterations
    save_interval = 50         # Save partial results every 50 iterations
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for n_est in [100, 400, 800]:
        for min_split in [50]:
            for min_leaf in [50]:
                for max_feat in ["sqrt"]:
                    model = RandomSurvivalForest(
                        n_estimators=n_est,
                        min_samples_split=min_split,
                        min_samples_leaf=min_leaf,
                        random_state=42,
                        n_jobs=-1,
                        max_features=max_feat
                    )
                    model.fit(X_train, y_train)
                    test_pred = model.predict(X_test)
                    c_index_test = concordance_index_censored(y_test['OS_STATUS'], y_test['OS_MONTHS'], test_pred)[0]
                    
                    print(f"Iteration {iteration}: n_estimators={n_est}, min_samples_split={min_split}, min_samples_leaf={min_leaf}, max_features={max_feat}, Test C-index: {c_index_test:.3f}")
                    
                    cv_scores = []
                    for train_idx, val_idx in kf.split(X_train):
                        X_cv_train = X_train.iloc[train_idx]
                        y_cv_train = y_train[train_idx]
                        X_cv_val  = X_train.iloc[val_idx]
                        y_cv_val  = y_train[val_idx]
                        model_cv = RandomSurvivalForest(
                            n_estimators=n_est,
                            min_samples_split=min_split,
                            min_samples_leaf=min_leaf,
                            random_state=42,
                            n_jobs=-1,
                            max_features=max_feat
                        )
                        model_cv.fit(X_cv_train, y_cv_train)
                        pred_cv = model_cv.predict(X_cv_val)
                        cv_score = concordance_index_censored(
                            y_cv_val['OS_STATUS'], y_cv_val['OS_MONTHS'], pred_cv)[0]
                        cv_scores.append(cv_score)
                    cv_mean = np.mean(cv_scores)
                    cv_std = np.std(cv_scores)
                    
                    results.append({
                        "n_estimators": n_est,
                        "min_samples_split": min_split,
                        "min_samples_leaf": min_leaf,
                        "max_features": max_feat,
                        "cv_mean": cv_mean,
                        "cv_std": cv_std,
                        "test_c_index": c_index_test,
                        "model_obj": model  # store model for later use
                    })
                    print(f"CV mean: {cv_mean:.3f}, CV std: {cv_std:.3f}, Test C-index: {c_index_test:.3f}")
                    
                    if cv_mean > best_c_index:
                        best_c_index = cv_mean
                        best_cv_std = cv_std  # store the corresponding std error
                        best_params = {"n_estimators": n_est, "min_samples_split": min_split,
                                       "min_samples_leaf": min_leaf, "max_features": max_feat}
                        best_model = model
                    iteration += 1
                    if iteration % save_interval == 0:
                        partial_csv = os.path.join(output_dir, f"{name}_rsf_hyperparameter_results_partial.csv")
                        pd.DataFrame(results).to_csv(partial_csv, index=False)
                        print(f"Partial hyperparameter results saved to {partial_csv}")
    results_df = pd.DataFrame(results)
    results_csv = os.path.join(output_dir, f"{name}_rsf_hyperparameter_results.csv")
    results_df.to_csv(results_csv, index=False)
    print(f"RSF hyperparameter results saved to {results_csv}")
    print(f"Best RSF hyperparameters: {best_params} with CV mean C-index: {best_c_index:.3f}")
    
    # ---------------- Select simplest model using the 1 SE rule ---------------- #
    one_se_threshold = best_c_index - best_cv_std
    one_se_candidates = [res for res in results if res["cv_mean"] >= one_se_threshold]
    # Define simplicity: lower n_estimators is simpler; if tied, prefer a higher (min_samples_split + min_samples_leaf)
    one_se_candidates_sorted = sorted(one_se_candidates, key=lambda r: (r["n_estimators"], -(r["min_samples_split"] + r["min_samples_leaf"]), r["max_features"]))
    one_se_candidate = one_se_candidates_sorted[0]
    one_se_model = one_se_candidate["model_obj"]
    one_se_params = {"n_estimators": one_se_candidate["n_estimators"],
                     "min_samples_split": one_se_candidate["min_samples_split"],
                     "min_samples_leaf": one_se_candidate["min_samples_leaf"],
                     "max_features": one_se_candidate["max_features"]}
    print(f"1 SE RSF hyperparameters: {one_se_params} with CV mean C-index: {one_se_candidate['cv_mean']:.3f} (threshold: {one_se_threshold:.3f})")
    
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
                                           scoring=rsf_concordance_score,
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
    print("Loading train data from: alltrain.csv")
    train = pd.read_csv("allTrain.csv")
    
    print(f"Number of events in training set: {train['OS_STATUS'].sum()} | Censored cases: {train.shape[0] - train['OS_STATUS'].sum()}")
    print("Train data shape:", train.shape)
    
    # Data Loading and Preprocessing for validation data
    print("Loading validation data from: allvalidation.csv")
    valid = pd.read_csv("allValidation.csv")
    
    print(f"Number of events in validation set: {valid['OS_STATUS'].sum()} | Censored cases: {valid.shape[0] - valid['OS_STATUS'].sum()}")
    print("Validation data shape:", valid.shape)
    
    # Run the RSF pipeline with provided train and validation data
    create_rsf(train, valid, 'all 3-18-25')