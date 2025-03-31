import os
import time
import pandas as pd
import numpy as np
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored
from sklearn.model_selection import KFold
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.metrics import make_scorer
import joblib
import sys

# Redirect console output to both the terminal and a log file
class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
    def flush(self):
        for f in self.files:
            f.flush()

# Define custom concordance metric function
def rsf_concordance_metric(y, y_pred):
    return concordance_index_censored(y['OS_STATUS'], y['OS_MONTHS'], y_pred)[0]

# Function to perform nested cross-validation using HalvingGridSearchCV
def nested_cv_rsf(X, y, param_distributions):
    outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
    inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)
    outer_scores = []
    outer_best_params = []
    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X)):
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y[train_idx]
        X_test_fold = X.iloc[test_idx]
        y_test_fold = y[test_idx]

        # Set up inner CV hyperparameter search
        rsf_score = make_scorer(rsf_concordance_metric, greater_is_better=True)
        halving_search = HalvingGridSearchCV(
            estimator=RandomSurvivalForest(random_state=42, n_jobs=-1),
            param_grid=param_distributions,
            factor=2,
            cv=inner_cv,
            scoring=rsf_score,
            random_state=42,
            n_jobs=-1,
            verbose=1, 
            min_resources = 150,
            error_score=np.nan
        )
        halving_search.fit(X_train_fold, y_train_fold)
        best_estimator = halving_search.best_estimator_
        y_pred_test = best_estimator.predict(X_test_fold)
        c_index = concordance_index_censored(y_test_fold['OS_STATUS'], y_test_fold['OS_MONTHS'], y_pred_test)[0]
        y_pred_train = best_estimator.predict(X_train_fold)
        c_index_train = concordance_index_censored(y_train_fold['OS_STATUS'], y_train_fold['OS_MONTHS'], y_pred_train)[0]
        outer_scores.append(c_index)
        outer_best_params.append(halving_search.best_params_)
        print(f"Fold {fold_idx+1}: Test C-index = {c_index:.3f}")
        print(f"Fold {fold_idx+1}: Train C-index = {c_index_train:.3f}")
        print(f"Fold {fold_idx+1}: Best params = {halving_search.best_params_}")
    mean_score = np.mean(outer_scores)
    se_score = np.std(outer_scores) / np.sqrt(len(outer_scores))
    best_fold = np.argmax(outer_scores)
    return mean_score, se_score, outer_best_params[best_fold]

if __name__ == "__main__":
    # Set directories and load data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "rsf_results_optimal")
    os.makedirs(output_dir, exist_ok=True)
    
    log_file = open(os.path.join(output_dir, "LOG-rsf-feature-search.txt"), "w")
    sys.stdout = Tee(sys.stdout, log_file)
    
    train = pd.read_csv("allTrain.csv")
    valid = pd.read_csv("allValidation.csv")
    
    start = time.time()
    
    # Load preselection CSV from the 1SE model and filter features with >0 importance
    imp_csv_path = os.path.join("rsf_results", "ALL 3-29-25 RS_rsf_preselection_importances_1SE.csv")
    imp_df = pd.read_csv(imp_csv_path)
    imp_df = imp_df[imp_df["Importance"] > 0].sort_values(by="Importance", ascending=False)
    selected_features_all = imp_df["Feature"].tolist()
    
    # Define hyperparameter grid for RSF
    param_distributions = {
        "n_estimators": [500, 750, 1000],
        "min_samples_leaf": list(range(60, 81, 5)),
        "max_features": ["sqrt", "log2", 500],
        "max_depth": [10],
    }
    
    # Evaluate different percentages of the >0 importance features (from 10% to 100%)
    percentages = [0.1 * i for i in range(1, 11)]
    results = []
    for p in percentages:
        num_features = max(1, int(len(selected_features_all) * p))
        features_subset = selected_features_all[:num_features]
        print(f"\nEvaluating with {num_features} features ({p*100:.0f}%)")
        
        # Prepare training subset using the selected features plus survival columns
        train_subset = train[['OS_STATUS', 'OS_MONTHS'] + features_subset].copy()
        for col in ['Adjuvant Chemo', 'IS_MALE']:
            if col in train_subset.columns:
                train_subset[col] = train_subset[col].astype(int)
        y_train = Surv.from_dataframe('OS_STATUS', 'OS_MONTHS', train_subset)
        X_train = train_subset.drop(columns=['OS_STATUS', 'OS_MONTHS'])
        
        mean_cv_score, se_cv_score, best_params = nested_cv_rsf(X_train, y_train, param_distributions)
        results.append((p, num_features, mean_cv_score, se_cv_score, best_params))
        print(f"Percentage {p*100:.0f}% ({num_features} features): Mean CV C-index = {mean_cv_score:.3f} (SE: {se_cv_score:.3f})")
    
    # Find best model based on mean CV score
    best_result = max(results, key=lambda x: x[2])
    best_percentage, best_num_features, best_cv_score, best_cv_se, best_hyperparams = best_result
    print("\n==========================")
    print(f"Best Model - Optimal percentage: {best_percentage*100:.0f}% ({best_num_features} features)")
    print(f"Best Model - Hyperparameters: {best_hyperparams}")
    print(f"Best Model - Mean CV C-index: {best_cv_score:.3f} (SE: {best_cv_se:.3f})")
    print("==========================\n")

    # Apply 1SE rule: select the simplest model (lowest num_features) with mean CV score >= (best_cv_score - best_cv_se)
    one_se_candidates = [res for res in results if res[2] >= best_cv_score - best_cv_se]
    one_se_result = min(one_se_candidates, key=lambda x: x[1]) if one_se_candidates else best_result
    one_se_percentage, one_se_num_features, one_se_cv_score, one_se_cv_se, one_se_hyperparams = one_se_result
    print("\n==========================")
    print(f"1SE Model - Optimal percentage: {one_se_percentage*100:.0f}% ({one_se_num_features} features)")
    print(f"1SE Model - Hyperparameters: {one_se_hyperparams}")
    print(f"1SE Model - Mean CV C-index: {one_se_cv_score:.3f} (SE: {one_se_cv_se:.3f})")
    print("==========================\n")
    
    # Save results to CSV
    results_df = pd.DataFrame(results, columns=["Percentage", "Num_Features", "Mean_CV_C_index", "SE_CV_C_index", "Best_Params"])
    
    # Save
    results_csv_path = os.path.join(output_dir, "rsf_feature_selection_results.csv")
    results_df.to_csv(results_csv_path, index=False)
    print(f"Results saved to {results_csv_path}") 
    
    print(f"Total time taken: {time.time() - start:.2f} seconds")
    
    # Train final Best model on full training set using the optimal feature subset and hyperparameters
    final_features_best = selected_features_all[:best_num_features]
    train_subset_best = train[['OS_STATUS', 'OS_MONTHS'] + final_features_best].copy()
    for col in ['Adjuvant Chemo', 'IS_MALE']:
        if col in train_subset_best.columns:
            train_subset_best[col] = train_subset_best[col].astype(int)
    y_train_best = Surv.from_dataframe('OS_STATUS', 'OS_MONTHS', train_subset_best)
    X_train_best = train_subset_best.drop(columns=['OS_STATUS', 'OS_MONTHS'])
    final_model_best = RandomSurvivalForest(random_state=42, n_jobs=-1, **best_hyperparams)
    final_model_best.fit(X_train_best, y_train_best)
    
    # Evaluate Best model on the validation set
    valid_subset_best = valid[['OS_STATUS', 'OS_MONTHS'] + final_features_best].copy()
    for col in ['Adjuvant Chemo', 'IS_MALE']:
        if col in valid_subset_best.columns:
            valid_subset_best[col] = valid_subset_best[col].astype(int)
    y_valid_best = Surv.from_dataframe('OS_STATUS', 'OS_MONTHS', valid_subset_best)
    X_valid_best = valid_subset_best.drop(columns=['OS_STATUS', 'OS_MONTHS'])
    valid_pred_best = final_model_best.predict(X_valid_best)
    valid_c_index_best = concordance_index_censored(y_valid_best['OS_STATUS'], y_valid_best['OS_MONTHS'], valid_pred_best)[0]
    print(f"Validation C-index for Best model: {valid_c_index_best:.3f}")
    
    # Train final 1SE model on full training set using the optimal feature subset and hyperparameters
    final_features_1se = selected_features_all[:one_se_num_features]
    train_subset_1se = train[['OS_STATUS', 'OS_MONTHS'] + final_features_1se].copy()
    for col in ['Adjuvant Chemo', 'IS_MALE']:
        if col in train_subset_1se.columns:
            train_subset_1se[col] = train_subset_1se[col].astype(int)
    y_train_1se = Surv.from_dataframe('OS_STATUS', 'OS_MONTHS', train_subset_1se)
    X_train_1se = train_subset_1se.drop(columns=['OS_STATUS', 'OS_MONTHS'])
    final_model_1se = RandomSurvivalForest(random_state=42, n_jobs=-1, **one_se_hyperparams)
    final_model_1se.fit(X_train_1se, y_train_1se)
    
    # Evaluate 1SE model on the validation set
    valid_subset_1se = valid[['OS_STATUS', 'OS_MONTHS'] + final_features_1se].copy()
    for col in ['Adjuvant Chemo', 'IS_MALE']:
        if col in valid_subset_1se.columns:
            valid_subset_1se[col] = valid_subset_1se[col].astype(int)
    y_valid_1se = Surv.from_dataframe('OS_STATUS', 'OS_MONTHS', valid_subset_1se)
    X_valid_1se = valid_subset_1se.drop(columns=['OS_STATUS', 'OS_MONTHS'])
    valid_pred_1se = final_model_1se.predict(X_valid_1se)
    valid_c_index_1se = concordance_index_censored(y_valid_1se['OS_STATUS'], y_valid_1se['OS_MONTHS'], valid_pred_1se)[0]
    print(f"Validation C-index for 1SE model: {valid_c_index_1se:.3f}")
    
    # Save final models
    model_filename_best = os.path.join(output_dir, f"final_rsf_model_best_{int(best_percentage*100)}perc.pkl")
    joblib.dump(final_model_best, model_filename_best)
    print(f"Final Best model saved to {model_filename_best}")
    
    model_filename_1se = os.path.join(output_dir, f"final_rsf_model_1SE_{int(one_se_percentage*100)}perc.pkl")
    joblib.dump(final_model_1se, model_filename_1se)
    print(f"Final 1SE model saved to {model_filename_1se}")