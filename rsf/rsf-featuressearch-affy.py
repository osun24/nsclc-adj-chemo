import os
import time
import pandas as pd
import numpy as np
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.metrics import make_scorer
import joblib
import sys
import datetime

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
    outer_cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)  # Repeated CV for robustness
    inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)
    outer_scores = []
    outer_best_params = []
    fold_metrics = []  # New list to store fold-level metrics
    inner_cv_results_list = []  # New list to store detailed inner CV results per outer fold
    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X)):
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y[train_idx]
        X_test_fold = X.iloc[test_idx]
        y_test_fold = y[test_idx]

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
            min_resources=250,
            error_score=np.nan
        )
        
        # Configure joblib to use a longer timeout for workers
        with joblib.parallel_backend('loky', timeout=300):  # 5-minute timeout
            halving_search.fit(X_train_fold, y_train_fold)

        # Save detailed inner CV results for the current fold
        inner_cv_results_list.append({
            'fold': fold_idx + 1,
            'cv_results': halving_search.cv_results_
        })

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
        # Save fold-level metrics for later analysis
        fold_metrics.append({
            'fold': fold_idx + 1,
            'test_c_index': c_index,
            'train_c_index': c_index_train,
            'best_params': halving_search.best_params_
        })
    mean_score = np.mean(outer_scores)
    se_score = np.std(outer_scores) / np.sqrt(len(outer_scores))
    best_fold = np.argmax(outer_scores)
    return mean_score, se_score, outer_best_params[best_fold], fold_metrics, inner_cv_results_list

if __name__ == "__main__":
    # Set directories and load data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "rsf_results_optimal")
    os.makedirs(output_dir, exist_ok=True)
    
    current_date = datetime.datetime.now().strftime("%Y%m%d")  # Added current date for file naming
    
    log_file = open(os.path.join(output_dir, f"{current_date}_LOG-rsf-feature-search.txt"), "w")
    sys.stdout = Tee(sys.stdout, log_file)
    
    print("Loading train data from: affyTrain.csv")
    train = pd.read_csv("affyTrain.csv")
    
    print(f"Number of events in training set: {train['OS_STATUS'].sum()} | Censored cases: {train.shape[0] - train['OS_STATUS'].sum()}")
    print("Train data shape:", train.shape)
    
    print("Loading validation data from: affyValidation.csv")
    valid = pd.read_csv("affyValidation.csv")
    
    print(f"Number of events in validation set: {valid['OS_STATUS'].sum()} | Censored cases: {valid.shape[0] - valid['OS_STATUS'].sum()}")
    print("Validation data shape:", valid.shape)
    
    start = time.time()
    
    # Load preselection CSV from the 1SE model and filter features with >0 importance
    imp_csv_path = os.path.join("rsf/rsf_results_affy", "Affy RS_rsf_ensemble_feature_importance.csv")
    imp_df = pd.read_csv(imp_csv_path)
    imp_df = imp_df[imp_df["Importance"] > 0].sort_values(by="Importance", ascending=False)
    selected_features_all = imp_df["Feature"].tolist()
    
    # Handle Adjuvant Chemo column with dummies to have OBS as baseline
    if 'Adjuvant Chemo' in train.columns:
        train['Adjuvant Chemo'] = train['Adjuvant Chemo'].replace({'OBS': 0, 'ACT': 1})
    
    if 'Adjuvant Chemo' in valid.columns:
        valid['Adjuvant Chemo'] = valid['Adjuvant Chemo'].replace({'OBS': 0, 'ACT': 1})
    
    # Define hyperparameter grid for RSF (modified for the Affy dataset)
    param_distributions = {
        "n_estimators": [500, 625, 750, 875], # previously 1000
        "min_samples_leaf": [50, 60, 65, 70, 80], # 90, 100
        "max_features": ["sqrt", "log2", 500, 0.1],
        "max_depth": [10],
    }
    
    all_fold_metrics = []  # Initialize list to collect all fold-level metrics
    all_inner_cv_results = []  # Initialize list to collect all inner CV results
    
    # Evaluate different percentages of the >0 importance features (from 1% to 100%)
    number_of_features = [5000]
    results = []
    for p in number_of_features:
        num_features = p
        features_subset = selected_features_all[:num_features]
        print(f"\nEvaluating with {num_features} features ({p*100:.0f}%)")
        
        # Prepare training subset using the selected features plus survival columns
        train_subset = train[['OS_STATUS', 'OS_MONTHS'] + features_subset].copy()
        # Make sure to do the string replacement before converting to int
        for col in ['Adjuvant Chemo', 'IS_MALE']:
            if col in train_subset.columns:
                # First replace string values with numeric values
                if col == 'Adjuvant Chemo':
                    train_subset[col] = train_subset[col].replace({'OBS': 0, 'ACT': 1})
                # Then convert to int
                train_subset[col] = train_subset[col].astype(int)
        y_train = Surv.from_dataframe('OS_STATUS', 'OS_MONTHS', train_subset)
        X_train = train_subset.drop(columns=['OS_STATUS', 'OS_MONTHS'])
        
        mean_cv_score, se_cv_score, best_params, fold_metrics, inner_cv_results = nested_cv_rsf(X_train, y_train, param_distributions)
        
        # Train a candidate model on full training set using the best parameters from CV
        final_model_candidate = RandomSurvivalForest(random_state=42, n_jobs=-1, **best_params)
        final_model_candidate.fit(X_train, y_train)
        
        # Evaluate candidate model on the validation set
        valid_subset_candidate = valid[['OS_STATUS', 'OS_MONTHS'] + features_subset].copy()
        for col in ['Adjuvant Chemo', 'IS_MALE']:
            if col in valid_subset_candidate.columns:
                # First replace string values with numeric values 
                if col == 'Adjuvant Chemo':
                    valid_subset_candidate[col] = valid_subset_candidate[col].replace({'OBS': 0, 'ACT': 1})
                # Then convert to int
                valid_subset_candidate[col] = valid_subset_candidate[col].astype(int)
        y_valid_candidate = Surv.from_dataframe('OS_STATUS', 'OS_MONTHS', valid_subset_candidate)
        X_valid_candidate = valid_subset_candidate.drop(columns=['OS_STATUS', 'OS_MONTHS'])
        valid_pred_candidate = final_model_candidate.predict(X_valid_candidate)
        valid_c_index_candidate = concordance_index_censored(y_valid_candidate['OS_STATUS'], y_valid_candidate['OS_MONTHS'], valid_pred_candidate)[0]
        print(f"Validation C-index for candidate model with {p} features: {valid_c_index_candidate:.3f}")
        
        results.append((p, num_features, mean_cv_score, se_cv_score, best_params, valid_c_index_candidate))
        # Annotate each fold's metrics with the current percentage and feature count
        for metric in fold_metrics:
            metric['percentage'] = p
            metric['num_features'] = num_features
        all_fold_metrics.extend(fold_metrics)
        # Also annotate and collect inner CV detailed results
        for item in inner_cv_results:
            item['percentage'] = p
            item['num_features'] = num_features
        all_inner_cv_results.extend(inner_cv_results)
        print(f"Percentage {p*100:.0f}% ({num_features} features): Mean CV C-index = {mean_cv_score:.3f} (SE: {se_cv_score:.3f})")
    
    # Find best model based on mean CV score
    best_result = max(results, key=lambda x: x[2])
    best_percentage, best_num_features, best_cv_score, best_cv_se, best_hyperparams, best_valid_c_index = best_result
    print("\n==========================")
    print(f"Best Model - Optimal percentage: {best_percentage*100:.0f}% ({best_num_features} features)")
    print(f"Best Model - Hyperparameters: {best_hyperparams}")
    print(f"Best Model - Mean CV C-index: {best_cv_score:.3f} (SE: {best_cv_se:.3f})")
    print(f"Best Model - Validation C-index: {best_valid_c_index:.3f}")
    print("==========================\n")

    # Apply 1SE rule: select the simplest model (lowest num_features) with mean CV score >= (best_cv_score - best_cv_se)
    one_se_candidates = [res for res in results if res[2] >= best_cv_score - best_cv_se]
    one_se_result = min(one_se_candidates, key=lambda x: x[1]) if one_se_candidates else best_result
    one_se_percentage, one_se_num_features, one_se_cv_score, one_se_cv_se, one_se_hyperparams, one_se_valid_c_index = one_se_result
    print("\n==========================")
    print(f"1SE Model - Optimal percentage: {one_se_percentage*100:.0f}% ({one_se_num_features} features)")
    print(f"1SE Model - Hyperparameters: {one_se_hyperparams}")
    print(f"1SE Model - Mean CV C-index: {one_se_cv_score:.3f} (SE: {one_se_cv_se:.3f})")
    print(f"1SE Model - Validation C-index: {one_se_valid_c_index:.3f}")
    print("==========================\n")
    
    # Save results to CSV
    results_df = pd.DataFrame(results, columns=["Percentage", "Num_Features", "Mean_CV_C_index", "SE_CV_C_index", "Best_Params", "Validation_C_index"])
    
    # Save
    results_csv_path = os.path.join(output_dir, f"{current_date}_rsf_feature_selection_results.csv")
    results_df.to_csv(results_csv_path, index=False)
    print(f"Results saved to {results_csv_path}") 
    
    # Combine outer and inner fold metrics into a single CSV file
    combined_rows = []
    
    # Process outer fold metrics (fold_metrics from all_fold_metrics)
    for row in all_fold_metrics:
        row_copy = row.copy()
        row_copy['fold_type'] = 'outer'
        # Rename 'fold' to 'outer_fold' for clarity
        row_copy['outer_fold'] = row_copy.pop('fold')
        combined_rows.append(row_copy)
    
    # Process inner CV results (from all_inner_cv_results)
    for item in all_inner_cv_results:
        outer_fold = item.get('fold')
        percentage = item.get('percentage')
        num_features = item.get('num_features')
        cv_results = item.get('cv_results', {})
        n_candidates = len(cv_results.get('mean_test_score', []))
        for i in range(n_candidates):
            inner_row = {
                'fold_type': 'inner',
                'outer_fold': outer_fold,
                'candidate_index': i,
                'percentage': percentage,
                'num_features': num_features,
                'mean_test_score': cv_results.get('mean_test_score', [None]*n_candidates)[i],
                'std_test_score': cv_results.get('std_test_score', [None]*n_candidates)[i],
                'rank_test_score': cv_results.get('rank_test_score', [None]*n_candidates)[i],
                'params': cv_results.get('params', [None]*n_candidates)[i]
            }
            combined_rows.append(inner_row)
    
    combined_df = pd.DataFrame(combined_rows)
    combined_csv_path = os.path.join(output_dir, f"{current_date}_rsf_all_fold_results.csv")
    combined_df.to_csv(combined_csv_path, index=False)
    print(f"Combined fold-level metrics (inner and outer) saved to {combined_csv_path}")
    
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
    model_filename_best = os.path.join(output_dir, f"{current_date}_final_rsf_model_best_{int(best_percentage*100)}perc.pkl")
    joblib.dump(final_model_best, model_filename_best)
    print(f"Final Best model saved to {model_filename_best}")
    
    model_filename_1se = os.path.join(output_dir, f"{current_date}_final_rsf_model_1SE_{int(one_se_percentage*100)}perc.pkl")
    joblib.dump(final_model_1se, model_filename_1se)
    print(f"Final 1SE model saved to {model_filename_1se}")