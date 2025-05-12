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
from sklearn.utils import shuffle

# Determine script directory for consistent path handling in a VM
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "rsf_results_affy")
os.makedirs(output_dir, exist_ok=True)

# Create a subdirectory for trial results
trials_dir = os.path.join(output_dir, "trials")
os.makedirs(trials_dir, exist_ok=True)

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
log_file = open(os.path.join(output_dir, "rsf_run_log-Affy.txt"), "w")
sys.stdout = Tee(sys.stdout, log_file)

def rsf_concordance_metric(y, y_pred):
    return concordance_index_censored(y['OS_STATUS'], y['OS_MONTHS'], y_pred)[0]

def find_highest_completed_trial(trials_dir, name):
    """Find the highest trial number that has completed by checking saved files."""
    import glob
    import re
    
    # Look for model files which indicate completed trials
    pattern = os.path.join(trials_dir, f"{name}_final_rsf_model_1se_trial*.pkl")
    files = glob.glob(pattern)
    
    if not files:
        return 0  # No completed trials found
    
    # Extract trial numbers from filenames
    trial_numbers = []
    for file in files:
        match = re.search(r'trial(\d+)\.pkl$', file)
        if match:
            trial_numbers.append(int(match.group(1)))
    
    return max(trial_numbers) if trial_numbers else 0

def create_rsf(train_df, test_df, name):
    # Number of trials and folds for robust feature selection
    n_trials = 20
    n_folds = 3
    
    # Check for existing trial results and determine start point
    highest_completed_trial = find_highest_completed_trial(trials_dir, name)
    start_trial = highest_completed_trial + 1
    
    if start_trial > n_trials:
        print(f"All {n_trials} trials have already been completed. Proceeding to ensemble analysis.")
        # We need to load the saved trial importances
        all_trial_importances = []
        
        for trial in range(1, n_trials + 1):
            # Load each trial's feature importance results
            importance_csv = os.path.join(trials_dir, f"{name}_rsf_preselection_importances_1SE_trial{trial}.csv")
            if os.path.exists(importance_csv):
                print(f"Loading importance results for trial {trial}...")
                importance_df = pd.read_csv(importance_csv)
                all_trial_importances.append(importance_df[["Feature", "Importance", "Std", "Trial"]])
            else:
                print(f"Warning: Could not find importance results for trial {trial}")
    else:
        print(f"Auto-pickup: Found {highest_completed_trial} completed trials. Starting from trial {start_trial}.")
        
        # Store feature importances from all trials
        all_trial_importances = []
        
        # Load importance results from completed trials
        for trial in range(1, start_trial):
            importance_csv = os.path.join(trials_dir, f"{name}_rsf_preselection_importances_1SE_trial{trial}.csv")
            if os.path.exists(importance_csv):
                print(f"Loading importance results for trial {trial}...")
                importance_df = pd.read_csv(importance_csv)
                all_trial_importances.append(importance_df[["Feature", "Importance", "Std", "Trial"]])
            else:
                print(f"Warning: Could not find importance results for trial {trial}")
        
        for trial in range(start_trial, n_trials + 1):
            print(f"\n{'='*50}\nStarting Trial {trial}/{n_trials}\n{'='*50}")
            
            # Shuffle training data for this trial
            trial_train_df = shuffle(train_df, random_state=trial*42)
            
            # Create structured arrays for survival analysis
            y_train = Surv.from_dataframe('OS_STATUS', 'OS_MONTHS', trial_train_df)
            y_test  = Surv.from_dataframe('OS_STATUS', 'OS_MONTHS', test_df)
            
            # Define covariates: affy columns except survival columns
            binary_columns = ['Adjuvant Chemo', 'IS_MALE']
            trial_train_df[binary_columns] = trial_train_df[binary_columns].astype(int)
            test_df[binary_columns] = test_df[binary_columns].astype(int)
            
            for col in binary_columns:
                assert trial_train_df[col].max() <= 1 and trial_train_df[col].min() >= 0, f"{col} should only contain binary values (0/1)."
                
            continuous_columns = trial_train_df.columns.difference(['OS_STATUS', 'OS_MONTHS', *binary_columns])
            features = continuous_columns.union(binary_columns)
            
            # Create feature matrices from provided train and test data
            X_train = trial_train_df[features]
            X_test  = test_df[features]
            
            # Count number of events in training and test sets
            train_events = np.sum(y_train['OS_STATUS'])
            test_events = np.sum(y_test['OS_STATUS'])
            print(f"Number of events in trial {trial} training set: {train_events}")
            print(f"Number of events in test set: {test_events}")
            
            # ---------------- Nested Cross-Validation for Hyperparameter Tuning ---------------- #
            # Define custom scoring function for concordance index
            rsf_score = make_scorer(rsf_concordance_metric, greater_is_better=True)

            # Define parameter grid for grid search
            param_grid = {
                "n_estimators": [500, 750],
                "min_samples_leaf": [50, 60, 70, 80],    
                "max_features": ["sqrt", 500, 0.1], # 0.1 * 13062 = 1306
                "max_depth": [10],
            }

            # Set up outer and inner KFold CV for nested CV (using 3 folds instead of 5)
            outer_cv = KFold(n_splits=n_folds, shuffle=True, random_state=trial*100)
            inner_cv = KFold(n_splits=n_folds, shuffle=True, random_state=trial*100+1)

            outer_scores = []
            outer_best_params = []
            # Initialize lists to collect fold-level metrics and inner CV results
            outer_fold_metrics = []
            inner_cv_results_list = []

            print(f"Starting Nested Cross-Validation for trial {trial}...")
            outer_start_time = time.time()
            
            for fold_idx, (outer_train_idx, outer_test_idx) in enumerate(outer_cv.split(X_train)):
                print(f"[Fold {fold_idx+1}/{outer_cv.get_n_splits()}] Starting outer fold...")
                X_train_outer = X_train.iloc[outer_train_idx]
                y_train_outer = y_train[outer_train_idx]
                X_test_outer = X_train.iloc[outer_test_idx]
                y_test_outer = y_train[outer_test_idx]
                
                # Inner CV for hyperparameter tuning using GridSearchCV
                grid_search_inner = GridSearchCV(
                    estimator=RandomSurvivalForest(random_state=trial*42+fold_idx, n_jobs=-1),
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
                row_copy['trial'] = trial
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
                        'trial': trial,
                        'candidate_index': i,
                        'mean_test_score': cv_results.get('mean_test_score', [None]*n_candidates)[i],
                        'std_test_score': cv_results.get('std_test_score', [None]*n_candidates)[i],
                        'rank_test_score': cv_results.get('rank_test_score', [None]*n_candidates)[i],
                        'params': cv_results.get('params', [None]*n_candidates)[i]
                    }
                    combined_rows.append(inner_row)
            
            combined_df = pd.DataFrame(combined_rows)
            combined_csv_path = os.path.join(trials_dir, f"{name}_rsf_all_fold_results_trial{trial}_{current_date}.csv")
            combined_df.to_csv(combined_csv_path, index=False)
            print(f"Combined fold-level metrics (inner and outer) saved to {combined_csv_path}")

            # ---------------- Final Model Training using Nested CV Results ---------------- #
            # Select best hyperparameters from nested CV outer folds (fold with highest Test C-index)
            best_fold_idx = np.argmax(outer_scores)
            best_params_nested = outer_best_params[best_fold_idx]
            print(f"Selected hyperparameters from nested CV: {best_params_nested}")
            
            # Train final model on full training set using nested CV best hyperparameters
            best_model = RandomSurvivalForest(random_state=trial*42, n_jobs=-1, **best_params_nested)
            best_model.fit(X_train, y_train)
            
            # Save nested CV outer fold summary results to CSV
            cv_results_df = pd.DataFrame({
                "fold": list(range(1, len(outer_scores)+1)),
                "Test_C_index": outer_scores,
                "trial": trial,
                "Hyperparameters": outer_best_params
            })
            results_csv = os.path.join(trials_dir, f"{name}_rsf_nested_cv_results_trial{trial}.csv")
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
                # Custom sort key function to handle mixed types in max_features
                def sort_key(r):
                    # Handle n_estimators and min_samples_leaf normally
                    n_est = r["n_estimators"]
                    min_leaf = r["min_samples_leaf"]
                    
                    # For max_features, convert to a comparable form:
                    # Use a tuple with a type indicator as first element for sorting
                    max_feat = r["max_features"]
                    if isinstance(max_feat, str):
                        # Strings like "sqrt" come first
                        return (n_est, min_leaf, (0, max_feat))
                    elif isinstance(max_feat, int):
                        # Integers come second, sorted by value
                        return (n_est, min_leaf, (1, max_feat))
                    else:
                        # Floats (or other types) come last, sorted by value
                        return (n_est, min_leaf, (2, float(max_feat)))
                
                one_se_candidates_sorted = sorted(candidates, key=sort_key)
                one_se_candidate = one_se_candidates_sorted[0]
            else:
                print("No candidate model met the 1 SE threshold. Defaulting to the best nested CV model.")
                one_se_candidate = best_params_nested
                
            # Train 1 SE model on full training set using 1 SE hyperparameters
            one_se_candidate = {k: v for k, v in one_se_candidate.items() if k != "score"}
            print(f"1 SE RSF hyperparameters from nested CV: {one_se_candidate} (threshold: {one_se_threshold:.3f})")
            one_se_model = RandomSurvivalForest(random_state=trial*42, n_jobs=-1, **one_se_candidate)
            one_se_model.fit(X_train, y_train)

            # Save best model for this trial
            best_model_file = os.path.join(trials_dir, f"{name}_final_rsf_model_trial{trial}.pkl")
            joblib.dump(best_model, best_model_file)
            print(f"Best RSF model for trial {trial} saved to {best_model_file}")
            
            # Save simplest 1 SE model for this trial
            one_se_model_file = os.path.join(trials_dir, f"{name}_final_rsf_model_1se_trial{trial}.pkl")
            joblib.dump(one_se_model, one_se_model_file)
            print(f"1 SE RSF model for trial {trial} saved to {one_se_model_file}")

            # ---------------- Evaluate Best and 1 SE RSF Models ---------------- #
            best_pred_train = best_model.predict(X_train)
            best_train_c_index = concordance_index_censored(y_train['OS_STATUS'], y_train['OS_MONTHS'], best_pred_train)[0]
            best_pred_test = best_model.predict(X_test)
            best_test_c_index = concordance_index_censored(y_test['OS_STATUS'], y_test['OS_MONTHS'], best_pred_test)[0]
            print(f"Best RSF: Train C-index: {best_train_c_index:.3f}, Test C-index: {best_test_c_index:.3f}")
            
            # Calculate 1SE model performance
            one_se_pred_train = one_se_model.predict(X_train)
            one_se_train_c_index = concordance_index_censored(y_train['OS_STATUS'], y_train['OS_MONTHS'], one_se_pred_train)[0]
            one_se_pred_test = one_se_model.predict(X_test)
            one_se_test_c_index = concordance_index_censored(y_test['OS_STATUS'], y_test['OS_MONTHS'], one_se_pred_test)[0]
            print(f"1SE RSF: Train C-index: {one_se_train_c_index:.3f}, Test C-index: {one_se_test_c_index:.3f}")
            
            # ---------------- Calculate Feature Importance for 1SE Model ---------------- #
            print(f"Calculating permutation importance for 1SE model in trial {trial}...")
            perm_start_time = time.time()
            
            # Use a smaller number of iterations for permutation importance to save time
            perm_importance = permutation_importance(one_se_model, X_train, y_train,
                                       scoring=lambda est, X, y: rsf_concordance_metric(y, est.predict(X)),
                                       n_repeats=5, random_state=42, n_jobs=-1)
            
            perm_end_time = time.time()
            print(f"Permutation importance calculated in {perm_end_time - perm_start_time:.2f} seconds.")
            
            # Create and save feature importance DataFrame
            importances = pd.DataFrame({
                'Feature': X_train.columns,
                'Importance': perm_importance.importances_mean,
                'Std': perm_importance.importances_std,
                'Trial': trial
            })
            
            # Sort by importance
            importances = importances.sort_values('Importance', ascending=False)
            
            # Save feature importances for this trial
            importance_file = os.path.join(trials_dir, f"{name}_rsf_preselection_importances_1SE_trial{trial}.csv")
            importances.to_csv(importance_file, index=False)
            print(f"Feature importances for trial {trial} saved to {importance_file}")
            
            # Append to the list of all trial importances
            all_trial_importances.append(importances)
    
    # After completing all trials (or loading existing ones), combine all importance results
    if all_trial_importances:
        # Concatenate all trial importance dataframes
        all_importances_df = pd.concat(all_trial_importances, axis=0, ignore_index=True)
        
        # Calculate ensemble feature importance
        ensemble_importances = all_importances_df.groupby("Feature").agg({
            "Importance": ["mean", "std", "count"],
            "Std": "mean"
        }).reset_index()
        
        # Flatten the multi-index columns
        ensemble_importances.columns = ["Feature", "Mean_Importance", "Std_Importance", "Num_Trials", "Mean_Std"]
        ensemble_importances = ensemble_importances.sort_values(by="Mean_Importance", ascending=False)
        
        # Save the ensemble feature importance results
        ensemble_csv = os.path.join(output_dir, f"{name}_rsf_ensemble_feature_importance.csv")
        ensemble_importances.to_csv(ensemble_csv, index=False)
        print(f"Ensemble feature importance saved to {ensemble_csv}")
        
        # Plot the top features from ensemble importance
        top_ensemble = ensemble_importances.head(50)
        plt.figure(figsize=(12, 10))
        plt.barh(top_ensemble["Feature"][::-1], top_ensemble["Mean_Importance"][::-1],
                xerr=top_ensemble["Std_Importance"][::-1], color=(9/255, 117/255, 181/255))
        plt.xlabel("Ensemble Permutation Importance")
        plt.title(f"RSF Ensemble Feature Importance (Top 50 Features, {n_trials} Trials)")
        plt.tight_layout()
        ensemble_plot = os.path.join(output_dir, f"{name}_rsf_ensemble_feature_importance.png")
        plt.savefig(ensemble_plot)
        plt.close()
        
        # Select top features based on ensemble importance
        top_n_rsf = min(1000, len(ensemble_importances))
        selected_features_ensemble = ensemble_importances.iloc[:top_n_rsf]["Feature"].tolist()
        print(f"Ensemble feature selection: selected top {len(selected_features_ensemble)} features.")
        
        # Save the selected feature list
        selected_features_file = os.path.join(output_dir, f"{name}_rsf_ensemble_selected_features.txt")
        with open(selected_features_file, 'w') as f:
            for feature in selected_features_ensemble:
                f.write(f"{feature}\n")
        print(f"Selected features saved to {selected_features_file}")
        
        # Return the ensemble selected features for further analysis
        return selected_features_ensemble
    else:
        print("WARNING: No feature importance data was collected or loaded. Cannot perform ensemble analysis.")
        return []

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
    
    selected_features = create_rsf(train, valid, 'Affy RS')
    print(f"Ensemble feature selection process complete. Selected {len(selected_features)} features for future analysis.")