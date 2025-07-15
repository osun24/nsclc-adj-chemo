"""
Iterative Feature Selection with Nested Cross-Validation for Random Survival Forest

This script performs iterative feature selection by:
1. Running nested 5-fold cross-validation 
2. Using pre-ranked features from previous permutation importance analysis
3. Removing bottom 20% of features in each iteration based on ranking
4. Repeating until performance degrades or minimum features reached

Uses pre-ranked features instead of recalculating feature importance.
Fixed hyperparameters: 70 min_samples_leaf, 750 n_estimators
Search space: max_features in ["sqrt", "log2", 0.1, 0.2, 0.5]
"""

import os
import time
import datetime
import pandas as pd
import numpy as np
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored
import matplotlib.pyplot as plt
# Feature importance removed - using pre-ranked features
import joblib
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

# Setup directories and logging
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "rsf_results_affy", "iterative_feature_selection")
os.makedirs(output_dir, exist_ok=True)

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
log_file = open(os.path.join(output_dir, f"{current_date}_iterative_feature_selection_log.txt"), "w")
sys.stdout = Tee(sys.stdout, log_file)

def rsf_concordance_metric(y, y_pred):
    """Custom concordance metric for RSF."""
    return concordance_index_censored(y['OS_STATUS'], y['OS_MONTHS'], y_pred)[0]

def run_nested_cv_iteration(X_train, y_train, features_to_use, iteration_num):
    """
    Run nested cross-validation for one iteration of feature selection.
    
    Returns:
    --------
    dict: Results including CV scores and best parameters
    """
    print(f"\n{'='*60}")
    print(f"ITERATION {iteration_num}: Using {len(features_to_use)} features")
    print(f"{'='*60}")
    
    # Filter features
    X_filtered = X_train[features_to_use]
    
    # Define parameter grid (fixed: n_estimators=750, min_samples_leaf=70)
    param_grid = {
        "n_estimators": [500, 750], 
        "min_samples_leaf": [70, 90], 
        "max_features": ["sqrt", 0.1, 0.2, 0.5],
        "max_depth": [3]
    }
    
    # Setup cross-validation
    outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
    inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    rsf_score = make_scorer(rsf_concordance_metric, greater_is_better=True)
    
    # Store results
    outer_scores = []
    outer_train_scores = []
    best_params_list = []
    
    print("Starting Nested Cross-Validation...")
    
    for fold_idx, (outer_train_idx, outer_test_idx) in enumerate(outer_cv.split(X_filtered)):
        # Calculate actual number of model fits for this fold
        n_param_combinations = len(param_grid["n_estimators"]) * len(param_grid["min_samples_leaf"]) * len(param_grid["max_features"])
        inner_cv_fits = n_param_combinations * inner_cv.get_n_splits()
        total_fits_this_fold = inner_cv_fits + 1  # +1 for final model
        
        print(f"[Fold {fold_idx+1}/5] Training samples: {len(outer_train_idx)}, Test samples: {len(outer_test_idx)}")
        print(f"[Fold {fold_idx+1}/5] Model fits: {total_fits_this_fold} ({inner_cv_fits} inner CV + 1 final)")
        
        # Split data
        X_train_outer = X_filtered.iloc[outer_train_idx]
        y_train_outer = y_train[outer_train_idx]
        X_test_outer = X_filtered.iloc[outer_test_idx]
        y_test_outer = y_train[outer_test_idx]
        
        # Inner CV for hyperparameter tuning
        grid_search = GridSearchCV(
            estimator=RandomSurvivalForest(random_state=42, n_jobs=-1),
            param_grid=param_grid,
            cv=inner_cv,
            scoring=rsf_score,
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_train_outer, y_train_outer)
        best_params = grid_search.best_params_
        best_params_list.append(best_params)
        
        # Train final model with best params
        final_model = RandomSurvivalForest(random_state=42, n_jobs=-1, **best_params)
        final_model.fit(X_train_outer, y_train_outer)
        
        # Evaluate on outer test and train sets
        y_pred_test = final_model.predict(X_test_outer)
        y_pred_train = final_model.predict(X_train_outer)
        
        test_c_index = concordance_index_censored(
            y_test_outer['OS_STATUS'], y_test_outer['OS_MONTHS'], y_pred_test
        )[0]
        train_c_index = concordance_index_censored(
            y_train_outer['OS_STATUS'], y_train_outer['OS_MONTHS'], y_pred_train
        )[0]
        
        outer_scores.append(test_c_index)
        outer_train_scores.append(train_c_index)
        
        print(f"[Fold {fold_idx+1}] Best params: {best_params}")
        print(f"[Fold {fold_idx+1}] Test C-index: {test_c_index:.4f}, Train C-index: {train_c_index:.4f}")
    
    # Aggregate results
    mean_test_score = np.mean(outer_scores)
    std_test_score = np.std(outer_scores)
    mean_train_score = np.mean(outer_train_scores)
    std_train_score = np.std(outer_train_scores)
    
    print(f"\nIteration {iteration_num} Results:")
    print(f"Test C-index: {mean_test_score:.4f} ± {std_test_score:.4f}")
    print(f"Train C-index: {mean_train_score:.4f} ± {std_train_score:.4f}")
    
    # Save iteration results
    iteration_results = {
        'iteration': iteration_num,
        'n_features': len(features_to_use),
        'features_used': features_to_use.copy(),
        'mean_test_score': mean_test_score,
        'std_test_score': std_test_score,
        'mean_train_score': mean_train_score,
        'std_train_score': std_train_score,
        'test_scores': outer_scores,
        'train_scores': outer_train_scores,
        'best_params_list': best_params_list
    }
    
    # Save detailed results for this iteration
    iteration_dir = os.path.join(output_dir, f"iteration_{iteration_num:02d}")
    os.makedirs(iteration_dir, exist_ok=True)
    
    return iteration_results

def main():
    """Main function to run iterative feature selection."""
    print("="*80)
    print("ITERATIVE FEATURE SELECTION WITH NESTED CROSS-VALIDATION")
    print("="*80)
    
    # Load data
    print("Loading training data...")
    train_orig = pd.read_csv("affyTrain.csv")
    valid_orig = pd.read_csv("affyValidation.csv")
    
    # Combine train and validation
    train = pd.concat([train_orig, valid_orig], axis=0, ignore_index=True)
    
    print(f"Combined training data shape: {train.shape}")
    print(f"Number of events: {train['OS_STATUS'].sum()}")
    print(f"Number of censored: {train.shape[0] - train['OS_STATUS'].sum()}")
    
    # Preprocess data
    train['Adjuvant Chemo'] = train['Adjuvant Chemo'].replace({'OBS': 0, 'ACT': 1})
    
    # Create survival arrays
    y_train = Surv.from_dataframe('OS_STATUS', 'OS_MONTHS', train)
    
    # Define features (exclude survival columns)
    binary_columns = ['Adjuvant Chemo', 'IS_MALE']
    train[binary_columns] = train[binary_columns].astype(int)
    
    continuous_columns = train.columns.difference(['OS_STATUS', 'OS_MONTHS'] + binary_columns)
    all_features = list(continuous_columns) + binary_columns
    
    X_train = train[all_features]
    
    print(f"Total features available: {len(all_features)}")
    
    # Define the features to force-include in every model iteration
    forced_features_base = [
        'Age', 'Stage_IA', 'Stage_IB', 'Stage_II', 'Stage_III', 'Stage_IV', 'Stage_Unknown',
        'Histology_Adenocarcinoma', 'Histology_Adenosquamous Carcinoma', 
        'Histology_Large Cell Carcinoma', 'Histology_Squamous Cell Carcinoma',
        'Race_African American', 'Race_Asian', 'Race_Caucasian', 
        'Race_Native Hawaiian or Other Pacific Islander', 'Race_Unknown',
        'Smoked?_No', 'Smoked?_Unknown', 'Smoked?_Yes', "Adjuvant Chemo", "IS_MALE"
    ]
    # Ensure forced features actually exist in the dataset
    forced_features = [f for f in forced_features_base if f in all_features]
    print(f"\nForcing inclusion of {len(forced_features)} clinical features.")

    # Initialize with median-ranked features from previous analysis
    print("\nLoading pre-ranked features from median importance...")
    try:
        preranked_features_df = pd.read_csv("cph/cox_prescreen_results/20250713_significant_interactions_p_0.05_selection_results.csv") # 20x 10-fold CV results
        preranked_features = preranked_features_df['selected_gene'].tolist()
        
        # Separate pre-ranked features into clinical (forced) and genomic (selectable)
        selectable_features = [f for f in preranked_features if f not in forced_features and f in all_features]
        
        print(f"Using {len(selectable_features)} pre-ranked selectable (genomic) features.")
        print(f"The iterative process will operate on these {len(selectable_features)} features.")

    except FileNotFoundError:
        print("Pre-ranked features not found, using all non-forced features as selectable.")
        selectable_features = [f for f in all_features if f not in forced_features]
    
    # Iterative feature selection
    current_selectable_features = selectable_features.copy()
    all_iterations = []
    best_score = -np.inf
    best_iteration = 0
    patience = 5  # Stop if no improvement for 5 iterations (adjustable: 2-5 depending on computational budget)
    no_improvement_count = 0
    min_genes_to_keep = 3  # Minimum number of selectable (gene) features to keep
    
    iteration = 10
    
    while len(current_selectable_features) >= min_genes_to_keep:
        # Combine forced features with the current set of selectable features
        current_features_for_iteration = forced_features + current_selectable_features
        
        print(f"\nStarting iteration {iteration} with {len(current_features_for_iteration)} total features ({len(forced_features)} forced + {len(current_selectable_features)} selectable)...")
        
        # Run nested CV for current feature set
        results = run_nested_cv_iteration(X_train, y_train, current_features_for_iteration, iteration)
        all_iterations.append(results)
        
        # Check if this is the best performance so far
        current_score = results['mean_test_score']
        if current_score > best_score:
            best_score = current_score
            best_iteration = iteration
            no_improvement_count = 0
            print(f"*** NEW BEST PERFORMANCE: {best_score:.4f} ***")
        else:
            no_improvement_count += 1
            print(f"No improvement for {no_improvement_count} iterations")
        
        # Early stopping check
        if no_improvement_count >= patience:
            print(f"\nEarly stopping: No improvement for {patience} iterations")
            break
        
        # Remove bottom 20% of selectable features based on pre-ranked order
        n_features_to_remove = max(1, int(len(current_selectable_features) * 0.2))
        n_features_to_keep = len(current_selectable_features) - n_features_to_remove
        
        if n_features_to_keep < min_genes_to_keep:
            print(f"Would reduce to {n_features_to_keep} selectable features, below minimum of {min_genes_to_keep}")
            break
        
        # Keep top selectable features based on pre-ranked order
        top_selectable_features = current_selectable_features[:n_features_to_keep]
        
        print(f"Removing {n_features_to_remove} selectable features (keeping top {n_features_to_keep})")
        
        current_selectable_features = top_selectable_features
        iteration += 1
    
    # Summary and final results
    print("\n" + "="*80)
    print("ITERATIVE FEATURE SELECTION COMPLETE")
    print("="*80)
    
    # Create summary dataframe
    summary_data = []
    for i, result in enumerate(all_iterations):
        summary_data.append({
            'Iteration': result['iteration'],
            'N_Features': result['n_features'],
            'Mean_Test_C_Index': result['mean_test_score'],
            'Std_Test_C_Index': result['std_test_score'],
            'Mean_Train_C_Index': result['mean_train_score'],
            'Std_Train_C_Index': result['std_train_score']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(output_dir, f"{current_date}_iteration_summary.csv"), index=False)
    
    print("\nIteration Summary:")
    print(summary_df.to_string(index=False))
    
    print(f"\nBest performance: {best_score:.4f} at iteration {best_iteration}")
    print(f"Best feature set size: {all_iterations[best_iteration-1]['n_features']}")
    
    # Plot 1: Performance over iterations
    plt.figure(figsize=(12, 6))
    plt.errorbar(summary_df['Iteration'], summary_df['Mean_Test_C_Index'], 
                yerr=summary_df['Std_Test_C_Index'], marker='o', capsize=5, label='Test C-Index', 
                linewidth=2, markersize=8)
    plt.errorbar(summary_df['Iteration'], summary_df['Mean_Train_C_Index'], 
                yerr=summary_df['Std_Train_C_Index'], marker='s', capsize=5, label='Train C-Index',
                linewidth=2, markersize=8)
    plt.xlabel('Iteration')
    plt.ylabel('C-Index')
    plt.title('Performance vs Iteration (with Feature Count)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add feature count as secondary x-axis
    ax2 = plt.gca().twiny()
    ax2.set_xlim(plt.gca().get_xlim())
    ax2.set_xticks(summary_df['Iteration'])
    ax2.set_xticklabels(summary_df['N_Features'])
    ax2.set_xlabel('Number of Features')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{current_date}_performance_vs_iterations.png"), dpi=300)
    plt.close()
    
    # Plot 2: Performance relative to number of features (primary focus)
    plt.figure(figsize=(12, 8))
    
    # Main plot with number of features on x-axis
    plt.errorbar(summary_df['N_Features'], summary_df['Mean_Test_C_Index'], 
                yerr=summary_df['Std_Test_C_Index'], marker='o', capsize=5, label='Test C-Index',
                linewidth=2, markersize=10, color='#1f77b4')
    plt.errorbar(summary_df['N_Features'], summary_df['Mean_Train_C_Index'], 
                yerr=summary_df['Std_Train_C_Index'], marker='s', capsize=5, label='Train C-Index',
                linewidth=2, markersize=10, color='#ff7f0e')
    
    # Highlight the best performance point
    best_idx = summary_df['Mean_Test_C_Index'].idxmax()
    best_n_features = summary_df.loc[best_idx, 'N_Features']
    best_score = summary_df.loc[best_idx, 'Mean_Test_C_Index']
    plt.scatter(best_n_features, best_score, color='red', s=150, marker='*', 
                label=f'Best: {best_score:.4f} ({best_n_features} features)', zorder=5)
    
    plt.xlabel('Number of Features', fontsize=12)
    plt.ylabel('C-Index', fontsize=12)
    plt.title('Performance vs Number of Features\n(Iterative Feature Selection)', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Add annotations for each point showing iteration number
    for i, row in summary_df.iterrows():
        plt.annotate(f'Iter {int(row["Iteration"])}', 
                    (row['N_Features'], row['Mean_Test_C_Index']),
                    xytext=(5, 5), textcoords='offset points', 
                    fontsize=9, alpha=0.7)
    
    # Invert x-axis to show feature reduction from left to right
    plt.gca().invert_xaxis()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{current_date}_performance_vs_n_features.png"), dpi=300)
    plt.close()
    
    # Plot 3: Combined view with dual y-axes
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # Plot C-index performance
    line1 = ax1.errorbar(summary_df['N_Features'], summary_df['Mean_Test_C_Index'], 
                        yerr=summary_df['Std_Test_C_Index'], marker='o', capsize=5, 
                        color='#1f77b4', linewidth=2, markersize=8, label='Test C-Index')
    line2 = ax1.errorbar(summary_df['N_Features'], summary_df['Mean_Train_C_Index'], 
                        yerr=summary_df['Std_Train_C_Index'], marker='s', capsize=5, 
                        color='#ff7f0e', linewidth=2, markersize=8, label='Train C-Index')
    
    ax1.set_xlabel('Number of Features', fontsize=12)
    ax1.set_ylabel('C-Index', color='black', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True, alpha=0.3)
    ax1.invert_xaxis()
    
    # Add second y-axis for overfitting gap
    ax2 = ax1.twinx()
    overfitting_gap = summary_df['Mean_Train_C_Index'] - summary_df['Mean_Test_C_Index']
    line3 = ax2.plot(summary_df['N_Features'], overfitting_gap, 
                     color='red', marker='^', linewidth=2, markersize=6, 
                     label='Overfitting Gap (Train - Test)', alpha=0.7)
    
    ax2.set_ylabel('Overfitting Gap (Train - Test C-Index)', color='red', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='red')
    
    # Combine legends
    lines = [line1, line2, line3[0]]
    labels = ['Test C-Index', 'Train C-Index', 'Overfitting Gap']
    ax1.legend(lines, labels, loc='upper left', fontsize=11)
    
    plt.title('Feature Selection: Performance and Overfitting Analysis', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{current_date}_performance_overfitting_analysis.png"), dpi=300)
    plt.close()
    
    # Train final best model on full dataset
    print("\n" + "="*60)
    print("TRAINING FINAL MODEL WITH BEST FEATURE SET")
    print("="*60)
    
    best_result = all_iterations[best_iteration-1]
    best_features = best_result['features_used']
    
    print(f"Training final model with {len(best_features)} features from iteration {best_iteration}")
    print(f"Best cross-validation performance: {best_score:.4f}")
    
    # Get most common hyperparameters from best iteration
    best_params_list = best_result['best_params_list']
    
    # Find most frequent hyperparameter combination
    from collections import Counter
    param_combinations = [tuple(sorted(params.items())) for params in best_params_list]
    most_common_params = Counter(param_combinations).most_common(1)[0][0]
    best_params_final = dict(most_common_params)
    
    print(f"Using most frequent hyperparameters from best iteration: {best_params_final}")
    
    # Filter training data to best features
    X_train_best = X_train[best_features]
    
    # Train final model
    print("Training final Random Survival Forest...")
    final_rsf = RandomSurvivalForest(random_state=42, n_jobs=-1, **best_params_final)
    final_rsf.fit(X_train_best, y_train)
    
    # Evaluate on training set
    train_pred = final_rsf.predict(X_train_best)
    final_train_c_index = concordance_index_censored(
        y_train['OS_STATUS'], y_train['OS_MONTHS'], train_pred
    )[0]
    
    print(f"Final model training C-index: {final_train_c_index:.4f}")
    
    # Load and evaluate on test set
    print("Loading and evaluating on test set...")
    try:
        test_data = pd.read_csv("affyTest.csv")
        test_data['Adjuvant Chemo'] = test_data['Adjuvant Chemo'].replace({'OBS': 0, 'ACT': 1})
        test_data[binary_columns] = test_data[binary_columns].astype(int)
        
        # Filter test data to best features only
        available_test_features = [f for f in best_features if f in test_data.columns]
        missing_features = [f for f in best_features if f not in test_data.columns]
        
        if missing_features:
            print(f"Warning: {len(missing_features)} features missing in test set: {missing_features[:5]}...")
            print(f"Using {len(available_test_features)} available features for test evaluation")
            
            # Retrain model with only available features if needed
            if len(available_test_features) < len(best_features):
                print("Retraining model with available features only...")
                X_train_available = X_train[available_test_features]
                final_rsf_test = RandomSurvivalForest(random_state=42, n_jobs=-1, **best_params_final)
                final_rsf_test.fit(X_train_available, y_train)
                
                # Update model for saving
                final_rsf = final_rsf_test
                best_features = available_test_features
                X_train_best = X_train_available
        else:
            available_test_features = best_features
        
        # Prepare test data
        y_test = Surv.from_dataframe('OS_STATUS', 'OS_MONTHS', test_data)
        X_test_best = test_data[available_test_features]
        
        # Evaluate on test set
        test_pred = final_rsf.predict(X_test_best)
        final_test_c_index = concordance_index_censored(
            y_test['OS_STATUS'], y_test['OS_MONTHS'], test_pred
        )[0]
        
        print(f"Final model test C-index: {final_test_c_index:.4f}")
        print(f"Test set: {test_data.shape[0]} samples, {test_data['OS_STATUS'].sum()} events")
        
    except FileNotFoundError:
        print("Test file not found, skipping test evaluation")
        final_test_c_index = None
        y_test = None
        X_test_best = None
    
    # Save the final model
    model_filename = f"{current_date}_best_rsf_model_iter_{best_iteration}_{len(best_features)}_features.pkl"
    model_path = os.path.join(output_dir, model_filename)
    
    print(f"\nSaving final model to: {model_filename}")
    joblib.dump(final_rsf, model_path)
    
    # Create model specification file
    model_spec = {
        'model_file': model_filename,
        'model_path': model_path,
        'best_iteration': best_iteration,
        'n_features': len(best_features),
        'features': best_features,
        'hyperparameters': best_params_final,
        'cv_performance': {
            'mean_test_c_index': best_score,
            'std_test_c_index': all_iterations[best_iteration-1]['std_test_score']
        },
        'final_performance': {
            'train_c_index': final_train_c_index,
            'test_c_index': final_test_c_index if final_test_c_index else 'Not available'
        },
        'training_data_shape': X_train.shape,
        'date_created': current_date
    }
    
    # Save model specification
    spec_filename = f"{current_date}_best_model_specification.txt"
    spec_path = os.path.join(output_dir, spec_filename)
    
    with open(spec_path, 'w') as f:
        f.write("BEST MODEL SPECIFICATION\n")
        f.write("="*50 + "\n\n")
        f.write(f"Model File: {model_spec['model_file']}\n")
        f.write(f"Best Iteration: {model_spec['best_iteration']}\n")
        f.write(f"Number of Features: {model_spec['n_features']}\n")
        f.write(f"Training Data Shape: {model_spec['training_data_shape']}\n")
        f.write(f"Date Created: {model_spec['date_created']}\n\n")
        
        f.write("HYPERPARAMETERS:\n")
        f.write("-" * 20 + "\n")
        for key, value in model_spec['hyperparameters'].items():
            f.write(f"{key}: {value}\n")
        
        f.write(f"\nCROSS-VALIDATION PERFORMANCE:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Mean Test C-Index: {model_spec['cv_performance']['mean_test_c_index']:.4f}\n")
        f.write(f"Std Test C-Index: {model_spec['cv_performance']['std_test_c_index']:.4f}\n")
        
        f.write(f"\nFINAL MODEL PERFORMANCE:\n")
        f.write("-" * 25 + "\n")
        f.write(f"Training C-Index: {model_spec['final_performance']['train_c_index']:.4f}\n")
        f.write(f"Test C-Index: {model_spec['final_performance']['test_c_index']}\n")
        
        f.write(f"\nFEATURES ({len(best_features)}):\n")
        f.write("-" * 15 + "\n")
        for i, feature in enumerate(best_features, 1):
            f.write(f"{i:3d}. {feature}\n")
    
    print(f"Model specification saved to: {spec_filename}")
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL MODEL SUMMARY")
    print("="*60)
    print(f"✓ Best iteration: {best_iteration}")
    print(f"✓ Features selected: {len(best_features)}")
    print(f"✓ Cross-validation C-index: {best_score:.4f}")
    print(f"✓ Final training C-index: {final_train_c_index:.4f}")
    if final_test_c_index:
        print(f"✓ Final test C-index: {final_test_c_index:.4f}")
    print(f"✓ Model saved: {model_filename}")
    print(f"✓ Specification saved: {spec_filename}")
    
    # Save best feature set
    best_result = all_iterations[best_iteration-1]
    best_features_df = pd.DataFrame({
        'Feature': best_result['features_used'],
        'Rank': range(1, len(best_result['features_used']) + 1)
    })
    best_features_df.to_csv(
        os.path.join(output_dir, f"{current_date}_best_features_iteration_{best_iteration}.csv"), 
        index=False
    )
    
    print(f"\nBest features saved to: {current_date}_best_features_iteration_{best_iteration}.csv")
    print(f"Results saved to: {output_dir}")
    
    # Close log file
    sys.stdout = sys.__stdout__
    log_file.close()

if __name__ == "__main__":
    main()
