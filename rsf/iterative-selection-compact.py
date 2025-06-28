#!/usr/bin/env python3
"""
Streamlined Iterative Feature Selection with Nested 5-Fold Cross-Validation

Key features:
- Fixed hyperparameters: n_estimators=750, min_samples_leaf=70
- Search space: max_features in ["sqrt", "log2", 0.1, 0.2, 0.5] 
- Iterative removal of bottom 20% features based on median importance
- Early stopping when performance plateaus
"""

import os
import pandas as pd
import numpy as np
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold, GridSearchCV
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

def rsf_concordance_metric(y, y_pred):
    return concordance_index_censored(y['OS_STATUS'], y['OS_MONTHS'], y_pred)[0]

def run_nested_cv(X, y, features, iteration):
    """Run nested 5-fold CV with specified features."""
    print(f"\nIteration {iteration}: Testing {len(features)} features")
    
    X_subset = X[features]
    
    # Fixed hyperparameters + search space
    param_grid = {
        "n_estimators": [750],
        "min_samples_leaf": [70], 
        "max_features": ["sqrt", "log2", 0.1, 0.2, 0.5]
    }
    
    outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
    inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scorer = make_scorer(rsf_concordance_metric, greater_is_better=True)
    
    test_scores = []
    all_importances = []
    
    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X_subset)):
        # Split data
        X_train, X_test = X_subset.iloc[train_idx], X_subset.iloc[test_idx]
        y_train_fold, y_test_fold = y[train_idx], y[test_idx]
        
        # Inner CV for hyperparameter selection
        grid_search = GridSearchCV(
            RandomSurvivalForest(random_state=42, n_jobs=-1),
            param_grid, cv=inner_cv, scoring=scorer, n_jobs=-1
        )
        grid_search.fit(X_train, y_train_fold)
        
        # Evaluate on outer test fold
        best_model = grid_search.best_estimator_
        test_pred = best_model.predict(X_test)
        test_c_index = concordance_index_censored(
            y_test_fold['OS_STATUS'], y_test_fold['OS_MONTHS'], test_pred
        )[0]
        test_scores.append(test_c_index)
        
        # Compute feature importance on test fold
        perm_imp = permutation_importance(
            best_model, X_test, y_test_fold,
            scoring=lambda est, X, y: rsf_concordance_metric(y, est.predict(X)),
            n_repeats=3, random_state=42, n_jobs=-1
        )
        
        fold_importance = pd.DataFrame({
            'Feature': features,
            'Importance': perm_imp.importances_mean,
            'Fold': fold + 1
        })
        all_importances.append(fold_importance)
        
        print(f"  Fold {fold+1}: C-index = {test_c_index:.4f}, Best params = {grid_search.best_params_}")
    
    # Aggregate results
    mean_score = np.mean(test_scores)
    std_score = np.std(test_scores)
    
    # Calculate median importance across folds
    combined_imp = pd.concat(all_importances, ignore_index=True)
    median_imp = combined_imp.groupby('Feature')['Importance'].median().reset_index()
    median_imp = median_imp.sort_values('Importance', ascending=False)
    
    print(f"  Result: {mean_score:.4f} ± {std_score:.4f}")
    
    return {
        'iteration': iteration,
        'n_features': len(features),
        'mean_score': mean_score,
        'std_score': std_score,
        'test_scores': test_scores,
        'feature_importance': median_imp,
        'features': features.copy()
    }

def main():
    print("ITERATIVE FEATURE SELECTION - NESTED 5-FOLD CV")
    print("=" * 50)
    
    # Load and combine data
    train_orig = pd.read_csv("affyTrain.csv")
    valid_orig = pd.read_csv("affyValidation.csv") 
    data = pd.concat([train_orig, valid_orig], ignore_index=True)
    
    # Preprocess
    data['Adjuvant Chemo'] = data['Adjuvant Chemo'].replace({'OBS': 0, 'ACT': 1})
    data[['Adjuvant Chemo', 'IS_MALE']] = data[['Adjuvant Chemo', 'IS_MALE']].astype(int)
    
    # Create survival target
    y = Surv.from_dataframe('OS_STATUS', 'OS_MONTHS', data)
    
    # Define features
    feature_cols = data.columns.difference(['OS_STATUS', 'OS_MONTHS']).tolist()
    X = data[feature_cols]
    
    print(f"Dataset: {data.shape[0]} samples, {len(feature_cols)} features")
    print(f"Events: {data['OS_STATUS'].sum()}, Censored: {data.shape[0] - data['OS_STATUS'].sum()}")
    
    # Initialize with pre-ranked features if available
    try:
        preranked = pd.read_csv("rsf_results_affy/Affy RS_combined_fold_permutation_importance_median_ranked.csv")
        current_features = [f for f in preranked['Feature'].tolist() if f in feature_cols]
        print(f"Starting with {len(current_features)} pre-ranked features")
    except:
        current_features = feature_cols.copy()
        print("Using all features (no pre-ranking found)")
    
    # Iterative feature selection
    results = []
    best_score = -np.inf
    patience_counter = 0
    min_features = 2
    
    iteration = 1
    while len(current_features) >= min_features:
        # Run nested CV
        result = run_nested_cv(X, y, current_features, iteration)
        results.append(result)
        
        # Check for improvement
        if result['mean_score'] > best_score:
            best_score = result['mean_score']
            patience_counter = 0
            print(f"  *** NEW BEST: {best_score:.4f} ***")
        else:
            patience_counter += 1
            print(f"  No improvement (patience: {patience_counter}/3)")
        
        # Early stopping
        if patience_counter >= 3:
            print("\nEarly stopping: No improvement for 3 iterations")
            break
        
        # Remove bottom 20% features
        n_remove = max(1, int(len(current_features) * 0.2))
        n_keep = len(current_features) - n_remove
        
        if n_keep < min_features:
            print(f"Would go below minimum {min_features} features, stopping")
            break
        
        # Keep top features by median importance
        top_features = result['feature_importance'].head(n_keep)['Feature'].tolist()
        removed_features = result['feature_importance'].tail(n_remove)['Feature'].tolist()
        
        print(f"  Removing {n_remove} features: {removed_features[:3]}{'...' if n_remove > 3 else ''}")
        current_features = top_features
        iteration += 1
    
    # Summary
    print("\n" + "=" * 50)
    print("RESULTS SUMMARY")
    print("=" * 50)
    
    # Create summary dataframe  
    summary = pd.DataFrame([{
        'Iteration': r['iteration'],
        'N_Features': r['n_features'],
        'Mean_C_Index': r['mean_score'],
        'Std_C_Index': r['std_score']
    } for r in results])
    
    print(summary.to_string(index=False, float_format='%.4f'))
    
    # Find best iteration
    best_idx = summary['Mean_C_Index'].idxmax()
    best_result = results[best_idx]
    
    print(f"\nBest performance: {best_result['mean_score']:.4f} ± {best_result['std_score']:.4f}")
    print(f"Best iteration: {best_result['iteration']} ({best_result['n_features']} features)")
    
    # Save results
    output_dir = "rsf_results_affy/iterative_selection"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save summary
    summary.to_csv(f"{output_dir}/iteration_summary.csv", index=False)
    
    # Save best features
    best_features = pd.DataFrame({
        'Feature': best_result['features'],
        'Rank': range(1, len(best_result['features']) + 1)
    })
    best_features.to_csv(f"{output_dir}/best_features.csv", index=False)
    
    # Plot performance
    plt.figure(figsize=(10, 6))
    plt.errorbar(summary['Iteration'], summary['Mean_C_Index'], 
                yerr=summary['Std_C_Index'], marker='o', capsize=5)
    plt.xlabel('Iteration')
    plt.ylabel('Cross-Validation C-Index')
    plt.title('Feature Selection Performance')
    plt.grid(True, alpha=0.3)
    
    # Add feature count on top
    ax2 = plt.gca().twiny()
    ax2.set_xlim(plt.gca().get_xlim())
    ax2.set_xticks(summary['Iteration'])
    ax2.set_xticklabels(summary['N_Features'])
    ax2.set_xlabel('Number of Features')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/performance_plot.png", dpi=300)
    plt.show()
    
    print(f"\nResults saved to: {output_dir}/")

if __name__ == "__main__":
    main()
