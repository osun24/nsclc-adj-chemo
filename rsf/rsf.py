import os
import time
import pandas as pd
import numpy as np
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
import joblib
from mpl_toolkits.mplot3d import Axes3D    # new import for 3D plotting

def rsf_concordance_score(estimator, X, y):
    preds = estimator.predict(X)
    return concordance_index_censored(y['OS_STATUS'], y['OS_MONTHS'], preds)[0]

def create_rsf(train_df, valid_df, name, trees=1000):
    # Subset using preselected covariates
    selected_csv = pd.read_csv('rsf/rsf_results_GPL570 3-13-25 RS_rsf_preselection_importances.csv', index_col=0)
    """[:10]
    Validation C-index: 0.728
    Train C-index: 0.821"""
    """[:10]
    n_estimators=500, min_samples_split=35, min_samples_leaf=30,
                               random_state=42, n_jobs=-1, max_features="sqrt"
    Validation C-index: 0.732
Train C-index: 0.816"""
    top_100 = selected_csv.index[:4]
    
    """[:0] - no genetic
    Validation C-index: 0.697
    Train C-index: 0.686"""
    clinical_vars = ["Adjuvant Chemo", "Age", "Stage", "IS_MALE", "Histology", "Race", "Smoked?"] + \
                    [col for col in train_df.columns if col.startswith('Race')] + \
                    [col for col in train_df.columns if col.startswith('Histology')] + \
                    [col for col in train_df.columns if col.startswith('Smoked?')]
    selected_covariates = list(set(top_100).union(set(clinical_vars)))
    selected_covariates = [col for col in selected_covariates if col in train_df.columns]
    
    # Keep only the selected columns in both train and validation datasets
    train_df = train_df[['OS_STATUS', 'OS_MONTHS'] + selected_covariates]
    valid_df = valid_df[['OS_STATUS', 'OS_MONTHS'] + selected_covariates]
    
    # Create structured arrays for survival analysis
    y_train = Surv.from_dataframe('OS_STATUS', 'OS_MONTHS', train_df)
    y_valid = Surv.from_dataframe('OS_STATUS', 'OS_MONTHS', valid_df)
    
    # Process binary columns (convert and assert binary values)
    binary_columns = ['Adjuvant Chemo', 'IS_MALE']
    for df_ in [train_df, valid_df]:
        df_[binary_columns] = df_[binary_columns].astype(int)
        for col in binary_columns:
            assert df_[col].max() <= 1 and df_[col].min() >= 0, f"{col} should only contain binary values (0/1)."
    
    features = train_df.columns.difference(['OS_STATUS', 'OS_MONTHS'])
    X_train = train_df[features]
    X_valid = valid_df[features]
    
    # Fit the Random Survival Forest model
    rsf = RandomSurvivalForest(n_estimators=500, min_samples_split=35, min_samples_leaf=30,
                               random_state=42, n_jobs=-1, max_features="sqrt")
    rsf.fit(X_train, y_train)
    
    # Evaluate model performance on validation and training sets
    c_index_valid = concordance_index_censored(y_valid['OS_STATUS'], y_valid['OS_MONTHS'], rsf.predict(X_valid))
    print(f"Validation C-index: {c_index_valid[0]:.3f}")
    train_c_index = concordance_index_censored(y_train['OS_STATUS'], y_train['OS_MONTHS'], rsf.predict(X_train))
    print(f"Train C-index: {train_c_index[0]:.3f}")
    
    # Save the RSF model to a file
    #joblib.dump(rsf, f'rsf_model-{trees}-c{c_index_valid[0]:.3f}.pkl')
    
    # Permutation importance on the validation set
    result = permutation_importance(rsf, X_valid, y_valid, n_repeats=5, random_state=42)
    importances_df = pd.DataFrame({
        "importances_mean": result.importances_mean,
        "importances_std": result.importances_std
    }, index=X_valid.columns).sort_values(by="importances_mean", ascending=False)
    print(importances_df)
    
    # Save full importances to CSV
    importances_df.to_csv(f'rsf/rsf_results_{name}_rsf_importances.csv')
    
    # Plot top 30 feature importances
    top_importances = importances_df.head(30).sort_values(by="importances_mean", ascending=True)
    plt.figure(figsize=(12, 8))
    plt.rcParams.update({'font.size': 14})
    plt.barh(top_importances.index, top_importances["importances_mean"],
             xerr=top_importances["importances_std"])
    plt.xlabel("Permutation Feature Importance")
    plt.ylabel("Feature")
    plt.title(f"Random Survival Forest: Feature Importances (Validation C-index: {c_index_valid[0]:.3f})")
    plt.tight_layout()
    name_clean = name.replace(' ', '-')
    plt.savefig(f'rsf-importances-{name_clean}-{trees}trees-validation.png')
    plt.show()

def create_rsf_custom(train_df, valid_df, covariates, name, trees=1000):
    # Select only the survival columns and the passed covariates
    train_df = train_df[['OS_STATUS', 'OS_MONTHS'] + covariates]
    valid_df = valid_df[['OS_STATUS', 'OS_MONTHS'] + covariates]
    
    # Process binary columns if they exist
    binary_columns = ['Adjuvant Chemo', 'IS_MALE']
    for df_ in [train_df, valid_df]:
        common_bins = [col for col in binary_columns if col in df_.columns]
        if common_bins:
            df_[common_bins] = df_[common_bins].astype(int)
            for col in common_bins:
                assert df_[col].max() <= 1 and df_[col].min() >= 0, f"{col} should only contain binary values (0/1)."
                
    # Create structured arrays for survival analysis
    from sksurv.util import Surv  # in case not imported at module top
    y_train = Surv.from_dataframe('OS_STATUS', 'OS_MONTHS', train_df)
    y_valid = Surv.from_dataframe('OS_STATUS', 'OS_MONTHS', valid_df)
    
    features = train_df.columns.difference(['OS_STATUS', 'OS_MONTHS'])
    X_train = train_df[features]
    X_valid = valid_df[features]
    
    # Fit the Random Survival Forest model with passed trees argument
    from sksurv.ensemble import RandomSurvivalForest
    rsf = RandomSurvivalForest(n_estimators=trees, min_samples_split=35, min_samples_leaf=30,
                               random_state=42, n_jobs=-1, max_features="sqrt")
    rsf.fit(X_train, y_train)
    
    # Evaluate model performance on validation and training sets
    from sksurv.metrics import concordance_index_censored
    c_index_valid = concordance_index_censored(y_valid['OS_STATUS'], y_valid['OS_MONTHS'], rsf.predict(X_valid))
    print(f"Validation C-index: {c_index_valid[0]:.3f}")
    train_c_index = concordance_index_censored(y_train['OS_STATUS'], y_train['OS_MONTHS'], rsf.predict(X_train))
    print(f"Train C-index: {train_c_index[0]:.3f}")
    
    # Permutation importance on the validation set
    from sklearn.inspection import permutation_importance
    result = permutation_importance(rsf, X_valid, y_valid, n_repeats=5, random_state=42)
    importances_df = pd.DataFrame({
        "importances_mean": result.importances_mean,
        "importances_std": result.importances_std
    }, index=X_valid.columns).sort_values(by="importances_mean", ascending=False)
    print(importances_df)
    
    # Save full importances to CSV
    importances_df.to_csv(f'rsf/rsf_results_{name}_rsf_custom_importances.csv')
    
    # Plot top 30 feature importances
    top_importances = importances_df.head(30).sort_values(by="importances_mean", ascending=True)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 8))
    plt.rcParams.update({'font.size': 14})
    plt.barh(top_importances.index, top_importances["importances_mean"],
             xerr=top_importances["importances_std"])
    plt.xlabel("Permutation Feature Importance")
    plt.ylabel("Feature")
    plt.title(f"RSF Custom: Feature Importances (Validation C-index: {c_index_valid[0]:.3f}, Train: {train_c_index[0]:.3f})")
    plt.tight_layout()
    name_clean = name.replace(' ', '-')
    plt.savefig(f'rsf-importances-{name_clean}-{trees}trees-validation.png')
    plt.show()

def evaluate_features_range(train_df, valid_df, selected_features, name):
    # Filter out features not present in the DataFrame columns to avoid KeyError
    selected_features = [f for f in selected_features if f in train_df.columns]
    
    # Define clinical variables to be included by default
    clinical_vars = ["Adjuvant Chemo", "Age", "Stage", "IS_MALE", "Histology", "Race", "Smoked?"] + \
                    [col for col in train_df.columns if col.startswith('Race')] + \
                    [col for col in train_df.columns if col.startswith('Histology')] + \
                    [col for col in train_df.columns if col.startswith('Smoked?')]
    clinical_vars = [var for var in clinical_vars if var in train_df.columns]
    
    # Define feature counts: 0-25 (each integer) then 25-250 in 10 equally spaced steps
    range1 = list(range(0, 26))  # start at 1 to avoid empty feature set
    range2 = list(np.linspace(25, 1000, num=10, dtype=int))
    candidate_counts = sorted(set(range1 + range2))
    
    feature_counts = []
    train_c_indices = []
    valid_c_indices = []
    
    print("Evaluating RSF model performance over different number of features:")
    for count in candidate_counts:
        # Use first 'count' features from ranking and include clinical variables
        features_subset = selected_features[:count]
        features_subset = list(set(features_subset).union(set(clinical_vars)))
        
        # Subset data (include survival columns) and ensure binary processing as needed
        train_subset = train_df[['OS_STATUS', 'OS_MONTHS'] + features_subset].copy()
        valid_subset = valid_df[['OS_STATUS', 'OS_MONTHS'] + features_subset].copy()
        binary_columns = ['Adjuvant Chemo', 'IS_MALE']
        for df_ in [train_subset, valid_subset]:
            common_bins = [col for col in binary_columns if col in df_.columns]
            if common_bins:
                df_[common_bins] = df_[common_bins].astype(int)
        
        # Create structured arrays for survival analysis
        from sksurv.util import Surv
        y_train = Surv.from_dataframe('OS_STATUS', 'OS_MONTHS', train_subset)
        y_valid = Surv.from_dataframe('OS_STATUS', 'OS_MONTHS', valid_subset)
        features = train_subset.columns.difference(['OS_STATUS', 'OS_MONTHS'])
        X_train = train_subset[features]
        X_valid = valid_subset[features]
        
        # Use fixed RSF parameters (as in create_rsf)
        from sksurv.ensemble import RandomSurvivalForest
        rsf = RandomSurvivalForest(n_estimators=500, min_samples_split=35,
                                   min_samples_leaf=30, random_state=42,
                                   n_jobs=-1, max_features="sqrt")
        rsf.fit(X_train, y_train)
        from sksurv.metrics import concordance_index_censored
        train_score = concordance_index_censored(y_train['OS_STATUS'], y_train['OS_MONTHS'], rsf.predict(X_train))[0]
        valid_score = concordance_index_censored(y_valid['OS_STATUS'], y_valid['OS_MONTHS'], rsf.predict(X_valid))[0]
        
        print(f"Features: {count}, Train C-index: {train_score:.3f}, Validation C-index: {valid_score:.3f}")
        feature_counts.append(count)
        train_c_indices.append(train_score)
        valid_c_indices.append(valid_score)
    
    # Plot the performance
    plt.figure(figsize=(10,6))
    plt.plot(feature_counts, train_c_indices, marker='o', label='Train C-index')
    plt.plot(feature_counts, valid_c_indices, marker='s', label='Validation C-index')
    plt.xlabel("Number of Features")
    plt.ylabel("C-index")
    plt.title("RSF Performance vs. Number of Features")
    plt.legend()
    plt.tight_layout()
    plot_filename = f"rsf/feature_range_{name}.png"
    plt.savefig(plot_filename)
    plt.show()
    print(f"Plot saved to {plot_filename}")

def open_rsf(train_df, valid_df, filepath, feature_selected=False):
    # Load the saved RSF model
    rsf = joblib.load(filepath)
    
    if feature_selected:
        # Subset using preselected covariates
        selected_csv = pd.read_csv('rsf/rsf_results_GPL570_rsf_preselection_importances.csv', index_col=0)
        top_100 = selected_csv.index[:75]
        clinical_vars = ["Adjuvant Chemo", "Age", "Stage", "IS_MALE", "Histology", "Race", "Smoked?"] + \
                        [col for col in train_df.columns if col.startswith('Race')] + \
                        [col for col in train_df.columns if col.startswith('Histology')] + \
                        [col for col in train_df.columns if col.startswith('Smoked?')]
        selected_covariates = list(set(top_100[:77]).union(set(clinical_vars)))
        selected_covariates = [col for col in selected_covariates if col in train_df.columns]
        
        # Keep only the selected columns in both train and validation datasets
        train_df = train_df[['OS_STATUS', 'OS_MONTHS'] + selected_covariates]
        valid_df = valid_df[['OS_STATUS', 'OS_MONTHS'] + selected_covariates]
    else:
        # Use all columns (assumes OS_STATUS and OS_MONTHS are present)
        train_df = train_df.copy()
        valid_df = valid_df.copy()
    
    # Process binary columns if they exist
    binary_columns = ['Adjuvant Chemo', 'IS_MALE']
    for df_ in [train_df, valid_df]:
        common_bin = [col for col in binary_columns if col in df_.columns]
        if common_bin:
            df_[common_bin] = df_[common_bin].astype(int)
    
    features = train_df.columns.difference(['OS_STATUS', 'OS_MONTHS'])
    X_train = train_df[features]
    X_valid = valid_df[features]
    
    # Create structured arrays for survival analysis
    y_train = Surv.from_dataframe('OS_STATUS', 'OS_MONTHS', train_df)
    y_valid = Surv.from_dataframe('OS_STATUS', 'OS_MONTHS', valid_df)
    
    # Get predictions and compute concordance indices
    train_pred = rsf.predict(X_train)
    valid_pred = rsf.predict(X_valid)
    train_c_index = concordance_index_censored(y_train['OS_STATUS'], y_train['OS_MONTHS'], train_pred)
    valid_c_index = concordance_index_censored(y_valid['OS_STATUS'], y_valid['OS_MONTHS'], valid_pred)
    
    # Print model training parameters
    print(f"Estimators: {rsf.n_estimators}")
    print(f"Min samples split: {rsf.min_samples_split}")
    print(f"Min samples leaf: {rsf.min_samples_leaf}")
    print(f"Max features: {rsf.max_features}")
    
    print(f"Loaded RSF Model from {filepath}")
    print(f"Train C-index: {train_c_index[0]:.3f}")
    print(f"Validation C-index: {valid_c_index[0]:.3f}")
    
    # Run permutation feature importance
    perm_result = permutation_importance(rsf, X_train, y_train,
                                           scoring=rsf_concordance_score,
                                           n_repeats=5, random_state=42, n_jobs=-1)
    importances = perm_result.importances_mean
    importance_df = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": importances,
        "Std": perm_result.importances_std
    }).sort_values(by="Importance", ascending=False)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "rsf_results")
    os.makedirs(output_dir, exist_ok=True)
    
    preselect_csv = os.path.join(output_dir, f"1SE_rsf_preselection_importances.csv")
    importance_df.to_csv(preselect_csv, index=False)
    print(f"RSF pre-selection importances saved to {preselect_csv}")
    
    top_preselect = importance_df.head(50)
    plt.figure(figsize=(12, 8))
    plt.barh(top_preselect["Feature"][::-1], top_preselect["Importance"][::-1],
             xerr=top_preselect["Std"][::-1], color=(9/255, 117/255, 181/255))
    plt.xlabel("Permutation Importance")
    plt.title("RSF Pre-Selection (Top 50 Features)")
    plt.tight_layout()
    preselect_plot = os.path.join(output_dir, f"1SE_rsf_preselection_importances.png")
    plt.savefig(preselect_plot)
    plt.close()
    print(f"RSF pre-selection plot saved to {preselect_plot}")

def search_optimal_feature_permutation(train_df, valid_df, selected_features, name, max_features=50, candidate_limit=50):
    """
    Greedy forward selection of features.
    Starting with a base set, iteratively add the feature from the top candidate_limit features
    (i.e. the pre-ranked list restricted to candidate_limit features, minus those already selected)
    that improves the RSF validation C-index the most.
    
    Parameters:
      train_df, valid_df: DataFrames with 'OS_STATUS' and 'OS_MONTHS'
      selected_features: Pre-ranked list of features (subset of columns)
      name: used for saving the selection plot
      max_features: maximum number of features to add
      candidate_limit: the number of top features to consider at each step
      
    Returns:
      best_subset: list of selected features
      best_valid_cindex: corresponding validation C-index
      history: list of tuples (step, feature_added, validation C-index, train C-index)
    """
    # Filter to only features present in the DataFrame and use only the top candidate_limit.
    selected_features = [f for f in selected_features if f in train_df.columns]
    
    best_subset = ["Adjuvant Chemo"]
    best_valid_cindex = 0.0
    history = []  # (step, feature_added, valid_c, train_c)
    
    def evaluate_features(features):
        train_subset = train_df[['OS_STATUS', 'OS_MONTHS'] + features].copy()
        valid_subset = valid_df[['OS_STATUS', 'OS_MONTHS'] + features].copy()
        binary_columns = ['Adjuvant Chemo', 'IS_MALE']
        for df_ in [train_subset, valid_subset]:
            common_bins = [col for col in binary_columns if col in df_.columns]
            if common_bins:
                df_[common_bins] = df_[common_bins].astype(int)
        from sksurv.util import Surv
        y_train = Surv.from_dataframe('OS_STATUS', 'OS_MONTHS', train_subset)
        y_valid = Surv.from_dataframe('OS_STATUS', 'OS_MONTHS', valid_subset)
        features_used = train_subset.columns.difference(['OS_STATUS', 'OS_MONTHS'])
        X_train = train_subset[features_used]
        X_valid = valid_subset[features_used]
        from sksurv.ensemble import RandomSurvivalForest
        rsf = RandomSurvivalForest(n_estimators=500, min_samples_split=35, 
                                   min_samples_leaf=30, random_state=42, 
                                   n_jobs=-1, max_features="sqrt")
        rsf.fit(X_train, y_train)
        from sksurv.metrics import concordance_index_censored
        valid_c = concordance_index_censored(y_valid['OS_STATUS'], y_valid['OS_MONTHS'], rsf.predict(X_valid))[0]
        train_c = concordance_index_censored(y_train['OS_STATUS'], y_train['OS_MONTHS'], rsf.predict(X_train))[0]
        return valid_c, train_c

    print("Starting greedy forward selection of features:")
    for i in range(max_features):
        # At each step, restrict candidates to the top candidate_limit features minus those already selected.
        candidate_pool = [f for f in selected_features[:candidate_limit] if f not in best_subset]
        if not candidate_pool:
            print("No remaining candidates to evaluate.")
            break
        
        improvement = False
        best_candidate = None
        best_candidate_valid = best_valid_cindex
        best_candidate_train = 0.0
        
        print(f"Step {i+1}/{max_features}: Evaluating {len(candidate_pool)} candidate features...")
        for feature in candidate_pool:
            candidate_features = best_subset + [feature]
            valid_c, train_c = evaluate_features(candidate_features)
            print(f"Evaluated feature '{feature}': Validation C-index = {valid_c:.3f}, Train C-index = {train_c:.3f}")
            if valid_c > best_candidate_valid:
                best_candidate_valid = valid_c
                best_candidate_train = train_c
                best_candidate = feature
                improvement = True
        if improvement and best_candidate is not None:
            best_subset.append(best_candidate)
            best_valid_cindex = best_candidate_valid
            history.append((len(best_subset), best_candidate, best_candidate_valid, best_candidate_train))
            print(f"Step {len(best_subset)}: Added feature '{best_candidate}' -> Validation C-index: {best_candidate_valid:.3f}, Train C-index: {best_candidate_train:.3f}")
        else:
            print(f"No improvement at step {i+1}. Terminating selection.")
            break

    print("Final selected features:", best_subset)
    
    # Final permutation importance on the selected features
    final_features = best_subset
    train_subset = train_df[['OS_STATUS', 'OS_MONTHS'] + final_features].copy()
    valid_subset = valid_df[['OS_STATUS', 'OS_MONTHS'] + final_features].copy()
    for df_ in [train_subset, valid_subset]:
        common_bins = [col for col in ['Adjuvant Chemo', 'IS_MALE'] if col in df_.columns]
        if common_bins:
            df_[common_bins] = df_[common_bins].astype(int)
    from sksurv.util import Surv
    y_train = Surv.from_dataframe('OS_STATUS', 'OS_MONTHS', train_subset)
    y_valid = Surv.from_dataframe('OS_STATUS', 'OS_MONTHS', valid_subset)
    features_used = train_subset.columns.difference(['OS_STATUS', 'OS_MONTHS'])
    X_train = train_subset[features_used]
    X_valid = valid_subset[features_used]
    from sksurv.ensemble import RandomSurvivalForest
    final_rsf = RandomSurvivalForest(n_estimators=500, min_samples_split=35, 
                                     min_samples_leaf=30, random_state=42, 
                                     n_jobs=-1, max_features="sqrt")
    final_rsf.fit(X_train, y_train)
    from sklearn.inspection import permutation_importance
    perm_result = permutation_importance(final_rsf, X_valid, y_valid, n_repeats=5, random_state=42)
    imp_df = pd.DataFrame({
        "importances_mean": perm_result.importances_mean,
        "importances_std": perm_result.importances_std
    }, index=X_valid.columns).sort_values(by="importances_mean", ascending=False)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 8))
    plt.barh(imp_df.index, imp_df["importances_mean"], xerr=imp_df["importances_std"])
    plt.xlabel("Permutation Feature Importance")
    plt.title(f"Final RSF Permutation Feature Importances (Validation C-index: {best_valid_cindex:.3f})")
    plt.tight_layout()
    perm_plot_filename = f"rsf/optimal_permutation_importances_{name}.png"
    plt.savefig(perm_plot_filename)
    plt.show()
    print(f"Final permutation importances plot saved to {perm_plot_filename}")
    
    return best_subset, best_valid_cindex, history

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
    #create_rsf(train, valid, 'GPL570', trees=100)
    
    # Example usage of open_rsf:
    # Update the filepath below to the actual saved model filename if different.
    model_filepath = "/Users/owensun/Downloads/code/nsclc-adj-chemo/rsf/rsf_results_GPL570 3-15-25 RS_final_rsf_model_1se.pkl"
    # Set feature_selected=True to use preselected features, or False to use all features.
    #open_rsf(train, valid, model_filepath, feature_selected=False)
    
    #create_rsf(train, valid, 'GPL570-315', trees=500)
    selected_csv = pd.read_csv('rsf/rsf_results_GPL570 3-13-25 RS_rsf_preselection_importances.csv', index_col=0)
    selected_features = list(selected_csv.index)
    #evaluate_features_range(train, valid,selected_features, 'GPL570-315')
    
    # Create RSF model with custom covariates
    selected_covariates = ['Stage_IA', 'Smoked?_Unknown', 'RTL3', 'Age', 'GDF9', 'LOC105375172', 'LINC01352', 'NDFIP2', 'Smoked?_Yes', 'MAN2B2', "Adjuvant Chemo"]
    #Validation C-index: 0.776 | Train C-index: 0.785
    
    # Final selected genomic features: ['Stage_IA', 'GDF9', 'LOC105375172', 'RTL3', 'NDFIP2', 'LOC105378231']
    
    # selected_covariates = ["Stage_IA", "Smoked?_Unknown", "Age", "RTL3", "LOC105375172", "Smoked?_Yes", "IQCF6", "Adjuvant Chemo"]
    #Validation C-index: 0.753 | Train C-index: 0.758

    #create_rsf_custom(train, valid, selected_covariates, 'GPL570-O1 Selected', trees=500)
    
    # Define clinical variables to always include
    clinical_vars = ["Adjuvant Chemo", "Age", "Stage", "IS_MALE", "Histology", "Race", "Smoked?"] + \
                   [col for col in train.columns if col.startswith('Race')] + \
                   [col for col in train.columns if col.startswith('Histology')] + \
                   [col for col in train.columns if col.startswith('Smoked?')]
    clinical_vars = [var for var in clinical_vars if var in train.columns]
    
    genomic = ['GDF9', 'LOC105375172', 'RTL3', 'NDFIP2', 'LOC105378231']
    selected_covariates = list(set(genomic).union(set(clinical_vars)))
    
    selected_features = selected_covariates
    
    selected_features = ['Smoked?_Unknown', 'Smoked?_Yes', 'Age', 'GDF9', 'RTL3', 'LOC105375172', 'NDFIP2', 'IS_MALE', "Adjuvant Chemo"]
    
    #Validation C-index: 0.774 | Train C-index: 0.762
    #create_rsf_custom(train, valid, selected_features, 'GPL570-GREEDY FEATURES', trees=500)
    
    #create_rsf_custom(train, valid, selected_covariates, 'GPL570-CLINICAL + GREEDY SELECTION', trees=500)
    
    #Final selected features: ['Adjuvant Chemo', 'Smoked?_Unknown', 'LOC105375172', 'Age', 'RTL3', 'GDF9', 'NDFIP2', 'IS_MALE'] 
    
    
    search_optimal_feature_permutation(train, valid, selected_features, 'GPL570-SEARCH', max_features=50)


    """Loading train data from: GPL570train.csv
Number of events in training set: 122 | Censored cases: 224
Train data shape: (346, 21379)
Loading validation data from: GPL570validation.csv
Number of events in validation set: 41 | Censored cases: 75
Validation data shape: (116, 21379)
Loaded RSF Model from rsf/rsf_results_GPL570 3-13-25 RS_final_rsf_model_1se.pkl
Train C-index: 0.914
Validation C-index: 0.620
    """
    
"""                                   importances_mean  importances_std
Stage_IA                                   0.067403         0.016835
Smoked?_Unknown                            0.040974         0.008458
RTL3                                       0.029286         0.011824
Age                                        0.027857         0.012742
GDF9                                       0.015065         0.003403
LOC105375172                               0.011948         0.006165
LINC01352                                  0.005519         0.003753
NDFIP2                                     0.004091         0.003427
Smoked?_Yes                                0.002078         0.001650
MAN2B2                                     0.001753         0.002610
Smoked?_No                                 0.000390         0.001747
Race_Caucasian                             0.000000         0.000000
Histology_Adenosquamous Carcinoma          0.000000         0.000000
Adjuvant Chemo                             0.000000         0.000000
Race_African American                      0.000000         0.000000
Histology_Large Cell Carcinoma             0.000000         0.000000
Race_Asian                                 0.000000         0.000000
Race_Unknown                               0.000000         0.000000
IS_MALE                                   -0.002662         0.005036
CASP1                                     -0.003312         0.003947
IQCF6                                     -0.003961         0.001711
Histology_Adenocarcinoma                  -0.004156         0.013431
Histology_Squamous Cell Carcinoma         -0.008831         0.004505
SKI                                       -0.010844         0.011668"""