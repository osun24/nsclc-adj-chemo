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
    top_100 = selected_csv.index[:0]
    
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
    rsf = RandomSurvivalForest(n_estimators=100, min_samples_split=30, min_samples_leaf=25,
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
    model_filepath = "rsf/rsf_results_GPL570 3-13-25 RS_final_rsf_model_1se.pkl"
    # Set feature_selected=True to use preselected features, or False to use all features.
    open_rsf(train, valid, model_filepath, feature_selected=False)


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