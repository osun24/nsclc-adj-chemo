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

def create_rsf(df, name, trees=1000):
    # Subset df using preselected covariates and clinical variables
    selected_csv = pd.read_csv('rsf_results_GPL570_rsf_preselection_importances.csv', index_col=0)
    top_100 = list(selected_csv.index[:150])
    clinical_vars = ["Adjuvant Chemo", "Age", "Stage", "Sex", "Histology", "Race", "Smoked?"] + [col for col in surv.columns if col.startswith('Race')]
    selected_covariates = list(set(top_100).union(set(clinical_vars)))
    selected_covariates = [col for col in selected_covariates if col in df.columns]
    df = df[['OS_STATUS', 'OS_MONTHS'] + selected_covariates]
    
    # Create structured array for survival analysis
    surv_data = Surv.from_dataframe('OS_STATUS', 'OS_MONTHS', df)
    
    covariates = df.columns.difference(['OS_STATUS', 'OS_MONTHS'])
    
    # Identify binary columns (assuming these are already binary 0/1)
    # please change sex to IS_FEMALE/IS_MALE for clarity
    binary_columns = ['Adjuvant Chemo', 'Sex'] 
    df[binary_columns] = df[binary_columns].astype(int)
    
    continuous_columns = df.columns.difference(['OS_STATUS', 'OS_MONTHS', *binary_columns])
    
    # Check that binary columns are not scaled
    for col in binary_columns:
        assert df[col].max() <= 1 and df[col].min() >= 0, f"{col} should only contain binary values (0/1)."

    test_size = 0.2
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df[continuous_columns.union(binary_columns)], surv_data, test_size=test_size, random_state=42)

    print(X_train.columns)
    # Fit the Random Survival Forest model
    rsf = RandomSurvivalForest(n_estimators=trees, min_samples_split=75, min_samples_leaf=30, random_state=42, n_jobs= -1, max_features = "sqrt") # run on all processors
    rsf.fit(X_train, y_train)

    # Evaluate model performance
    c_index = concordance_index_censored(y_test['OS_STATUS'], y_test['OS_MONTHS'], rsf.predict(X_test))
    print(f"C-index: {c_index[0]:.3f}")
    
    # Train c-index
    train_c_index = concordance_index_censored(y_train['OS_STATUS'], y_train['OS_MONTHS'], rsf.predict(X_train))
    print(f"Train C-index: {train_c_index[0]:.3f}")

    # Save the RSF model to a file
    #joblib.dump(rsf, f'rsf_model-{trees}-c{c_index[0]:.3f}.pkl')

    result = permutation_importance(rsf, X_test, y_test, n_repeats=5, random_state=42)

    importances_df = pd.DataFrame({
        "importances_mean": result.importances_mean,
        "importances_std": result.importances_std
    }, index=X_test.columns).sort_values(by="importances_mean", ascending=False)

    print(importances_df)

    importances_df = importances_df.sort_values(by="importances_mean", ascending=True)  # Ascending for better barh plot

    # Plot the feature importances
    plt.figure(figsize=(12, 8))
    
    # Increase font size
    plt.rcParams.update({'font.size': 14})
    plt.barh(importances_df.index, importances_df["importances_mean"], xerr=importances_df["importances_std"], color=(9/255,117/255,181/255))
    plt.xlabel("Permutation Feature Importance")
    plt.ylabel("Feature")
    plt.title(f"Random Survival Forest: Feature Importances (C-index: {c_index[0]:.3f})")
    plt.tight_layout()
    name = name.replace(' ', '-')
    plt.savefig(f'rsf-importances-{name}-{trees}trees-{test_size}testsize.png')
    plt.show()

def search_feature_space_rsf(df, name, trees=300):
    # Load preselected gene features (use top 30)
    selected_csv = pd.read_csv('rsf_results_GPL570_rsf_preselection_importances.csv', index_col=0)
    candidate_genes = list(selected_csv.index[:30])
    # Clinical variables (must be included)
    clinical_vars = ["Adjuvant Chemo", "Age", "Stage", "Sex", "Histology", "Race", "Smoked?"]
    clinical_vars = [var for var in clinical_vars if var in df.columns]
    
    test_size = 0.2
    # Lists to hold results for 3D plotting
    gene_features_list = []
    n_estimators_list = []
    test_c_indexes = []
    train_c_indexes = []
    
    # Iterate over different numbers of gene features and n_estimators
    for m in np.unique(np.linspace(0, 40, num=50, dtype=int)):
        for n_est in np.unique(np.linspace(50, 500, num=10, dtype=int)):
            # Select top m genes and add clinical variables (if m exceeds candidate_genes length, use all available)
            selected_covariates = list(set(candidate_genes[:m]).union(set(clinical_vars)))
            df_subset = df[['OS_STATUS', 'OS_MONTHS'] + [col for col in selected_covariates if col in df.columns]].copy()
            
            # Create structured survival array
            surv_data = Surv.from_dataframe('OS_STATUS', 'OS_MONTHS', df_subset)
            # Define feature columns (exclude OS_STATUS and OS_MONTHS)
            feature_cols = df_subset.columns.difference(['OS_STATUS', 'OS_MONTHS'])
            
            # Convert binary columns if present
            binary_columns = [col for col in ['Adjuvant Chemo', 'Sex'] if col in feature_cols]
            df_subset[binary_columns] = df_subset[binary_columns].astype(int)
            
            # Split into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(df_subset[feature_cols], surv_data,
                                                                test_size=test_size, random_state=42)
            # Fit the Random Survival Forest model
            rsf = RandomSurvivalForest(n_estimators=n_est, min_samples_split=75, min_samples_leaf=30,
                                       random_state=42, n_jobs=-1, max_features="sqrt")
            rsf.fit(X_train, y_train)
            
            # Evaluate model performance on test set
            test_ci = concordance_index_censored(y_test['OS_STATUS'], y_test['OS_MONTHS'], rsf.predict(X_test))[0]
            # Train c-index
            train_ci = concordance_index_censored(y_train['OS_STATUS'], y_train['OS_MONTHS'], rsf.predict(X_train))[0]
            gene_features_list.append(m)
            n_estimators_list.append(n_est)
            test_c_indexes.append(test_ci)
            train_c_indexes.append(train_ci)
            print(f"Gene features: {m}, Estimators: {n_est}, Total features: {len(feature_cols)}, Test C-index: {test_ci:.3f}, Train C-index: {train_ci:.3f}")
    
    # Find optimal setting based on test performance
    optimal_index = np.argmax(test_c_indexes)
    optimal_genes = gene_features_list[optimal_index]
    optimal_n_est = n_estimators_list[optimal_index]
    optimal_c_index = test_c_indexes[optimal_index]
    print(f"\nOptimal: Gene features: {optimal_genes}, Estimators: {optimal_n_est} (Test C-index: {optimal_c_index:.3f})")
    
    # 3D Plot performance vs number of gene features and n_estimators
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    sc1 = ax.scatter(gene_features_list, n_estimators_list, test_c_indexes, c='blue', marker='o', label="Test C-index")
    sc2 = ax.scatter(gene_features_list, n_estimators_list, train_c_indexes, c='green', marker='^', label="Train C-index")
    ax.set_xlabel("Number of Gene Features")
    ax.set_ylabel("n_estimators")
    ax.set_zlabel("C-index")
    ax.set_title("RSF Performance vs Gene Features and n_estimators")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'rsf_feature_search_3D_{name}_{trees}trees.png')
    plt.show()
    
    return optimal_genes

# Data preprocessing (unchanged)
surv = pd.read_csv('GPL570merged.csv')
surv = pd.get_dummies(surv, columns=["Stage", "Histology", "Race"])
surv = surv.drop(columns=['PFS_MONTHS','RFS_MONTHS'])
print(surv.columns[surv.isna().any()].tolist())
print(surv['Smoked?'].isna().sum())  # 121
surv = surv.dropna()  # left with 457 samples

# Run feature space search
optimal_features = search_feature_space_rsf(surv, 'GPL570', trees=300)

print(optimal_features)
#create_rsf(surv, 'GPL570', 300)