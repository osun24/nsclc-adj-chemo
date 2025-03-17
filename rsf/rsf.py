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
    selected_csv = pd.read_csv('rsf/rsf_results_GPL570_rsf_preselection_importances.csv', index_col=0)
    top_100 = selected_csv.index[:75]
    
    clinical_vars = ["Adjuvant Chemo", "Age", "Stage", "Sex", "Histology", "Race", "Smoked?"] + [col for col in surv.columns if col.startswith('Race')] + [col for col in surv.columns if col.startswith('Histology')] 
    selected_covariates = list(set(top_100[:77]).union(set(clinical_vars)))
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
    rsf = RandomSurvivalForest(n_estimators=trees, min_samples_split=75, min_samples_leaf=30, random_state=42, n_jobs= -1, max_features = None) # run on all processors
    rsf.fit(X_train, y_train)

    # Evaluate model performance
    c_index = concordance_index_censored(y_test['OS_STATUS'], y_test['OS_MONTHS'], rsf.predict(X_test))
    print(f"C-index: {c_index[0]:.3f}")
    
    # Train c-index
    train_c_index = concordance_index_censored(y_train['OS_STATUS'], y_train['OS_MONTHS'], rsf.predict(X_train))
    print(f"Train C-index: {train_c_index[0]:.3f}")

    # Save the RSF model to a file
    joblib.dump(rsf, f'rsf_model-{trees}-c{c_index[0]:.3f}.pkl')

    result = permutation_importance(rsf, X_test, y_test, n_repeats=5, random_state=42)

    importances_df = pd.DataFrame({
        "importances_mean": result.importances_mean,
        "importances_std": result.importances_std
    }, index=X_test.columns).sort_values(by="importances_mean", ascending=False)

    print(importances_df)
    
    # save to csv   
    importances_df.to_csv(f'rsf/rsf_results_{name}_rsf_importances.csv')
    
    # take only top 30
    importances_df = importances_df.head(30)

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

# Data preprocessing (unchanged)
surv = pd.read_csv('GPL570merged.csv')
surv = pd.get_dummies(surv, columns=["Stage", "Histology", "Race"])
surv = surv.drop(columns=['PFS_MONTHS','RFS_MONTHS'])
print(surv.columns[surv.isna().any()].tolist())
print(surv['Smoked?'].isna().sum())  # 121
surv = surv.dropna()  # left with 457 samples

#print(optimal_features)
#create_rsf(surv, 'GPL570', 50)
# Optimal: Gene features: 77, Estimators: 50 (Test C-index: 0.641)