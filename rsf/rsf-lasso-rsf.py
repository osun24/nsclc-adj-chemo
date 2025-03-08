import os
import time
import pandas as pd
import numpy as np
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
import joblib

# Determine script directory for consistent path handling in a VM
script_dir = os.path.dirname(os.path.abspath(__file__))
data_file = "GPL570merged.csv"
output_dir = os.path.join(script_dir, "rsf_results")
os.makedirs(output_dir, exist_ok=True)

def rsf_concordance_score(estimator, X, y):
    preds = estimator.predict(X)
    return concordance_index_censored(y['OS_STATUS'], y['OS_MONTHS'], preds)[0]

def create_rsf(df, name, trees=300):
    overall_start = time.time()
    # Create structured array for survival analysis
    surv_data = Surv.from_dataframe('OS_STATUS', 'OS_MONTHS', df)
    
    # Define covariates: all columns except survival columns
    covariates = df.columns.difference(['OS_STATUS', 'OS_MONTHS'])
    
    # Specify binary and continuous columns
    binary_columns = ['Adjuvant Chemo', 'Sex']
    df[binary_columns] = df[binary_columns].astype(int)
    continuous_columns = df.columns.difference(['OS_STATUS', 'OS_MONTHS', *binary_columns])
    
    for col in binary_columns:
        assert df[col].max() <= 1 and df[col].min() >= 0, f"{col} should only contain binary values (0/1)."
    
    test_size = 0.2
    features = continuous_columns.union(binary_columns)
    X_train, X_test, y_train, y_test = train_test_split(df[features], surv_data, test_size=test_size, random_state=42)
    
    print("Initial features:", X_train.columns.tolist())
    
    # ---------------- Step 1: RSF Pre-Selection with Permutation Importance ---------------- #
    step1_start = time.time()
    initial_rsf = RandomSurvivalForest(
        n_estimators=trees,
        min_samples_split=75,
        min_samples_leaf=30,
        random_state=42,
        n_jobs=-1,
        max_features="sqrt"
    )
    initial_rsf.fit(X_train, y_train)
    
    # print concordance indexes
    test_pred = initial_rsf.predict(X_test)
    c_index = concordance_index_censored(y_test['OS_STATUS'], y_test['OS_MONTHS'], test_pred)
    print(f"Initial RSF Test C-index: {c_index[0]:.3f}")
    train_pred = initial_rsf.predict(X_train)
    train_c_index = concordance_index_censored(y_train['OS_STATUS'], y_train['OS_MONTHS'], train_pred)
    print(f"Initial RSF Train C-index: {train_c_index[0]:.3f}")
    
    perm_result = permutation_importance(initial_rsf, X_train, y_train,
                                           scoring=rsf_concordance_score,
                                           n_repeats=5, random_state=42, n_jobs = -1)
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
             xerr=top_preselect["Std"][::-1], color=(9/255,117/255,181/255))
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
    
    # ---------------- Step 2: LASSO-Cox Feature Selection ---------------- #
    step2_start = time.time()
    alphas = np.logspace(-4, 4, 100)
    lasso = CoxnetSurvivalAnalysis(l1_ratio=1.0, alphas=alphas, max_iter=10000)
    lasso.fit(X_train_rsf, y_train)
    
    coefs = None
    for coef in lasso.coef_:
        if np.sum(coef != 0) > 0:
            coefs = coef
            break
    if coefs is None:
        raise ValueError("LASSO did not select any features.")
    
    nonzero_mask = coefs != 0
    selected_features_lasso = X_train_rsf.columns[nonzero_mask].tolist()
    
    if len(selected_features_lasso) > 450:
        abs_coefs = np.abs(coefs[nonzero_mask])
        sorted_idx = np.argsort(abs_coefs)[-450:]
        selected_features_lasso = np.array(selected_features_lasso)[sorted_idx].tolist()
    
    lasso_df = pd.DataFrame({
        "Feature": X_train_rsf.columns,
        "Coefficient": coefs
    })
    lasso_csv = os.path.join(output_dir, f"{name}_lasso_coefficients.csv")
    lasso_df.to_csv(lasso_csv, index=False)
    print(f"LASSO coefficients saved to {lasso_csv}")
    
    print(f"LASSO selection: {len(selected_features_lasso)} features selected.")
    
    lasso_nonzero = lasso_df[lasso_df["Coefficient"] != 0].sort_values(by="Coefficient", key=np.abs)
    plt.figure(figsize=(12, 8))
    plt.barh(lasso_nonzero["Feature"], lasso_nonzero["Coefficient"], color=(9/255,117/255,181/255))
    plt.xlabel("LASSO Coefficient")
    plt.title("LASSO-Cox Selected Feature Coefficients")
    plt.tight_layout()
    lasso_plot = os.path.join(output_dir, f"{name}_lasso_coefficients.png")
    plt.savefig(lasso_plot)
    plt.close()
    print(f"LASSO coefficients plot saved to {lasso_plot}")
    
    X_train_final = X_train_rsf[selected_features_lasso]
    X_test_final = X_test_rsf[selected_features_lasso]
    step2_end = time.time()
    print(f"Step 2 completed in {step2_end - step2_start:.2f} seconds.")
    
    # ---------------- Step 3: Retrain RSF with Final Selected Features ---------------- #
    step3_start = time.time()
    final_rsf = RandomSurvivalForest(
        n_estimators=trees,
        min_samples_split=75,
        min_samples_leaf=30,
        random_state=42,
        n_jobs=-1,
        max_features="sqrt"
    )
    final_rsf.fit(X_train_final, y_train)
    
    test_pred = final_rsf.predict(X_test_final)
    c_index = concordance_index_censored(y_test['OS_STATUS'], y_test['OS_MONTHS'], test_pred)
    print(f"Final RSF Test C-index: {c_index[0]:.3f}")
    train_pred = final_rsf.predict(X_train_final)
    train_c_index = concordance_index_censored(y_train['OS_STATUS'], y_train['OS_MONTHS'], train_pred)
    print(f"Final RSF Train C-index: {train_c_index[0]:.3f}")
    
    perf_file = os.path.join(output_dir, f"{name}_performance.txt")
    with open(perf_file, "w") as f:
        f.write(f"Final RSF Test C-index: {c_index[0]:.3f}\n")
        f.write(f"Final RSF Train C-index: {train_c_index[0]:.3f}\n")
    print(f"Performance metrics saved to {perf_file}")
    
    final_params = final_rsf.get_params()
    params_file = os.path.join(output_dir, f"{name}_final_rsf_params.txt")
    with open(params_file, "w") as f:
        for key, value in final_params.items():
            f.write(f"{key}: {value}\n")
    print(f"Final RSF parameters saved to {params_file}")
    
    result = permutation_importance(final_rsf, X_test_final, y_test,
                                    scoring=rsf_concordance_score, n_repeats=5, random_state=42, n_jobs = -1)
    importances_df_final = pd.DataFrame({
        "importances_mean": result.importances_mean,
        "importances_std": result.importances_std
    }, index=X_test_final.columns).sort_values(by="importances_mean", ascending=False)
    
    final_importance_csv = os.path.join(output_dir, f"{name}_final_rsf_importances.csv")
    importances_df_final.to_csv(final_importance_csv)
    print(f"Final RSF permutation importances saved to {final_importance_csv}")
    
    importances_df_final = importances_df_final.sort_values(by="importances_mean", ascending=True)
    plt.figure(figsize=(12, 8))
    plt.rcParams.update({'font.size': 14})
    plt.barh(importances_df_final.index,
             importances_df_final["importances_mean"],
             xerr=importances_df_final["importances_std"],
             color=(9/255, 117/255, 181/255))
    plt.xlabel("Permutation Feature Importance")
    plt.ylabel("Feature")
    plt.title(f"Final RSF: Feature Importances (C-index: {c_index[0]:.3f})")
    plt.tight_layout()
    final_importance_plot = os.path.join(output_dir, f"{name}_final_rsf_importances.png")
    plt.savefig(final_importance_plot)
    plt.close()
    print(f"Final RSF permutation importance plot saved to {final_importance_plot}")
    
    step3_end = time.time()
    print(f"Step 3 completed in {step3_end - step3_start:.2f} seconds.")
    overall_end = time.time()
    print(f"Total pipeline completed in {overall_end - overall_start:.2f} seconds.")
    
    # Optionally, save the final RSF model
    # model_file = os.path.join(output_dir, f"{name}_final_rsf_model.pkl")
    # joblib.dump(final_rsf, model_file)
    # print(f"Final RSF model saved to {model_file}")

if __name__ == "__main__":
    # Data Loading and Preprocessing
    print(f"Loading data from: {data_file}")
    surv = pd.read_csv(data_file)
    surv = pd.get_dummies(surv, columns=["Stage", "Histology", "Race"])
    surv = surv.drop(columns=['PFS_MONTHS', 'RFS_MONTHS'])
    print("Columns with NA values:", surv.columns[surv.isna().any()].tolist())
    print("Number of missing 'Smoked?' entries:", surv['Smoked?'].isna().sum())
    surv = surv.dropna()  # Retain 457 samples

    # Run the RSF pipeline with feature selection and periodic time estimation
    create_rsf(surv, 'GPL570', trees=300)