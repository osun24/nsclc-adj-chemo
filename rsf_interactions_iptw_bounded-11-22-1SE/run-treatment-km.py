import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import inspect
import warnings

from sklearn.linear_model import LogisticRegression
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv
from sksurv.ensemble import RandomSurvivalForest
from lifelines import KaplanMeierFitter
from lifelines.utils import restricted_mean_survival_time
from lifelines.statistics import logrank_test
from lifelines.plotting import add_at_risk_counts
from threadpoolctl import threadpool_limits

warnings.filterwarnings(
    "ignore",
    message="Ties in event time detected; using efron's method to handle ties."
)

np.random.seed(42)

# ============================================================
# Configuration
# ============================================================
TRAIN_CSV = "affyfRMATrain.csv"
VALID_CSV = "affyfRMAValidation.csv"
TEST_CSV  = "affyfRMATest.csv"

GENES_CSV = "/mnt/data/LOOCV_Genes2.csv"
if not os.path.exists(GENES_CSV):
    if os.path.exists("/content/LOOCV_Genes2.csv"):
        GENES_CSV = "/content/LOOCV_Genes2.csv"
    elif os.path.exists("../LOOCV_Genes2.csv"):
        GENES_CSV = "../LOOCV_Genes2.csv"

CLINICAL_VARS = [
    "Adjuvant Chemo","Age","IS_MALE",
    "Stage_IA","Stage_IB","Stage_II","Stage_III",
    "Histology_Adenocarcinoma","Histology_Large Cell Carcinoma","Histology_Squamous Cell Carcinoma",
    "Race_African American","Race_Asian","Race_Caucasian","Race_Native Hawaiian or Other Pacific Islander","Race_Unknown",
    "Smoked?_No","Smoked?_Unknown","Smoked?_Yes"
]

CLIN_FEATS_PRETX = [c for c in CLINICAL_VARS if c != "Adjuvant Chemo"]

# ============================================================
# Helper Functions (from RSF-Optuna.py)
# ============================================================
def preprocess_split(df, clinical_vars, gene_names):
    """Preprocess a split: map treatment, handle missing values."""
    if "Adjuvant Chemo" in df.columns:
        df["Adjuvant Chemo"] = df["Adjuvant Chemo"].map({"OBS": 0, "ACT": 1})
    for col in ["Adjuvant Chemo", "IS_MALE"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    keep_cols = [c for c in clinical_vars if c in df.columns] + [g for g in gene_names if g in df.columns]
    cols = ["OS_STATUS", "OS_MONTHS"] + keep_cols
    return df[cols].copy()


def build_features_with_interactions(df, main_genes, inter_genes,
                                    act_col="Adjuvant Chemo", dup_inter=1,
                                    clin_cols=None):
    """Build [clinical + genes_main + gene*ACT] features."""
    if clin_cols is None:
        clin_cols = CLINICAL_VARS

    base_cols = list(clin_cols) + list(main_genes)
    X_base = df[base_cols].to_numpy(dtype=np.float64)
    A = df[act_col].to_numpy(dtype=np.float64).reshape(-1, 1)

    names = list(base_cols)
    blocks = [X_base]

    if len(inter_genes) > 0:
        X_int = df[list(inter_genes)].to_numpy(dtype=np.float64) * A
        names_int = [f"{g}*ACT" for g in inter_genes]
        blocks.append(X_int)
        names += names_int

        if int(dup_inter) > 1:
            for d in range(1, int(dup_inter)):
                blocks.append(X_int.copy())
                names += [f"{g}*ACT#dup{d}" for g in inter_genes]

    X = np.concatenate(blocks, axis=1) if len(blocks) > 1 else X_base
    return X, names


def cindex(pred, time, event):
    """Compute C-index."""
    return float(concordance_index_censored(event.astype(bool), time.astype(float), pred)[0])


def recommend_treatment(risk_treated, risk_untreated, eps=1e-100):
    """Deterministic treatment recommendation with epsilon tie-break."""
    diff = risk_untreated - risk_treated  # positive => treated better (lower risk)
    return np.where(diff > eps, 1, np.where(diff < -eps, 0, 0))


def compare_treatment_recommendation_km_rsf(model, df, genes_main, genes_inter, dup_inter,
                                           clin_cols,
                                           time_col="OS_MONTHS", event_col="OS_STATUS", p = 0, q= 0):
    """KM comparison for alignment with model's treatment recommendation."""
    df = df.copy()
    df["Adjuvant Chemo"] = df["Adjuvant Chemo"].astype(int)

    # Counterfactual feature matrices with ACT forced to 1 vs 0
    df_treated = df.copy()
    df_treated["Adjuvant Chemo"] = 1

    df_untreated = df.copy()
    df_untreated["Adjuvant Chemo"] = 0

    X_treated, _ = build_features_with_interactions(
        df_treated, genes_main, genes_inter, dup_inter=dup_inter, clin_cols=clin_cols
    )
    X_untreated, _ = build_features_with_interactions(
        df_untreated, genes_main, genes_inter, dup_inter=dup_inter, clin_cols=clin_cols
    )

    # RSF predict() returns a risk score: higher = worse
    with threadpool_limits(limits=1):
        risk_treated = model.predict(X_treated.astype(np.float64))
        risk_untreated = model.predict(X_untreated.astype(np.float64))

    model_rec = recommend_treatment(risk_treated, risk_untreated, eps=1e-100)
    actual = df["Adjuvant Chemo"].to_numpy(int)
    alignment = actual == model_rec

    df["model_rec"] = model_rec
    df["alignment"] = alignment

    mask_aligned = df["alignment"]
    mask_not_aligned = ~df["alignment"]

    kmf_aligned = KaplanMeierFitter()
    kmf_not_aligned = KaplanMeierFitter()

    plt.figure(figsize=(10, 6))

    kmf_aligned.fit(
        durations=df.loc[mask_aligned, time_col],
        event_observed=df.loc[mask_aligned, event_col],
        label="Aligned with RSF recommendation"
    )
    ax = kmf_aligned.plot(ci_show=True)

    kmf_not_aligned.fit(
        durations=df.loc[mask_not_aligned, time_col],
        event_observed=df.loc[mask_not_aligned, event_col],
        label="Not aligned with RSF recommendation"
    )
    kmf_not_aligned.plot(ax=ax, ci_show=True)

    results = logrank_test(
        df.loc[mask_aligned, time_col],
        df.loc[mask_not_aligned, time_col],
        event_observed_A=df.loc[mask_aligned, event_col],
        event_observed_B=df.loc[mask_not_aligned, event_col],
        weightings="fleming-harrington",
        p =p,
        q=q
    )
    print("Log-rank test p-value:", results.p_value)

    plt.title("Kaplan-Meier Survival Curves by Treatment Alignment")
    plt.xlabel("Time (months)")
    plt.ylabel("Survival Probability")
    add_at_risk_counts(kmf_aligned, kmf_not_aligned)
    if p > 0 or q > 0:
        text = f"Weighted (p = {p}, q = {q})"
    else: text=""
    plt.text(0.1, 0.1, f"{text} Log-rank p-value: {results.p_value:.4f}", transform=plt.gca().transAxes)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(f"km_alignment_rsf_recommendation_{text}.png", dpi=600)
    print("Saved KM plot to km_alignment_rsf_recommendation.png")
    plt.show()

    print(f"\nAligned patients: {mask_aligned.sum()}")
    print(f"Not aligned patients: {mask_not_aligned.sum()}")


def rmst_by_treatment_recommendation(model, df, genes_main, genes_inter, dup_inter,
                                     clin_cols, tau=60,
                                     time_col="OS_MONTHS", event_col="OS_STATUS"):
    """Compute RMST for patients aligned vs not aligned with model recommendation."""
    df = df.copy()
    df["Adjuvant Chemo"] = df["Adjuvant Chemo"].astype(int)

    # Counterfactual feature matrices with ACT forced to 1 vs 0
    df_treated = df.copy()
    df_treated["Adjuvant Chemo"] = 1

    df_untreated = df.copy()
    df_untreated["Adjuvant Chemo"] = 0

    X_treated, _ = build_features_with_interactions(
        df_treated, genes_main, genes_inter, dup_inter=dup_inter, clin_cols=clin_cols
    )
    X_untreated, _ = build_features_with_interactions(
        df_untreated, genes_main, genes_inter, dup_inter=dup_inter, clin_cols=clin_cols
    )

    # RSF predict() returns a risk score: higher = worse
    with threadpool_limits(limits=1):
        risk_treated = model.predict(X_treated.astype(np.float64))
        risk_untreated = model.predict(X_untreated.astype(np.float64))

    model_rec = recommend_treatment(risk_treated, risk_untreated, eps=1e-100)
    actual = df["Adjuvant Chemo"].to_numpy(int)
    alignment = actual == model_rec

    mask_aligned = alignment
    mask_not_aligned = ~alignment

    if tau is None:
        tau = float(df[time_col].max())

    kmf_aligned = KaplanMeierFitter()
    kmf_not_aligned = KaplanMeierFitter()

    kmf_aligned.fit(
        durations=df.loc[mask_aligned, time_col],
        event_observed=df.loc[mask_aligned, event_col],
        label="Aligned with RSF recommendation"
    )
    kmf_not_aligned.fit(
        durations=df.loc[mask_not_aligned, time_col],
        event_observed=df.loc[mask_not_aligned, event_col],
        label="Not aligned with RSF recommendation"
    )

    rmst_aligned = float(restricted_mean_survival_time(kmf_aligned, t=tau))
    rmst_not_aligned = float(restricted_mean_survival_time(kmf_not_aligned, t=tau))
    rmst_diff = rmst_aligned - rmst_not_aligned

    print("\n[RMST by Treatment Recommendation]")
    print(f"RMST (aligned) at tau={tau:.2f}: {rmst_aligned:.4f}")
    print(f"RMST (not aligned) at tau={tau:.2f}: {rmst_not_aligned:.4f}")
    print(f"RMST difference (aligned - not aligned): {rmst_diff:.4f}")

    return {
        "tau": tau,
        "rmst_aligned": rmst_aligned,
        "rmst_not_aligned": rmst_not_aligned,
        "rmst_diff": rmst_diff,
        "n_aligned": int(mask_aligned.sum()),
        "n_not_aligned": int(mask_not_aligned.sum())
    }


# ============================================================
# Main Script
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Loading RSF model and evaluating on test set with KM analysis")
    print("=" * 60)

    # Load data
    print("\n[Loading Data]")
    test_raw = pd.read_csv(TEST_CSV)
    
    # Load list of genes used
    with open("rsf_interactions_iptw_bounded-11-22-1SE/features_used.txt", "r") as f:
        feat_lines = [line.strip() for line in f.readlines()]
    
    # Features are in order: clinical first, then genes
    # Clinical vars are the first ones in features_used.txt
    clin_used = [f for f in feat_lines if f in CLINICAL_VARS]
    genes_used = [f for f in feat_lines if f not in CLINICAL_VARS]
    
    print(f"Features loaded: {len(clin_used)} clinical, {len(genes_used)} genes")
    
    # Preprocess test set
    test_df = preprocess_split(test_raw, CLINICAL_VARS, genes_used)
    test_df = test_df.sort_values(
        by=["OS_MONTHS", "OS_STATUS"],
        ascending=[False, False],
        kind="mergesort"
    ).reset_index(drop=True)
    
    print(f"Test set: {len(test_df)} samples")
    print(f"Test OS_STATUS value counts:\n{test_df['OS_STATUS'].value_counts()}")
    print(f"Test Adjuvant Chemo value counts:\n{test_df['Adjuvant Chemo'].value_counts()}")
    
    # Load RSF model
    print("\n[Loading RSF Model]")
    rsf_model = joblib.load("rsf_interactions_iptw_bounded-11-22-1SE/rsf_final.joblib")
    print(f"Model loaded: {rsf_model}")
    try:
        rsf_model.set_params(n_jobs=1)
    except Exception:
        print("Could not set n_jobs=1 on RSF model, will rely on threadpool_limits to control threading.")
        pass
    
    # Parameters from chosen trial
    # From chosen_params.txt: top_k_genes=16, inter_ratio=1.113452335944007
    # k_main=16, k_int = floor(1.113 * 16) = 17, clamped to 16
    k_main = 16
    k_int = 0
    dup_inter = 1
    
    genes_main = genes_used[:k_main]
    genes_inter = genes_main[:k_int]
    
    print(f"k_main={k_main}, k_int={k_int}, dup_inter={dup_inter}")
    print(f"Genes (main): {genes_main}")
    print(f"Genes (inter): {genes_inter}")
    
    # Evaluate on test set
    print("\n[Evaluating Test Set]")
    X_te_raw, feat_names = build_features_with_interactions(
        test_df, genes_main, genes_inter, dup_inter=dup_inter, clin_cols=clin_used
    )
    X_te = X_te_raw.astype(np.float64)
    
    with threadpool_limits(limits=1):
        pred_te = rsf_model.predict(X_te)
    t_te = test_df["OS_MONTHS"].values.astype(float)
    e_te = test_df["OS_STATUS"].values.astype(int)
    
    ci_te = cindex(pred_te, t_te, e_te)
    print(f"Test C-index: {ci_te:.4f}")
    
    # Per-arm C-indices
    act_te = test_df["Adjuvant Chemo"].to_numpy(int)
    def ci_by_arm(pred, t, e, arm):
        out = {}
        for label, mask in [("ACT=1", arm == 1), ("ACT=0", arm == 0)]:
            out[label] = cindex(pred[mask], t[mask], e[mask]) if mask.sum() >= 3 else np.nan
        return out
    print(f"Test CI by arm: {ci_by_arm(pred_te, t_te, e_te, act_te)}")
    
    # Kaplan-Meier alignment on test set
    print("\n[KM Alignment Analysis]")
    compare_treatment_recommendation_km_rsf(
        rsf_model,
        test_df,
        genes_main=genes_main,
        genes_inter=genes_inter,
        dup_inter=dup_inter,
        clin_cols=clin_used,
    )

    rmst_by_treatment_recommendation(
        rsf_model,
        test_df,
        genes_main=genes_main,
        genes_inter=genes_inter,
        dup_inter=dup_inter,
        clin_cols=clin_used,
    )
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)
