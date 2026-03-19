# ============================================================
# Colab-ready SINGLE CELL (CPU)
# XGBoost (gbtree) Cox PH with IPTW, feature budgets,
# emphasized gene×ACT interactions (incl. optional duplication),
# early stopping on C-index, and multi-objective Pareto selection
# ============================================================

# ---------- Imports ----------
import os, math, gc, time, random, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv
from sksurv.linear_model import CoxPHSurvivalAnalysis
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.plotting import add_at_risk_counts
from lifelines.utils import restricted_mean_survival_time
from threadpoolctl import threadpool_limits

from sksurv.compare import compare_survival

import optuna
from optuna.samplers import NSGAIISampler

warnings.filterwarnings("ignore",
    message="Ties in event time detected; using efron's method to handle ties.")

np.random.seed(42); random.seed(42)

# ---------- Paths ----------
# Update these to your files if needed
TRAIN_CSV = "affyfRMATrain.csv"
VALID_CSV = "affyfRMAValidation.csv"
TEST_CSV  = "affyfRMATest.csv"

GENES_CSV = "/mnt/data/LOOCV_Genes2.csv"
if not os.path.exists(GENES_CSV):
    if os.path.exists("/content/LOOCV_Genes2.csv"):
        GENES_CSV = "/content/LOOCV_Genes2.csv"
    elif os.path.exists("LOOCV_Genes2.csv"):
        GENES_CSV = "LOOCV_Genes2.csv"
print("LOOCV_Genes2.csv path:", GENES_CSV)

# ---------- Clinical columns ----------
CLINICAL_VARS = [
    "Adjuvant Chemo","Age","IS_MALE",
    "Stage_IA","Stage_IB","Stage_II","Stage_III",
    "Histology_Adenocarcinoma","Histology_Large Cell Carcinoma","Histology_Squamous Cell Carcinoma",
    "Race_African American","Race_Asian","Race_Caucasian","Race_Native Hawaiian or Other Pacific Islander","Race_Unknown",
    "Smoked?_No","Smoked?_Unknown","Smoked?_Yes"
]
CLIN_FEATS_PRETX = [c for c in CLINICAL_VARS if c != "Adjuvant Chemo"]  # reset after feature intersection

# ============================================================
# Helpers: IO, preprocessing, ranking, features, IPTW, metrics
# ============================================================
def load_genes_list(genes_csv):
    g = pd.read_csv(genes_csv)
    if "Prop" not in g.columns or "Gene" not in g.columns:
        raise ValueError("LOOCV_Genes2.csv must have columns 'Gene' and 'Prop'.")
    g["Prop"] = pd.to_numeric(g["Prop"], errors="coerce").fillna(0)
    genes = g.loc[g["Prop"] == 1, "Gene"].astype(str).tolist()
    print(f"[Genes] Selected {len(genes)} genes with Prop == 1")
    return genes

def preprocess_split(df, clinical_vars, gene_names):
    # Avoid pandas FutureWarning by mapping (we coerce to numeric right after)
    if "Adjuvant Chemo" in df.columns:
        df["Adjuvant Chemo"] = df["Adjuvant Chemo"].map({"OBS": 0, "ACT": 1})
    for col in ["Adjuvant Chemo","IS_MALE"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    keep_cols = [c for c in clinical_vars if c in df.columns] + [g for g in gene_names if g in df.columns]
    cols = ["OS_STATUS","OS_MONTHS"] + keep_cols
    return df[cols].copy()

def rank_genes_univariate(train_df, gene_cols):
    y = Surv.from_arrays(event=train_df["OS_STATUS"].astype(bool).values,
                         time=train_df["OS_MONTHS"].values.astype(float))
    ranks = []
    for g in gene_cols:
        Xg = train_df[[g]].to_numpy(dtype=np.float32)
        try:
            model = CoxPHSurvivalAnalysis(alpha=1e-12)
            model.fit(Xg, y)
            pred = model.predict(Xg)
            ci = concordance_index_censored(y["event"], y["time"], pred)[0]
            ranks.append((g, float(ci)))
        except Exception:
            ranks.append((g, 0.5))
    ranks.sort(key=lambda z: z[1], reverse=True)
    return [g for g, _ in ranks]

# === Emphasize interactions via optional duplication ===
def build_features_with_interactions(df, main_genes, inter_genes,
                                     act_col="Adjuvant Chemo", dup_inter=1,
                                     clin_cols=None):
    """
    Build [clinical + genes_main + gene*ACT] features.
    If dup_inter>1, duplicate interaction columns with unique names to bias column sampling.
    """
    if clin_cols is None:
        clin_cols = CLINICAL_VARS

    base_cols = list(clin_cols) + list(main_genes)  # keep ACT main effect in clinicals
    X_base = df[base_cols].to_numpy(dtype=np.float32)
    A = df[act_col].to_numpy(dtype=np.float32).reshape(-1, 1)

    names = list(base_cols)
    blocks = [X_base]

    if len(inter_genes) > 0:
        X_int = df[list(inter_genes)].to_numpy(dtype=np.float32) * A
        names_int = [f"{g}*ACT" for g in inter_genes]
        blocks.append(X_int); names += names_int

        # Duplicate interaction columns to increase selection chance
        if int(dup_inter) > 1:
            for d in range(1, int(dup_inter)):
                blocks.append(X_int.copy())
                names += [f"{g}*ACT#dup{d}" for g in inter_genes]

    X = np.concatenate(blocks, axis=1) if len(blocks) > 1 else X_base
    return X, names

def compute_iptw(df, covariate_cols, act_col="Adjuvant Chemo",
                 ps_clip=(0.05, 0.95), w_clip=(0.1, 10.0),
                 ref_prev=None, model=None):
    A = df[act_col].astype(int).values
    X = df[covariate_cols].astype(float).values
    if model is None:
        model = LogisticRegression(max_iter=2000, solver="lbfgs", class_weight="balanced")
        model.fit(X, A)
    ps = model.predict_proba(X)[:, 1]
    ps = np.clip(ps, ps_clip[0], ps_clip[1])
    if ref_prev is None:
        ref_prev = A.mean()
    w = np.where(A == 1, ref_prev / ps, (1 - ref_prev) / (1 - ps))
    w = np.clip(w, w_clip[0], w_clip[1])
    return w.astype(np.float32), model, float(ref_prev)

def pack_cox_labels(time, event):
    """XGBoost Cox: encode event as sign of time for our custom metric."""
    time = np.asarray(time, dtype=np.float32)
    event = np.asarray(event, dtype=int)
    return np.where(event == 1, time, -time).astype(np.float32)

def cindex(pred, time, event):
    return float(concordance_index_censored(event.astype(bool), time.astype(float), pred)[0])


def predict_xgb_risk(booster, X, feature_names, best_ntree):
    dmat = xgb.DMatrix(X.astype(np.float32), feature_names=feature_names)
    return booster.predict(dmat, iteration_range=(0, int(best_ntree)), output_margin=True)


def slice_booster_to_best_iteration(booster, best_ntree):
    if best_ntree is None:
        return booster
    try:
        return booster[:int(best_ntree)]
    except Exception:
        return booster


def compare_treatment_recommendation_km_xgb(booster, df, genes_main, genes_inter, dup_inter,
                                            feature_names, best_ntree, clin_cols,
                                            time_col="OS_MONTHS", event_col="OS_STATUS",
                                            path="", includeRMST=False, p=None, q=None, Cindex=None):
    """KM comparison for alignment with model's treatment recommendation on provided data."""
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

    with threadpool_limits(limits=1):
        risk_treated = predict_xgb_risk(booster, X_treated, feature_names, best_ntree)
        risk_untreated = predict_xgb_risk(booster, X_untreated, feature_names, best_ntree)
        
        delta = risk_treated - risk_untreated
        delta_mean = float(np.mean(delta))
        delta_std = float(np.std(delta))
        print(f"[Delta risk treated-untreated] mean={delta_mean:.6f}, sd={delta_std:.6f}, n_interactions={len(genes_inter)}")
        if np.isclose(delta_std, 0.0, atol=1e-10):
            print("[Warning] Near-zero treatment heterogeneity detected (counterfactual risks are almost identical).")

    model_rec = np.where(risk_treated < risk_untreated, 1, 0)  # choose lower risk arm
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
        label="Aligned with XGBoost recommendation"
    )
    ax = kmf_aligned.plot(ci_show=True)

    kmf_not_aligned.fit(
        durations=df.loc[mask_not_aligned, time_col],
        event_observed=df.loc[mask_not_aligned, event_col],
        label="Not aligned with XGBoost recommendation"
    )
    kmf_not_aligned.plot(ax=ax, ci_show=True)

    results = logrank_test(
        df.loc[mask_aligned, time_col],
        df.loc[mask_not_aligned, time_col],
        event_observed_A=df.loc[mask_aligned, event_col],
        event_observed_B=df.loc[mask_not_aligned, event_col],
    )
    fh_pvalue = None
    if p is not None and q is not None:
        fh_results = logrank_test(
            df.loc[mask_aligned, time_col],
            df.loc[mask_not_aligned, time_col],
            event_observed_A=df.loc[mask_aligned, event_col],
            event_observed_B=df.loc[mask_not_aligned, event_col],
            weightings="fleming-harrington",
            p=p,
            q=q,
        )
        fh_pvalue = float(fh_results.p_value)
    
    # Run with scikit-survival compare_survival to cross-check log-rank p-value
    y_all = Surv.from_arrays(
        event=df[event_col].astype(bool).values,
        time=df[time_col].values.astype(float),
    )
    group_indicator = np.where(mask_aligned.to_numpy(bool), "aligned", "not_aligned")
    chisq, pvalue = compare_survival(y_all, group_indicator)
    print("scikit-survival compare_survival log-rank p-value:", pvalue)

    tau = 60  # 5 year survival
    rmst_aligned = float(restricted_mean_survival_time(kmf_aligned, t=tau))
    rmst_not_aligned = float(restricted_mean_survival_time(kmf_not_aligned, t=tau))
    rmst_diff = rmst_aligned - rmst_not_aligned

    print("\n[RMST by Treatment Recommendation]")
    print(f"RMST (aligned) at tau={tau:.2f}: {rmst_aligned:.4f}")
    print(f"RMST (not aligned) at tau={tau:.2f}: {rmst_not_aligned:.4f}")
    print(f"RMST difference (aligned - not aligned): {rmst_diff:.4f}")

    # Ordered metrics: Log-rank p-value; Fleming-Harrington p-value; Test C-index; 5-year RMST difference
    print(f"Log-rank test p-value: {float(results.p_value):.4f}")
    if fh_pvalue is not None:
        print(f"Fleming-Harrington test p-value (p={p}, q={q}): {fh_pvalue:.4f}")
    if Cindex is not None:
        print(f"C-index: {float(Cindex):.4f}")
    print(f"5-year RMST difference: {float(rmst_diff):.4f} months")

    plt.title("Kaplan-Meier Survival Curves by Treatment Alignment")
    plt.xlabel("Time")
    plt.ylabel("Survival Probability")
    add_at_risk_counts(kmf_aligned, kmf_not_aligned)
    plt.text(0.1, 0.2, f"Log-rank p-value: {results.p_value:.4f}", transform=plt.gca().transAxes)
    if fh_pvalue is not None:
        plt.text(0.1, 0.15, f"FH({p}, {q}) log-rank p-value: {fh_pvalue:.4f}", transform=plt.gca().transAxes)
    if Cindex is not None:
        plt.text(0.1, 0.1, f"C-index: {float(Cindex):.4f}", transform=plt.gca().transAxes)
    if includeRMST:
        plt.text(0.1, 0.05, f"5-year RMST difference: {rmst_diff:.2f} months", transform=plt.gca().transAxes)

    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(path + "km_alignment_xgb_recommendation.png", dpi=600)
    plt.show(block=False)

    return {
        "tau": tau,
        "rmst_aligned": rmst_aligned,
        "rmst_not_aligned": rmst_not_aligned,
        "rmst_diff": rmst_diff,
        "n_aligned": int(mask_aligned.sum()),
        "n_not_aligned": int(mask_not_aligned.sum()),
        "logrank_pvalue": float(results.p_value),
        "fh_pvalue": fh_pvalue,
    }


def compute_alignment_rmst_diff_xgb(booster, df, genes_main, genes_inter, dup_inter,
                                    feature_names, best_ntree, clin_cols, tau=60,
                                    time_col="OS_MONTHS", event_col="OS_STATUS"):
    """Compute RMST difference (aligned - not aligned) without plotting/log-rank output."""
    df = df.copy()
    df["Adjuvant Chemo"] = df["Adjuvant Chemo"].astype(int)

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

    with threadpool_limits(limits=None):
        risk_treated = predict_xgb_risk(booster, X_treated, feature_names, best_ntree)
        risk_untreated = predict_xgb_risk(booster, X_untreated, feature_names, best_ntree)

    model_rec = np.where(risk_treated < risk_untreated, 1, 0)
    actual = df["Adjuvant Chemo"].to_numpy(int)
    alignment = actual == model_rec

    mask_aligned = alignment
    mask_not_aligned = ~alignment

    if int(mask_aligned.sum()) == 0 or int(mask_not_aligned.sum()) == 0:
        return 0.0

    kmf_aligned = KaplanMeierFitter()
    kmf_not_aligned = KaplanMeierFitter()

    kmf_aligned.fit(
        durations=df.loc[mask_aligned, time_col],
        event_observed=df.loc[mask_aligned, event_col],
    )
    kmf_not_aligned.fit(
        durations=df.loc[mask_not_aligned, time_col],
        event_observed=df.loc[mask_not_aligned, event_col],
    )

    rmst_aligned = float(restricted_mean_survival_time(kmf_aligned, t=tau))
    rmst_not_aligned = float(restricted_mean_survival_time(kmf_not_aligned, t=tau))
    return float(rmst_aligned - rmst_not_aligned)

# ============================================================
# Load data, rank genes (TRAIN only), set budgets
# ============================================================
train_raw = pd.read_csv(TRAIN_CSV)
valid_raw = pd.read_csv(VALID_CSV)
test_raw  = pd.read_csv(TEST_CSV)

# Print number of censored and non-censored in each split and adjuivant chemotherapy status in each
print("Train OS_STATUS value counts:")
print(train_raw["OS_STATUS"].value_counts())
print("Train Adjuvant Chemo value counts:")
print(train_raw["Adjuvant Chemo"].value_counts())

print("Valid OS_STATUS value counts:")
print(valid_raw["OS_STATUS"].value_counts())
print("Valid Adjuvant Chemo value counts:")
print(valid_raw["Adjuvant Chemo"].value_counts())

print("Test OS_STATUS value counts:")
print(test_raw["OS_STATUS"].value_counts())
print("Test Adjuvant Chemo value counts:")
print(test_raw["Adjuvant Chemo"].value_counts())

GENE_LIST = load_genes_list(GENES_CSV)

train_df = preprocess_split(train_raw, CLINICAL_VARS, GENE_LIST)
valid_df = preprocess_split(valid_raw, CLINICAL_VARS, GENE_LIST)
test_df  = preprocess_split(test_raw,  CLINICAL_VARS, GENE_LIST)

# Keep only features present in all splits
feat_candidates = [c for c in (CLINICAL_VARS + GENE_LIST)
                   if c in train_df.columns and c in valid_df.columns and c in test_df.columns]
CLIN_FEATS = [c for c in CLINICAL_VARS if c in feat_candidates]
GENE_FEATS = [g for g in GENE_LIST if g in feat_candidates]
CLIN_FEATS_PRETX = [c for c in CLIN_FEATS if c != "Adjuvant Chemo"]

# Sort by time/status for stability
train_df = train_df.sort_values(by=["OS_MONTHS","OS_STATUS"], ascending=[False, False]).reset_index(drop=True)
valid_df = valid_df.sort_values(by=["OS_MONTHS","OS_STATUS"], ascending=[False, False]).reset_index(drop=True)

# Train-only univariate ranking for genes
GENE_RANK = rank_genes_univariate(train_df, GENE_FEATS)
MAX_GENES = len(GENE_RANK)
print(f"[Gene Ranking] Ranked {MAX_GENES} genes on TRAIN")

# ====== Capacity budgets tied to event count ======
N_EVENTS_TR = int(train_df["OS_STATUS"].sum())
FEAT_EVENT_FRACTION = 0.50
FEAT_BUDGET = max(24, int(FEAT_EVENT_FRACTION * N_EVENTS_TR))   # total inputs incl. clinical
print(f"[Budgets] events(train)={N_EVENTS_TR} → feature budget ≤ {FEAT_BUDGET}")

# ============================================================
# XGBoost Cox setup: DMatrix, custom C-index eval, train helper
# ============================================================
def make_dmatrix(X, time, event, weight=None, feature_names=None):
    y_signed = pack_cox_labels(time, event)  # +t if event, -t if censored
    dm = xgb.DMatrix(X, label=y_signed, weight=weight, feature_names=feature_names)
    return dm

def xgb_cindex_eval(predt, dtrain):
    y_signed = dtrain.get_label()
    times = np.abs(y_signed)
    events = (y_signed > 0.0).astype(int)
    ci = cindex(predt, times, events)
    return ("cindex", ci)  # XGBoost 2.x custom_metric signature

def train_xgb_cox(dtrain, dvalid, params, num_boost_round, early_stopping_rounds):
    evals_result = {}
    with xgb.config_context(verbosity=1):
        booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            evals=[(dtrain, "train"), (dvalid, "valid")],
            custom_metric=xgb_cindex_eval,      # <- custom metric (CPU)
            evals_result=evals_result,
            early_stopping_rounds=early_stopping_rounds,
            maximize=True,                      # higher C-index is better
        )
    return booster, evals_result

# ============================================================
# Optuna: NSGA-II (Val CI ↑, RMST diff ↑) + deterministic bootstrap tuning
# ============================================================
def suggest_hparams(trial):
    max_nonclin = max(8, FEAT_BUDGET - len(CLIN_FEATS))

    # Reserve headroom for interactions; otherwise k_int is forced to zero.
    max_main = max(1, max_nonclin - 1)

    base_main  = [16, 32, 64, 96, 128, 192, 256, 384, 512, MAX_GENES]
    TOPK_MAIN_CHOICES = tuple(sorted({k for k in base_main if 1 <= k <= min(MAX_GENES, max_main)}))
    if len(TOPK_MAIN_CHOICES) == 0:
        TOPK_MAIN_CHOICES = (min(MAX_GENES, max_main),)
    top_k_genes = int(trial.suggest_categorical("top_k_genes", TOPK_MAIN_CHOICES))

    k_main = int(min(top_k_genes, max_main))
    k_int_cap = int(max(0, min(k_main, max_nonclin - k_main)))

    base_inter = [0, 8, 16, 32, 64]
    TOPK_INTER_CHOICES = tuple(sorted({k for k in base_inter if 0 <= k <= k_int_cap}))
    if len(TOPK_INTER_CHOICES) == 0:
        TOPK_INTER_CHOICES = (0,)
    inter_param_name = f"top_k_inter_cap_{k_int_cap}"
    top_k_inter = int(trial.suggest_categorical(inter_param_name, TOPK_INTER_CHOICES))

    k_int = int(min(top_k_inter, k_int_cap))

    # Optional duplication of interaction columns (1 = off)
    dup_inter = 1 #trial.suggest_int("dup_inter", 1, 3)

    # XGBoost params (CPU)
    params = {
        "objective": "survival:cox",
        "booster": "gbtree",
        "tree_method": "hist",                  # CPU histogram
        "disable_default_eval_metric": True,    # c-index
        "eta": trial.suggest_float("eta", 0.01, 0.12, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 6),
        "min_child_weight": trial.suggest_float("min_child_weight", 2.0, 30.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 6.0),
        "subsample": trial.suggest_float("subsample", 0.65, 0.95),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.6, 1.0),
        "colsample_bynode": trial.suggest_float("colsample_bynode", 0.6, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 3.0, 60.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 3.0),
        "max_bin": trial.suggest_int("max_bin", 128, 512),
        "seed": 42,
        "nthread": -1,
    }
    num_boost_round = trial.suggest_int("num_boost_round", 600, 3200, step=100)
    early_stopping_rounds = trial.suggest_int("early_stopping_rounds", 75, 175, step=25)

    return k_main, k_int, dup_inter, params, num_boost_round, early_stopping_rounds

# Precompute IPTW on Train and Valid (prevalence anchored to TRAIN)
w_tr, ps_model, pi_tr = compute_iptw(train_df, covariate_cols=CLIN_FEATS_PRETX, act_col="Adjuvant Chemo")
w_va, _, _ = compute_iptw(valid_df, covariate_cols=CLIN_FEATS_PRETX, act_col="Adjuvant Chemo",
                          ref_prev=pi_tr, model=ps_model)

# Bootstrap tuning on TRAIN; each fitted model is scored on the fixed VALID split.
BOOTSTRAP_TUNE_N = 5
BOOTSTRAP_BASE_SEED = 42
BOOTSTRAP_MAX_RESAMPLE_TRIES = 256

def bootstrap_resample_df(df, seed, require_two_arms=False, require_event=True):
    rng = np.random.default_rng(int(seed))
    n = len(df)
    for _ in range(BOOTSTRAP_MAX_RESAMPLE_TRIES):
        idx = rng.integers(0, n, size=n)
        boot = df.iloc[idx].copy()
        if require_event and int(boot["OS_STATUS"].sum()) == 0:
            continue
        if require_two_arms and boot["Adjuvant Chemo"].nunique() < 2:
            continue
        return boot.sort_values(by=["OS_MONTHS", "OS_STATUS"], ascending=[False, False]).reset_index(drop=True)
    raise RuntimeError("Could not draw a valid bootstrap sample with events and both treatment arms.")

def build_trial_mats_for_splits(train_fit_df, valid_eval_df, w_fit, w_eval, k_main, k_int, dup_inter):
    genes_main  = GENE_RANK[:k_main]
    genes_inter = genes_main[:k_int]
    Xtr_raw, feat_names = build_features_with_interactions(
        train_fit_df, genes_main, genes_inter, dup_inter=dup_inter, clin_cols=CLIN_FEATS
    )
    Xva_raw, _ = build_features_with_interactions(
        valid_eval_df, genes_main, genes_inter, dup_inter=dup_inter, clin_cols=CLIN_FEATS
    )

    Xtr = Xtr_raw.astype(np.float32)
    Xva = Xva_raw.astype(np.float32)

    ytr = pack_cox_labels(train_fit_df["OS_MONTHS"].values, train_fit_df["OS_STATUS"].values)
    yva = pack_cox_labels(valid_eval_df["OS_MONTHS"].values,  valid_eval_df["OS_STATUS"].values)

    dtr = xgb.DMatrix(Xtr, label=ytr, weight=w_fit, feature_names=feat_names)
    dva = xgb.DMatrix(Xva, label=yva, weight=w_eval, feature_names=feat_names)
    return dtr, dva, feat_names, genes_main, genes_inter

def objective(trial):
    k_main, k_int, dup_inter, params, num_boost_round, esr = suggest_hparams(trial)
    boot_val_cis = []
    boot_rmst_diffs = []
    boot_best_ntrees = []
    n_features = None

    for b in range(BOOTSTRAP_TUNE_N):
        boot_seed = BOOTSTRAP_BASE_SEED + trial.number * 1000 + b
        boot_train_df = bootstrap_resample_df(
            train_df,
            seed=boot_seed,
            require_two_arms=True,
            require_event=True,
        )
        w_tr_boot, _, _ = compute_iptw(
            boot_train_df,
            covariate_cols=CLIN_FEATS_PRETX,
            act_col="Adjuvant Chemo",
        )
        dtr, dva, feat_names, genes_main, genes_inter = build_trial_mats_for_splits(
            boot_train_df,
            valid_df,
            w_fit=w_tr_boot,
            w_eval=None,
            k_main=k_main,
            k_int=k_int,
            dup_inter=dup_inter,
        )
        params_boot = dict(params)
        params_boot["seed"] = int(boot_seed)

        booster, _ = train_xgb_cox(dtr, dva, params_boot, num_boost_round, esr)

        yva_signed = dva.get_label()
        tva = np.abs(yva_signed)
        eva = (yva_signed > 0).astype(int)

        best_ntree = booster.best_iteration + 1 if booster.best_iteration is not None else num_boost_round
        va_pred = booster.predict(dva, iteration_range=(0, best_ntree), output_margin=True)
        val_ci = float(cindex(va_pred, tva, eva))
        rmst_diff = compute_alignment_rmst_diff_xgb(
            booster,
            valid_df,
            genes_main=genes_main,
            genes_inter=genes_inter,
            dup_inter=dup_inter,
            feature_names=feat_names,
            best_ntree=best_ntree,
            clin_cols=CLIN_FEATS,
            tau=60,
        )

        boot_val_cis.append(float(val_ci))
        boot_rmst_diffs.append(float(rmst_diff))
        boot_best_ntrees.append(int(best_ntree))
        n_features = len(dtr.feature_names)

    median_val_ci = float(np.median(boot_val_cis))
    median_rmst_diff = float(np.median(boot_rmst_diffs))
    val_ci_se = float(np.std(boot_val_cis, ddof=1)) if len(boot_val_cis) > 1 else 0.0
    rmst_iqr = float(np.percentile(boot_rmst_diffs, 75) - np.percentile(boot_rmst_diffs, 25)) if len(boot_rmst_diffs) > 1 else 0.0
    best_ntree_median = int(round(np.median(boot_best_ntrees))) if boot_best_ntrees else int(num_boost_round)

    trial.set_user_attr("n_features", int(n_features))
    trial.set_user_attr("k_main", int(k_main))
    trial.set_user_attr("k_int", int(k_int))
    trial.set_user_attr("dup_inter", int(dup_inter))
    trial.set_user_attr("best_ntree", int(best_ntree_median))
    trial.set_user_attr("val_ci", float(median_val_ci))
    trial.set_user_attr("rmst_diff", float(median_rmst_diff))
    trial.set_user_attr("val_ci_boot_median", float(median_val_ci))
    trial.set_user_attr("val_ci_boot_se", float(val_ci_se))
    trial.set_user_attr("rmst_diff_boot_median", float(median_rmst_diff))
    trial.set_user_attr("rmst_diff_boot_iqr", float(rmst_iqr))
    trial.set_user_attr("bootstrap_n", int(BOOTSTRAP_TUNE_N))

    print(
        f"[Trial {trial.number:03d}] Boot Val CI median({BOOTSTRAP_TUNE_N})={median_val_ci:.4f} "
        f"(SE={val_ci_se:.4f}), RMST diff median({BOOTSTRAP_TUNE_N})={float(median_rmst_diff):.4f} "
        f"(IQR={rmst_iqr:.4f}), N_feats={n_features}, K_main={k_main}, K_int={k_int}, Dup={dup_inter}, "
        f"N_tree={best_ntree_median}"
    )

    return median_val_ci, float(median_rmst_diff)

# ---- Run study ----
RUN_TAG = f"3-13-mo-ci-rmst-boot{BOOTSTRAP_TUNE_N}"
storage = f"sqlite:///xgb_cox_optuna_mar12_multiobj_boot{BOOTSTRAP_TUNE_N}.db"
study_name = f"xgb_cox_{RUN_TAG}"
study = optuna.create_study(
    directions=["maximize", "maximize"],
    study_name=study_name,
    storage=storage,
    load_if_exists=True,
    sampler=NSGAIISampler(seed=42),
)
N_TRIALS = 0 # adjust as needed
print(f"Starting bootstrap optimization: {N_TRIALS} trials x {BOOTSTRAP_TUNE_N} bootstraps/trial")
study.optimize(objective, n_trials=N_TRIALS, gc_after_trial=True)

# ---- Multi-objective selection from Pareto front ----
candidates = [
    t for t in study.best_trials
    if t.state == optuna.trial.TrialState.COMPLETE and t.values is not None
]

if not candidates:
    candidates = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.values is not None
    ]

if not candidates:
    raise RuntimeError("No completed multi-objective trials found. Increase N_TRIALS and rerun.")

ci_vals = np.array([float(t.values[0]) for t in candidates], dtype=float)
rmst_vals = np.array([float(t.values[1]) for t in candidates], dtype=float)

ci_min, ci_max = float(ci_vals.min()), float(ci_vals.max())
rmst_min, rmst_max = float(rmst_vals.min()), float(rmst_vals.max())

def _norm(x, lo, hi):
    return 0.0 if hi <= lo else (x - lo) / (hi - lo)

scored = []
for t in candidates:
    ci = float(t.values[0])
    rmst = float(t.values[1])
    score = _norm(ci, ci_min, ci_max) + _norm(rmst, rmst_min, rmst_max)
    scored.append((score, t))

scored.sort(key=lambda z: z[0], reverse=True)
chosen = scored[0][1]

print(f"\n[Pareto] Candidates: {len(candidates)}")
print(f"[Pareto] Selected compromise trial #{chosen.number}")
print(f"[Pareto] Selected values: Boot median Val CI={float(chosen.values[0]):.4f}, Boot median RMST diff={float(chosen.values[1]):.4f}")
print("[Chosen Params]", chosen.params)
print("[Chosen Attrs] k_main=%s k_int=%s dup_inter=%s best_ntree=%s ci_se=%s rmst_iqr=%s boot_n=%s" %
      (str(chosen.user_attrs.get("k_main")), str(chosen.user_attrs.get("k_int")),
       str(chosen.user_attrs.get("dup_inter")), str(chosen.user_attrs.get("best_ntree")),
       str(chosen.user_attrs.get("val_ci_boot_se")), str(chosen.user_attrs.get("rmst_diff_boot_iqr")),
       str(chosen.user_attrs.get("bootstrap_n"))))

# ============================================================
# Final training on Train+Val with chosen hyperparams + IPTW
# ============================================================
best_hp = chosen.params
k_main = int(chosen.user_attrs["k_main"])
k_int  = int(chosen.user_attrs["k_int"])
dup_inter = int(chosen.user_attrs.get("dup_inter", 1))

# Assemble Train+Val and Test
trainval_df = pd.concat([train_df, valid_df], axis=0, ignore_index=True)
trainval_df = trainval_df.sort_values(by=["OS_MONTHS","OS_STATUS"], ascending=[False, False]).reset_index(drop=True)

genes_main  = GENE_RANK[:k_main]
genes_inter = genes_main[:k_int]
X_trv_raw, feat_names = build_features_with_interactions(
    trainval_df, genes_main, genes_inter, dup_inter=dup_inter, clin_cols=CLIN_FEATS
)
X_te_raw, _ = build_features_with_interactions(
    test_df, genes_main, genes_inter, dup_inter=dup_inter, clin_cols=CLIN_FEATS
)

X_trv = X_trv_raw.astype(np.float32)
X_te  = X_te_raw.astype(np.float32)

# Labels and IPTW
y_trv = pack_cox_labels(trainval_df["OS_MONTHS"].values, trainval_df["OS_STATUS"].values)
y_te  = pack_cox_labels(test_df["OS_MONTHS"].values,     test_df["OS_STATUS"].values)

w_trv, ps_model_fin, pi_fin = compute_iptw(trainval_df, covariate_cols=CLIN_FEATS_PRETX)
w_te, _, _ = compute_iptw(test_df, covariate_cols=CLIN_FEATS_PRETX, ref_prev=pi_fin, model=ps_model_fin)

# DMatrices
d_trv = xgb.DMatrix(X_trv, label=y_trv, weight=w_trv, feature_names=feat_names)
d_te  = xgb.DMatrix(X_te,  label=y_te,  weight=w_te,  feature_names=feat_names)

# Params and rounds (CPU)
params_fin = {
    "objective": "survival:cox",
    "booster": "gbtree",
    "tree_method": "hist",               # CPU histogram
    "disable_default_eval_metric": True, # custom_metric only
    "seed": 7,
    "nthread": -1,
    **{
        k: v for k, v in best_hp.items()
        if k in {
            "eta", "max_depth", "min_child_weight", "gamma",
            "subsample", "colsample_bytree", "colsample_bylevel",
            "colsample_bynode", "reg_lambda", "reg_alpha", "max_bin"
        }
    }
}
num_boost_round_fin = int(best_hp.get("num_boost_round", 1600))
early_stopping_rounds_fin = int(best_hp.get("early_stopping_rounds", 125))

# Split Train+Val for ES
event_mask = (y_trv > 0).astype(int)
idx = np.arange(len(y_trv))
tr_idx, va_idx = train_test_split(idx, test_size=0.25, random_state=42, stratify=event_mask)
d_tr_es = xgb.DMatrix(X_trv[tr_idx], label=y_trv[tr_idx], weight=w_trv[tr_idx], feature_names=feat_names)
d_va_es = xgb.DMatrix(X_trv[va_idx], label=y_trv[va_idx], weight=w_trv[va_idx], feature_names=feat_names)

booster_final, evr_final = train_xgb_cox(d_tr_es, d_va_es, params_fin,
                                         num_boost_round_fin, early_stopping_rounds_fin)
best_ntree_final = booster_final.best_iteration + 1 if booster_final.best_iteration is not None else num_boost_round_fin
booster_final_best = slice_booster_to_best_iteration(booster_final, best_ntree_final)

# Evaluate Train+Val and Test at the best iteration
pred_trv = booster_final.predict(d_trv, iteration_range=(0, best_ntree_final), output_margin=True)
pred_te  = booster_final.predict(d_te,  iteration_range=(0, best_ntree_final), output_margin=True)

t_trv = np.abs(y_trv); e_trv = (y_trv > 0).astype(int)
t_te  = np.abs(y_te);  e_te  = (y_te  > 0).astype(int)

ci_trv = cindex(pred_trv, t_trv, e_trv)
ci_te  = cindex(pred_te,  t_te,  e_te)

print(f"\n[Final XGB-Cox] Train+Val CI: {ci_trv:.4f}")
print(f"[Final XGB-Cox] Test CI:      {ci_te:.4f}")

# Per-arm C-indices (sanity)
act_trv = trainval_df["Adjuvant Chemo"].to_numpy(int)
act_te  = test_df["Adjuvant Chemo"].to_numpy(int)
def ci_by_arm(pred, t, e, arm):
    out = {}
    for label, mask in [("ACT=1", arm==1), ("ACT=0", arm==0)]:
        out[label] = cindex(pred[mask], t[mask], e[mask]) if mask.sum() >= 3 else np.nan
    return out
print("[Train+Val] CI by arm:", ci_by_arm(pred_trv, t_trv, e_trv, act_trv))
print("[Test]      CI by arm:", ci_by_arm(pred_te,  t_te,  e_te,  act_te))

# Save artifacts
OUT_DIR = f"xgb_cox_interactions_iptw_bounded-{RUN_TAG}-pareto"
os.makedirs(OUT_DIR, exist_ok=True)
booster_final_best.save_model(os.path.join(OUT_DIR, "xgb_cox_final.json"))
with open(os.path.join(OUT_DIR, "chosen_params.txt"), "w") as f:
    f.write(str(best_hp))
with open(os.path.join(OUT_DIR, "features_used.txt"), "w") as f:
    f.write("\n".join(feat_names))
print("Saved final model and parameters to:", OUT_DIR)

# Kaplan-Meier alignment on test set using final model
print("\n[KM Alignment] Test set vs. XGBoost recommendation")
compare_treatment_recommendation_km_xgb(
    booster_final_best,
    test_df,
    genes_main=genes_main,
    genes_inter=genes_inter,
    dup_inter=dup_inter,
    feature_names=feat_names,
    best_ntree=best_ntree_final,
    clin_cols=CLIN_FEATS,
    path=OUT_DIR + "/",
    includeRMST=True,
    p=1,
    q=0,
    Cindex=ci_te,
)
