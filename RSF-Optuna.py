import os, math, gc, time, random, warnings, inspect
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest

from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.plotting import add_at_risk_counts
from lifelines.utils import restricted_mean_survival_time
from threadpoolctl import threadpool_limits
from sklearn.inspection import permutation_importance

import optuna

warnings.filterwarnings(
    "ignore",
    message="Ties in event time detected; using efron's method to handle ties."
)

np.random.seed(42)
random.seed(42)

# ---------- Paths ----------
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
    # Map treatment to 0/1 if needed
    if "Adjuvant Chemo" in df.columns:
        df["Adjuvant Chemo"] = df["Adjuvant Chemo"].map({"OBS": 0, "ACT": 1})

    for col in ["Adjuvant Chemo", "IS_MALE"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    keep_cols = [c for c in clinical_vars if c in df.columns] + [g for g in gene_names if g in df.columns]
    cols = ["OS_STATUS", "OS_MONTHS"] + keep_cols
    return df[cols].copy()


def rank_genes_univariate(train_df, gene_cols):
    y = Surv.from_arrays(
        event=train_df["OS_STATUS"].astype(bool).values,
        time=train_df["OS_MONTHS"].values.astype(float),
    )
    ranks = []
    for g in gene_cols:
        Xg = train_df[[g]].to_numpy(dtype=np.float64)
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
        clin_cols = CLINICAL_VARS  # original behavior

    base_cols = list(clin_cols) + list(main_genes)  # keep ACT main effect in clinicals
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
    return w.astype(np.float64), model, float(ref_prev)


def cindex(pred, time, event):
    return float(concordance_index_censored(event.astype(bool), time.astype(float), pred)[0])


def cindex_bootstrap(pred, time, event, n_bootstraps=100, seed=42):
    """Computes C-index and its bootstrap standard error."""
    rng = np.random.RandomState(seed)
    n_samples = len(time)
    cis = []
    for _ in range(n_bootstraps):
        indices = rng.choice(np.arange(n_samples), size=n_samples, replace=True)
        if len(np.unique(event[indices])) < 2:
            continue
        try:
            ci = cindex(pred[indices], time[indices], event[indices])
            cis.append(ci)
        except Exception:
            continue

    if not cis:
        return 0.5, 0.5
    return float(np.mean(cis)), float(np.std(cis))


# RSF constructor that filters params to whatever your installed version supports.
def make_rsf(**params):
    sig = inspect.signature(RandomSurvivalForest.__init__)
    allowed = set(sig.parameters.keys())
    allowed.discard("self")
    filtered = {k: v for k, v in params.items() if k in allowed}
    return RandomSurvivalForest(**filtered)


def compare_treatment_recommendation_km_rsf(model, df, genes_main, genes_inter, dup_inter,
                                           clin_cols,
                                           time_col="OS_MONTHS", event_col="OS_STATUS", path =""):
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

    with threadpool_limits(limits=None):
        # RSF predict() returns a risk score: higher = worse
        risk_treated = model.predict(X_treated.astype(np.float64))
        risk_untreated = model.predict(X_untreated.astype(np.float64))

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
    )
    print("Log-rank test p-value:", results.p_value)
    
    tau = 60 # 5 year survival
    rmst_aligned = float(restricted_mean_survival_time(kmf_aligned, t=tau))
    rmst_not_aligned = float(restricted_mean_survival_time(kmf_not_aligned, t=tau))
    rmst_diff = rmst_aligned - rmst_not_aligned

    print("\n[RMST by Treatment Recommendation]")
    print(f"RMST (aligned) at tau={tau:.2f}: {rmst_aligned:.4f}")
    print(f"RMST (not aligned) at tau={tau:.2f}: {rmst_not_aligned:.4f}")
    print(f"RMST difference (aligned - not aligned): {rmst_diff:.4f}")

    plt.title("Kaplan-Meier Survival Curves by Treatment Alignment")
    plt.xlabel("Time")
    plt.ylabel("Survival Probability")
    add_at_risk_counts(kmf_aligned, kmf_not_aligned)
    plt.text(0.1, 0.1, f"Log-rank p-value: {results.p_value:.4f}", transform=plt.gca().transAxes)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(path + "km_alignment_rsf_recommendation.png", dpi=600)
    plt.show(block=False)
    
    return {
        "tau": tau,
        "rmst_aligned": rmst_aligned,
        "rmst_not_aligned": rmst_not_aligned,
        "rmst_diff": rmst_diff,
        "n_aligned": int(mask_aligned.sum()),
        "n_not_aligned": int(mask_not_aligned.sum()),
        "logrank_pvalue": float(results.p_value)
    }


def compute_alignment_rmst_diff_rsf(model, df, genes_main, genes_inter, dup_inter,
                                   clin_cols, tau=60,
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
        risk_treated = model.predict(X_treated.astype(np.float64))
        risk_untreated = model.predict(X_untreated.astype(np.float64))

    model_rec = np.where(risk_treated < risk_untreated, 1, 0)
    actual = df["Adjuvant Chemo"].to_numpy(int)
    alignment = actual == model_rec

    mask_aligned = alignment
    mask_not_aligned = ~alignment

    # Guard against degenerate splits
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
CLIN_FEATS_PRETX = [c for c in CLIN_FEATS if c != "Adjuvant Chemo"]  # safer than original

# Sort by time/status for stability
train_df = train_df.sort_values(by=["OS_MONTHS", "OS_STATUS"], ascending=[False, False]).reset_index(drop=True)
valid_df = valid_df.sort_values(by=["OS_MONTHS", "OS_STATUS"], ascending=[False, False]).reset_index(drop=True)

# Train-only univariate ranking for genes
GENE_RANK = rank_genes_univariate(train_df, GENE_FEATS)
MAX_GENES = len(GENE_RANK)
print(f"[Gene Ranking] Ranked {MAX_GENES} genes on TRAIN")

# ====== Capacity budgets tied to event count ======
N_EVENTS_TR = int(train_df["OS_STATUS"].sum())
FEAT_EVENT_FRACTION = 0.50
FEAT_BUDGET = max(24, int(FEAT_EVENT_FRACTION * N_EVENTS_TR))  # total inputs incl. clinical
print(f"[Budgets] events(train)={N_EVENTS_TR} → feature budget ≤ {FEAT_BUDGET}")

# Multi-objective RMST settings (used inside Optuna objective)
RMST_OPT_N_RUNS = 10
RMST_OPT_SEEDS = list(range(1, RMST_OPT_N_RUNS + 1))


# ============================================================
# Optuna: single-objective (Val CI ↑) with RSF + interactions
# ============================================================
def suggest_hparams(trial):
    max_nonclin = max(8, FEAT_BUDGET - len(CLIN_FEATS))

    # Same gene menus
    base_main = [16, 32, 64, 96, 128, 192, 256, 384, 512, MAX_GENES]
    TOPK_MAIN_CHOICES = tuple(sorted({k for k in base_main if k <= MAX_GENES}))
    top_k_genes = int(trial.suggest_categorical("top_k_genes", TOPK_MAIN_CHOICES))

    base_inter = [0]
    TOPK_INTER_CHOICES = tuple(sorted({k for k in base_inter if k <= MAX_GENES}))
    inter_ratio = trial.suggest_float("inter_ratio", 0.75, 1.25)
    top_k_inter_raw = int(min(int(round(inter_ratio * top_k_genes)), max(base_inter)))

    # Budget clamp
    k_main = int(min(top_k_genes, max_nonclin))
    k_int  = int(min(top_k_inter_raw, k_main, max_nonclin - k_main))

    # Optional duplication of interaction columns (1 = off)
    dup_inter = 1  # trial.suggest_int("dup_inter", 1, 3)

    # ---- RSF hyperparameters ----
    # Important knobs: n_estimators, max_features, min_samples_leaf/split, max_depth
    n_estimators = trial.suggest_int("n_estimators", 200, 2000, step=100)

    max_depth = trial.suggest_int("max_depth", 2, 10)

    min_samples_split = trial.suggest_int("min_samples_split", 2, 50)
    min_samples_leaf  = trial.suggest_int("min_samples_leaf", 20, 150)

    # Choose how to set max_features
    mf_mode = trial.suggest_categorical("max_features_mode", ["sqrt", "log2", "all", "frac"])
    if mf_mode == "all":
        max_features = None
    elif mf_mode == "frac":
        max_features = trial.suggest_float("max_features_frac", 0.2, 1.0)
    else:
        max_features = mf_mode

    rsf_params = dict(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        min_weight_fraction_leaf=0.0,  # can tune if you want weights to constrain growth more
        max_features=max_features,
        max_leaf_nodes=None,
        bootstrap=True,
        oob_score=False,
        n_jobs=-1,
        random_state=42,
        verbose=0,
        warm_start=False,
    )

    return k_main, k_int, dup_inter, rsf_params


# Precompute IPTW on Train and Valid (prevalence anchored to TRAIN)
w_tr, ps_model, pi_tr = compute_iptw(train_df, covariate_cols=CLIN_FEATS_PRETX, act_col="Adjuvant Chemo")
w_va, _, _ = compute_iptw(valid_df, covariate_cols=CLIN_FEATS_PRETX, act_col="Adjuvant Chemo",
                          ref_prev=pi_tr, model=ps_model)


def build_trial_mats(k_main, k_int, dup_inter):
    genes_main  = GENE_RANK[:k_main]
    genes_inter = genes_main[:k_int]

    Xtr_raw, feat_names = build_features_with_interactions(
        train_df, genes_main, genes_inter, dup_inter=dup_inter, clin_cols=CLIN_FEATS
    )
    Xva_raw, _ = build_features_with_interactions(
        valid_df, genes_main, genes_inter, dup_inter=dup_inter, clin_cols=CLIN_FEATS
    )

    Xtr = Xtr_raw.astype(np.float64)
    Xva = Xva_raw.astype(np.float64)

    ytr = Surv.from_arrays(
        event=train_df["OS_STATUS"].astype(bool).values,
        time=train_df["OS_MONTHS"].astype(float).values
    )
    yva = Surv.from_arrays(
        event=valid_df["OS_STATUS"].astype(bool).values,
        time=valid_df["OS_MONTHS"].astype(float).values
    )

    return Xtr, ytr, w_tr, Xva, yva, w_va, feat_names, genes_main, genes_inter


def objective(trial):
    k_main, k_int, dup_inter, rsf_params = suggest_hparams(trial)
    Xtr, ytr, wtr, Xva, yva, wva, feat_names, genes_main, genes_inter = build_trial_mats(k_main, k_int, dup_inter)

    tva = valid_df["OS_MONTHS"].values.astype(float)
    eva = valid_df["OS_STATUS"].values.astype(int)

    va_cis = []
    rmst_diffs = []

    for seed in RMST_OPT_SEEDS:
        rsf_params["random_state"] = int(seed)
        rsf = make_rsf(**rsf_params)
        rsf.fit(Xtr, ytr, sample_weight=wtr)

        va_pred = rsf.predict(Xva)
        va_ci = cindex(va_pred, tva, eva)
        va_cis.append(float(va_ci))

        rmst_diff = compute_alignment_rmst_diff_rsf(
            rsf,
            valid_df,
            genes_main=genes_main,
            genes_inter=genes_inter,
            dup_inter=dup_inter,
            clin_cols=CLIN_FEATS,
            tau=60,
        )
        rmst_diffs.append(float(rmst_diff))

    median_val_ci = float(np.median(va_cis))
    median_rmst_diff = float(np.median(rmst_diffs))
    val_ci_se = float(np.std(va_cis, ddof=1)) if len(va_cis) > 1 else 0.0
    rmst_iqr = float(np.percentile(rmst_diffs, 75) - np.percentile(rmst_diffs, 25)) if len(rmst_diffs) > 1 else 0.0

    trial.set_user_attr("n_features", int(Xtr.shape[1]))
    trial.set_user_attr("k_main", int(k_main))
    trial.set_user_attr("k_int", int(k_int))
    trial.set_user_attr("dup_inter", int(dup_inter))
    trial.set_user_attr("val_ci_median_10runs", float(median_val_ci))
    trial.set_user_attr("val_ci_se_10runs", float(val_ci_se))
    trial.set_user_attr("median_rmst_diff_10runs", float(median_rmst_diff))
    trial.set_user_attr("iqr_rmst_diff_10runs", float(rmst_iqr))
    trial.set_user_attr("rmst_runs_n", int(RMST_OPT_N_RUNS))

    print(
        f"[Trial {trial.number:03d}] Val CI median(10)={median_val_ci:.4f} (SE={val_ci_se:.4f}), "
        f"RMST diff median(10)={median_rmst_diff:.4f} (IQR={rmst_iqr:.4f}), "
        f"N_feats={Xtr.shape[1]}, K_main={k_main}, K_int={k_int}, Dup={dup_inter}, "
        f"n_estimators={rsf_params['n_estimators']}"
    )

    # Multi-objective: maximize validation C-index and median RMST difference
    return median_val_ci, median_rmst_diff


# ---- Run study ----
RUN_TAG = "3-8-mo-ci-rmst10"
storage = "sqlite:///rsf_optuna_mar8_multiobj.db"
study_name = f"rsf_{RUN_TAG}"

study = optuna.create_study(
    directions=["maximize", "maximize"],
    study_name=study_name,
    storage=storage,
    load_if_exists=True,
    sampler=optuna.samplers.NSGAIISampler(seed=42),
)

N_TRIALS = 100
print(f"Starting optimization: {N_TRIALS} trials")
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
best_trial = chosen

print(f"\n[Pareto] Candidates: {len(candidates)}")
print(f"[Pareto] Selected compromise trial #{chosen.number}")
print(f"[Pareto] Selected values: Val CI={float(chosen.values[0]):.4f}, Median RMST diff={float(chosen.values[1]):.4f}")
print("[Chosen Params]", chosen.params)
print("[Chosen Attrs] k_main=%s k_int=%s dup_inter=%s" %
      (str(chosen.user_attrs.get("k_main")),
       str(chosen.user_attrs.get("k_int")),
       str(chosen.user_attrs.get("dup_inter"))))


# ============================================================
# Model selection: Pareto compromise, Pareto best CI, Highest median RMST diff
# ============================================================
def build_rsf_params_from_trial(trial, n_jobs=1):
    hp = trial.params
    mf_mode = hp.get("max_features_mode", "sqrt")
    if mf_mode == "all":
        max_features = None
    elif mf_mode == "frac":
        max_features = float(hp.get("max_features_frac", 1.0))
    else:
        max_features = mf_mode

    return dict(
        n_estimators=int(hp.get("n_estimators", 1000)),
        max_depth=hp.get("max_depth", None),
        min_samples_split=int(hp.get("min_samples_split", 6)),
        min_samples_leaf=int(hp.get("min_samples_leaf", 3)),
        min_weight_fraction_leaf=0.0,
        max_features=max_features,
        max_leaf_nodes=None,
        bootstrap=True,
        oob_score=False,
        n_jobs=n_jobs,
        random_state=None,
        verbose=0,
        warm_start=False,
    )


def get_trial_gene_sets(trial):
    k_main = int(trial.user_attrs["k_main"])
    k_int = int(trial.user_attrs["k_int"])
    dup_inter = int(trial.user_attrs.get("dup_inter", 1))
    genes_main = GENE_RANK[:k_main]
    genes_inter = genes_main[:k_int]
    return k_main, k_int, dup_inter, genes_main, genes_inter


def build_feature_mats_for_trial(trial, train_fit_df, eval_df):
    _, _, dup_inter, genes_main, genes_inter = get_trial_gene_sets(trial)
    X_trv_raw, feat_names = build_features_with_interactions(
        train_fit_df, genes_main, genes_inter, dup_inter=dup_inter, clin_cols=CLIN_FEATS
    )
    X_te_raw, _ = build_features_with_interactions(
        eval_df, genes_main, genes_inter, dup_inter=dup_inter, clin_cols=CLIN_FEATS
    )
    X_trv = X_trv_raw.astype(np.float64)
    X_te = X_te_raw.astype(np.float64)
    y_trv = Surv.from_arrays(
        event=train_fit_df["OS_STATUS"].astype(bool).values,
        time=train_fit_df["OS_MONTHS"].astype(float).values
    )
    y_te = Surv.from_arrays(
        event=eval_df["OS_STATUS"].astype(bool).values,
        time=eval_df["OS_MONTHS"].astype(float).values
    )
    return X_trv, X_te, y_trv, y_te, feat_names, genes_main, genes_inter, dup_inter


def evaluate_trial_seeds(
    trial,
    train_fit_df,
    eval_df,
    w_fit,
    seed_list,
    label="",
    eval_label="Eval",
    n_jobs=1,
    checkpoint_csv_path=None,
    checkpoint_every=1,
):
    rsf_params = build_rsf_params_from_trial(trial, n_jobs=n_jobs)
    X_trv, X_te, y_trv, y_te, feat_names, genes_main, genes_inter, dup_inter = build_feature_mats_for_trial(
        trial, train_fit_df, eval_df
    )

    t_trv = train_fit_df["OS_MONTHS"].values.astype(float)
    e_trv = train_fit_df["OS_STATUS"].values.astype(int)
    t_te = eval_df["OS_MONTHS"].values.astype(float)
    e_te = eval_df["OS_STATUS"].values.astype(int)

    all_results = []
    all_pvalues = []
    all_ci_test = []
    all_rmst_following = []
    all_rmst_not_following = []
    all_rmst_diffs = []

    print(f"\n{'='*60}")
    print(f"Running RSF {len(seed_list)} seeds for {label}")
    print(f"{'='*60}\n")

    for run_idx, seed in enumerate(seed_list, 1):
        rsf_params["random_state"] = seed

        with threadpool_limits(limits=None):
            rsf_final = make_rsf(**rsf_params)
            rsf_final.fit(X_trv, y_trv, sample_weight=w_fit)

            pred_trv = rsf_final.predict(X_trv)
            pred_te = rsf_final.predict(X_te)

            ci_trv = cindex(pred_trv, t_trv, e_trv)
            ci_te = cindex(pred_te, t_te, e_te)
            all_ci_test.append(ci_te)

            km_results = compare_treatment_recommendation_km_rsf(
                rsf_final,
                eval_df,
                genes_main=genes_main,
                genes_inter=genes_inter,
                dup_inter=dup_inter,
                clin_cols=CLIN_FEATS,
                path=""
            )

            pvalue = km_results["logrank_pvalue"]
            rmst_diff = km_results["rmst_diff"]
            rmst_aligned = km_results["rmst_aligned"]
            rmst_not_aligned = km_results["rmst_not_aligned"]

            all_pvalues.append(pvalue)
            all_rmst_following.append(rmst_aligned)
            all_rmst_not_following.append(rmst_not_aligned)
            all_rmst_diffs.append(rmst_diff)

            all_results.append({
                "run": run_idx,
                "seed": seed,
                "ci_trainval": ci_trv,
                "ci_test": ci_te,
                "logrank_pvalue": pvalue,
                "rmst_following": rmst_aligned,
                "rmst_not_following": rmst_not_aligned,
                "rmst_diff": rmst_diff,
                "n_aligned": km_results["n_aligned"],
                "n_not_aligned": km_results["n_not_aligned"],
            })

            if checkpoint_csv_path is not None and (run_idx % int(checkpoint_every) == 0 or run_idx == len(seed_list)):
                pd.DataFrame(all_results).to_csv(checkpoint_csv_path, index=False)

            plt.close("all")

    return {
        "all_results": all_results,
        "all_pvalues": all_pvalues,
        "all_ci_test": all_ci_test,
        "all_rmst_following": all_rmst_following,
        "all_rmst_not_following": all_rmst_not_following,
        "all_rmst_diffs": all_rmst_diffs,
        "feat_names": feat_names,
        "genes_main": genes_main,
        "genes_inter": genes_inter,
        "dup_inter": dup_inter,
    }

# Model 1: Best model
best_model_trial = best_trial

# Model 2: Pareto candidate with highest validation CI
if candidates:
    one_se_nearest = max(candidates, key=lambda t: float(t.values[0]))
else:
    print("**PAY ATTENTION: No Pareto CI-max candidate found, defaulting to best trial.**")
    one_se_nearest = best_trial

# Model 3: Highest median RMST diff among Pareto candidates
seed_list = list(range(1, 31))
candidate_rmst_summary = []
best_rmst_trial = one_se_nearest

OUT_DIR = f"rsf_interactions_iptw_bounded-{RUN_TAG}-pareto"
os.makedirs(OUT_DIR, exist_ok=True)
candidate_summary_progress_csv = os.path.join(OUT_DIR, "candidate_rmst_summary_progress.csv")

if os.path.exists(candidate_summary_progress_csv):
    existing_summary_df = pd.read_csv(candidate_summary_progress_csv)
    if not existing_summary_df.empty:
        candidate_rmst_summary = existing_summary_df.to_dict("records")
        print(f"Loaded existing candidate RMST progress: {len(candidate_rmst_summary)} rows")

if candidates:
    done_trial_numbers = {int(d["trial_number"]) for d in candidate_rmst_summary if "trial_number" in d}

    for t in candidates:
        if t.number in done_trial_numbers:
            print(f"Skipping candidate trial #{t.number} (already in progress file).")
            continue

        # Print current out of total candidates for progress
        print(f"\nEvaluating candidate trial #{t.number} ({candidates.index(t)+1}/{len(candidates)}) for RMST summary...")

        candidate_runs_csv = os.path.join(OUT_DIR, f"candidate_trial_{t.number}_validation_runs.csv")
        
        eval_out = evaluate_trial_seeds(
            t,
            train_fit_df=train_df,
            eval_df=valid_df,
            w_fit=w_tr,
            seed_list=seed_list,
            label=f"candidate trial #{t.number} (validation)",
            eval_label="Validation",
            n_jobs=-1, # -1 to parallelize across seeds but results in non-determinism
            checkpoint_csv_path=candidate_runs_csv,
            checkpoint_every=1,
        )
        rmst_diffs = np.array(eval_out["all_rmst_diffs"], dtype=float)
        med = float(np.median(rmst_diffs))
        q1 = float(np.percentile(rmst_diffs, 25))
        q3 = float(np.percentile(rmst_diffs, 75))
        iqr = q3 - q1
        candidate_rmst_summary.append({
            "trial_number": t.number,
            "val_ci": float(t.values[0]),
            "median_rmst_diff": med,
            "iqr_rmst_diff": iqr,
        })

        pd.DataFrame(candidate_rmst_summary).to_csv(candidate_summary_progress_csv, index=False)

    candidate_rmst_summary.sort(key=lambda d: (-d["median_rmst_diff"], d["iqr_rmst_diff"]))
    pd.DataFrame(candidate_rmst_summary).to_csv(os.path.join(OUT_DIR, "candidate_rmst_summary_final.csv"), index=False)
    best_rmst_trial_number = candidate_rmst_summary[0]["trial_number"]
    best_rmst_trial = next(t for t in candidates if t.number == best_rmst_trial_number)

    # Plot distribution of median and IQR RMST diffs across candidates
    medians = [d["median_rmst_diff"] for d in candidate_rmst_summary]
    iqrs = [d["iqr_rmst_diff"] for d in candidate_rmst_summary]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(medians, bins=10, color=(100/255, 150/255, 255/255), alpha=0.7, edgecolor="black")
    axes[0].set_title("Candidate Median RMST Diff (Validation, 60 months)")
    axes[0].set_xlabel("Median RMST diff (months)")
    axes[0].set_ylabel("Count")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].hist(iqrs, bins=10, color=(255/255, 150/255, 100/255), alpha=0.7, edgecolor="black")
    axes[1].set_title("Candidate IQR RMST Diff (Validation, 60 months)")
    axes[1].set_xlabel("IQR RMST diff (months)")
    axes[1].set_ylabel("Count")
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "candidate_rmst_median_iqr_hist.png"), dpi=600)
    print(f"Saved candidate RMST median/IQR histogram to: {os.path.join(OUT_DIR, 'candidate_rmst_median_iqr_hist.png')}")
    plt.show(block=False)

model_specs = [
    ("best_model", best_model_trial),
    ("pareto_best_ci", one_se_nearest),
    ("highest_median_rmst", best_rmst_trial),
]

# Dataset for final training/eval
trainval_df = pd.concat([train_df, valid_df], axis=0, ignore_index=True)
trainval_df = trainval_df.sort_values(by=["OS_MONTHS", "OS_STATUS"], ascending=[False, False]).reset_index(drop=True)

# IPTW (train prevalence anchor)
w_trv, ps_model_fin, pi_fin = compute_iptw(trainval_df, covariate_cols=CLIN_FEATS_PRETX)
w_te, _, _ = compute_iptw(test_df, covariate_cols=CLIN_FEATS_PRETX, ref_prev=pi_fin, model=ps_model_fin)


BASE_OUT_DIR = f"rsf_multiselect-{RUN_TAG}-pareto"
os.makedirs(BASE_OUT_DIR, exist_ok=True)

for model_label, trial in model_specs:
    print(f"\n{'='*60}")
    print(f"MODEL: {model_label} (trial #{trial.number})")
    print(f"{'='*60}\n")

    model_out_dir = os.path.join(BASE_OUT_DIR, model_label)
    os.makedirs(model_out_dir, exist_ok=True)

    eval_out = evaluate_trial_seeds(
        trial,
        train_fit_df=trainval_df,
        eval_df=test_df,
        w_fit=w_trv,
        seed_list=seed_list,
        label=f"{model_label}",
        eval_label="Test",
        n_jobs=-1, # -1 to parallelize across seeds but results in non-determinism
        checkpoint_csv_path=os.path.join(model_out_dir, "multiple_runs_results_progress.csv"),
        checkpoint_every=1,
    )

    all_results = eval_out["all_results"]
    all_pvalues = eval_out["all_pvalues"]
    all_ci_test = eval_out["all_ci_test"]
    all_rmst_following = eval_out["all_rmst_following"]
    all_rmst_not_following = eval_out["all_rmst_not_following"]
    all_rmst_diffs = eval_out["all_rmst_diffs"]

    median_pvalue = float(np.median(all_pvalues))
    mean_pvalue = float(np.mean(all_pvalues))
    std_pvalue = float(np.std(all_pvalues))
    median_ci = float(np.median(all_ci_test))
    mean_ci = float(np.mean(all_ci_test))
    std_ci = float(np.std(all_ci_test))

    median_rmst_following = float(np.median(all_rmst_following))
    mean_rmst_following = float(np.mean(all_rmst_following))
    std_rmst_following = float(np.std(all_rmst_following))
    median_rmst_not_following = float(np.median(all_rmst_not_following))
    mean_rmst_not_following = float(np.mean(all_rmst_not_following))
    std_rmst_not_following = float(np.std(all_rmst_not_following))
    median_rmst_diff = float(np.median(all_rmst_diffs))
    mean_rmst_diff = float(np.mean(all_rmst_diffs))
    std_rmst_diff = float(np.std(all_rmst_diffs))

    print(f"\n{model_label} Test C-index:")
    print(f"Median: {median_ci:.4f}")
    print(f"Mean:   {mean_ci:.4f} ± {std_ci:.4f}")

    print(f"\n{model_label} 60-Month RMST (Following):")
    print(f"Median: {median_rmst_following:.4f} months")
    print(f"Mean:   {mean_rmst_following:.4f} ± {std_rmst_following:.4f} months")

    print(f"\n{model_label} 60-Month RMST (NOT Following):")
    print(f"Median: {median_rmst_not_following:.4f} months")
    print(f"Mean:   {mean_rmst_not_following:.4f} ± {std_rmst_not_following:.4f} months")

    print(f"\n{model_label} 60-Month RMST Difference (Following - Not Following):")
    print(f"Median: {median_rmst_diff:.4f} months")
    print(f"Mean:   {mean_rmst_diff:.4f} ± {std_rmst_diff:.4f} months")

    pd.DataFrame(all_results).to_csv(os.path.join(model_out_dir, "multiple_runs_results.csv"), index=False)

    runs = np.arange(1, len(seed_list) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    ax1 = axes[0, 0]
    ax1.bar(runs, all_pvalues, color=(9/255,117/255,181/255), alpha=0.7, edgecolor='black')
    ax1.axhline(y=median_pvalue, color='red', linestyle='--', linewidth=2, label=f'Median: {median_pvalue:.6f}')
    ax1.axhline(y=0.05, color='orange', linestyle=':', linewidth=2, label='α = 0.05')
    ax1.set_xlabel('Run Number', fontsize=12)
    ax1.set_ylabel('Log-rank p-value', fontsize=12)
    ax1.set_title(f'{model_label} Log-rank P-values', fontsize=14, fontweight='bold')
    ax1.set_xticks(runs[::5])
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3)

    ax2 = axes[0, 1]
    ax2.hist(all_pvalues, bins=10, color=(9/255,117/255,181/255), alpha=0.7, edgecolor='black')
    ax2.axvline(x=median_pvalue, color='red', linestyle='--', linewidth=2, label=f'Median: {median_pvalue:.6f}')
    ax2.axvline(x=mean_pvalue, color='green', linestyle='--', linewidth=2, label=f'Mean: {mean_pvalue:.6f}')
    ax2.axvline(x=0.05, color='orange', linestyle=':', linewidth=2, label='α = 0.05')
    ax2.set_xlabel('Log-rank p-value', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title(f'{model_label} P-value Distribution', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3)

    ax3 = axes[1, 0]
    ax3.bar(runs, all_rmst_diffs, color=(220/255,20/255,60/255), alpha=0.7, edgecolor='black')
    ax3.axhline(y=median_rmst_diff, color='darkred', linestyle='--', linewidth=2, label=f'Median: {median_rmst_diff:.4f}')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax3.set_xlabel('Run Number', fontsize=12)
    ax3.set_ylabel('RMST Difference (months)', fontsize=12)
    ax3.set_title(f'{model_label} RMST Difference', fontsize=14, fontweight='bold')
    ax3.set_xticks(runs[::5])
    ax3.legend(fontsize=10)
    ax3.grid(axis='y', alpha=0.3)

    ax4 = axes[1, 1]
    ax4.hist(all_rmst_diffs, bins=10, color=(220/255,20/255,60/255), alpha=0.7, edgecolor='black')
    ax4.axvline(x=median_rmst_diff, color='darkred', linestyle='--', linewidth=2, label=f'Median: {median_rmst_diff:.4f}')
    ax4.axvline(x=mean_rmst_diff, color='darkgreen', linestyle='--', linewidth=2, label=f'Mean: {mean_rmst_diff:.4f}')
    ax4.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax4.set_xlabel('RMST Difference (months)', fontsize=12)
    ax4.set_ylabel('Frequency', fontsize=12)
    ax4.set_title(f'{model_label} RMST Diff Distribution', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(model_out_dir, "pvalue_and_rmst_distribution.png"), dpi=600)
    plt.show(block = False)

    # Feature importance and final KM plot using median p-value seed
    median_run_idx = np.argsort(all_pvalues)[len(all_pvalues)//2]
    median_seed = seed_list[median_run_idx]
    rsf_params = build_rsf_params_from_trial(trial)
    rsf_params["random_state"] = median_seed

    X_trv, X_te, y_trv, y_te, feat_names, genes_main, genes_inter, dup_inter = build_feature_mats_for_trial(
        trial, trainval_df, test_df
    )

    with threadpool_limits(limits=None):
        rsf_final = make_rsf(**rsf_params)
        rsf_final.fit(X_trv, y_trv, sample_weight=w_trv)

        compare_treatment_recommendation_km_rsf(
            rsf_final,
            test_df,
            genes_main=genes_main,
            genes_inter=genes_inter,
            dup_inter=dup_inter,
            clin_cols=CLIN_FEATS,
            path=model_out_dir + "/"
        )

        result = permutation_importance(
            rsf_final, X_te, y_te, n_repeats=30, random_state=42, n_jobs=-1
        )

    importances_df = pd.DataFrame({
        "importances_mean": result.importances_mean,
        "importances_std": result.importances_std
    }, index=feat_names).sort_values(by="importances_mean", ascending=False)

    importances_df.to_csv(os.path.join(model_out_dir, "feature_importances.csv"), index=True)

    importances_df = importances_df.sort_values(by="importances_mean", ascending=True)
    plt.figure(figsize=(12, 8))
    plt.rcParams.update({'font.size': 14})
    plt.barh(importances_df.index, importances_df["importances_mean"],
             xerr=importances_df["importances_std"], color=(9/255,117/255,181/255))
    plt.xlabel("Permutation Feature Importance")
    plt.ylabel("Feature")
    plt.title(f"Random Survival Forest: Feature Importances on Test Set ({model_label})")
    plt.tight_layout()
    plt.savefig(os.path.join(model_out_dir, "feature_importances.png"), dpi=600)
    plt.show(block = False)
