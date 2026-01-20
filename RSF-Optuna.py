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

import optuna
# from optuna.samplers import NSGAIISampler  # keep if you later switch back to MO

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
                                           time_col="OS_MONTHS", event_col="OS_STATUS"):
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

    plt.title("Kaplan-Meier Survival Curves by Treatment Alignment")
    plt.xlabel("Time")
    plt.ylabel("Survival Probability")
    add_at_risk_counts(kmf_aligned, kmf_not_aligned)
    plt.text(0.1, 0.1, f"Log-rank p-value: {results.p_value:.4f}", transform=plt.gca().transAxes)
    plt.tight_layout()
    plt.savefig("km_alignment_rsf_recommendation.png", dpi=600)
    plt.show()


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


# ============================================================
# Optuna: single-objective (Val CI ↑) with RSF + interactions
# ============================================================
def suggest_hparams(trial):
    max_nonclin = max(8, FEAT_BUDGET - len(CLIN_FEATS))

    # Same gene menus
    base_main = [16, 32, 64, 96, 128, 192, 256, 384, 512, MAX_GENES]
    TOPK_MAIN_CHOICES = tuple(sorted({k for k in base_main if k <= MAX_GENES}))
    top_k_genes = int(trial.suggest_categorical("top_k_genes", TOPK_MAIN_CHOICES))

    base_inter = [0, 8, 16, 32, 64, 96, 128, 192, 256, 384, 512]
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

    rsf = make_rsf(**rsf_params)
    rsf.fit(Xtr, ytr, sample_weight=wtr)

    tr_pred = rsf.predict(Xtr)
    va_pred = rsf.predict(Xva)

    ttr = train_df["OS_MONTHS"].values.astype(float)
    etr = train_df["OS_STATUS"].values.astype(int)
    tva = valid_df["OS_MONTHS"].values.astype(float)
    eva = valid_df["OS_STATUS"].values.astype(int)

    tr_ci = cindex(tr_pred, ttr, etr)
    va_ci, va_ci_se = cindex_bootstrap(va_pred, tva, eva)
    gap = max(0.0, tr_ci - va_ci)

    trial.set_user_attr("n_features", int(Xtr.shape[1]))
    trial.set_user_attr("k_main", int(k_main))
    trial.set_user_attr("k_int", int(k_int))
    trial.set_user_attr("dup_inter", int(dup_inter))
    trial.set_user_attr("val_ci_se", float(va_ci_se))

    print(
        f"[Trial {trial.number:03d}] Val CI={va_ci:.4f} (SE={va_ci_se:.4f}), "
        f"Gap={gap:.4f}, N_feats={Xtr.shape[1]}, K_main={k_main}, K_int={k_int}, Dup={dup_inter}, "
        f"n_estimators={rsf_params['n_estimators']}"
    )

    return va_ci


# ---- Run study ----
storage = "sqlite:///rsf_optuna.db"
study_name = "rsf_jan_19_1se"

study = optuna.create_study(
    direction="maximize",
    study_name=study_name,
    storage=storage,
    load_if_exists=True,
)

N_TRIALS = 100
print(f"Starting optimization: {N_TRIALS} trials")
study.optimize(objective, n_trials=N_TRIALS, gc_after_trial=True)


# ---- Choose the best trial using the 1-Standard-Error (1-SE) rule ----
best_trial = study.best_trial
best_ci = float(best_trial.value)
best_ci_se = float(best_trial.user_attrs.get("val_ci_se", 0.0))

one_se_threshold = best_ci - best_ci_se
print(f"\n[1-SE Rule] Best Val CI: {best_ci:.4f} (SE: {best_ci_se:.4f})")
print(f"[1-SE Rule] Threshold: {one_se_threshold:.4f}")

candidates = [
    t for t in study.trials
    if t.state == optuna.trial.TrialState.COMPLETE and float(t.value) >= one_se_threshold
]

if not candidates:
    chosen = best_trial
    print("[1-SE Rule] No candidates found above threshold, falling back to best trial.")
else:
    candidates.sort(key=lambda t: t.user_attrs.get("n_features", float("inf")))
    chosen = candidates[0]
    print(f"[1-SE Rule] Found {len(candidates)} candidates. "
          f"Chose trial #{chosen.number} with {chosen.user_attrs.get('n_features')} features.")

print("\n[Chosen Trial] #", chosen.number, "Val CI:", float(chosen.value))
print("[Chosen Params]", chosen.params)
print("[Chosen Attrs] k_main=%s k_int=%s dup_inter=%s" %
      (str(chosen.user_attrs.get("k_main")),
       str(chosen.user_attrs.get("k_int")),
       str(chosen.user_attrs.get("dup_inter"))))


# ============================================================
# Final training on Train+Val with chosen hyperparams + IPTW
# ============================================================
best_hp = chosen.params
k_main = int(chosen.user_attrs["k_main"])
k_int  = int(chosen.user_attrs["k_int"])
dup_inter = int(chosen.user_attrs.get("dup_inter", 1))

trainval_df = pd.concat([train_df, valid_df], axis=0, ignore_index=True)
trainval_df = trainval_df.sort_values(by=["OS_MONTHS", "OS_STATUS"], ascending=[False, False]).reset_index(drop=True)

genes_main  = GENE_RANK[:k_main]
genes_inter = genes_main[:k_int]

X_trv_raw, feat_names = build_features_with_interactions(
    trainval_df, genes_main, genes_inter, dup_inter=dup_inter, clin_cols=CLIN_FEATS
)
X_te_raw, _ = build_features_with_interactions(
    test_df, genes_main, genes_inter, dup_inter=dup_inter, clin_cols=CLIN_FEATS
)

X_trv = X_trv_raw.astype(np.float64)
X_te  = X_te_raw.astype(np.float64)

y_trv = Surv.from_arrays(
    event=trainval_df["OS_STATUS"].astype(bool).values,
    time=trainval_df["OS_MONTHS"].astype(float).values
)
y_te = Surv.from_arrays(
    event=test_df["OS_STATUS"].astype(bool).values,
    time=test_df["OS_MONTHS"].astype(float).values
)

# IPTW (train prevalence anchor)
w_trv, ps_model_fin, pi_fin = compute_iptw(trainval_df, covariate_cols=CLIN_FEATS_PRETX)
w_te, _, _ = compute_iptw(test_df, covariate_cols=CLIN_FEATS_PRETX, ref_prev=pi_fin, model=ps_model_fin)

# Extract RSF params from chosen trial params
rsf_param_names = {
    "n_estimators", "max_depth", "min_samples_split", "min_samples_leaf",
    "min_weight_fraction_leaf", "max_features", "max_leaf_nodes",
    "bootstrap", "oob_score", "n_jobs", "random_state", "verbose", "warm_start"
}

# Rebuild max_features from mode params if needed
mf_mode = best_hp.get("max_features_mode", "sqrt")
if mf_mode == "all":
    max_features = None
elif mf_mode == "frac":
    max_features = float(best_hp.get("max_features_frac", 1.0))
else:
    max_features = mf_mode

rsf_params_fin = dict(
    n_estimators=int(best_hp.get("n_estimators", 1000)),
    max_depth=best_hp.get("max_depth", None),
    min_samples_split=int(best_hp.get("min_samples_split", 6)),
    min_samples_leaf=int(best_hp.get("min_samples_leaf", 3)),
    min_weight_fraction_leaf=0.0,
    max_features=max_features,
    max_leaf_nodes=None,
    bootstrap=True,
    oob_score=False,
    n_jobs=-1,
    random_state=7,
    verbose=0,
    warm_start=False,
)

rsf_final = make_rsf(**rsf_params_fin)
rsf_final.fit(X_trv, y_trv, sample_weight=w_trv)

pred_trv = rsf_final.predict(X_trv)
pred_te  = rsf_final.predict(X_te)

t_trv = trainval_df["OS_MONTHS"].values.astype(float)
e_trv = trainval_df["OS_STATUS"].values.astype(int)
t_te  = test_df["OS_MONTHS"].values.astype(float)
e_te  = test_df["OS_STATUS"].values.astype(int)

ci_trv = cindex(pred_trv, t_trv, e_trv)
ci_te  = cindex(pred_te,  t_te,  e_te)

print(f"\n[Final RSF] Train+Val CI: {ci_trv:.4f}")
print(f"[Final RSF] Test CI:      {ci_te:.4f}")

# Per-arm C-indices (sanity)
act_trv = trainval_df["Adjuvant Chemo"].to_numpy(int)
act_te  = test_df["Adjuvant Chemo"].to_numpy(int)

def ci_by_arm(pred, t, e, arm):
    out = {}
    for label, mask in [("ACT=1", arm == 1), ("ACT=0", arm == 0)]:
        out[label] = cindex(pred[mask], t[mask], e[mask]) if mask.sum() >= 3 else np.nan
    return out

print("[Train+Val] CI by arm:", ci_by_arm(pred_trv, t_trv, e_trv, act_trv))
print("[Test]      CI by arm:", ci_by_arm(pred_te,  t_te,  e_te,  act_te))

date = "11-22"

# Save artifacts
import joblib
OUT_DIR = f"rsf_interactions_iptw_bounded-{date}-1SE"
os.makedirs(OUT_DIR, exist_ok=True)

joblib.dump(rsf_final, os.path.join(OUT_DIR, "rsf_final.joblib"))

with open(os.path.join(OUT_DIR, "chosen_params.txt"), "w") as f:
    f.write(str(best_hp) + "\n\n")
    f.write("RSF params used:\n")
    f.write(str(rsf_params_fin))

with open(os.path.join(OUT_DIR, "features_used.txt"), "w") as f:
    f.write("\n".join(feat_names))

print("Saved final model and parameters to:", OUT_DIR)

# Kaplan-Meier alignment on test set using final model
print("\n[KM Alignment] Test set vs. RSF recommendation")
compare_treatment_recommendation_km_rsf(
    rsf_final,
    test_df,
    genes_main=genes_main,
    genes_inter=genes_inter,
    dup_inter=dup_inter,
    clin_cols=CLIN_FEATS,
)