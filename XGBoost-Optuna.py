# ============================================================
# Colab-ready SINGLE CELL (CPU)
# XGBoost (gbtree) Cox PH with IPTW, feature budgets,
# emphasized gene×ACT interactions (incl. optional duplication),
# early stopping on C-index, and conservative Pareto selection
# ============================================================

# ---------- Imports ----------
import os, math, gc, time, random, warnings
import numpy as np
import pandas as pd

import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv
from sksurv.linear_model import CoxPHSurvivalAnalysis

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

GENES_CSV = "/mnt/data/Genes.csv"
if not os.path.exists(GENES_CSV):
    if os.path.exists("/content/Genes.csv"):
        GENES_CSV = "/content/Genes.csv"
    elif os.path.exists("Genes.csv"):
        GENES_CSV = "Genes.csv"
print("Genes.csv path:", GENES_CSV)

# ---------- Clinical columns ----------
CLINICAL_VARS = [
    "Adjuvant Chemo","Age","IS_MALE",
    "Stage_IA","Stage_IB","Stage_II","Stage_III",
    "Histology_Adenocarcinoma","Histology_Large Cell Carcinoma","Histology_Squamous Cell Carcinoma",
    "Race_African American","Race_Asian","Race_Caucasian","Race_Native Hawaiian or Other Pacific Islander","Race_Unknown",
    "Smoked?_No","Smoked?_Unknown","Smoked?_Yes"
]
CLIN_FEATS_PRETX = [c for c in CLINICAL_VARS if c != "Adjuvant Chemo"]  # for IPTW only

# ============================================================
# Helpers: IO, preprocessing, ranking, features, IPTW, metrics
# ============================================================
def load_genes_list(genes_csv):
    g = pd.read_csv(genes_csv)
    if "Prop" not in g.columns or "Gene" not in g.columns:
        raise ValueError("Genes.csv must have columns 'Gene' and 'Prop'.")
    g["Prop"] = pd.to_numeric(g["Prop"], errors="coerce").fillna(0)
    genes = g.loc[g["Prop"] == 1, "Gene"].astype(str).tolist()
    print(f"[Genes] Selected {len(genes)} genes with Prop == 1")
    return genes

def coerce_survival_cols(df):
    if df["OS_STATUS"].dtype == object:
        df["OS_STATUS"] = df["OS_STATUS"].replace({"DECEASED":1,"LIVING":0,"Dead":1,"Alive":0}).astype(int)
    else:
        df["OS_STATUS"] = pd.to_numeric(df["OS_STATUS"], errors="coerce").fillna(0).astype(int)
    df["OS_MONTHS"] = pd.to_numeric(df["OS_MONTHS"], errors="coerce").fillna(0.0).astype(float)
    return df

def preprocess_split(df, clinical_vars, gene_names):
    # Avoid pandas FutureWarning by mapping (we coerce to numeric right after)
    if "Adjuvant Chemo" in df.columns:
        df["Adjuvant Chemo"] = df["Adjuvant Chemo"].map({"OBS": 0, "ACT": 1})
    for col in ["Adjuvant Chemo","IS_MALE"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    df = coerce_survival_cols(df)
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
def build_features_with_interactions(df, main_genes, inter_genes, act_col="Adjuvant Chemo", dup_inter=1):
    """
    Build [clinical + genes_main + gene*ACT] features.
    If dup_inter>1, duplicate interaction columns with unique names to bias column sampling.
    """
    base_cols = CLINICAL_VARS + list(main_genes)  # keep ACT main effect in clinicals
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

# ============================================================
# Load data, rank genes (TRAIN only), set budgets
# ============================================================
train_raw = pd.read_csv(TRAIN_CSV)
valid_raw = pd.read_csv(VALID_CSV)
test_raw  = pd.read_csv(TEST_CSV)

GENE_LIST = load_genes_list(GENES_CSV)

train_df = preprocess_split(train_raw, CLINICAL_VARS, GENE_LIST)
valid_df = preprocess_split(valid_raw, CLINICAL_VARS, GENE_LIST)
test_df  = preprocess_split(test_raw,  CLINICAL_VARS, GENE_LIST)

# Keep only features present in all splits
feat_candidates = [c for c in (CLINICAL_VARS + GENE_LIST)
                   if c in train_df.columns and c in valid_df.columns and c in test_df.columns]
CLIN_FEATS = [c for c in CLINICAL_VARS if c in feat_candidates]
GENE_FEATS = [g for g in GENE_LIST if g in feat_candidates]

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
    with xgb.config_context(verbosity=2):
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
# Optuna: NSGA-II (Val CI ↑, Gap ↓) + emphasized interactions
# ============================================================
def suggest_hparams(trial):
    max_nonclin = max(8, FEAT_BUDGET - len(CLIN_FEATS))

    # Larger menu; interactions can be as numerous as mains (within budget)
    base_main  = [16, 32, 64, 96, 128, 192, 256, 384, 512, MAX_GENES]
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
    dup_inter = trial.suggest_int("dup_inter", 1, 3)

    # XGBoost params (CPU)
    params = {
        "objective": "survival:cox",
        "booster": "gbtree",
        "tree_method": "hist",                  # CPU histogram
        "disable_default_eval_metric": True,    # we'll use custom_metric only
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

def build_trial_mats(k_main, k_int, dup_inter):
    genes_main  = GENE_RANK[:k_main]
    genes_inter = genes_main[:k_int]
    Xtr_raw, feat_names = build_features_with_interactions(train_df, genes_main, genes_inter, dup_inter=dup_inter)
    Xva_raw, _          = build_features_with_interactions(valid_df, genes_main, genes_inter, dup_inter=dup_inter)

    Xtr = Xtr_raw.astype(np.float32)
    Xva = Xva_raw.astype(np.float32)

    ytr = pack_cox_labels(train_df["OS_MONTHS"].values, train_df["OS_STATUS"].values)
    yva = pack_cox_labels(valid_df["OS_MONTHS"].values,  valid_df["OS_STATUS"].values)

    dtr = xgb.DMatrix(Xtr, label=ytr, weight=w_tr, feature_names=feat_names)
    dva = xgb.DMatrix(Xva, label=yva, weight=w_va, feature_names=feat_names)
    return dtr, dva, feat_names

def objective(trial):
    k_main, k_int, dup_inter, params, num_boost_round, esr = suggest_hparams(trial)
    dtr, dva, feat_names = build_trial_mats(k_main, k_int, dup_inter)
    booster, evr = train_xgb_cox(dtr, dva, params, num_boost_round, esr)

    # Best iteration
    best_ntree = booster.best_ntree_limit or (booster.best_iteration + 1 if booster.best_iteration is not None else num_boost_round)
    tr_pred = booster.predict(dtr, iteration_range=(0, best_ntree), output_margin=True)
    va_pred = booster.predict(dva, iteration_range=(0, best_ntree), output_margin=True)

    ytr_signed = dtr.get_label(); ttr = np.abs(ytr_signed); etr = (ytr_signed > 0).astype(int)
    yva_signed = dva.get_label(); tva = np.abs(yva_signed); eva = (yva_signed > 0).astype(int)

    tr_ci = cindex(tr_pred, ttr, etr)
    va_ci = cindex(va_pred, tva, eva)
    gap = max(0.0, tr_ci - va_ci)

    trial.set_user_attr("n_features", int(len(dtr.feature_names)))
    trial.set_user_attr("k_main", int(k_main))
    trial.set_user_attr("k_int", int(k_int))
    trial.set_user_attr("dup_inter", int(dup_inter))
    trial.set_user_attr("best_ntree", int(best_ntree))

    print(f"[Trial {trial.number:03d}] Val CI={va_ci:.4f}, Gap={gap:.4f}, N_feats={len(dtr.feature_names)}, "
          f"K_main={k_main}, K_int={k_int}, Dup={dup_inter}, N_tree={best_ntree}")

    return va_ci, gap

# ---- Run study ----
storage = "sqlite:///xgb_cox_optuna.db"
study_name = "xgb_cox_mo_gap_bounded_interactions_emphasis"
sampler = NSGAIISampler(seed=42, population_size=24)
study = optuna.create_study(
    directions=["maximize", "minimize"],
    study_name=study_name, storage=storage, load_if_exists=True, sampler=sampler
)
N_TRIALS = 100  # adjust as needed
print(f"Starting multi-objective optimization: {N_TRIALS} trials")
study.optimize(objective, n_trials=N_TRIALS, gc_after_trial=True)

# ---- Choose a robust solution from Pareto front (conservative) ----
pareto = study.best_trials
best_val = max(tr.values[0] for tr in pareto)
TOL = 0.015  # within 1.5pp absolute C-index of best
cands = [tr for tr in pareto if (best_val - tr.values[0]) <= TOL]
# prefer smaller gap, then fewer features
cands.sort(key=lambda tr: (tr.values[1], tr.user_attrs.get("n_features", 10**9)))
chosen = cands[0]

print("\n[Chosen Pareto] Val CI=%.4f | Gap=%.4f | n_features=%d" %
      (chosen.values[0], chosen.values[1], chosen.user_attrs.get("n_features", -1)))
print("[Chosen Params]", chosen.params)
print("[Chosen Attrs] k_main=%s k_int=%s dup_inter=%s best_ntree=%s" %
      (str(chosen.user_attrs.get("k_main")), str(chosen.user_attrs.get("k_int")),
       str(chosen.user_attrs.get("dup_inter")), str(chosen.user_attrs.get("best_ntree"))))

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
X_trv_raw, feat_names = build_features_with_interactions(trainval_df, genes_main, genes_inter, dup_inter=dup_inter)
X_te_raw,  _          = build_features_with_interactions(test_df,      genes_main, genes_inter, dup_inter=dup_inter)

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
    **{k: v for k, v in best_hp.items() if k not in ["num_boost_round", "early_stopping_rounds"]}
}
num_boost_round_fin = int(best_hp.get("num_boost_round", 1600))
early_stopping_rounds_fin = int(best_hp.get("early_stopping_rounds", 125))

# Split Train+Val for ES
event_mask = (y_trv > 0).astype(int)
idx = np.arange(len(y_trv))
tr_idx, va_idx = train_test_split(idx, test_size=0.25, random_state=7, stratify=event_mask)
d_tr_es = xgb.DMatrix(X_trv[tr_idx], label=y_trv[tr_idx], weight=w_trv[tr_idx], feature_names=feat_names)
d_va_es = xgb.DMatrix(X_trv[va_idx], label=y_trv[va_idx], weight=w_trv[va_idx], feature_names=feat_names)

booster_final, evr_final = train_xgb_cox(d_tr_es, d_va_es, params_fin,
                                         num_boost_round_fin, early_stopping_rounds_fin)
best_ntree_final = booster_final.best_ntree_limit or (booster_final.best_iteration + 1 if booster_final.best_iteration is not None else num_boost_round_fin)

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
OUT_DIR = "xgb_cox_interactions_iptw_bounded"
os.makedirs(OUT_DIR, exist_ok=True)
booster_final.save_model(os.path.join(OUT_DIR, "xgb_cox_final.json"))
with open(os.path.join(OUT_DIR, "chosen_params.txt"), "w") as f:
    f.write(str(best_hp))
with open(os.path.join(OUT_DIR, "features_used.txt"), "w") as f:
    f.write("\n".join(feat_names))
print("Saved final model and parameters to:", OUT_DIR)
