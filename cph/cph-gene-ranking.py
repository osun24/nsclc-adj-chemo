import os, math, gc, time, random, warnings, inspect
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv
from sksurv.linear_model import CoxPHSurvivalAnalysis

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
    return [g for g, _ in ranks], ranks

def rank_genes_multivariate(train_df, gene_cols, clin_cols):
    y = Surv.from_arrays(
        event=train_df["OS_STATUS"].astype(bool).values,
        time=train_df["OS_MONTHS"].values.astype(float),
    )
    ranks = []
    for g in gene_cols:
        Xg = train_df[[g] + clin_cols].to_numpy(dtype=np.float64)
        try:
            model = CoxPHSurvivalAnalysis(alpha=1e-12)
            model.fit(Xg, y)
            pred = model.predict(Xg)
            ci = concordance_index_censored(y["event"], y["time"], pred)[0]
            ranks.append((g, float(ci)))
        except Exception:
            ranks.append((g, 0.5))
    ranks.sort(key=lambda z: z[1], reverse=True)
    return [g for g, _ in ranks], ranks

def rank_genes_multivariate_cv(train_df, gene_cols, clin_cols, n_splits=5):
    from sklearn.model_selection import KFold

    ranks = []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for g in gene_cols:
        ci_folds = []
        for train_idx, test_idx in kf.split(train_df):
            tr = train_df.iloc[train_idx]
            te = train_df.iloc[test_idx]

            y_tr = Surv.from_arrays(
                event=tr["OS_STATUS"].astype(bool).values,
                time=tr["OS_MONTHS"].values.astype(float),
            )
            y_te = Surv.from_arrays(
                event=te["OS_STATUS"].astype(bool).values,
                time=te["OS_MONTHS"].values.astype(float),
            )

            Xg_tr = tr[[g] + clin_cols].to_numpy(dtype=np.float64)
            Xg_te = te[[g] + clin_cols].to_numpy(dtype=np.float64)

            try:
                model = CoxPHSurvivalAnalysis(alpha=1e-12)
                model.fit(Xg_tr, y_tr)
                pred = model.predict(Xg_te)
                ci = concordance_index_censored(y_te["event"], y_te["time"], pred)[0]
                ci_folds.append(ci)
            except Exception:
                ci_folds.append(0.5)
        mean_ci = float(np.mean(ci_folds))
        ranks.append((g, mean_ci))
    ranks.sort(key=lambda z: z[1], reverse=True)
    return [g for g, _ in ranks], ranks


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

GENE_RANK, MULTIV_GENE_CIS = rank_genes_multivariate_cv(train_df, GENE_FEATS, CLIN_FEATS_PRETX, n_splits=5)
print(f"[Gene Ranking CV] Ranked {len(GENE_RANK)} genes on TRAIN (multivariate 5-fold CV)")

# Save gene-C-index pairs to CSV
pd.DataFrame(MULTIV_GENE_CIS, columns=["Gene", "C-index"]).to_csv(
    "cph/gene_cindex_multivariate_cv.csv", index=False
)
print("Saved gene C-indices to cph/gene_cindex_multivariate_cv.csv")

"""# Train-only univariate and multivariate rankings for genes
GENE_RANK, MULTIV_GENE_CIS = rank_genes_multivariate(train_df, GENE_FEATS, CLIN_FEATS_PRETX)
GENE_RANK_UNI, UNIV_GENE_CIS = rank_genes_univariate(train_df, GENE_FEATS)
print(f"[Gene Ranking] Ranked {len(GENE_RANK)} genes on TRAIN (multivariate)")
print(f"[Gene Ranking] Ranked {len(GENE_RANK_UNI)} genes on TRAIN (univariate)")

# Save gene-C-index pairs to CSV
pd.DataFrame(MULTIV_GENE_CIS, columns=["Gene", "C-index"]).to_csv(
    "cph/gene_cindex_multivariate.csv", index=False
)
pd.DataFrame(UNIV_GENE_CIS, columns=["Gene", "C-index"]).to_csv(
    "cph/gene_cindex_univariate.csv", index=False
)
print("Saved gene C-indices to cph/gene_cindex_multivariate.csv and cph/gene_cindex_univariate.csv")

# Plot histogram of C-indices for multivariate vs univariate rankings
plt.figure(figsize=(8, 6))
plt.hist([ci for _, ci in MULTIV_GENE_CIS], bins=20, alpha=0.6, label="Multivariate")
plt.hist([ci for _, ci in UNIV_GENE_CIS], bins=20, alpha=0.6, label="Univariate")
plt.xlabel("C-index")
plt.ylabel("Frequency")
plt.title("Gene-level C-index distributions")
plt.legend()
plt.tight_layout()
plt.savefig("cph/gene_cindex_histogram.png", dpi=300)
plt.close()
print("Saved histogram to cph/gene_cindex_histogram.png")"""