# Minimal Cox PH interaction LOOCV prescreen (alpha=0.05)
# Model per gene: treatment*gene + treatment + gene (no other covariates)
# Train + validation combined; fold result = {1 sig, 0 not sig, NaN fold not evaluable/fit failed}
# Output: CSV with genes as rows, LOOCV trials as columns, and rightmost n_significant (NaN ignored)

import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
from lifelines import CoxPHFitter
import warnings, time

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# --------------------
# Load & base preprocessing
# --------------------
train_df = pd.read_csv("affyTrain_z.csv")
valid_df = pd.read_csv("affyValidation_z.csv")

for df in (train_df, valid_df):
    if 'Adjuvant Chemo' in df.columns:
        df['Adjuvant Chemo'] = df['Adjuvant Chemo'].replace({'OBS': 0, 'ACT': 1})
    elif "Adjuvant.Chemo" in df.columns:
        df['Adjuvant Chemo'] = df['Adjuvant.Chemo'].replace({'OBS': 0, 'ACT': 1})
        

binary_columns = ['Adjuvant Chemo', 'IS_MALE']
for col in binary_columns:
    if col in train_df.columns:
        train_df[col] = train_df[col].astype(int)
    if col in valid_df.columns:
        valid_df[col] = valid_df[col].astype(int)

survival_cols = ['OS_STATUS', 'OS_MONTHS']

# Print counts
print("Train df:")
print(pd.crosstab(train_df["OS_STATUS"], train_df["Adjuvant Chemo"]))
print("Validation df:")
print(pd.crosstab(valid_df["OS_STATUS"], valid_df["Adjuvant Chemo"]))

# --------------------
# Combine train + validation
# --------------------
df_all = pd.concat([train_df, valid_df], ignore_index=True)

need_base = [c for c in ['OS_STATUS', 'OS_MONTHS', 'Adjuvant Chemo'] if c in df_all.columns]
df_all = df_all.dropna(subset=need_base).copy()
df_all['OS_STATUS'] = df_all['OS_STATUS'].astype(int)
df_all['OS_MONTHS'] = df_all['OS_MONTHS'].astype(float)
df_all['Adjuvant Chemo'] = df_all['Adjuvant Chemo'].astype(int)

# --------------------
# Exclude known clinical columns from the gene list
# --------------------
clin_covars = [
    'Adjuvant Chemo','Age','IS_MALE','Stage_IB','Stage_II',
    'Histology_Adenocarcinoma','Histology_Adenosquamous Carcinoma',
    'Histology_Large Cell Carcinoma','Histology_Squamous Cell Carcinoma',
    'Race_African American','Race_Asian','Race_Caucasian',
    'Race_Native Hawaiian or Other Pacific Islander','Race_Unknown',
    'Smoked?_No','Smoked?_Unknown','Smoked?_Yes'
]
present_covars = [c for c in clin_covars if c in df_all.columns]

exclude_cols = set(survival_cols + present_covars)
numeric_cols = df_all.select_dtypes(include=[np.number]).columns.tolist()
gene_cols = [c for c in numeric_cols if c not in exclude_cols]

# --------------------
# LOOCV splits
# --------------------
n_samples = len(df_all)
loo = LeaveOneOut()
splits = list(loo.split(np.arange(n_samples)))  # (train_idx, test_idx)

# --------------------
# Per-gene LOOCV significance of the interaction term
# --------------------
def loocv_significance_for_gene(gene_name, alpha=0.05):
    start = time.perf_counter()
    sig_flags = np.full(len(splits), np.nan, dtype=float)  # NaN means fold not evaluable / fit failed
    col_dur, col_evt, col_trt = 'OS_MONTHS', 'OS_STATUS', 'Adjuvant Chemo'

    for k, (train_idx, _) in enumerate(splits):
        tr = df_all.iloc[train_idx, :].copy()
        tr = tr.dropna(subset=[col_dur, col_evt, col_trt, gene_name])
        if tr.empty:
            continue
        if (tr[col_evt].sum() == 0) or (tr[col_trt].nunique() < 2) or (len(tr) < 5):
            continue

        # Scale gene within the fold
        g_scaled = StandardScaler().fit_transform(tr[[gene_name]]).ravel()

        design = pd.DataFrame({
            'duration': tr[col_dur].astype(float).values,
            'event': tr[col_evt].astype(int).values,
            'treatment': tr[col_trt].astype(int).values,
            'g': g_scaled
        }, index=tr.index)
        design['txg'] = design['treatment'] * design['g']

        try:
            cph = CoxPHFitter()
            cols = ['duration', 'event', 'treatment', 'g', 'txg']
            cph.fit(design[cols], duration_col='duration', event_col='event')
            if 'txg' in cph.summary.index:
                p = float(cph.summary.loc['txg', 'p'])
                sig_flags[k] = 1.0 if p < alpha else 0.0
        except Exception:
            # leave as NaN
            pass
    
    elapsed = time.perf_counter() - start
    print(f"Finished {gene_name} in {elapsed:.2f}s", flush=True)
    return gene_name, sig_flags

# --------------------
# Run prescreen (parallel over genes)
# --------------------
results = Parallel(n_jobs=-1, prefer="processes")(
    delayed(loocv_significance_for_gene)(g) for g in gene_cols
)

# --------------------
# Assemble output matrix
# --------------------
trial_cols = [f"trial_{i+1:03d}" for i in range(len(splits))]
out = pd.DataFrame(np.nan, index=gene_cols, columns=trial_cols)

for gene_name, flags in results:
    out.loc[gene_name, :] = flags

out['n_significant'] = np.nansum(out.values == 1.0, axis=1).astype(int)

# Save CSV
out_path = "interaction_loocv_fRMA_11-17-25.csv"
out.to_csv(out_path, index_label="gene")
print(f"Saved: {out_path}")

# Quick diagnostics
evaluable_counts = np.sum(~out[trial_cols].isna(), axis=1)
print(f"Genes with ≥1 evaluable fold: {(evaluable_counts>0).sum()} / {len(out)}")