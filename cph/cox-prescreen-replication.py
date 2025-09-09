#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cox Proportional Hazards LOOCV Prescreening with ACT × Gene Interactions

Implements leave-one-out cross-validation (LOOCV) on the TRAINING dataset only.
For each probe set def main():
    parser = argparse.ArgumentParser(description="CoxPH LOOCV prescreen with ACT×Gene interactions")
    parser.add_argument('--train', type=str, default='train_merged.csv', help='Path to training CSV')ne), fits a CoxPH model with:
    OS ~ Adjuvant Chemo + Gene + (Adjuvant Chemo × Gene)

Excludes probe sets with median LOOCV train-fold p-value for the interaction term > α (default α=0.05).
Outputs:
- Selected genes (median p ≤ α)
- LOOCV summary table (median p, proportion significant)
- LOOCV p-value matrix (N_genes × N_samples; NaNs allowed)
- Final screened feature list = selected genes + Age, Sex, Treatment, Stage

Expected columns in affyTrain.csv:
- Survival: OS_MONTHS, OS_STATUS (1=event, 0=censored)
- Treatment: 'Adjuvant Chemo' as {'OBS','ACT'} or {0,1}
- Demographics: Age, Sex or IS_MALE
- Stage: either one-hot Stage_* columns or a single Stage/Overall.Stage column
- Genomic probe sets: all other columns after exclusions

Usage:
    python cox_prescreen_loocv.py \
        --train affyTrain.csv \
        --alpha 0.05 \
        --n-jobs -1 \
        --no-parallel   # (optional) to disable multiprocessing
"""
import os
import sys
import time
import datetime
import argparse
import warnings
import numpy as np
import pandas as pd
import multiprocessing
from joblib import Parallel, delayed
from sklearn.model_selection import LeaveOneOut
from lifelines import CoxPHFitter

warnings.filterwarnings("ignore")
np.random.seed(42)

# --------------------------- Utilities ---------------------------

class Tee:
    def __init__(self, *files): self.files = files
    def write(self, obj):
        for f in self.files: f.write(obj)
    def flush(self):
        for f in self.files: f.flush()

def nowstamp(fmt="%Y%m%d_%H%M%S"):
    return datetime.datetime.now().strftime(fmt)

# --------------------------- Data Loading & Preprocess ---------------------------

def load_training(train_path: str) -> pd.DataFrame:
    print("="*80)
    print("LOADING TRAINING DATA ONLY (LOOCV will use this dataset)")
    print("="*80)
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training file not found: {train_path}")
    df = pd.read_csv(train_path, index_col=0)
    print(f"Initial train shape: {df.shape}")

    # Drop other survival variables to focus only on Overall Survival (OS)
    other_survival_cols = ['PFS_MONTHS', 'PFS_STATUS', 'RFS_MONTHS', 'RFS_STATUS']
    cols_to_drop = [col for col in other_survival_cols if col in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        print(f"Dropped other survival columns: {cols_to_drop}")
    
    print(f"Train shape after dropping columns: {df.shape}")

    # Normalize treatment to {0,1}
    if 'Adjuvant Chemo' not in df.columns:
        raise KeyError("Missing 'Adjuvant Chemo' column.")
    if df['Adjuvant Chemo'].dtype == object:
        df['Adjuvant Chemo'] = df['Adjuvant Chemo'].replace({'OBS': 0, 'ACT': 1}).astype(float)
    else:
        df['Adjuvant Chemo'] = df['Adjuvant Chemo'].astype(float)

    # Normalize sex
    if 'IS_MALE' not in df.columns:
        if 'Sex' in df.columns:
            # Accept common encodings
            sex_map = {'M': 1, 'Male': 1, 'F': 0, 'Female': 0}
            if df['Sex'].dtype == object:
                df['IS_MALE'] = df['Sex'].map(sex_map).astype(float)
            else:
                # Assume already binary {0,1} meaning male?
                df['IS_MALE'] = df['Sex'].astype(float)
        else:
            print("Warning: No 'IS_MALE' or 'Sex' column found; sex will not be added to final clinical set.")

    # Normalize stage (prefer Stage_* dummies if present; else one-hot from Stage/Overall.Stage)
    stage_dummy_cols = [c for c in df.columns if c.startswith('Stage_')]
    if not stage_dummy_cols:
        stage_col = None
        for candidate in ['Overall.Stage', 'Overall_Stage', 'Stage']:
            if candidate in df.columns:
                stage_col = candidate
                break
        if stage_col:
            dummies = pd.get_dummies(df[stage_col], prefix='Stage', dummy_na=False)
            # Drop a reference category to avoid collinearity (keep all for feature list output; CoxPH formulas can handle full rank if reference not included elsewhere)
            df = pd.concat([df, dummies], axis=1)
            stage_dummy_cols = list(dummies.columns)
        else:
            print("Warning: No stage variable found; stage will not be added to final clinical set.")

    # Basic survival checks
    for col in ['OS_MONTHS', 'OS_STATUS']:
        if col not in df.columns:
            raise KeyError(f"Missing survival column: {col}")
    print(f"Events: {int(df['OS_STATUS'].sum())} | Censored: {df.shape[0] - int(df['OS_STATUS'].sum())}")
    return df

def identify_feature_sets(df: pd.DataFrame):
    """
    Identifies genomic, clinical, and survival features from the dataframe.
    Assumes that clinical/survival columns are known, and the rest are genomic.
    """
    # Define all known non-genomic columns
    survival_cols = ['OS_MONTHS', 'OS_STATUS'] # Use ONLY Overall Survival
    clinical_base = ['Adjuvant Chemo', 'Age', 'IS_MALE', 'Stage', 'Histology']
    other_exclude = ['Batch'] # Exclude batch identifiers if they exist

    # Stage columns are handled separately as they are generated dynamically
    stage_dummies = [c for c in df.columns if c.startswith('Stage_')]
    
    # Combine all columns to be excluded from the genomic set
    exclude_cols = set(survival_cols + clinical_base + stage_dummies + other_exclude)

    # Identify genomic features: anything not in the exclude list and is numeric
    genomic_features = [
        c for c in df.columns 
        if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])
    ]

    # Identify clinical features that are actually present in the dataframe for final model
    clinical_present = [c for c in clinical_base if c in df.columns]
    clinical_present.extend(stage_dummies)
    # Ensure 'Adjuvant Chemo' is not listed twice if it was in clinical_base
    clinical_present = sorted(list(set(clinical_present)))

    print("\n" + "="*80)
    print("IDENTIFIED FEATURE SETS")
    print("="*80)
    print(f"Genomic features (candidate probe sets): {len(genomic_features)}")
    print(f"Clinical/demographic variables to append: {clinical_present}")
    return genomic_features, clinical_present

# --------------------------- LOOCV Screening ---------------------------

def loocv_interaction_screen(df: pd.DataFrame,
                             genomic_features,
                             alpha: float = 0.05,
                             use_parallel: bool = True,
                             n_jobs: int = -1):
    """
    LOOCV on training data; for each left-out sample, fit Cox on remaining N-1.
    Model per gene: `Gene` + `Adjuvant Chemo` + `Adjuvant_Chemo_x_Gene`
    Retain genes with median LOOCV p-value (interaction) ≤ alpha.
    """
    print("\n" + "="*80)
    print(f"RUNNING LOOCV TREATMENT×GENE INTERACTION SCREENING (α = {alpha:.2f})")
    print("="*80)

    n = df.shape[0]
    g = len(genomic_features)
    pmat = np.full((g, n), np.nan, dtype=float)

    loo = LeaveOneOut()
    splits = list(loo.split(df))
    n_cores = multiprocessing.cpu_count() if n_jobs == -1 else min(n_jobs, multiprocessing.cpu_count())
    if use_parallel:
        print(f"- Parallelization: Enabled on {n_cores} cores")
    else:
        print("- Parallelization: Disabled")

    def _fold_eval(fold_idx):
        train_idx, _ = splits[fold_idx]
        train = df.iloc[train_idx]
        base = train[['OS_MONTHS', 'OS_STATUS', 'Adjuvant Chemo']].copy()

        fold_res = []
        for gi, gene in enumerate(genomic_features):
            try:
                m = base.copy()
                m[gene] = train[gene]
                m[f'Adjuvant_Chemo_x_{gene}'] = m['Adjuvant Chemo'] * m[gene]
                cph = CoxPHFitter()
                cph.fit(m,
                        duration_col='OS_MONTHS',
                        event_col='OS_STATUS',
                        show_progress=False,
                        formula=f"`{gene}` + `Adjuvant Chemo` + `Adjuvant_Chemo_x_{gene}`")
                p = float(cph.summary.loc[f'Adjuvant_Chemo_x_{gene}', 'p'])
            except Exception:
                p = np.nan
            fold_res.append((gi, p))
        return fold_res

    if use_parallel:
        all_results = Parallel(n_jobs=n_cores, verbose=0)(
            delayed(_fold_eval)(k) for k in range(n)
        )
    else:
        all_results = [_fold_eval(k) for k in range(n)]

    for fold_idx, res in enumerate(all_results):
        for gi, p in res:
            pmat[gi, fold_idx] = p

    median_p = np.nanmedian(pmat, axis=1)
    prop_sig = np.nanmean((pmat <= alpha), axis=1)

    summary = (pd.DataFrame({
        'gene': genomic_features,
        'median_p': median_p,
        'prop_significant_folds': prop_sig,
        'n_folds_non_nan': np.sum(~np.isnan(pmat), axis=1)
    })
    .sort_values(['median_p', 'prop_significant_folds'], ascending=[True, False])
    .reset_index(drop=True))

    selected = summary[summary['median_p'] <= alpha].copy()

    print(f"\nSelection (α={alpha:.2f}): kept {len(selected)} / {len(genomic_features)} probe sets "
          f"({100.0*len(selected)/max(1,len(genomic_features)):.1f}%).")
    if len(selected) > 0:
        print("Top 10 by median p:")
        print(selected[['gene','median_p','prop_significant_folds']].head(10).to_string(index=False))

    return selected, summary, pmat

# --------------------------- Main Pipeline ---------------------------

def main():
    parser = argparse.ArgumentParser(description="CoxPH LOOCV prescreen with ACTxGene interactions")
    parser.add_argument('--train', type=str, default='train_merged.csv', help='Path to training CSV')
    parser.add_argument('--alpha', type=float, default=0.05, help='Significance level for exclusion (median p ≤ α kept)')
    parser.add_argument('--n-jobs', type=int, default=-1, help='Cores for parallel processing (-1 = all)')
    parser.add_argument('--no-parallel', action='store_true', help='Disable parallelization')
    parser.add_argument('--outdir', type=str, default='cox_prescreen_results', help='Output directory')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    outdir = os.path.join(script_dir, args.outdir)
    os.makedirs(outdir, exist_ok=True)

    log_path = os.path.join(outdir, f"{nowstamp('%Y%m%d')}_cox_prescreen_log.txt")
    log_file = open(log_path, "w")
    sys.stdout = Tee(sys.stdout, log_file)

    print("="*80)
    print("COX PROPORTIONAL HAZARDS LOOCV INTERACTION PRESCREEN")
    print("="*80)
    print(f"Train file: {args.train}")
    print(f"Alpha: {args.alpha}")
    print(f"Parallel: {not args.no_parallel} | n_jobs: {args.n_jobs}")
    print(f"Output: {outdir}")
    print("="*80)

    t0 = time.time()

    # Load + identify features
    df = load_training(args.train)
    genomic_features, clinical_present = identify_feature_sets(df)

    # LOOCV screen
    selected, summary, pmat = loocv_interaction_screen(
        df=df,
        genomic_features=genomic_features,
        alpha=args.alpha,
        use_parallel=not args.no_parallel,
        n_jobs=args.n_jobs
    )

    # Save outputs
    stamp = nowstamp("%Y%m%d")
    sel_path = os.path.join(outdir, f"{stamp}_loocv_selected_genes_alpha_{args.alpha:.2f}.csv")
    sum_path = os.path.join(outdir, f"{stamp}_loocv_summary_alpha_{args.alpha:.2f}.csv")
    pval_path = os.path.join(outdir, f"{stamp}_loocv_pvals_alpha_{args.alpha:.2f}.npy")
    selected.to_csv(sel_path, index=False)
    summary.to_csv(sum_path, index=False)
    np.save(pval_path, pmat)
    print(f"\nSaved:")
    print(f"- Selected genes: {sel_path}")
    print(f"- LOOCV summary: {sum_path}")
    print(f"- LOOCV p-matrix: {pval_path}")

    # Build final screened feature list = selected genes + Age, Sex, Treatment, Stage
    base_vars = []
    if 'Age' in df.columns: base_vars.append('Age')
    if 'IS_MALE' in df.columns: base_vars.append('IS_MALE')  # sex
    if 'Adjuvant Chemo' in df.columns: base_vars.append('Adjuvant Chemo')  # treatment
    stage_vars = [c for c in df.columns if c.startswith('Stage_')]
    base_vars.extend(stage_vars)

    final_list = pd.DataFrame({
        'feature': list(selected['gene']) + base_vars,
        'source': (['gene_selected_loocv'] * len(selected)) + (['clinical_demographic'] * len(base_vars))
    })
    final_path = os.path.join(outdir, f"{stamp}_final_screened_features_for_models.csv")
    final_list.to_csv(final_path, index=False)
    print(f"- Final screened features (genes + Age/Sex/Treatment/Stage): {final_path}")

    # Short textual summary report
    report = os.path.join(outdir, f"{stamp}_loocv_screening_summary.txt")
    with open(report, 'w') as f:
        f.write("COX LOOCV INTERACTION SCREENING (TRAIN ONLY)\n")
        f.write("="*60 + "\n")
        f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Alpha: {args.alpha}\n")
        f.write(f"Train shape: {df.shape}\n")
        f.write(f"Events: {int(df['OS_STATUS'].sum())}; Censored: {df.shape[0] - int(df['OS_STATUS'].sum())}\n")
        f.write(f"Genomic features evaluated: {len(genomic_features)}\n")
        f.write(f"Genes retained (median p ≤ α): {len(selected)}\n")
        f.write(f"Clinical/demographic appended: {base_vars}\n")
        f.write("\nTop 20 retained genes:\n")
        f.write(selected[['gene','median_p','prop_significant_folds']].head(20).to_string(index=False))
        f.write("\n")
    print(f"- Summary report: {report}")

    # Done
    total_min = (time.time() - t0) / 60.0
    print("\n" + "="*80)
    print("LOOCV PRESCREEN COMPLETE")
    print("="*80)
    print(f"Elapsed: {total_min:.1f} min")
    print(f"Results saved to: {outdir}")

    # Reset stdout
    sys.stdout = sys.__stdout__
    log_file.close()

if __name__ == "__main__":
    main()