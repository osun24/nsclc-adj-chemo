#!/usr/bin/env python3
"""
Benchmark & extrapolate runtime for LOOCV univariate CoxPH (with ACT×Gene interaction)
Total fits = n_samples * n_genes. Measures (1) per-fit latency and (2) parallel efficiency,
then estimates wall time for the full 2.6M-model workload.

Usage:
  python bench_cox_loocv.py --train train_merged.csv --cores 14 --sample-genes 300
"""
import argparse, time, numpy as np, pandas as pd, multiprocessing as mp
from joblib import Parallel, delayed
from lifelines import CoxPHFitter

def identify_genomic_features(df: pd.DataFrame):
    survival_cols = ['OS_MONTHS', 'OS_STATUS']
    clinical_base = ['Adjuvant Chemo', 'Age', 'IS_MALE', 'Histology']
    stage_dummies = [c for c in df.columns if c.startswith('Stage_')]
    exclude = set(survival_cols + clinical_base + stage_dummies + ['Batch'])
    genomic = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    return genomic

def load_df(path: str):
    df = pd.read_csv(path, index_col=0)
    # normalize treatment
    if df['Adjuvant Chemo'].dtype == object:
        df['Adjuvant Chemo'] = df['Adjuvant Chemo'].replace({'OBS': 0, 'ACT': 1}).astype(float)
    else:
        df['Adjuvant Chemo'] = df['Adjuvant Chemo'].astype(float)
    # ensure survival columns present
    for c in ['OS_MONTHS','OS_STATUS']:
        if c not in df.columns:
            raise KeyError(f"Missing column: {c}")
    return df

def per_fit_latency_one_fold(df: pd.DataFrame, genomic_features, k_genes: int):
    """Single-core latency over one LOOCV training fold; reuse a fixed design frame."""
    n = df.shape[0]
    tr = df.iloc[: n-1]  # simulate leaving last sample out
    t  = tr['OS_MONTHS'].to_numpy()
    e  = tr['OS_STATUS'].to_numpy()
    z  = tr['Adjuvant Chemo'].to_numpy()
    # design template
    m = pd.DataFrame({
        'OS_MONTHS': t, 'OS_STATUS': e, 'Z': z,
        'G': np.empty_like(z), 'ZXG': np.empty_like(z)
    })
    # sample genes
    rng = np.random.default_rng(0)
    k = min(k_genes, len(genomic_features))
    genes = rng.choice(genomic_features, size=k, replace=False)

    cph = CoxPHFitter()
    t0 = time.time()
    for gname in genes:
        g = tr[gname].to_numpy()
        m['G'] = g; m['ZXG'] = z * g
        cph.fit(m, duration_col='OS_MONTHS', event_col='OS_STATUS',
                formula="G + Z + ZXG", robust=False)
        _ = float(cph.summary.loc['ZXG','p'])
    elapsed = time.time() - t0
    return elapsed / k  # seconds per fit

def _mini_fold_task(df, fold_idx, genes):
    tr = df.drop(df.index[fold_idx])
    t  = tr['OS_MONTHS'].to_numpy()
    e  = tr['OS_STATUS'].to_numpy()
    z  = tr['Adjuvant Chemo'].to_numpy()
    m  = pd.DataFrame({'OS_MONTHS': t, 'OS_STATUS': e, 'Z': z,
                       'G': np.empty_like(z), 'ZXG': np.empty_like(z)})
    cph = CoxPHFitter()
    for gname in genes:
        g = tr[gname].to_numpy(); m['G'] = g; m['ZXG'] = z * g
        cph.fit(m, duration_col='OS_MONTHS', event_col='OS_STATUS',
                formula="G + Z + ZXG", robust=False)
        _ = float(cph.summary.loc['ZXG','p'])
    return None

def parallel_efficiency(df: pd.DataFrame, genomic_features, cores: int, folds: int, genes_total: int):
    """Run a tiny parallel job to estimate efficiency vs ideal linear scaling."""
    n = df.shape[0]
    folds = min(folds, n)
    rng = np.random.default_rng(1)
    sample_genes = rng.choice(genomic_features, size=min(genes_total, len(genomic_features)), replace=False)
    # split sampled genes across folds (roughly equal)
    chunk = max(1, len(sample_genes) // folds)
    batches = [sample_genes[i*chunk:(i+1)*chunk] for i in range(folds)]
    batches = [b for b in batches if len(b) > 0]

    t0 = time.time()
    Parallel(n_jobs=cores, backend="loky")(
        delayed(_mini_fold_task)(df, i, batches[i]) for i in range(len(batches))
    )
    elapsed = time.time() - t0
    fits_done = sum(len(b) for b in batches)
    fits_per_sec_parallel = fits_done / max(elapsed, 1e-9)

    # single-core reference from the same tiny workload
    t1 = time.time()
    for i in range(len(batches)):
        _mini_fold_task(df, i, batches[i])
    elapsed_single = time.time() - t1
    fits_per_sec_single = fits_done / max(elapsed_single, 1e-9)

    ideal = cores * fits_per_sec_single
    eff = fits_per_sec_parallel / max(ideal, 1e-9)
    return eff

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', required=True, help='Path to training CSV')
    ap.add_argument('--cores', type=int, default=min(14, mp.cpu_count()))
    ap.add_argument('--sample-genes', type=int, default=300, help='Genes for per-fit latency sample')
    ap.add_argument('--eff-folds', type=int, default=8, help='Folds for efficiency micro-benchmark')
    ap.add_argument('--eff-genes', type=int, default=200, help='Total genes in efficiency micro-benchmark')
    args = ap.parse_args()

    df = load_df(args.train)
    genomic_features = identify_genomic_features(df)
    n, g = df.shape[0], len(genomic_features)
    total_fits = n * g

    print(f"Samples: {n} | Candidate genes: {g} | Total fits (n*g): {total_fits:,}")
    print(f"Cores requested: {args.cores}")

    per_fit_sec = per_fit_latency_one_fold(df, genomic_features, args.sample_genes)
    print(f"Per-fit latency (single core, one fold): {per_fit_sec:.5f} s")

    eff = parallel_efficiency(df, genomic_features, args.cores, args.eff_folds, args.eff_genes)
    print(f"Measured parallel efficiency vs ideal linear: {eff:.2f}")

    fits_per_sec_single = 1.0 / max(per_fit_sec, 1e-12)
    total_seconds = total_fits / (args.cores * fits_per_sec_single * max(eff, 1e-3))

    hrs = total_seconds / 3600.0
    print("\n===== ESTIMATE =====")
    print(f"Estimated wall time: {hrs:.2f} hours")
    print(f"(Assumes same per-fit cost across folds/genes and efficiency={eff:.2f} on {args.cores} cores)")

if __name__ == "__main__":
    main()