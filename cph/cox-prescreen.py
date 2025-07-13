"""
Cox Proportional Hazards Model with Interactions for Genomic Feature Prescreening

This script prescreens genomic features using Cox regression with treatment-gene interactions.
For each genomic feature, we test the interaction between adjuvant chemotherapy and the gene.
Only genomic features with significant treatment interactions are retained.

Two methodologies are supported:
1. CV-based screening (default): 20 trials × 10-fold CV (similar to RSF approach)
2. Single-analysis screening: One analysis on combined dataset

Approach:
- For each gene: Cox model with Adjuvant Chemo + Gene + Adjuvant Chemo * Gene
- CV method: Count selection frequency across CV folds
- Single method: Test significance of interaction term at alpha = 0.05 and 0.10
- Use train + validation data (follows rsf-gridsearch-featureimp-affy.py)
- Exclude clinical features from screening

Model: Cox(Adjuvant Chemo + Gene + Adjuvant Chemo * Gene)

Key Features:
- ROBUST CV-BASED GENE SELECTION (similar to R implementation)
- OPTIMIZED FOR COMPUTATIONAL EFFICIENCY with parallel processing
- Uses joblib for multiprocessing with n_jobs=-1 (all CPU cores)
- Processes thousands of genes simultaneously instead of sequentially
- Maintains identical preprocessing as rsf-gridsearch-featureimp-affy.py
- Excludes clinical features from screening (genomic features only)
- Provides comprehensive results with multiple significance levels

Performance:
- CV method: ~20-200 CV folds × genes/second (depends on CPU cores)
- Single method: ~10-50 genes/second (depends on CPU cores)
- For ~22,000 genes: CV ~1-4 hours vs Single ~10-40 minutes

Usage:
    python cox-prescreen.py                       # CV method (default)
    python cox-prescreen.py --no-cv               # Single analysis
    python cox-prescreen.py --n-trials 10         # 10 trials x 10 folds
    python cox-prescreen.py --n-jobs 4            # Use 4 cores
    python cox-prescreen.py --no-parallel         # Sequential processing
    python cox-prescreen.py --alpha 0.05 0.01     # Custom alpha levels
"""

import os
import sys
import time
import datetime
import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test # THINK ABOUT THIS!
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from joblib import Parallel, delayed
import multiprocessing
from sklearn.model_selection import KFold

np.random.seed(42)

# Setup directories and logging
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "cox_prescreen_results")
os.makedirs(output_dir, exist_ok=True)

class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
    def flush(self):
        for f in self.files:
            f.flush()

import sys
current_date = datetime.datetime.now().strftime("%Y%m%d")
log_file = open(os.path.join(output_dir, f"{current_date}_cox_prescreen_log.txt"), "w")
sys.stdout = Tee(sys.stdout, log_file)

def load_and_combine_data():
    """Load and combine all datasets following rsf-gridsearch-featureimp-affy.py approach"""
    print("="*80)
    print("LOADING AND COMBINING DATASETS")
    print("="*80)
    
    # Load datasets exactly as in rsf-gridsearch-featureimp-affy.py
    print("Loading train data from: affyTrain.csv")
    train = pd.read_csv("affyTrain.csv")
    print(f"Train data shape: {train.shape}")
    print(f"Train events: {train['OS_STATUS'].sum()} | Censored: {train.shape[0] - train['OS_STATUS'].sum()}")
    
    print("Loading validation data from: affyValidation.csv")
    valid = pd.read_csv("affyValidation.csv")
    print(f"Validation data shape: {valid.shape}")
    print(f"Validation events: {valid['OS_STATUS'].sum()} | Censored: {valid.shape[0] - valid['OS_STATUS'].sum()}")
    
    # Combine train and validation first (as in rsf script)
    train_combined = pd.concat([train, valid], ignore_index=True)
    print(f"Combined train+validation shape: {train_combined.shape}")

    # Apply same preprocessing as in rsf script
    train_combined['Adjuvant Chemo'] = train_combined['Adjuvant Chemo'].replace({'OBS': 0, 'ACT': 1})
    
    return train_combined

def identify_feature_types(df):
    """Identify clinical vs genomic features"""
    print("\n" + "="*80)
    print("IDENTIFYING FEATURE TYPES")
    print("="*80)
    
    # Define clinical features to exclude from screening
    clinical_features = [
        'Age', 'Stage_IA', 'Stage_IB', 'Stage_II', 'Stage_III', 'Stage_IV', 'Stage_Unknown',
        'Histology_Adenocarcinoma', 'Histology_Adenosquamous Carcinoma', 
        'Histology_Large Cell Carcinoma', 'Histology_Squamous Cell Carcinoma',
        'Race_African American', 'Race_Asian', 'Race_Caucasian', 
        'Race_Native Hawaiian or Other Pacific Islander', 'Race_Unknown',
        'Smoked?_No', 'Smoked?_Unknown', 'Smoked?_Yes'
    ]
    
    # Define other non-genomic columns
    non_genomic_features = ['OS_STATUS', 'OS_MONTHS', 'Adjuvant Chemo', 'IS_MALE']
    
    # All columns excluding survival, treatment, and clinical features
    all_columns = set(df.columns)
    exclude_columns = set(clinical_features + non_genomic_features)
    
    # Genomic features are everything else
    genomic_features = list(all_columns - exclude_columns)
    
    # Remove any clinical features that might not exist in the data
    existing_clinical = [f for f in clinical_features if f in df.columns]
    missing_clinical = [f for f in clinical_features if f not in df.columns]
    
    print(f"Total columns in dataset: {len(all_columns)}")
    print(f"Clinical features found: {len(existing_clinical)}")
    print(f"Clinical features missing: {len(missing_clinical)}")
    print(f"Genomic features to screen: {len(genomic_features)}")
    
    if missing_clinical:
        print(f"Missing clinical features: {missing_clinical[:10]}{'...' if len(missing_clinical) > 10 else ''}")
    
    # Verify genomic features
    print(f"First 10 genomic features: {genomic_features[:10]}")
    
    return genomic_features, existing_clinical

def run_interaction_screening_cv(df, genomic_features, n_trials=20, n_folds=10, alpha_levels=[0.05, 0.10], use_parallel=True, n_jobs=-1):
    """
    Run interaction screening using 20 trials of 10-fold CV methodology (similar to RSF script)
    
    This approach provides more robust gene selection by:
    1. Running multiple trials with different random seeds
    2. Using k-fold cross-validation within each trial
    3. Counting how many times each gene is selected across all CV folds
    4. Only retaining genes that are consistently selected
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Combined dataset with survival outcomes and genomic features
    genomic_features : list
        List of genomic feature names to test
    n_trials : int
        Number of CV trials to run (default: 20)
    n_folds : int
        Number of folds per trial (default: 10)
    alpha_levels : list
        Significance levels for interaction term (default: [0.05, 0.10])
    use_parallel : bool
        Whether to use parallel processing (default: True)
    n_jobs : int
        Number of cores to use for parallel processing (default: -1)
    
    Returns:
    --------
    selection_results : pandas.DataFrame
        DataFrame with selection frequencies for each gene across all CV folds
    detailed_results : list
        List of detailed results for each significant interaction found
    """
    print("\n" + "="*80)
    print("RUNNING CV-BASED TREATMENT-GENE INTERACTION SCREENING")
    print("="*80)
    
    print(f"Configuration:")
    print(f"- Number of trials: {n_trials}")
    print(f"- Folds per trial: {n_folds}")
    print(f"- Total CV iterations: {n_trials * n_folds}")
    print(f"- Genes to test: {len(genomic_features)}")
    print(f"- Alpha levels: {alpha_levels}")
    print(f"- Model: Cox(Adjuvant Chemo + Gene + Adjuvant Chemo * Gene)")
    
    n_cores = multiprocessing.cpu_count() if n_jobs == -1 else min(n_jobs, multiprocessing.cpu_count())
    print(f"- Parallelization: {'Enabled' if use_parallel else 'Disabled'}")
    if use_parallel:
        print(f"- CPU cores: {n_cores}")
    
    total_cv_folds = n_trials * n_folds
    selection_matrix = np.zeros((len(genomic_features), total_cv_folds))
    detailed_results = []
    start_time = time.time()
    
    print(f"\nStarting {n_trials} trials of {n_folds}-fold CV...")
    
    for trial in range(n_trials):
        print(f"\n[Trial {trial+1}/{n_trials}] Starting...")
        trial_seed = 2000 + trial
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=trial_seed)
        shuffled_data = df.sample(frac=1, random_state=trial_seed).reset_index(drop=True)
        
        for fold_idx, (train_idx, _) in enumerate(kfold.split(shuffled_data)):
            cv_fold_counter = trial * n_folds + fold_idx
            print(f"[Trial {trial+1}, Fold {fold_idx+1}] Processing CV fold {cv_fold_counter + 1}/{total_cv_folds}")
            
            train_fold = shuffled_data.iloc[train_idx]
            
            clinical_data = train_fold[['OS_STATUS', 'OS_MONTHS', 'Adjuvant Chemo']]
            gene_data_list = [
                (idx, gene, train_fold[gene], clinical_data, alpha_levels)
                for idx, gene in enumerate(genomic_features)
            ]

            if use_parallel:
                fold_results = Parallel(n_jobs=n_cores)(
                    delayed(_test_gene_interaction)(data) for data in gene_data_list
                )
            else:
                fold_results = [_test_gene_interaction(data) for data in gene_data_list]

            for result in fold_results:
                if result:
                    gene_idx, _, is_significant, interaction_result = result
                    if is_significant:
                        selection_matrix[gene_idx, cv_fold_counter] = 1
                        interaction_result.update({'trial': trial + 1, 'fold': fold_idx + 1, 'cv_fold': cv_fold_counter + 1})
                        detailed_results.append(interaction_result)

        if trial > 0:
            elapsed = time.time() - start_time
            avg_time_per_trial = elapsed / (trial + 1)
            eta = (n_trials - (trial + 1)) * avg_time_per_trial
            print(f"[Trial {trial+1}] Completed. ETA: {eta/60:.1f} minutes")

    total_time = time.time() - start_time
    print(f"\nCV screening completed in {total_time/60:.1f} minutes")

    selection_summary = pd.DataFrame({
        'gene': genomic_features,
        'total_selections': selection_matrix.sum(axis=1),
        'selection_frequency': selection_matrix.sum(axis=1) / total_cv_folds,
        'selected_in_trials': [
            np.sum(selection_matrix[i, :].reshape(n_trials, n_folds).any(axis=1))
            for i in range(len(genomic_features))
        ]
    }).sort_values('total_selections', ascending=False).reset_index(drop=True)

    selected_genes = selection_summary[selection_summary['total_selections'] >= 1].copy()
    
    print(f"\nCV Selection Results:")
    print(f"- Total genes tested: {len(genomic_features)}")
    print(f"- Genes selected ≥1 time: {len(selected_genes)}")
    
    if not selected_genes.empty:
        top_gene = selected_genes.iloc[0]
        print(f"- Most selected gene: {top_gene['gene']} ({int(top_gene['total_selections'])}/{total_cv_folds} folds)")

    return selected_genes, detailed_results

def _test_gene_interaction(gene_data):
    """Helper function to test gene interaction for a single gene."""
    gene_idx, gene_name, gene_series, clinical_data, alpha_levels = gene_data
    
    try:
        model_data = clinical_data.copy()
        # This is safe because pandas aligns on index.
        # The bug was resetting the index, which is now removed.
        model_data[gene_name] = gene_series
        model_data[f'Adjuvant_Chemo_x_{gene_name}'] = model_data['Adjuvant Chemo'] * model_data[gene_name]

        cph = CoxPHFitter()
        cph.fit(model_data, duration_col='OS_MONTHS', event_col='OS_STATUS', show_progress=False,
                formula=f"`{gene_name}` + `Adjuvant Chemo` + `Adjuvant_Chemo_x_{gene_name}`")

        interaction_p_value = cph.summary.loc[f'Adjuvant_Chemo_x_{gene_name}', 'p']
        is_significant = any(interaction_p_value <= alpha for alpha in alpha_levels)

        interaction_result = {
            'gene': gene_name,
            'interaction_coef': cph.params_[f'Adjuvant_Chemo_x_{gene_name}'],
            'interaction_p_value': interaction_p_value,
            'interaction_hr': np.exp(cph.params_[f'Adjuvant_Chemo_x_{gene_name}']),
            'n_observations': len(model_data),
            'n_events': model_data['OS_STATUS'].sum(),
        }
        for alpha in alpha_levels:
            interaction_result[f'significant_at_{alpha:.2f}'] = interaction_p_value <= alpha
        
        return gene_idx, gene_name, is_significant, interaction_result
    except Exception:
        # Return None on any error during model fitting for a single gene
        return None

def analyze_cv_screening_results(selection_results, detailed_results, alpha_levels=[0.05, 0.10]):
    """Analyze and summarize CV-based screening results"""
    print("\n" + "="*80)
    print("ANALYZING CV SCREENING RESULTS")
    print("="*80)
    
    if selection_results.empty:
        print("No genes were selected during CV screening.")
        return selection_results

    # Summary statistics
    print(f"Total genes tested: {len(selection_results)}")
    print(f"Genes selected ≥1 time: {len(selection_results[selection_results['total_selections'] >= 1])}")
    print(f"Genes selected ≥5 times: {len(selection_results[selection_results['total_selections'] >= 5])}")
    print(f"Genes selected ≥10 times: {len(selection_results[selection_results['total_selections'] >= 10])}")
    
    # Show top selected genes
    print(f"\nTop 20 most frequently selected genes:")
    top_genes = selection_results.head(20).copy()
    
    # Format for display
    top_genes['selection_freq_pct'] = (top_genes['selection_frequency'] * 100).round(1)
    
    display_cols = ['gene', 'total_selections', 'selection_freq_pct', 'selected_in_trials']
    print(top_genes[display_cols].to_string(index=False))
    
    # Analyze detailed results if available
    if detailed_results:
        print(f"\nDetailed interaction results:")
        print(f"- Total significant interactions found: {len(detailed_results)}")
        
        # Convert to DataFrame for analysis
        detailed_df = pd.DataFrame(detailed_results)
        
        # Show distribution of p-values
        print(f"- Mean interaction p-value: {detailed_df['interaction_p_value'].mean():.4f}")
        print(f"- Median interaction p-value: {detailed_df['interaction_p_value'].median():.4f}")
        
        # Show top interactions by p-value
        print(f"\nTop 10 strongest interactions (lowest p-values):")
        top_interactions = detailed_df.nsmallest(10, 'interaction_p_value')[
            ['gene', 'interaction_p_value', 'interaction_hr', 'trial', 'fold']
        ]
        print(top_interactions.to_string(index=False))
    
    return selection_results

def save_cv_results(selection_results, detailed_results, output_dir, alpha_levels=[0.05, 0.10]):
    """Save CV-based screening results to files"""
    print("\n" + "="*80)
    print("SAVING CV RESULTS")
    print("="*80)
    
    # Save selection frequency results
    selection_path = os.path.join(output_dir, f"{current_date}_cox_interaction_cv_selection_results.csv")
    selection_results.to_csv(selection_path, index=False)
    print(f"Gene selection frequencies saved to: {selection_path}")
    
    # Save detailed interaction results
    if detailed_results:
        detailed_df = pd.DataFrame(detailed_results)
        detailed_path = os.path.join(output_dir, f"{current_date}_cox_interaction_cv_detailed_results.csv")
        detailed_df.to_csv(detailed_path, index=False)
        print(f"Detailed interaction results saved to: {detailed_path}")

def create_cv_visualizations(selection_results, detailed_results, output_dir):
    """Create visualizations for CV-based screening results"""
    print("\n" + "="*80)
    print("CREATING CV VISUALIZATIONS")
    print("="*80)
    
    # 1. Selection frequency histogram
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.hist(selection_results['total_selections'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Number of CV Folds Selected')
    plt.ylabel('Number of Genes')
    plt.title('Distribution of Gene Selection Frequencies')
    plt.axvline(x=selection_results['total_selections'].mean(), color='red', linestyle='--', label='Mean')
    plt.legend()
    
    # 2. Selection frequency vs trial consistency
    plt.subplot(2, 3, 2)
    plt.scatter(selection_results['total_selections'], selection_results['selected_in_trials'], 
                alpha=0.6, s=20)
    plt.xlabel('Total CV Folds Selected')
    plt.ylabel('Number of Trials Selected')
    plt.title('Selection Consistency Across Trials')
    
    # 3. Top genes bar plot
    plt.subplot(2, 3, 3)
    top_20 = selection_results.head(20)
    plt.barh(range(len(top_20)), top_20['total_selections'][::-1], color='lightgreen')
    plt.yticks(range(len(top_20)), top_20['gene'][::-1])
    plt.xlabel('Number of CV Folds Selected')
    plt.title('Top 20 Most Selected Genes')
    
    # 4. Selection frequency distribution by threshold
    plt.subplot(2, 3, 4)
    thresholds = [1, 5, 10, 25, 50, 100]
    counts = [len(selection_results[selection_results['total_selections'] >= t]) for t in thresholds]
    plt.bar(range(len(thresholds)), counts, color='orange')
    plt.xticks(range(len(thresholds)), thresholds)
    plt.xlabel('Minimum Selection Threshold')
    plt.ylabel('Number of Genes')
    plt.title('Genes by Selection Threshold')
    
    # 5. P-value distribution (if detailed results available)
    if detailed_results:
        plt.subplot(2, 3, 5)
        detailed_df = pd.DataFrame(detailed_results)
        plt.hist(detailed_df['interaction_p_value'], bins=50, alpha=0.7, color='pink', edgecolor='black')
        plt.xlabel('Interaction P-value')
        plt.ylabel('Frequency')
        plt.title('Distribution of Significant Interaction P-values')
        plt.axvline(x=0.05, color='red', linestyle='--', label='α = 0.05')
        plt.legend()
        
        # 6. Effect size distribution
        plt.subplot(2, 3, 6)
        plt.hist(np.log(detailed_df['interaction_hr']), bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
        plt.xlabel('Log(Interaction Hazard Ratio)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Interaction Effect Sizes')
        plt.axvline(x=0, color='red', linestyle='-', label='HR = 1')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{current_date}_cv_interaction_screening_overview.png"), dpi=300)
    plt.close()
    
    # Create selection frequency heatmap for top genes
    plt.figure(figsize=(12, 8))
    top_50 = selection_results.head(50)
    
    # Create a simple bar plot showing selection frequency
    plt.barh(range(len(top_50)), top_50['selection_frequency'][::-1], color='steelblue')
    plt.yticks(range(len(top_50)), top_50['gene'][::-1])
    plt.xlabel('Selection Frequency')
    plt.title('Top 50 Genes - Selection Frequency Across CV Folds')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{current_date}_cv_top_genes_selection_frequency.png"), dpi=300)
    plt.close()
    
    print("CV visualizations saved:")
    print(f"- {current_date}_cv_interaction_screening_overview.png")
    print(f"- {current_date}_cv_top_genes_selection_frequency.png")

def create_cv_summary_report(selection_results, detailed_results, output_dir, n_trials=20, n_folds=10):
    """Create a summary report for CV-based screening"""
    print("\n" + "="*80)
    print("CREATING CV SUMMARY REPORT")
    print("="*80)
    
    report_path = os.path.join(output_dir, f"{current_date}_cox_interaction_cv_screening_summary.txt")
    
    with open(report_path, 'w') as f:
        f.write("COX INTERACTION SCREENING - CROSS-VALIDATION SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Screening method: CV-based treatment-gene interaction testing\n")
        f.write(f"Model: Cox(Adjuvant Chemo + Gene + Adjuvant Chemo * Gene)\n")
        f.write(f"CV Configuration: {n_trials} trials × {n_folds} folds = {n_trials * n_folds} total CV iterations\n\n")
        
        f.write("DATASET SUMMARY:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total genes tested: {len(selection_results)}\n")
        
        # Selection frequency summary
        f.write(f"\nSELECTION FREQUENCY SUMMARY:\n")
        f.write("-" * 30 + "\n")
        for min_sel in [1, 5, 10, 25, 50, 100]:
            count = len(selection_results[selection_results['total_selections'] >= min_sel])
            pct = (count / len(selection_results)) * 100
            f.write(f"Genes selected ≥{min_sel:2d} times: {count:4d} ({pct:.1f}%)\n")
        
        f.write(f"\nTOP 20 MOST SELECTED GENES:\n")
        f.write("-" * 30 + "\n")
        top_20 = selection_results.head(20)
        for _, row in top_20.iterrows():
            f.write(f"{row['gene']:<15}: {row['total_selections']:3d}/{n_trials * n_folds} folds "
                   f"({row['selection_frequency']*100:.1f}%), "
                   f"in {row['selected_in_trials']:2d}/{n_trials} trials\n")
        
        if detailed_results:
            detailed_df = pd.DataFrame(detailed_results)
            f.write(f"\nINTERACTION STATISTICS:\n")
            f.write("-" * 25 + "\n")
            f.write(f"Total significant interactions: {len(detailed_results)}\n")
            f.write(f"Mean interaction p-value: {detailed_df['interaction_p_value'].mean():.4f}\n")
            f.write(f"Median interaction p-value: {detailed_df['interaction_p_value'].median():.4f}\n")
            f.write(f"Mean interaction HR: {detailed_df['interaction_hr'].mean():.3f}\n")
            f.write(f"Median interaction HR: {detailed_df['interaction_hr'].median():.3f}\n")
        
        f.write(f"\nFILES GENERATED:\n")
        f.write("-" * 20 + "\n")
        f.write(f"- {current_date}_cox_interaction_cv_selection_results.csv\n")
        f.write(f"- {current_date}_cox_interaction_cv_detailed_results.csv\n")
        f.write(f"- {current_date}_cv_interaction_screening_overview.png\n")
        f.write(f"- {current_date}_cv_top_genes_selection_frequency.png\n")
        for min_sel in [1, 5, 10, 25, 50]:
            f.write(f"- {current_date}_selected_genes_min_{min_sel}_selections.csv\n")
    
    print(f"CV summary report saved to: {report_path}")

def main(use_parallel=True, n_jobs=-1, alpha_levels=[0.05, 0.10], use_cv=True, n_trials=20, n_folds=10):
    """Main function to run Cox interaction screening"""
    print("="*80)
    print("COX PROPORTIONAL HAZARDS INTERACTION SCREENING")
    print("="*80)
    print("Prescreening genomic features using treatment-gene interactions")
    print("Model: Cox(Adjuvant Chemo + Gene + Adjuvant Chemo * Gene)")
    
    if use_cv:
        print(f"Method: {n_trials} trials x {n_folds}-fold CV (like RSF approach)")
        print("Selection criteria: Genes with significant interactions across CV folds")
    else:
        print("Method: Single analysis on combined dataset")
        print("Selection criteria: Significant interaction term (p < α)")
    
    print("="*80)
    
    # Track overall timing
    main_start_time = time.time()
    
    # Load and combine data
    full_data = load_and_combine_data()
    
    # Identify feature types
    genomic_features, clinical_features = identify_feature_types(full_data)
    
    # Use CV-based screening (similar to RSF and R implementation)
    selection_results, detailed_results = run_interaction_screening_cv(
        full_data,
        genomic_features,
        n_trials=n_trials,
        n_folds=n_folds,
        alpha_levels=alpha_levels,
        use_parallel=use_parallel,
        n_jobs=n_jobs
    )
    
    # Analyze CV results
    selection_results = analyze_cv_screening_results(selection_results, detailed_results, alpha_levels)
    
    # Create CV visualizations
    create_cv_visualizations(selection_results, detailed_results, output_dir)
    
    # Save CV results
    save_cv_results(selection_results, detailed_results, output_dir, alpha_levels)
    
    # Create CV summary report
    create_cv_summary_report(selection_results, detailed_results, output_dir, n_trials, n_folds)
    
    total_time = time.time() - main_start_time
    print("\n" + "="*80)
    print("COX INTERACTION SCREENING COMPLETE")
    print("="*80)
    print(f"Total pipeline time: {total_time/60:.1f} minutes")
    print(f"Results saved to: {output_dir}")
    
    # Close log file
    sys.stdout = sys.__stdout__
    log_file.close()

if __name__ == "__main__":
   main()