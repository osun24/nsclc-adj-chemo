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
import argparse
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

def test_interaction_for_gene_parallel(gene_data, alpha_levels=[0.05, 0.10]):
    """
    Test treatment-gene interaction for a single gene using Cox PH model.
    This function is designed to be called in parallel.
    
    Parameters:
    -----------
    gene_data : tuple
        (gene_name, gene_series, clinical_data)
    alpha_levels : list
        Significance levels to test
    
    Returns:
    --------
    dict or None : Results dictionary or None if fitting failed
    """
    gene_name, gene_series, clinical_data = gene_data
    
    try:
        # Prepare data for this gene
        model_data = clinical_data.copy()
        model_data[gene_name] = gene_series
        
        # Create interaction term
        model_data[f'Adjuvant_Chemo_x_{gene_name}'] = model_data['Adjuvant Chemo'] * model_data[gene_name]
        
        # Fit Cox model with interaction
        cph = CoxPHFitter()
        cph.fit(model_data, duration_col='OS_MONTHS', event_col='OS_STATUS', show_progress=False)
        
        # Extract interaction term results
        interaction_coef = cph.params_[f'Adjuvant_Chemo_x_{gene_name}']
        interaction_p_value = cph.summary.loc[f'Adjuvant_Chemo_x_{gene_name}', 'p']
        
        # Handle different column names for confidence intervals
        ci_cols = cph.confidence_intervals_.columns
        if 'coef lower 95%' in ci_cols:
            ci_lower_col = 'coef lower 95%'
            ci_upper_col = 'coef upper 95%'
        elif '95% CI lower' in ci_cols:
            ci_lower_col = '95% CI lower'
            ci_upper_col = '95% CI upper'
        else:
            # Try to find CI columns by pattern
            lower_cols = [col for col in ci_cols if 'lower' in col.lower()]
            upper_cols = [col for col in ci_cols if 'upper' in col.lower()]
            if lower_cols and upper_cols:
                ci_lower_col = lower_cols[0]
                ci_upper_col = upper_cols[0]
            else:
                raise KeyError(f"Could not find confidence interval columns. Available columns: {list(ci_cols)}")
        
        interaction_ci_lower = cph.confidence_intervals_.loc[f'Adjuvant_Chemo_x_{gene_name}', ci_lower_col]
        interaction_ci_upper = cph.confidence_intervals_.loc[f'Adjuvant_Chemo_x_{gene_name}', ci_upper_col]
        interaction_hr = np.exp(interaction_coef)
        interaction_hr_ci_lower = np.exp(interaction_ci_lower)
        interaction_hr_ci_upper = np.exp(interaction_ci_upper)
        
        # Also get main effects
        adjuvant_coef = cph.params_['Adjuvant Chemo']
        adjuvant_p_value = cph.summary.loc['Adjuvant Chemo', 'p']
        gene_coef = cph.params_[gene_name]
        gene_p_value = cph.summary.loc[gene_name, 'p']
        
        concordance = cph.concordance_index_
        log_likelihood = cph.log_likelihood_
        aic = cph.AIC_partial_
        
        # Determine significance at different alpha levels
        results = {
            'gene': gene_name,
            'n_observations': len(model_data),
            'n_events': model_data['OS_STATUS'].sum(),
            
            # Interaction term results
            'interaction_coef': interaction_coef,
            'interaction_p_value': interaction_p_value,
            'interaction_hr': interaction_hr,
            'interaction_hr_ci_lower': interaction_hr_ci_lower,
            'interaction_hr_ci_upper': interaction_hr_ci_upper,
            
            # Main effects
            'adjuvant_coef': adjuvant_coef,
            'adjuvant_p_value': adjuvant_p_value,
            'gene_coef': gene_coef,
            'gene_p_value': gene_p_value,
            
            # Model statistics
            'concordance': concordance,
            'log_likelihood': log_likelihood,
            'aic': aic,
        }
        
        # Add significance flags for different alpha levels
        for alpha in alpha_levels:
            results[f'significant_at_{alpha:.2f}'] = interaction_p_value <= alpha
        
        return results
        
    except KeyError as e:
        print(f"Error testing gene {gene_name}: Column access error - {str(e)}")
        # Print available columns for debugging
        if 'cph' in locals():
            print(f"Available confidence interval columns: {list(cph.confidence_intervals_.columns)}")
        return None
    except Exception as e:
        print(f"Error testing gene {gene_name}: {str(e)}")
        return None

def test_interaction_for_gene(df, gene, alpha_levels=[0.05, 0.10]):
    """
    Test treatment-gene interaction for a single gene using Cox PH model
    
    Model: Cox(Adjuvant Chemo + Gene + Adjuvant Chemo * Gene)
    """
    try:
        # Prepare data for this gene
        model_data = df[['OS_STATUS', 'OS_MONTHS', 'Adjuvant Chemo', gene]].copy()

        # Create interaction term
        model_data[f'Adjuvant_Chemo_x_{gene}'] = model_data['Adjuvant Chemo'] * model_data[gene]
        
        # Fit Cox model with interaction
        cph = CoxPHFitter()
        cph.fit(model_data, duration_col='OS_MONTHS', event_col='OS_STATUS')
        
        # Extract interaction term results
        interaction_coef = cph.params_[f'Adjuvant_Chemo_x_{gene}']
        interaction_p_value = cph.summary.loc[f'Adjuvant_Chemo_x_{gene}', 'p']
        
        # Handle different column names for confidence intervals
        ci_cols = cph.confidence_intervals_.columns
        if 'coef lower 95%' in ci_cols:
            ci_lower_col = 'coef lower 95%'
            ci_upper_col = 'coef upper 95%'
        elif '95% CI lower' in ci_cols:
            ci_lower_col = '95% CI lower'
            ci_upper_col = '95% CI upper'
        else:
            # Try to find CI columns by pattern
            lower_cols = [col for col in ci_cols if 'lower' in col.lower()]
            upper_cols = [col for col in ci_cols if 'upper' in col.lower()]
            if lower_cols and upper_cols:
                ci_lower_col = lower_cols[0]
                ci_upper_col = upper_cols[0]
            else:
                raise KeyError(f"Could not find confidence interval columns. Available columns: {list(ci_cols)}")
        
        interaction_ci_lower = cph.confidence_intervals_.loc[f'Adjuvant_Chemo_x_{gene}', ci_lower_col]
        interaction_ci_upper = cph.confidence_intervals_.loc[f'Adjuvant_Chemo_x_{gene}', ci_upper_col]
        interaction_hr = np.exp(interaction_coef)
        interaction_hr_ci_lower = np.exp(interaction_ci_lower)
        interaction_hr_ci_upper = np.exp(interaction_ci_upper)
        
        # Also get main effects
        adjuvant_coef = cph.params_['Adjuvant Chemo']
        adjuvant_p_value = cph.summary.loc['Adjuvant Chemo', 'p']
        gene_coef = cph.params_[gene]
        gene_p_value = cph.summary.loc[gene, 'p']
        
        # Model statistics
        concordance = cph.concordance_index_
        log_likelihood = cph.log_likelihood_
        aic = cph.AIC_partial_
        
        # Determine significance at different alpha levels
        results = {
            'gene': gene,
            'n_observations': len(model_data),
            'n_events': model_data['OS_STATUS'].sum(),
            
            # Interaction term results
            'interaction_coef': interaction_coef,
            'interaction_p_value': interaction_p_value,
            'interaction_hr': interaction_hr,
            'interaction_hr_ci_lower': interaction_hr_ci_lower,
            'interaction_hr_ci_upper': interaction_hr_ci_upper,
            
            # Main effects
            'adjuvant_coef': adjuvant_coef,
            'adjuvant_p_value': adjuvant_p_value,
            'gene_coef': gene_coef,
            'gene_p_value': gene_p_value,
            
            # Model statistics
            'concordance': concordance,
            'log_likelihood': log_likelihood,
            'aic': aic,
        }
        
        # Add significance flags for different alpha levels
        for alpha in alpha_levels:
            results[f'significant_at_{alpha:.2f}'] = interaction_p_value <= alpha
        
        return results
        
    except KeyError as e:
        print(f"Error testing gene {gene}: Column access error - {str(e)}")
        # Print available columns for debugging
        if 'cph' in locals():
            print(f"Available confidence interval columns: {list(cph.confidence_intervals_.columns)}")
        return None
    except Exception as e:
        print(f"Error testing gene {gene}: {str(e)}")
        return None

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
    
    # Determine number of cores
    if n_jobs == -1:
        n_cores = multiprocessing.cpu_count()
    else:
        n_cores = min(n_jobs, multiprocessing.cpu_count())
    
    print(f"- Parallelization: {'Enabled' if use_parallel else 'Disabled'}")
    if use_parallel:
        print(f"- CPU cores: {n_cores}")
    
    # Initialize selection matrix (genes × CV folds)
    total_cv_folds = n_trials * n_folds
    selection_matrix = np.zeros((len(genomic_features), total_cv_folds))
    
    # Store detailed results for significant interactions
    detailed_results = []
    
    # Track timing
    start_time = time.time()
    cv_fold_counter = 0
    
    print(f"\nStarting {n_trials} trials of {n_folds}-fold CV...")
    
    for trial in range(n_trials):
        print(f"\n[Trial {trial+1}/{n_trials}] Starting...")
        
        # Set seed for reproducibility (similar to RSF script)
        trial_seed = 2000 + trial
        np.random.seed(trial_seed)
        
        # Shuffle data for this trial
        n_patients = len(df)
        shuffled_idx = np.random.permutation(n_patients)
        shuffled_data = df.iloc[shuffled_idx].reset_index(drop=True)
        
        # Create K-fold splits
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=trial_seed)
        
        for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(shuffled_data)):
            cv_fold_counter += 1
            print(f"[Trial {trial+1}, Fold {fold_idx+1}] Processing CV fold {cv_fold_counter}/{total_cv_folds}")
            
            # Get training data for this fold
            train_fold = shuffled_data.iloc[train_idx].copy()
            
            # Process genes in parallel or sequentially
            if use_parallel:
                # Prepare data for parallel processing
                clinical_data = train_fold[['OS_STATUS', 'OS_MONTHS', 'Adjuvant Chemo']].copy()
                gene_data_list = []
                
                for gene_idx, gene in enumerate(genomic_features):
                    gene_series = train_fold[gene]
                    gene_data_list.append((gene_idx, gene, gene_series, clinical_data, alpha_levels))
                
                # Run parallel processing
                fold_results = Parallel(n_jobs=n_cores, verbose=0)(
                    delayed(test_gene_interaction_cv_fold)(gene_data)
                    for gene_data in gene_data_list
                )
                
                # Process results
                for result in fold_results:
                    if result is not None:
                        gene_idx, gene_name, is_significant, interaction_result = result
                        
                        # Mark as selected if significant at any alpha level
                        if is_significant:
                            selection_matrix[gene_idx, cv_fold_counter-1] = 1
                            
                            # Store detailed results
                            interaction_result['trial'] = trial + 1
                            interaction_result['fold'] = fold_idx + 1
                            interaction_result['cv_fold'] = cv_fold_counter
                            detailed_results.append(interaction_result)
            
            else:
                # Sequential processing
                for gene_idx, gene in enumerate(genomic_features):
                    result = test_gene_interaction_single_fold(
                        train_fold, gene, alpha_levels
                    )
                    
                    if result is not None:
                        is_significant = any(result[f'significant_at_{alpha:.2f}'] for alpha in alpha_levels)
                        
                        if is_significant:
                            selection_matrix[gene_idx, cv_fold_counter-1] = 1
                            
                            # Store detailed results
                            result['trial'] = trial + 1
                            result['fold'] = fold_idx + 1
                            result['cv_fold'] = cv_fold_counter
                            detailed_results.append(result)
        
        # Progress update
        elapsed = time.time() - start_time
        remaining_trials = n_trials - (trial + 1)
        if trial > 0:
            avg_time_per_trial = elapsed / (trial + 1)
            eta = remaining_trials * avg_time_per_trial
            print(f"[Trial {trial+1}] Completed. ETA: {eta/60:.1f} minutes")
    
    total_time = time.time() - start_time
    print(f"\nCV screening completed in {total_time/60:.1f} minutes")
    
    # Create selection results DataFrame
    selection_results = pd.DataFrame(selection_matrix.T, columns=genomic_features)
    
    # Add summary statistics
    selection_summary = pd.DataFrame({
        'gene': genomic_features,
        'total_selections': selection_matrix.sum(axis=1),
        'selection_frequency': selection_matrix.sum(axis=1) / total_cv_folds,
        'selected_in_trials': [np.sum(selection_matrix[i, :].reshape(n_trials, n_folds).any(axis=1)) 
                              for i in range(len(genomic_features))]
    })
    
    # Sort by selection frequency
    selection_summary = selection_summary.sort_values('total_selections', ascending=False)
    
    # Filter genes with at least 1 selection
    selected_genes = selection_summary[selection_summary['total_selections'] >= 1].copy()
    
    print(f"\nCV Selection Results:")
    print(f"- Total genes tested: {len(genomic_features)}")
    print(f"- Genes selected ≥1 time: {len(selected_genes)}")
    print(f"- Most selected gene: {selected_genes.iloc[0]['gene'] if len(selected_genes) > 0 else 'None'} "
          f"({selected_genes.iloc[0]['total_selections'] if len(selected_genes) > 0 else 0}/{total_cv_folds} folds)")
    
    return selected_genes, detailed_results, selection_results

def test_gene_interaction_cv_fold(gene_data):
    """
    Test gene interaction for a single CV fold (parallel processing version)
    
    Parameters:
    -----------
    gene_data : tuple
        (gene_idx, gene_name, gene_series, clinical_data, alpha_levels)
    
    Returns:
    --------
    tuple or None : (gene_idx, gene_name, is_significant, interaction_result)
    """
    gene_idx, gene_name, gene_series, clinical_data, alpha_levels = gene_data
    
    try:
        # Prepare data for this gene
        model_data = clinical_data.copy()
        model_data[gene_name] = gene_series.reset_index(drop=True)
        
        # Create interaction term
        model_data[f'Adjuvant_Chemo_x_{gene_name}'] = (
            model_data['Adjuvant Chemo'] * model_data[gene_name]
        )
        
        # Fit Cox model
        cph = CoxPHFitter()
        cph.fit(model_data, duration_col='OS_MONTHS', event_col='OS_STATUS', show_progress=False)
        
        # Extract interaction results
        interaction_coef = cph.params_[f'Adjuvant_Chemo_x_{gene_name}']
        interaction_p_value = cph.summary.loc[f'Adjuvant_Chemo_x_{gene_name}', 'p']
        
        # Check significance at any alpha level
        is_significant = any(interaction_p_value <= alpha for alpha in alpha_levels)
        
        # Create detailed result
        interaction_result = {
            'gene': gene_name,
            'interaction_coef': interaction_coef,
            'interaction_p_value': interaction_p_value,
            'interaction_hr': np.exp(interaction_coef),
            'n_observations': len(model_data),
            'n_events': model_data['OS_STATUS'].sum(),
        }
        
        # Add significance flags
        for alpha in alpha_levels:
            interaction_result[f'significant_at_{alpha:.2f}'] = interaction_p_value <= alpha
        
        return gene_idx, gene_name, is_significant, interaction_result
        
    except Exception as e:
        print(f"Error testing gene {gene_name}: {str(e)}")
        return None

def test_gene_interaction_single_fold(train_fold, gene, alpha_levels):
    """
    Test gene interaction for a single CV fold (sequential processing version)
    """
    try:
        # Prepare data for this gene
        model_data = train_fold[['OS_STATUS', 'OS_MONTHS', 'Adjuvant Chemo', gene]].copy()
        
        # Create interaction term
        model_data[f'Adjuvant_Chemo_x_{gene}'] = (
            model_data['Adjuvant Chemo'] * model_data[gene]
        )
        
        # Fit Cox model
        cph = CoxPHFitter()
        cph.fit(model_data, duration_col='OS_MONTHS', event_col='OS_STATUS', show_progress=False)
        
        # Extract results
        interaction_coef = cph.params_[f'Adjuvant_Chemo_x_{gene}']
        interaction_p_value = cph.summary.loc[f'Adjuvant_Chemo_x_{gene}', 'p']
        
        # Create result
        result = {
            'gene': gene,
            'interaction_coef': interaction_coef,
            'interaction_p_value': interaction_p_value,
            'interaction_hr': np.exp(interaction_coef),
            'n_observations': len(model_data),
            'n_events': model_data['OS_STATUS'].sum(),
        }
        
        # Add significance flags
        for alpha in alpha_levels:
            result[f'significant_at_{alpha:.2f}'] = interaction_p_value <= alpha
        
        return result
        
    except Exception as e:
        print(f"Error testing gene {gene}: {str(e)}")
        return None

def run_interaction_screening(df, genomic_features, alpha_levels=[0.05, 0.10], use_parallel=True, n_jobs=-1):
    """Run interaction screening for all genomic features with parallelization"""
    print("\n" + "="*80)
    print("RUNNING TREATMENT-GENE INTERACTION SCREENING")
    print("="*80)
    
    print(f"Testing {len(genomic_features)} genomic features for treatment interactions")
    print(f"Alpha levels: {alpha_levels}")
    print(f"Model for each gene: Cox(Adjuvant Chemo + Gene + Adjuvant Chemo * Gene)")
    
    # Determine number of cores to use
    if n_jobs == -1:
        n_cores = multiprocessing.cpu_count()
    else:
        n_cores = min(n_jobs, multiprocessing.cpu_count())
    
    print(f"Parallelization: {'Enabled' if use_parallel else 'Disabled'}")
    if use_parallel:
        print(f"Using {n_cores} CPU cores")
    
    results = []
    start_time = time.time()
    
    if use_parallel:
        # Prepare data for parallel processing
        print("Preparing data for parallel processing...")
        clinical_data = df[['OS_STATUS', 'OS_MONTHS', 'Adjuvant Chemo']].copy()
        gene_data_list = []
        
        for gene in genomic_features:
            gene_series = df[gene]
            gene_data_list.append((gene, gene_series, clinical_data))
        
        print(f"Running parallel Cox regression screening with {n_cores} cores...")
        
        # Use joblib for parallel processing
        parallel_results = Parallel(n_jobs=n_cores, verbose=1)(
            delayed(test_interaction_for_gene_parallel)(gene_data, alpha_levels)
            for gene_data in gene_data_list
        )
        
        # Filter out None results
        results = [r for r in parallel_results if r is not None]
        
    else:
        # Sequential processing (original approach)
        print("Running sequential Cox regression screening...")
        for i, gene in enumerate(genomic_features):
            if (i + 1) % 500 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                remaining = (len(genomic_features) - i - 1) / rate
                print(f"Progress: {i+1}/{len(genomic_features)} ({(i+1)/len(genomic_features)*100:.1f}%) - "
                      f"Rate: {rate:.1f} genes/sec - ETA: {remaining/60:.1f} min")
            
            result = test_interaction_for_gene(df, gene, alpha_levels)
            if result is not None:
                results.append(result)
    
    total_time = time.time() - start_time
    print(f"\nScreening completed in {total_time/60:.1f} minutes")
    print(f"Successfully tested {len(results)}/{len(genomic_features)} genes")
    
    if len(results) > 0:
        # Calculate summary statistics
        p_values = [r['interaction_p_value'] for r in results]
        significant_005 = sum(1 for p in p_values if p <= 0.05)
        significant_010 = sum(1 for p in p_values if p <= 0.10)
        
        print(f"Significant interactions (p <= 0.05): {significant_005} ({significant_005/len(results)*100:.1f}%)")
        print(f"Significant interactions (p <= 0.10): {significant_010} ({significant_010/len(results)*100:.1f}%)")
        print(f"Processing rate: {len(results)/total_time:.1f} genes/second")
    
    return pd.DataFrame(results), total_time, n_cores

def analyze_cv_screening_results(selection_results, detailed_results, alpha_levels=[0.05, 0.10]):
    """Analyze and summarize CV-based screening results"""
    print("\n" + "="*80)
    print("ANALYZING CV SCREENING RESULTS")
    print("="*80)
    
    # Summary statistics
    print(f"Total genes tested: {len(selection_results)}")
    print(f"Genes selected ≥1 time: {len(selection_results[selection_results['total_selections'] >= 1])}")
    print(f"Genes selected ≥5 times: {len(selection_results[selection_results['total_selections'] >= 5])}")
    print(f"Genes selected ≥10 times: {len(selection_results[selection_results['total_selections'] >= 10])}")
    
    # Show top selected genes
    print(f"\nTop 20 most frequently selected genes:")
    top_genes = selection_results.head(20)[['gene', 'total_selections', 'selection_frequency', 'selected_in_trials']].copy()
    
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

def analyze_screening_results(results_df, alpha_levels=[0.05, 0.10]):
    """Analyze and summarize single-analysis screening results"""
    print("\n" + "="*80)
    print("ANALYZING SCREENING RESULTS")
    print("="*80)
    
    # Sort by interaction p-value
    results_df = results_df.sort_values('interaction_p_value')
    
    # Summary statistics
    print(f"Total genes tested: {len(results_df)}")
    print(f"Median sample size: {results_df['n_observations'].median():.0f}")
    print(f"Median events: {results_df['n_events'].median():.0f}")
    
    for alpha in alpha_levels:
        sig_col = f'significant_at_{alpha:.2f}'
        n_significant = results_df[sig_col].sum()
        pct_significant = (n_significant / len(results_df)) * 100
        print(f"Significant interactions at alpha = {alpha}: {n_significant} ({pct_significant:.1f}%)")
    
    # Show top results
    print(f"\nTop 20 most significant interactions:")
    top_results = results_df.head(20)[['gene', 'interaction_p_value', 'interaction_hr', 
                                      'interaction_hr_ci_lower', 'interaction_hr_ci_upper',
                                      'significant_at_0.05', 'significant_at_0.10']].copy()
    
    # Format for display
    top_results['interaction_hr_formatted'] = top_results.apply(
        lambda row: f"{row['interaction_hr']:.3f} ({row['interaction_hr_ci_lower']:.3f}-{row['interaction_hr_ci_upper']:.3f})",
        axis=1
    )
    
    display_cols = ['gene', 'interaction_p_value', 'interaction_hr_formatted', 
                   'significant_at_0.05', 'significant_at_0.10']
    print(top_results[display_cols].to_string(index=False))
    
    return results_df

def create_visualizations(results_df, output_dir, alpha_levels=[0.05, 0.10]):
    """Create visualizations of screening results"""
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    
    # 1. P-value distribution
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.hist(results_df['interaction_p_value'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(x=0.05, color='red', linestyle='--', label='α = 0.05')
    plt.axvline(x=0.10, color='orange', linestyle='--', label='α = 0.10')
    plt.xlabel('Interaction P-value')
    plt.ylabel('Frequency')
    plt.title('Distribution of Treatment-Gene Interaction P-values')
    plt.legend()
    
    # 2. QQ plot for p-values
    plt.subplot(2, 2, 2)
    p_values = results_df['interaction_p_value'].values
    n = len(p_values)
    expected = np.linspace(1/n, 1, n)
    observed = np.sort(p_values)
    
    plt.scatter(expected, observed, alpha=0.6)
    plt.plot([0, 1], [0, 1], 'r--', label='Expected (uniform)')
    plt.xlabel('Expected P-value')
    plt.ylabel('Observed P-value')
    plt.title('QQ Plot: P-value Distribution')
    plt.legend()
    
    # 3. Effect size distribution
    plt.subplot(2, 2, 3)
    plt.hist(np.log(results_df['interaction_hr']), bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
    plt.axvline(x=0, color='red', linestyle='-', label='HR = 1 (no effect)')
    plt.xlabel('Log(Interaction Hazard Ratio)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Interaction Effect Sizes')
    plt.legend()
    
    # 4. Significance counts
    plt.subplot(2, 2, 4)
    alpha_counts = []
    alpha_labels = []
    for alpha in alpha_levels:
        sig_col = f'significant_at_{alpha:.2f}'
        count = results_df[sig_col].sum()
        alpha_counts.append(count)
        alpha_labels.append(f'α = {alpha}')
    
    plt.bar(alpha_labels, alpha_counts, color=['red', 'orange'])
    plt.ylabel('Number of Significant Interactions')
    plt.title('Significant Interactions by Alpha Level')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{current_date}_interaction_screening_overview.png"), dpi=300)
    plt.close()
    
    # 5. Volcano plot
    plt.figure(figsize=(10, 8))
    
    # Calculate -log10(p-value) for y-axis
    neg_log_p = -np.log10(results_df['interaction_p_value'])
    log_hr = np.log(results_df['interaction_hr'])
    
    # Color points by significance
    colors = []
    for _, row in results_df.iterrows():
        if row['significant_at_0.05']:
            colors.append('red')
        elif row['significant_at_0.10']:
            colors.append('orange')
        else:
            colors.append('gray')
    
    plt.scatter(log_hr, neg_log_p, c=colors, alpha=0.6, s=20)
    
    # Add significance lines
    plt.axhline(y=-np.log10(0.05), color='red', linestyle='--', label='α = 0.05')
    plt.axhline(y=-np.log10(0.10), color='orange', linestyle='--', label='α = 0.10')
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    
    plt.xlabel('Log(Interaction Hazard Ratio)')
    plt.ylabel('-Log10(P-value)')
    plt.title('Volcano Plot: Treatment-Gene Interactions')
    plt.legend()
    
    # Add text for counts
    n_sig_05 = results_df['significant_at_0.05'].sum()
    n_sig_10 = results_df['significant_at_0.10'].sum()
    plt.text(0.05, 0.95, f'Significant at α=0.05: {n_sig_05}', transform=plt.gca().transAxes)
    plt.text(0.05, 0.90, f'Significant at α=0.10: {n_sig_10}', transform=plt.gca().transAxes)
    
    plt.savefig(os.path.join(output_dir, f"{current_date}_interaction_volcano_plot.png"), dpi=300)
    plt.close()
    
    print("Visualizations saved:")
    print(f"- {current_date}_interaction_screening_overview.png")
    print(f"- {current_date}_interaction_volcano_plot.png")

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
    
    # Save filtered results by selection frequency
    for min_selections in [1, 5, 10, 25, 50]:
        filtered_genes = selection_results[selection_results['total_selections'] >= min_selections]
        
        if len(filtered_genes) > 0:
            filtered_path = os.path.join(output_dir, f"{current_date}_selected_genes_min_{min_selections}_selections.csv")
            filtered_genes.to_csv(filtered_path, index=False)
            print(f"Genes selected ≥{min_selections} times saved to: {filtered_path}")
            
            # Create gene list for downstream analysis
            gene_list_path = os.path.join(output_dir, f"{current_date}_selected_genes_min_{min_selections}_selections_list.txt")
            with open(gene_list_path, 'w') as f:
                for gene in filtered_genes['gene']:
                    f.write(f"{gene}\n")
            print(f"Gene list (≥{min_selections} selections) saved to: {gene_list_path}")

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

def save_results(results_df, output_dir, alpha_levels=[0.05, 0.10]):
    """Save screening results to files"""
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    # Save full results
    full_results_path = os.path.join(output_dir, f"{current_date}_cox_interaction_screening_full_results.csv")
    results_df.to_csv(full_results_path, index=False)
    print(f"Full results saved to: {full_results_path}")
    
    # Save significant results for each alpha level
    for alpha in alpha_levels:
        sig_col = f'significant_at_{alpha:.2f}'
        sig_results = results_df[results_df[sig_col] == True].copy()
        
        if len(sig_results) > 0:
            sig_path = os.path.join(output_dir, f"{current_date}_significant_interactions_alpha_{alpha:.2f}.csv")
            sig_results.to_csv(sig_path, index=False)
            print(f"Significant results (α={alpha}) saved to: {sig_path}")
            
            # Create feature list for downstream analysis
            feature_list_path = os.path.join(output_dir, f"{current_date}_significant_genes_alpha_{alpha:.2f}_list.txt")
            with open(feature_list_path, 'w') as f:
                for gene in sig_results['gene']:
                    f.write(f"{gene}\n")
            print(f"Significant gene list (α={alpha}) saved to: {feature_list_path}")
        else:
            print(f"No significant results found at α={alpha}")

def create_summary_report(results_df, output_dir, alpha_levels=[0.05, 0.10]):
    """Create a summary report"""
    print("\n" + "="*80)
    print("CREATING SUMMARY REPORT")
    print("="*80)
    
    report_path = os.path.join(output_dir, f"{current_date}_cox_interaction_screening_summary.txt")
    
    with open(report_path, 'w') as f:
        f.write("COX PROPORTIONAL HAZARDS INTERACTION SCREENING SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Screening method: Treatment-gene interaction testing\n")
        f.write(f"Model: Cox(Adjuvant Chemo + Gene + Adjuvant Chemo * Gene)\n\n")
        
        f.write("DATASET SUMMARY:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total genes tested: {len(results_df)}\n")
        f.write(f"Median sample size: {results_df['n_observations'].median():.0f}\n")
        f.write(f"Median events: {results_df['n_events'].median():.0f}\n\n")
        
        f.write("SCREENING RESULTS:\n")
        f.write("-" * 20 + "\n")
        for alpha in alpha_levels:
            sig_col = f'significant_at_{alpha:.2f}'
            n_significant = results_df[sig_col].sum()
            pct_significant = (n_significant / len(results_df)) * 100
            f.write(f"Significant interactions at α = {alpha}: {n_significant} ({pct_significant:.1f}%)\n")
        
        f.write(f"\nTOP 10 MOST SIGNIFICANT INTERACTIONS:\n")
        f.write("-" * 40 + "\n")
        top_10 = results_df.head(10)
        for _, row in top_10.iterrows():
            f.write(f"{row['gene']}: p = {row['interaction_p_value']:.2e}, "
                   f"HR = {row['interaction_hr']:.3f} "
                   f"({row['interaction_hr_ci_lower']:.3f}-{row['interaction_hr_ci_upper']:.3f})\n")
        
        f.write(f"\nFILES GENERATED:\n")
        f.write("-" * 20 + "\n")
        f.write(f"- {current_date}_cox_interaction_screening_full_results.csv\n")
        for alpha in alpha_levels:
            f.write(f"- {current_date}_significant_interactions_alpha_{alpha:.2f}.csv\n")
            f.write(f"- {current_date}_significant_genes_alpha_{alpha:.2f}_list.txt\n")
        f.write(f"- {current_date}_interaction_screening_overview.png\n")
        f.write(f"- {current_date}_interaction_volcano_plot.png\n")
    
    print(f"Summary report saved to: {report_path}")

def report_performance_metrics(total_time, n_genes, n_cores_used, use_parallel):
    """Report detailed performance metrics"""
    print("\n" + "="*80)
    print("PERFORMANCE METRICS")
    print("="*80)
    
    processing_rate = n_genes / total_time
    
    print(f"Total processing time: {total_time/60:.1f} minutes ({total_time:.1f} seconds)")
    print(f"Genes processed: {n_genes:,}")
    print(f"Processing rate: {processing_rate:.1f} genes/second")
    
    if use_parallel:
        print(f"CPU cores used: {n_cores_used}")
        print(f"Parallel efficiency: {processing_rate/n_cores_used:.1f} genes/second/core")
        
        # Estimate sequential time for comparison
        estimated_sequential_time = n_genes / 1.5  # Assume ~1.5 genes/sec sequential
        estimated_speedup = estimated_sequential_time / total_time
        print(f"Estimated speedup vs sequential: {estimated_speedup:.1f}x")
        print(f"Estimated sequential time: {estimated_sequential_time/60:.1f} minutes")
    else:
        print("Processing mode: Sequential")
        print("Consider using --parallel for faster processing on multi-core systems")
    
    print("="*80)

def main(use_parallel=True, n_jobs=-1, alpha_levels=[0.05, 0.10]):
    """Main function to run Cox interaction screening"""
    print("="*80)
    print("COX PROPORTIONAL HAZARDS INTERACTION SCREENING")
    print("="*80)
    print("Prescreening genomic features using treatment-gene interactions")
    print("Model: Cox(Adjuvant Chemo + Gene + Adjuvant Chemo * Gene)")
    print("Selection criteria: Significant interaction term (p < α)")
    print("="*80)
    
    # Load and combine data
    full_data = load_and_combine_data()
    
    # Identify feature types
    genomic_features, clinical_features = identify_feature_types(full_data)
    
def main(use_parallel=True, n_jobs=-1, alpha_levels=[0.05, 0.10], use_cv=True, n_trials=20, n_folds=10):
    """Main function to run Cox interaction screening"""
    print("="*80)
    print("COX PROPORTIONAL HAZARDS INTERACTION SCREENING")
    print("="*80)
    print("Prescreening genomic features using treatment-gene interactions")
    print("Model: Cox(Adjuvant Chemo + Gene + Adjuvant Chemo * Gene)")
    
    if use_cv:
        print(f"Method: {n_trials} trials × {n_folds}-fold CV (like RSF approach)")
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
    
    if use_cv:
        # Use CV-based screening (similar to RSF and R implementation)
        selection_results, detailed_results, _ = run_interaction_screening_cv(
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
        
    else:
        # Use single-analysis screening (original approach)
        results_df, screening_time, n_cores_used = run_interaction_screening(
            full_data, 
            genomic_features, 
            alpha_levels,
            use_parallel=use_parallel,
            n_jobs=n_jobs
        )
        
        # Report performance metrics
        report_performance_metrics(screening_time, len(genomic_features), n_cores_used, use_parallel)
        
        # Analyze results
        results_df = analyze_screening_results(results_df, alpha_levels)
        
        # Create visualizations
        create_visualizations(results_df, output_dir, alpha_levels)
        
        # Save results
        save_results(results_df, output_dir, alpha_levels)
        
        # Create summary report
        create_summary_report(results_df, output_dir, alpha_levels)
    
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
    parser = argparse.ArgumentParser(
        description="Cox Proportional Hazards Interaction Screening",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with CV (default) - 20 trials × 10 folds
  python cox-prescreen.py
  
  # Run with custom CV configuration
  python cox-prescreen.py --n-trials 10 --n-folds 5
  
  # Run single analysis (no CV)
  python cox-prescreen.py --no-cv
  
  # Run with 4 cores
  python cox-prescreen.py --n-jobs 4
  
  # Run sequentially (no parallelization)
  python cox-prescreen.py --no-parallel
        """
    )
    
    parser.add_argument(
        '--n-jobs', 
        type=int, 
        default=-1,
        help='Number of CPU cores to use (-1 for all cores, default: -1)'
    )
    
    parser.add_argument(
        '--no-parallel',
        action='store_true',
        help='Disable parallelization (run sequentially)'
    )
    
    parser.add_argument(
        '--alpha',
        type=float,
        nargs='+',
        default=[0.05, 0.10],
        help='Alpha levels for significance testing (default: 0.05 0.10)'
    )
    
    parser.add_argument(
        '--no-cv',
        action='store_true',
        help='Disable cross-validation (run single analysis on combined dataset)'
    )
    
    parser.add_argument(
        '--n-trials',
        type=int,
        default=20,
        help='Number of CV trials (default: 20)'
    )
    
    parser.add_argument(
        '--n-folds',
        type=int,
        default=10,
        help='Number of CV folds per trial (default: 10)'
    )
    
    args = parser.parse_args()
    
    # Pass arguments to main function
    main(
        use_parallel=not args.no_parallel,
        n_jobs=args.n_jobs,
        alpha_levels=args.alpha,
        use_cv=not args.no_cv,
        n_trials=args.n_trials,
        n_folds=args.n_folds
    )
