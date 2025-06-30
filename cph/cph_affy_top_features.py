import pandas as pd
from lifelines import CoxPHFitter
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.util import Surv
from sklearn.model_selection import GridSearchCV
from sksurv.metrics import concordance_index_censored
from sklearn.metrics import make_scorer
import os

def dataframe_to_latex(df, caption="Table Caption", label="table:label"):
    """
    Convert a pandas DataFrame to a LaTeX tabular environment and print it.
    
    Parameters:
    - df: pandas DataFrame
    - caption: The caption of the table (optional)
    - label: The label for referencing the table in LaTeX (optional)
    """
    # Start the LaTeX table environment
    latex_str = "\\begin{table}[H]\n"
    latex_str += "\\centering\n"
    latex_str += f"\\caption{{{caption}}}\n"
    latex_str += f"\\label{{{label}}}\n"
    # afterwards do l c c for left on covariates
    latex_str += "\\begin{tabular}{" + " | ".join(['c' for _ in df.columns]) + "}\n"
    latex_str += "\\hline\n"

    # Add column headers
    latex_str += " & ".join(df.columns) + " \\\\\n"
    latex_str += "\\hline\n"

    # Make row values title case but keep driver names in all caps, like keep EGFR Driver
    df['Covariate'] = df['Covariate'].str.title()
    
    for i in range(len(df)):
        if 'Driver' in df['Covariate'][i]:
            driverName = df['Covariate'][i].split(' ')[0].upper()
            df['Covariate'][i] = driverName + ' Driver'
        
    # Add table rows
    for _, row in df.iterrows():
        # Bold p-values less than 0.05, \textbf{ $\leq$ 0.001 } for p-values less than 0.001
        if row['p'] < 0.001:
            row['p'] = f"\\textbf{{ $\leq$ 0.001 }}"
        elif row['p'] < 0.05:
            row['p'] = f"\\textbf{{{round(row['p'], 3)}}}"
        else: 
            row['p'] = round(row['p'], 3)
        latex_str += " & ".join(map(str, row.values)) + " \\\\\n"
        # latex_str += "\\hline\n"

    # End the LaTeX table environment
    latex_str += "\\end{tabular}\n"
    latex_str += "\\end{table}\n"

    # Print the LaTeX table
    print(latex_str)

# Modified run_model: removed splitting; now expects separate train_df and test_df.
def run_model(train_df, test_df, name, penalizer=0.1, l1_ratio=1.0, duration_col='OS_MONTHS', event_col='OS_STATUS'):
    # Fit the Cox proportional hazards model on the provided training set using elastic net
    cph = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)
    cph.fit(train_df, duration_col=duration_col, event_col=event_col)

    # Print summary of the fitted model
    cph.print_summary(style="ascii")

    # Evaluate the model using the training set and the provided test set
    train_c_index = cph.concordance_index_
    print(f"Concordance Index on Training Set: {train_c_index:.3f}")
    test_c_index = cph.score(test_df, scoring_method="concordance_index")
    print(f"Concordance Index on Test Set: {test_c_index:.3f}")

    summary_df = cph.summary  # Get the summary as a DataFrame
    model_metrics = [cph.log_likelihood_, cph.concordance_index_]

    # Add a column for significance level
    def significance_stars(p):
        if p < 0.001:
            return "***"
        elif p < 0.01:
            return "**"
        elif p < 0.05:
            return "*"
        else:
            return ""

    summary_df['significance'] = summary_df['p'].apply(significance_stars)
    
    # Sort summary_df by the coefficient values (ascending order)
    summary_df = summary_df.sort_values(by='coef', ascending=False)

    print(summary_df)

    concordance = cph.concordance_index_
    partial_aic = cph.AIC_partial_
    log_likelihood_ratio_test = cph.log_likelihood_ratio_test().test_statistic
    ll_ratio_test_df = cph.log_likelihood_ratio_test().degrees_freedom
    neg_log2_p_ll_ratio_test = -np.log2(cph.log_likelihood_ratio_test().p_value)

    with open(f'cph-{test_c_index:.3f}-{name}-summary.txt', 'w') as f:
        f.write(summary_df.to_string())
        formatted_metrics = '\n'.join([f'{metric:.4f}' for metric in model_metrics])
        f.write(f"\n\nModel metrics: {formatted_metrics}")
        
        # Write additional metrics in the specified format
        f.write(f"\n\nTrain Concordance = {concordance:.3f}")
        # Test c-index
        f.write(f"\nConcordance on test set = {test_c_index:.3f}")
        f.write(f"\nPartial AIC = {partial_aic:.3f}")
        f.write(f"\nlog-likelihood ratio test = {log_likelihood_ratio_test:.3f} on {ll_ratio_test_df} df")
        f.write(f"\n-log2(p) of ll-ratio test = {neg_log2_p_ll_ratio_test:.3f}")
        
    # Extract the hazard ratios, confidence intervals, and p-values
    hazard_ratios = np.exp(cph.params_)  # Exponentiate the coefficients to get hazard ratios
    confidence_intervals = np.exp(cph.confidence_intervals_)  # Exponentiate confidence intervals
    p_values = cph.summary['p']  # Extract p-values from the summary
    
    # Round hazard ratio and confidence interval values to 2 decimal places
    summary_df['exp(coef)'] = summary_df['exp(coef)'].round(2)
    summary_df['exp(coef) lower 95%'] = summary_df['exp(coef) lower 95%'].round(2)
    summary_df['exp(coef) upper 95%'] = summary_df['exp(coef) upper 95%'].round(2)
    
    print(summary_df.info())
    
    # Keep only exp(coef) - hazard ratio, exp(coef) lower 95%, exp(coef) upper 95%, p-value
    latex_df = summary_df[['exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'p']]
    
    # Include covariate names in the table
    latex_df['Covariate'] = latex_df.index
    
    # Remove underscores in the covariate names
    latex_df['Covariate'] = latex_df['Covariate'].str.replace('_', ' ')
    
    # Combine lower and upper confidence interval bounds into hazard ratios 
    latex_df["Hazard Ratio"] = latex_df["exp(coef)"].astype(str) + " (" + latex_df["exp(coef) lower 95%"].astype(str) + "-" + latex_df["exp(coef) upper 95%"].astype(str) + ")"
    
    # Drop the columns that are no longer needed
    latex_df = latex_df.drop(columns=['exp(coef)', 'exp(coef) lower 95%'])
    
    # Arrange the columns in the correct order
    latex_df = latex_df[['Covariate', 'Hazard Ratio', 'p']]
    
    # dataframe_to_latex(latex_df, caption=f'{name} Model Summary', label='table:model-summary')

    # Sort the parameters by hazard ratio in descending order
    sorted_indices = np.argsort(hazard_ratios)[::-1]
    sorted_hazard_ratios = hazard_ratios[sorted_indices]
    sorted_confidence_intervals = confidence_intervals.iloc[sorted_indices]
    sorted_p_values = p_values.iloc[sorted_indices]
    sorted_params = cph.params_.index[sorted_indices]

    # Plotting the forest plot
    plt.figure(figsize=(12, 8))

    # Include lambda value in the title 
    plt.title(f'{name} Hazard Ratios (Test C-index: {test_c_index:.3f}, 95% CI)')

    # Generate a forest plot for hazard ratios with 95% confidence intervals
    plt.errorbar(sorted_hazard_ratios, range(len(sorted_hazard_ratios)), 
                xerr=[sorted_hazard_ratios - sorted_confidence_intervals.iloc[:, 0], sorted_confidence_intervals.iloc[:, 1] - sorted_hazard_ratios],
                fmt='o', color='black', ecolor='grey', capsize=5)

    # Add labels for the covariates
    plt.yticks(range(len(sorted_hazard_ratios)), sorted_params)
    plt.axvline(x=1, linestyle='--', color='red')  # Reference line at HR=1
    plt.xlabel('Hazard Ratio (log scale)')
    plt.xscale('log')  # Logarithmic scale for the hazard ratios

    x_pos = max(sorted_hazard_ratios) * 1.4  # Position the p-values far to the right
    for i, (hr, p) in enumerate(zip(sorted_hazard_ratios, sorted_p_values)):
        if p < 0.05:
            plt.text(x_pos, i, f'$\\mathbf{{p={p:.3f}}}$', va='center', ha='left', fontsize=8)
        else:
            plt.text(x_pos, i, f'p={p:.3f}', va='center', ha='left', fontsize=8)

    # Adjust x-axis limits to ensure p-values are visible
    plt.xlim(left=min(sorted_hazard_ratios) / 2, right=x_pos * 1.15)

    # Display the plot
    plt.tight_layout()
    name = name.replace(' ', '-')
    plt.savefig(f'cph-{test_c_index:.3f}-{name}-forest-plot.png')
    plt.show()

def c_index_score(estimator, X, y):
    # Predict risk scores (note: lower predicted values imply higher risk)
    risk_scores = -estimator.predict(X)
    return concordance_index_censored(y["OS_STATUS"], y["OS_MONTHS"], risk_scores)[0]

def optimize_penalties(train_df, test_df, name, duration_col='OS_MONTHS', event_col='OS_STATUS'):
    # Prepare training data: separate features and convert survival labels
    X_train = train_df.drop(columns=[duration_col, event_col])
    y_train = Surv.from_dataframe(event_col, duration_col, train_df)
    
    # Define parameter grid for l1_ratio; the alpha path is determined internally.
    param_grid = {
        "l1_ratio": [0.25, 0.5, 0.75, 1.0]
    }
    
    # Initialize Coxnet model; fit_baseline_model allows baseline survival estimation.
    coxnet = CoxnetSurvivalAnalysis(fit_baseline_model=True, alphas=np.logspace(-2, 1, 10))
    
    # Use a custom scoring function based on concordance index.
    scorer = c_index_score
    
    # Set up GridSearchCV with parallelization (using all available cores)
    grid_search = GridSearchCV(coxnet, param_grid, scoring=scorer, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print(f"Optimal parameters: {grid_search.best_params_} with Best CV Concordance = {grid_search.best_score_:.3f}")
    
    # Evaluate on the test set
    X_test = test_df.drop(columns=[duration_col, event_col])
    y_test = Surv.from_dataframe(event_col, duration_col, test_df)
    test_c_index = c_index_score(grid_search.best_estimator_, X_test, y_test)
    print(f"Concordance Index on Test Set: {test_c_index:.3f}")

def run_lasso_cox(train_df, test_df, name, alpha=0.1, duration_col='OS_MONTHS', event_col='OS_STATUS'):
    """
    Run Lasso Cox regression using scikit-survival's CoxnetSurvivalAnalysis.
    This provides pure L1 regularization (Lasso) for feature selection.
    """
    # Prepare training data
    X_train = train_df.drop(columns=[duration_col, event_col])
    y_train = Surv.from_dataframe(event_col, duration_col, train_df)
    
    # Prepare test data
    X_test = test_df.drop(columns=[duration_col, event_col])
    y_test = Surv.from_dataframe(event_col, duration_col, test_df)
    
    # Initialize Lasso Cox model (l1_ratio=1.0 for pure Lasso)
    lasso_cox = CoxnetSurvivalAnalysis(
        l1_ratio=1.0,  # Pure L1 regularization (Lasso)
        alpha_min_ratio=0.01,
        fit_baseline_model=True,
        alphas=[alpha]  # Use specific alpha value
    )
    
    # Fit the model
    print(f"Fitting Lasso Cox model with alpha={alpha}...")
    lasso_cox.fit(X_train, y_train)
    
    # Evaluate on training and test sets
    train_c_index = c_index_score(lasso_cox, X_train, y_train)
    test_c_index = c_index_score(lasso_cox, X_test, y_test)
    
    print(f"Lasso Cox Training C-index: {train_c_index:.3f}")
    print(f"Lasso Cox Test C-index: {test_c_index:.3f}")
    
    # Get coefficients (note: CoxnetSurvivalAnalysis returns coefficients directly)
    coef = lasso_cox.coef_
    feature_names = X_train.columns
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coef,
        'Hazard_Ratio': np.exp(coef),
        'Selected': np.abs(coef) > 1e-8  # Features with non-zero coefficients
    })
    
    # Sort by absolute coefficient value
    results_df = results_df.reindex(results_df['Coefficient'].abs().sort_values(ascending=False).index)
    
    print(f"\nLasso Cox Results ({name}):")
    print(f"Features selected: {results_df['Selected'].sum()} out of {len(feature_names)}")
    print("\nTop features (non-zero coefficients):")
    selected_features = results_df[results_df['Selected']].copy()
    for _, row in selected_features.iterrows():
        print(f"  {row['Feature']}: coef={row['Coefficient']:.4f}, HR={row['Hazard_Ratio']:.3f}")
    
    # Save results
    with open(f'lasso-cox-{test_c_index:.3f}-{name}-summary.txt', 'w') as f:
        f.write(f"Lasso Cox Regression Results - {name}\n")
        f.write("="*50 + "\n\n")
        f.write(f"Alpha (regularization): {alpha}\n")
        f.write(f"Training C-index: {train_c_index:.3f}\n")
        f.write(f"Test C-index: {test_c_index:.3f}\n")
        f.write(f"Features selected: {results_df['Selected'].sum()}/{len(feature_names)}\n\n")
        f.write("All Features:\n")
        f.write(results_df.to_string(index=False))
        
    # Create forest plot for selected features only
    if selected_features.shape[0] > 0:
        plt.figure(figsize=(12, 8))
        
        # Plot hazard ratios for selected features
        y_pos = np.arange(len(selected_features))
        hazard_ratios = selected_features['Hazard_Ratio'].values
        
        plt.barh(y_pos, hazard_ratios, alpha=0.7, 
                color=['red' if hr > 1 else 'blue' for hr in hazard_ratios])
        
        plt.yticks(y_pos, selected_features['Feature'])
        plt.xlabel('Hazard Ratio')
        plt.title(f'{name} - Lasso Cox Selected Features (Test C-index: {test_c_index:.3f})')
        plt.axvline(x=1, linestyle='--', color='black', alpha=0.5)
        
        # Add coefficient values as text
        for i, (hr, coef) in enumerate(zip(hazard_ratios, selected_features['Coefficient'])):
            plt.text(hr + 0.05, i, f'coef={coef:.3f}', va='center', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(f'lasso-cox-{test_c_index:.3f}-{name}-selected-features.png', dpi=300)
        plt.show()
    
    return results_df, train_c_index, test_c_index

if __name__ == "__main__":
    # Data preprocessing matching iterative-feature-selection-affy.py
    print("="*80)
    print("COX PROPORTIONAL HAZARDS MODEL - AFFYMETRIX TOP FEATURES")
    print("="*80)
    
    # Load data - same as iterative feature selection
    print("Loading training data...")
    train_orig = pd.read_csv("affyTrain.csv")
    valid_orig = pd.read_csv("affyValidation.csv")
    
    # Combine train and validation (same preprocessing as iterative feature selection)
    train_combined = pd.concat([train_orig, valid_orig], axis=0, ignore_index=True)
    
    print(f"Combined training data shape: {train_combined.shape}")
    print(f"Number of events in combined training: {train_combined['OS_STATUS'].sum()}")
    print(f"Number of censored in combined training: {train_combined.shape[0] - train_combined['OS_STATUS'].sum()}")
    
    # Load test data separately
    print("Loading test data...")
    test_data = pd.read_csv("affyTest.csv")
    print(f"Test data shape: {test_data.shape}")
    print(f"Number of events in test: {test_data['OS_STATUS'].sum()}")
    print(f"Number of censored in test: {test_data.shape[0] - test_data['OS_STATUS'].sum()}")
    
    # Preprocess data (same as iterative feature selection)
    train_combined['Adjuvant Chemo'] = train_combined['Adjuvant Chemo'].replace({'OBS': 0, 'ACT': 1})
    test_data['Adjuvant Chemo'] = test_data['Adjuvant Chemo'].replace({'OBS': 0, 'ACT': 1})
    
    # Define binary columns (same as iterative feature selection)
    binary_columns = ['Adjuvant Chemo', 'IS_MALE']
    train_combined[binary_columns] = train_combined[binary_columns].astype(int)
    test_data[binary_columns] = test_data[binary_columns].astype(int)
    
    # Load top features from best iteration (19 features from iteration 31)
    print("\nLoading top features from best iteration...")
    try:
        # Try to load from the RSF results directory
        best_features_path = "rsf/rsf_results_affy/iterative_feature_selection/20250628_best_features_iteration_31.csv"
        if not os.path.exists(best_features_path):
            # Fallback to current directory
            best_features_path = "20250628_best_features_iteration_31.csv"
        
        best_features_df = pd.read_csv(best_features_path)
        selected_features = best_features_df['Feature'].tolist()
        print(f"Using {len(selected_features)} top features from iteration 31")
        print("Top features:", selected_features)
        
    except FileNotFoundError:
        print("Best features file not found. Using predefined top 19 features from iteration 31...")
        # Hardcoded features from the CSV you provided
        selected_features = [
            'Stage_IA', 'FAM117A', 'CCNB1', 'PURA', 'PFKP', 'PARM1', 
            'ADGRF5', 'GUCY1A1', 'SLC1A4', 'TENT5C', 'Age', 'HILPDA', 
            'ETV5', 'STIM1', 'KDM5C', 'NCAPG2', 'ZFR2', 'SETBP1', 'RTCA'
        ]
        print(f"Using {len(selected_features)} hardcoded top features")
    
    # Filter data to selected features + survival columns
    survival_columns = ['OS_MONTHS', 'OS_STATUS']
    required_columns = selected_features + survival_columns
    
    # Check which features are available in both datasets
    train_available_features = [f for f in selected_features if f in train_combined.columns]
    test_available_features = [f for f in selected_features if f in test_data.columns]
    
    missing_train = [f for f in selected_features if f not in train_combined.columns]
    missing_test = [f for f in selected_features if f not in test_data.columns]
    
    if missing_train:
        print(f"Warning: {len(missing_train)} features missing in training data: {missing_train}")
    if missing_test:
        print(f"Warning: {len(missing_test)} features missing in test data: {missing_test}")
    
    # Use only features available in both datasets
    common_features = list(set(train_available_features) & set(test_available_features))
    print(f"Using {len(common_features)} features available in both datasets")
    
    # Prepare final datasets
    train_final = train_combined[common_features + survival_columns].copy()
    test_final = test_data[common_features + survival_columns].copy()
    
    print(f"\nFinal training data shape: {train_final.shape}")
    print(f"Final test data shape: {test_final.shape}")
    print(f"Features used: {common_features}")
    
    # Run Cox Proportional Hazards models
    print("\n" + "="*60)
    print("RUNNING COX PROPORTIONAL HAZARDS MODELS")
    print("="*60)
    
    # Model 1: No regularization (standard CPH)
    print("\n1. Standard CPH (no regularization):")
    run_model(train_final, test_final, 'Affy-Top-19-Standard', penalizer=0.0, l1_ratio=1.0)
    
    # Model 2: Ridge regression (L2 regularization)
    print("\n2. CPH with Ridge regularization (L2):")
    run_model(train_final, test_final, 'Affy-Top-19-Ridge', penalizer=0.1, l1_ratio=0.0)
    
    # Model 3: Lasso regression (L1 regularization)
    print("\n3. CPH with Lasso regularization (L1):")
    run_model(train_final, test_final, 'Affy-Top-19-Lasso', penalizer=0.1, l1_ratio=1.0)
    
    # Model 4: Elastic Net (L1 + L2 regularization)
    print("\n4. CPH with Elastic Net regularization:")
    run_model(train_final, test_final, 'Affy-Top-19-ElasticNet', penalizer=0.1, l1_ratio=0.5)
    
    # Model 5: Optimize penalties using cross-validation
    print("\n5. CPH with optimized penalties:")
    optimize_penalties(train_final, test_final, 'Affy-Top-19-Optimized')
    
    # Model 6: Pure Lasso Cox (scikit-survival implementation)
    print("\n6. Pure Lasso Cox (feature selection):")
    lasso_results, lasso_train_c, lasso_test_c = run_lasso_cox(train_final, test_final, 'Affy-Top-19-PureLasso', alpha=0.1)
    
    # Model 7: Lasso Cox with different alpha values
    print("\n7. Lasso Cox with multiple alpha values:")
    alpha_values = [0.01, 0.05, 0.1, 0.2, 0.5]
    lasso_results_summary = []
    
    for alpha in alpha_values:
        print(f"\n   Testing alpha = {alpha}:")
        results_df, train_c, test_c = run_lasso_cox(train_final, test_final, f'Affy-Lasso-alpha-{alpha}', alpha=alpha)
        n_selected = results_df['Selected'].sum()
        lasso_results_summary.append({
            'alpha': alpha,
            'train_c_index': train_c,
            'test_c_index': test_c,
            'n_features_selected': n_selected
        })
        print(f"   Alpha {alpha}: Test C-index = {test_c:.3f}, Features selected = {n_selected}")
    
    # Summary of Lasso results
    lasso_summary_df = pd.DataFrame(lasso_results_summary)
    print(f"\nLasso Alpha Comparison Summary:")
    print(lasso_summary_df.to_string(index=False))
    
    # Save Lasso summary
    lasso_summary_df.to_csv('lasso-alpha-comparison-summary.csv', index=False)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("Results saved:")
    print("- Summary files: cph-*-summary.txt")
    print("- Forest plots: cph-*-forest-plot.png")
    print("- Lasso results: lasso-cox-*-summary.txt")
    print("- Lasso plots: lasso-cox-*-selected-features.png")
    print("- Lasso comparison: lasso-alpha-comparison-summary.csv")
