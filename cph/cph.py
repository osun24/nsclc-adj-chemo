import pandas as pd
from lifelines import CoxPHFitter
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

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

def run_model(df, name, penalizer=0.1, l1_ratio=1.0, duration_col='OS_MONTHS', event_col='OS_STATUS'):
    # Train-test split
    # Split the data into training and testing sets
    test_size = 0.2
    # Split the data into training and testing sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Fit the Cox proportional hazards model on the training set using elastic net
    cph = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)
    cph.fit(train_df, duration_col=duration_col, event_col=event_col)

    # Print summary of the fitted model
    cph.print_summary(style="ascii")

    # Optional: Evaluate the model using the test set
    # For example, you can use the model's concordance index on the test set
    c_index = cph.concordance_index_
    print(f"Concordance Index on Training Set: {c_index:.3f}")

    # Evaluate the model on the test set
    c_index_test = cph.score(test_df, scoring_method="concordance_index")
    print(f"Concordance Index on Test Set: {c_index_test:.3f}")

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

    with open(f'cph-{c_index_test:.3f}-{name}-summary.txt', 'w') as f:
        f.write(summary_df.to_string())
        formatted_metrics = '\n'.join([f'{metric:.4f}' for metric in model_metrics])
        f.write(f"\n\nModel metrics: {formatted_metrics}")
        
        # Write additional metrics in the specified format
        f.write(f"\n\nTrain Concordance = {concordance:.3f}")
        # Test c-index
        f.write(f"\nConcordance on test set = {c_index_test:.3f}")
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
    
    #dataframe_to_latex(latex_df, caption=f'{name} Model Summary', label='table:model-summary')

    # Sort the parameters by hazard ratio in descending order
    sorted_indices = np.argsort(hazard_ratios)[::-1]
    sorted_hazard_ratios = hazard_ratios[sorted_indices]
    sorted_confidence_intervals = confidence_intervals.iloc[sorted_indices]
    sorted_p_values = p_values.iloc[sorted_indices]
    sorted_params = cph.params_.index[sorted_indices]

    # Plotting the forest plot
    plt.figure(figsize=(12, 8))

    # Include lambda value in the title 
    plt.title(f'{name} Hazard Ratios (Test Size: {test_size}, C-index: {c_index_test:.3f}, 95% CI)')

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
    plt.savefig(f'cph-{c_index_test:.3f}-{name}-forest-plot.png')
    plt.show()

def optimize_penalties(df, name, duration_col='OS_MONTHS', event_col='OS_STATUS'):
    # Split the data for tuning into train and test sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Define grid ranges for the combined penalty (lambda) and l1_ratio (L1 vs L2 trade-off)
    lambdas = np.logspace(-6, 0, 50, base=10)  # combined penalty strength (L1+L2)
    l1_ratios = [0.0, 0.25, 0.5, 0.75, 1.0]      # 0.0: pure L2, 1.0: pure L1
    best_score = -np.inf  # Higher test C-index is better
    best_lambda = None
    best_l1_ratio = None

    # Store scores for plotting
    results = {}

    for lr in l1_ratios:
        scores = []
        for lam in lambdas:
            cph = CoxPHFitter(penalizer=lam, l1_ratio=lr)
            cph.fit(train_df, duration_col=duration_col, event_col=event_col)
            # Use test set concordance index as the score
            score = cph.score(test_df, scoring_method="concordance_index")
            scores.append(score)
            if score > best_score:
                best_score = score
                best_lambda = lam
                best_l1_ratio = lr
        results[lr] = scores

    print(f"Optimal parameters: lambda (penalty strength) = {best_lambda} and l1_ratio (L1 weight) = {best_l1_ratio} with Test C-index = {best_score}")
    
    # Plot tuning results: log(lambda) vs test C-index for each l1_ratio
    plt.figure(figsize=(12, 8))
    for lr in l1_ratios:
        plt.plot(np.log10(lambdas), results[lr], marker='o', label=f"l1_ratio={lr}")
    plt.xlabel('log(lambda)')
    plt.ylabel('Test Concordance Index')
    plt.title(f'{name}: Elastic Net Penalties Tuning')
    plt.legend()
    plt.grid(True)
    plt.show()

# Without treatment data
surv = pd.read_csv('GPL570merged.csv') 
surv = pd.get_dummies(surv, columns=["Stage", "Histology", "Race"])
surv = surv.drop(columns=['PFS_MONTHS','RFS_MONTHS'])
print(surv.columns[surv.isna().any()].tolist())
print(surv['Smoked?'].isna().sum())  # 121
surv = surv.dropna()  # left with 457 samples

# --- New code to subset to clinical and top 100 genomic features ---
importances = pd.read_csv('rsf_results_GPL570_rsf_preselection_importances.csv')
selected_genes = importances['Feature'].tolist()[:80]
# Adjust clinical features for those that were one-hotencoded:
selected_clinical = ["Adjuvant Chemo", "Age", "Sex", "Smoked?"]
dummy_cols = [col for col in surv.columns if col.startswith('Stage_') or col.startswith('Histology_') or col.startswith('Race_')]
selected_clinical += dummy_cols
# Ensure survival columns are preserved
selected_survival = ['OS_MONTHS', 'OS_STATUS']
# Combine clinical, genomic, and survival features (using union)
selected_columns = list(set(selected_clinical + selected_genes + selected_survival))
surv = surv[selected_columns]

# LOW VARIANCE FILTER
from sklearn.feature_selection import VarianceThreshold
# Drop columns with low variance
selector = VarianceThreshold(threshold=0.01)
selector.fit(surv)
low_variance = surv.columns[~selector.get_support()]
print(f"Dropping low variance columns: {low_variance}")
surv = surv.drop(columns=low_variance)

#optimize_penalties(surv, 'GPL570')

# DROP those with low variance
# Optimal parameters: lambda (penalty strength) = 0.244205309454865 and l1_ratio (L1 weight) = 0.25 with Test C-index = 0.6658395368072787
run_model(surv, 'GPL570', penalizer=0.244205309454865, l1_ratio=0.25)