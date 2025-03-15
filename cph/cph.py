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

# Modified run_model: removed splitting; now expects separate train_df and valid_df.
def run_model(train_df, valid_df, name, penalizer=0.1, l1_ratio=1.0, duration_col='OS_MONTHS', event_col='OS_STATUS'):
    # Fit the Cox proportional hazards model on the provided training set using elastic net
    cph = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)
    cph.fit(train_df, duration_col=duration_col, event_col=event_col)

    # Print summary of the fitted model
    cph.print_summary(style="ascii")

    # Evaluate the model using the training set and the provided validation set
    train_c_index = cph.concordance_index_
    print(f"Concordance Index on Training Set: {train_c_index:.3f}")
    valid_c_index = cph.score(valid_df, scoring_method="concordance_index")
    print(f"Concordance Index on Validation Set: {valid_c_index:.3f}")

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

    with open(f'cph-{valid_c_index:.3f}-{name}-summary.txt', 'w') as f:
        f.write(summary_df.to_string())
        formatted_metrics = '\n'.join([f'{metric:.4f}' for metric in model_metrics])
        f.write(f"\n\nModel metrics: {formatted_metrics}")
        
        # Write additional metrics in the specified format
        f.write(f"\n\nTrain Concordance = {concordance:.3f}")
        # Test c-index
        f.write(f"\nConcordance on validation set = {valid_c_index:.3f}")
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
    plt.title(f'{name} Hazard Ratios (Validation C-index: {valid_c_index:.3f}, 95% CI)')

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
    plt.savefig(f'cph-{valid_c_index:.3f}-{name}-forest-plot.png')
    plt.show()

# Modified optimize_penalties: removed splitting; now receives train_df and valid_df.
def optimize_penalties(train_df, valid_df, name, duration_col='OS_MONTHS', event_col='OS_STATUS'):
    # Define grid ranges for the combined penalty (lambda) and l1_ratio (L1 vs L2 trade-off)
    lambdas = np.logspace(-6, 0, 50, base=10)
    l1_ratios = [0.0, 0.25, 0.5, 0.75, 1.0]
    best_score = -np.inf
    best_lambda = None
    best_l1_ratio = None
    results = {}

    for lr in l1_ratios:
        scores = []
        for lam in lambdas:
            cph = CoxPHFitter(penalizer=lam, l1_ratio=lr)
            cph.fit(train_df, duration_col=duration_col, event_col=event_col)
            score = cph.score(valid_df, scoring_method="concordance_index")
            scores.append(score)
            if score > best_score:
                best_score = score
                best_lambda = lam
                best_l1_ratio = lr
        results[lr] = scores

    print(f"Optimal parameters: lambda = {best_lambda} and l1_ratio = {best_l1_ratio} with Validation C-index = {best_score}")
    # ...existing plotting code...

if __name__ == "__main__":
    print("Loading train data from: GPL570train.csv")
    train = pd.read_csv("GPL570train.csv")
    print("Train data shape:", train.shape)

    print("Loading validation data from: GPL570validation.csv")
    valid = pd.read_csv("GPL570validation.csv")
    print("Validation data shape:", valid.shape)
    
    #"Smoked?_No",
    selected_columns = ["Stage_IA", "Smoked?_Unknown", "Age", "RTL3", "LOC105375172", "Smoked?_Yes", "IQCF6", "Adjuvant Chemo"]

    # LOW VARIANCE FILTER
    """from sklearn.feature_selection import VarianceThreshold
    # Drop columns with low variance
    selector = VarianceThreshold(threshold=0.01)
    selector.fit(surv)
    low_variance = surv.columns[~selector.get_support()]
    print(f"Dropping low variance columns: {low_variance}")
    surv = surv.drop(columns=low_variance)"""
    
    train = train[selected_columns + ['OS_MONTHS', 'OS_STATUS']] 
    valid = valid[selected_columns + ['OS_MONTHS', 'OS_STATUS']]
    # Call run_model with the separate train and validation datasets
    run_model(train, valid, 'GPL570 Model', penalizer=0.0, l1_ratio=0.25)
    # Optionally, call optimize_penalties using the provided CSVs
    # optimize_penalties(train, valid, 'GPL570')

# --- New code to subset to clinical and top 100 genomic features ---
importances = pd.read_csv('rsf/rsf_results_GPL570_rsf_preselection_importances.csv')
#selected_genes = importances['Feature'].tolist()[:80]
selected_genes = [
    "TP53",
    "KRAS",
    "EGFR",
    "ERCC1",
    "BRCA1",
    "RRM1",
    "BRAF",
    "MET",
    "ALK",
    "STK11",
    "RB1",
    "CCNB1",
    "CCND1",
    "CDKN2A",
    "CDK4",
    "CDK6",
    "MYC",
    "BCL2",
    "BAX",
    "MLH1",
    "MSH2",
    "MSH6",
    "ATM",
    "ATR",
    "CHEK1",
    "CHEK2",
    "FANCA",
    "FANCD2",
    "XRCC1",
    "XRCC2",
    "XRCC3",
    "RAD51",
    "TYMS",
    "TUBB3",
    "ABCC1",
    "ABCB1",
    "KEAP1",
    "NFE2L2",
    "PTEN",
    "PIK3CA",
    "AKT1",
    "ERBB2",
    "FGFR1",
    "CUL3",
    "GSTM1",
    "GSTP1",
    "SOD2",
    "CASP3",
    "CASP9",
    "MDM2",
    "CDKN1A",
    "CDKN1B",
    "PARP1",
    "MTHFR",
    "DUT",
    "SLFN11",
    "PDK1",
    "MCL1",
    "CCNE1",
    "PKM",
    "HIF1A",
    "VEGFA",
    "E2F1",
    "BRCC3",
    "MRE11",
    "NBN",
    "RAD50",
    "RAD17",     # Alternative to CHEK1 duplication
    "APAF1",
    "ATG5",
    "ATG7",
    "SIRT1",
    "MTHFD2",
    "DNMT1",
    "DNMT3A",
    "TLE1",
    "SOX2",
    "NKX2-1",
    "GTF2I",
    "PRC1",
    "KDM5B",
    "SMARCA4",
    "ARID1A",
    "BRIP1",
    "POLD1",
    "POLE",
    "MCM2",
    "MCM4",
    "CDC20",
    "CDH1",
    "VIM",
    "SPARC",
    "SNAI1",
    "TWIST1",
    "ERBB3",
    "HERPUD1",
    "GAPDH",
    "ACTB",
    "CD8A",
    "CD274"
]

selected_genes = [
    "TP53", "KRAS", "EGFR", "ALK", "MET", "STK11", "KEAP1", "BRAF",
    "PTEN", "RB1", "PIK3CA", "NF1", "SMARCA4", "MDM2", "CDKN2A",
    "MYC", "ATM", "BRCA1", "BRCA2", "PIK3R1"
]

# importances take top 500
selected_genes = importances['Feature'].tolist()[:50]


#optimize_penalties(surv, 'GPL570')

# DROP those with low variance
# Optimal parameters: lambda (penalty strength) = 0.244205309454865 and l1_ratio (L1 weight) = 0.25 with Test C-index = 0.6658395368072787
#run_model(surv, 'GPL570 - RSF Selected 50', penalizer=0.244205309454865, l1_ratio=0.25)