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

def run_model(df, name):
    # Train-test split
    # Split the data into training and testing sets
    test_size = 0.2
    # Split the data into training and testing sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Fit the Cox proportional hazards model on the training set
    cph = CoxPHFitter(penalizer=0.1, l1_ratio=1.0)
    cph.fit(train_df, duration_col='PFS_MONTHS', event_col='PFS_STATUS')

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
    latex_df = latex_df.drop(columns=['exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%'])
    
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

def optimize_l1_penalty(df, name):
    # Use training data for lambda optimization
    train_df, _ = train_test_split(df, test_size=0.2, random_state=42)

    # Define a range of lambda values (penalizer values)
    lambdas = np.logspace(-6, 0, 50, base = 10)
    lr_statistics = []

    for lam in lambdas:
        cph = CoxPHFitter(penalizer=lam, l1_ratio=1.0)
        cph.fit(train_df, duration_col='PFS_MONTHS', event_col='PFS_STATUS')
        # Use log_likelihood_ratio_test() to compute the test statistic,
        # which is 2*(logL(model) - logL(null)). This serves as our measure.
        deviance = -cph.log_likelihood_ratio_test().test_statistic
        lr_statistics.append(deviance)
    
    # Find the lambda with the lowest test statistic value
    best_idx = np.argmin(lr_statistics)
    best_lambda = lambdas[best_idx]
    best_stat = lr_statistics[best_idx]
    print(f"Lowest test statistic: {best_stat} at lambda: {best_lambda} (log10: {np.log10(best_lambda)})")

    # Plot log(lambda) vs likelihood ratio test statistic
    plt.figure(figsize=(12, 6))
    plt.plot(np.log10(lambdas), lr_statistics, marker='o')
    plt.xlabel('log(lambda)')
    plt.ylabel('Likelihood Ratio Test Statistic')
    plt.title(f'{name}: log(lambda) vs Likelihood Ratio Test Statistic')
    plt.grid(True)
    plt.show()

# Without treatment data
surv = pd.read_csv('GSE68465merged.csv')

surv.drop(columns=['Unnamed: 0'], inplace=True)

# rename "No" and "Yes" in adjuvant chemo to 0 and 1
surv['Adjuvant Chemo'] = surv['Adjuvant Chemo'].replace('No', 0)
surv['Adjuvant Chemo'] = surv['Adjuvant Chemo'].replace('Yes', 1)

# Drop those with "Unknown" or nan
surv = surv.dropna()
surv = surv[surv['Adjuvant Chemo'] != 'Unknown']

# Combine "unknown" and "not reported" into "unknown" for race
surv["Race"] = surv["Race"].replace("Not Reported", "Unknown")

# print proportion of male/female in sex
print(surv['Sex'].value_counts(normalize=True)) #slightly more female

# change sex to Is Female
surv['Is Female'] = surv['Sex'].replace({'Female': 1, 'Male': 0})
surv.drop(columns = ["Sex"], inplace=True)

# print value counts for smoking
print(surv['Smoked?'].value_counts(normalize=True)) # 0.5 never smoked, 0.3 former, 0.2 current

# change Smoked in the past, Currently smoking to 1 and Never smoked to 0
surv['Smoked?'] = surv['Smoked?'].replace({'Smoked in the past': 1, 'Currently smoking': 1, 'Never smoked': 0})

# drop those with Unknown smoked data **** (~20%!)
surv = surv[surv['Smoked?'] != 'Unknown'] 
surv = surv[surv["Smoked?"] != "--"]

# one hot encode race
surv = pd.get_dummies(surv, columns = ["Race"], drop_first=True)

# DROP HISTOLOGY because it is all LUNG ADENOCARCINOMA (ONLY FOR THIS DATASET!)
surv = surv.drop(columns = ["Histology"])

# print only those that are not float64:
for col in surv.columns:
    if surv[col].dtype != 'float64':
        print(f"{col}: {surv[col].unique()}")

# print number of rows and columns
print(surv.shape)
print(surv.head())

print("MISSING:")
# print columns with missing values
for col in surv.columns:
    if surv[col].isnull().sum() > 0:
        print(f"{col}: {surv[col].isnull().sum()}")

# Drop those with low variance
run_model(surv, 'GSE68465')