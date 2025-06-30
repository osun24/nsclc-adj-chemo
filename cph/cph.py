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

if __name__ == "__main__":
    print("Loading train data from: allTrain.csv")
    train_orig = pd.read_csv("allTrain.csv")
    print(f"Number of events in training set: {train_orig['OS_STATUS'].sum()} | Censored cases: {train_orig.shape[0] - train_orig['OS_STATUS'].sum()}")
    print("Train data shape:", train_orig.shape)
    
    # Data Loading and Preprocessing for validation data
    print("Loading validation data from: allValidation.csv")
    valid_orig = pd.read_csv("allValidation.csv")
    print(f"Number of events in validation set: {valid_orig['OS_STATUS'].sum()} | Censored cases: {valid_orig.shape[0] - valid_orig['OS_STATUS'].sum()}")
    print("Validation data shape:", valid_orig.shape)
    
    # Combine train and validation for training (matching iterative feature selection approach)
    train = pd.concat([train_orig, valid_orig], axis=0, ignore_index=True)
    print(f"Combined training data shape: {train.shape}")
    print(f"Number of events in combined training: {train['OS_STATUS'].sum()} | Censored cases: {train.shape[0] - train['OS_STATUS'].sum()}")
    
    # Load test data separately
    print("Loading test data from: allTest.csv")
    test = pd.read_csv("allTest.csv")
    print(f"Number of events in test set: {test['OS_STATUS'].sum()} | Censored cases: {test.shape[0] - test['OS_STATUS'].sum()}")
    print("Test data shape:", test.shape)
    
    # rsf/rsf-results/ALL 3-29-25 RS_rsf_preselection_importances_1SE.csv
    selected_columns = pd.read_csv("rsf/rsf_results/ALL 3-29-25 RS_rsf_preselection_importances_1SE.csv")['Feature'].tolist()
    
    # LOW VARIANCE FILTER
    from sklearn.feature_selection import VarianceThreshold
    # Drop columns with low variance
    selector = VarianceThreshold(threshold=0.01)
    selector.fit(train)
    low_variance = train.columns[~selector.get_support()]
    print(f"Dropping low variance columns: {low_variance}")
    train = train.drop(columns=low_variance)
    test = test.drop(columns=low_variance)
    
    # Take first 300 features 
    selected_columns = selected_columns[:300]
    
    train = train[selected_columns + ['OS_MONTHS', 'OS_STATUS']] 
    test = test[selected_columns + ['OS_MONTHS', 'OS_STATUS']]

    # Call run_model with the separate train and test datasets (updated variable names)
    #run_model(train, test, 'All, First 300 3-29 1SE', penalizer=0.0, l1_ratio=0.25)
    
    # Optionally, call optimize_penalties using the provided CSVs
    optimize_penalties(train, test, 'All - First 300 - 3-29 1SE')