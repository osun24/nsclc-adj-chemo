
"""
Extract unique genes with interaction p-values <= 0.05 from Cox interaction screening results
Format output to match selection_results.csv structure
"""

import pandas as pd
from datetime import datetime

# Load the full results
df = pd.read_csv('cph/cox_prescreen_results/20250712_cox_interaction_cv_detailed_results.csv')

# Filter for significant interactions (p <= 0.05)
significant_df = df[df['significant_at_0.05'] == True].copy()

# Extract number of occurrences for each gene (total_selections), selection_frequency, and unique genes
total_selections = significant_df['gene'].value_counts()
selection_frequency = total_selections / 200 # 20 trials x 10 folds
unique_genes = significant_df['gene'].unique()

# Create selection results format matching the template
selection_results = pd.DataFrame({
    'selected_gene': unique_genes,
    'total_selections': total_selections[unique_genes],
    'selection_frequency': selection_frequency[unique_genes]
})

# Sort by total_selections in descending order
selection_results = selection_results.sort_values(by='total_selections', ascending=False)

# Create output filename with current date
current_date = datetime.now().strftime("%Y%m%d")
output_file = f"cph/cox_prescreen_results/{current_date}_significant_interactions_p_0.05_selection_results.csv"

# Save results
selection_results.to_csv(output_file, index=False)

print(f"Extracted {len(unique_genes)} unique genes with interaction p-values <= 0.05")
print(f"Results saved to: {output_file}")
print(f"First 10 genes:")
print(selection_results.head(10))
