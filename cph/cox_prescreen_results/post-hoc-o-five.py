
"""
Extract unique genes with interaction p-values <= 0.05 from Cox interaction screening results
Format output to match selection_results.csv structure
"""

import pandas as pd
from datetime import datetime

# Load the full results
df = pd.read_csv('cph/cox_prescreen_results/20250712_cox_interaction_cv_detailed_results.csv')

# Filter for significant interactions (p <= 0.05)
significant_df = df[df['interaction_p_value'] <= 0.05].copy()

# Get unique genes only (remove duplicates if any)
unique_genes = significant_df['gene'].unique()

# Create selection results format matching the template
selection_results = pd.DataFrame({
    'gene': unique_genes,
    'total_selections': 1.0,  # Since each gene appears once
    'selection_frequency': 1.0,  # 100% selection frequency for p <= 0.05
    'selected_in_trials': 1  # Selected in 1 trial (the significance test)
})

# Sort by gene name for consistency
selection_results = selection_results.sort_values('gene')

# Create output filename with current date
current_date = datetime.now().strftime("%Y%m%d")
output_file = f"cph/cox_prescreen_results/{current_date}_significant_interactions_p_0.05_selection_results.csv"

# Save results
selection_results.to_csv(output_file, index=False)

print(f"Extracted {len(unique_genes)} unique genes with interaction p-values <= 0.05")
print(f"Results saved to: {output_file}")
print(f"First 10 genes:")
print(selection_results.head(10))
