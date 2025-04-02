import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import ast
import os

# Create output directory if it doesn't exist
output_dir = "cv_visualizations"
os.makedirs(output_dir, exist_ok=True)

# Load the nested CV results
results_path = "rsf/rsf_results/ALL 3-17-25 RS_rsf_nested_cv_results.csv"
df = pd.read_csv(results_path)

# Convert string representation of dictionaries to actual dictionaries
df['Hyperparameters'] = df['Hyperparameters'].apply(ast.literal_eval)

# Extract hyperparameters into separate columns
df['n_estimators'] = df['Hyperparameters'].apply(lambda x: x['n_estimators'])
df['min_samples_split'] = df['Hyperparameters'].apply(lambda x: x['min_samples_split'])
df['min_samples_leaf'] = df['Hyperparameters'].apply(lambda x: x['min_samples_leaf'])
df['max_features'] = df['Hyperparameters'].apply(lambda x: str(x['max_features']))

# 1. Plot C-index distribution across folds
plt.figure(figsize=(10, 6))
sns.barplot(x='fold', y='Test_C_index', data=df, palette='viridis')
plt.axhline(y=df['Test_C_index'].mean(), color='red', linestyle='--', 
            label=f'Mean C-index: {df["Test_C_index"].mean():.3f}')
plt.title('Cross-Validation C-index by Fold', fontsize=15)
plt.xlabel('Fold Number')
plt.ylabel('Concordance Index (C-index)')
plt.ylim(0.45, 0.7)
plt.legend()
plt.savefig(f"{output_dir}/c_index_by_fold.png", dpi=300, bbox_inches='tight')
plt.close()

# 2. Hyperparameter value distributions
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
sns.countplot(y='n_estimators', data=df, ax=axes[0, 0], palette='viridis')
axes[0, 0].set_title('Distribution of n_estimators')
axes[0, 0].set_ylabel('Number of Trees')

sns.countplot(y='min_samples_split', data=df, ax=axes[0, 1], palette='viridis')
axes[0, 1].set_title('Distribution of min_samples_split')

sns.countplot(y='min_samples_leaf', data=df, ax=axes[1, 0], palette='viridis')
axes[1, 0].set_title('Distribution of min_samples_leaf')

sns.countplot(y='max_features', data=df, ax=axes[1, 1], palette='viridis')
axes[1, 1].set_title('Distribution of max_features')

plt.tight_layout()
plt.savefig(f"{output_dir}/hyperparameter_distributions.png", dpi=300, bbox_inches='tight')
plt.close()

# 3. Relationship between n_estimators and performance
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='n_estimators', y='Test_C_index', size='min_samples_leaf', 
                hue='max_features', sizes=(50, 200), alpha=0.7)
plt.title('Impact of n_estimators on C-index', fontsize=15)
plt.xlabel('Number of Trees (n_estimators)')
plt.ylabel('Concordance Index (C-index)')
plt.grid(True, alpha=0.3)
plt.savefig(f"{output_dir}/n_estimators_vs_performance.png", dpi=300, bbox_inches='tight')
plt.close()

# 4. Hyperparameter correlation with performance
corr_data = df[['n_estimators', 'min_samples_split', 'min_samples_leaf', 'Test_C_index']].copy()
# Map max_features to numeric values for correlation analysis
feature_map = {'sqrt': 1, 'log2': 2, 'None': 3}
corr_data['max_features_numeric'] = df['max_features'].map(lambda x: feature_map.get(x, 0))

plt.figure(figsize=(8, 7))
correlation = corr_data.corr()
mask = np.triu(correlation)
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f", mask=mask)
plt.title('Correlation between Hyperparameters and Performance', fontsize=15)
plt.tight_layout()
plt.savefig(f"{output_dir}/hyperparameter_correlation.png", dpi=300, bbox_inches='tight')
plt.close()

# 5. Best performing configurations
top_n = 3
top_performers = df.sort_values('Test_C_index', ascending=False).head(top_n)

fig, ax = plt.subplots(figsize=(12, 6))
bars = sns.barplot(x='fold', y='Test_C_index', data=top_performers, palette='viridis', ax=ax)

# Add hyperparameter details as text annotations
for i, (_, row) in enumerate(top_performers.iterrows()):
    param_text = f"Trees: {row['n_estimators']}\nSplit: {row['min_samples_split']}\nLeaf: {row['min_samples_leaf']}\nFeatures: {row['max_features']}"
    ax.text(i, row['Test_C_index'] - 0.05, param_text, ha='center', va='top', fontsize=9)

plt.title('Top Performing Configurations', fontsize=15)
plt.xlabel('Fold')
plt.ylabel('Concordance Index (C-index)')
plt.ylim(0.5, 0.7)
plt.savefig(f"{output_dir}/top_performers.png", dpi=300, bbox_inches='tight')
plt.close()

# 6. Regularization vs Performance
plt.figure(figsize=(10, 6))
# Create a regularization score (higher = more regularized)
df['regularization_score'] = df['min_samples_split'] + df['min_samples_leaf']

sns.scatterplot(data=df, x='regularization_score', y='Test_C_index', 
                size='n_estimators', hue='max_features', sizes=(50, 200), alpha=0.7)
plt.title('Impact of Regularization on Performance', fontsize=15)
plt.xlabel('Regularization Strength (min_samples_split + min_samples_leaf)')
plt.ylabel('Concordance Index (C-index)')
plt.grid(True, alpha=0.3)
plt.savefig(f"{output_dir}/regularization_vs_performance.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"Visualizations have been saved to the '{output_dir}' directory")