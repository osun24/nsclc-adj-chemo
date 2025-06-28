import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from ast import literal_eval
import os

def analyze_rsf_hyperparameters(csv_file_path):
    """
    Comprehensive analysis of Random Survival Forest hyperparameter tuning results.
    
    Parameters:
    -----------
    csv_file_path : str
        Path to the CSV file with hyperparameter tuning results
    """
    print(f"Loading data from {csv_file_path}")
    # Load the data
    data = pd.read_csv(csv_file_path)
    
    # Create output directory
    output_dir = 'rsf/rsf_results_affy/rsf_analysis_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract outer fold data (where hyperparameter optimization results are stored)
    outer_fold_data = data[data['fold_type'] == 'outer'].copy()
    
    # Check if dataset has percentage and num_features columns
    has_feature_info = 'percentage' in outer_fold_data.columns and 'num_features' in outer_fold_data.columns
    
    if not has_feature_info:
        print("Note: This dataset doesn't contain feature percentage or count information.")
        # Add a dummy percentage column for visualizations
        outer_fold_data['percentage'] = 1.0
        outer_fold_data['num_features'] = 'unknown'
    
    # Extract hyperparameters from the string representation
    def extract_params(params_str):
        if pd.isna(params_str):
            return pd.Series([None, None, None, None])
        params = literal_eval(params_str)
        return pd.Series([
            params.get('max_depth'), 
            params.get('max_features'), 
            params.get('min_samples_leaf'),
            params.get('n_estimators')
        ])
    
    # Extract hyperparameters
    param_cols = outer_fold_data['best_params'].apply(extract_params)
    outer_fold_data[['max_depth', 'max_features', 'min_samples_leaf', 'n_estimators']] = param_cols
    
    # Convert max_features to standardized format
    outer_fold_data['max_features_std'] = outer_fold_data['max_features'].apply(
        lambda x: str(x) if isinstance(x, (int, float)) else x
    )
    
    # Basic statistics
    print("\n=== Overall Performance Statistics ===")
    print(outer_fold_data[['test_c_index', 'train_c_index']].describe())
    
    # Skip feature percentage specific plots if no data
    if has_feature_info:
        # 1. Performance across different feature percentages
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='percentage', y='test_c_index', data=outer_fold_data)
        plt.title('Test C-index by Feature Percentage')
        plt.xlabel('Percentage of Features')
        plt.ylabel('Test C-index')
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{output_dir}/test_cindex_by_percentage.png")
        plt.show()
    
    # 2. Train vs Test C-index (overfitting analysis)
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        outer_fold_data['train_c_index'], 
        outer_fold_data['test_c_index'],
        c=range(len(outer_fold_data)),  # Use index as color if no percentage
        cmap='viridis',
        alpha=0.7, 
        s=100
    )
    
    # Add fold labels
    for i, row in outer_fold_data.iterrows():
        plt.annotate(f"Fold {int(row['outer_fold'])}", 
                    (row['train_c_index'], row['test_c_index']),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=9)
    
    plt.colorbar(scatter, label='Fold index')
    
    # Add diagonal line (perfect generalization)
    min_val = min(outer_fold_data['train_c_index'].min(), outer_fold_data['test_c_index'].min()) - 0.05
    max_val = max(outer_fold_data['train_c_index'].max(), outer_fold_data['test_c_index'].max()) + 0.05
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
    
    plt.title('Train vs Test C-index')
    plt.xlabel('Train C-index')
    plt.ylabel('Test C-index')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/train_vs_test_cindex.png")
    plt.show()
    
    # 3. Calculate overfitting margin
    outer_fold_data['overfit_margin'] = outer_fold_data['train_c_index'] - outer_fold_data['test_c_index']
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(outer_fold_data)), outer_fold_data['overfit_margin'])
    plt.title('Overfitting Margin by Fold')
    plt.xlabel('Outer Fold')
    plt.ylabel('Train C-index - Test C-index')
    plt.xticks(range(len(outer_fold_data)), outer_fold_data['outer_fold'])
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/overfit_margin_by_fold.png")
    plt.show()
    
    # 4. Hyperparameter effects on performance
    for param in ['max_features_std', 'min_samples_leaf', 'n_estimators']:
        plt.figure(figsize=(12, 6))
        sns.boxplot(x=param, y='test_c_index', data=outer_fold_data)
        plt.title(f'Test C-index by {param}')
        plt.xticks(rotation=45 if param == 'max_features_std' else 0)
        plt.ylabel('Test C-index')
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{output_dir}/test_cindex_by_{param}.png")
        plt.show()
    
    # 4b. 3D plot of hyperparameter interactions
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(
        outer_fold_data['min_samples_leaf'],
        outer_fold_data['n_estimators'],
        outer_fold_data['test_c_index'],
        c=outer_fold_data['outer_fold'],
        cmap='viridis',
        s=80,
        alpha=0.7
    )
    
    ax.set_xlabel('min_samples_leaf')
    ax.set_ylabel('n_estimators')
    ax.set_zlabel('Test C-index')
    ax.set_title('3D Hyperparameter Interaction: min_samples_leaf vs n_estimators vs Performance')
    
    cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label('Fold')
    
    plt.savefig(f"{output_dir}/3d_hyperparameter_interaction.png")
    plt.show()
    
    # 6. Best configuration analysis
    best_config = outer_fold_data.loc[outer_fold_data['test_c_index'].idxmax()]
    
    print("\n=== Best Configuration ===")
    print(f"Test C-index: {best_config['test_c_index']:.4f}")
    print(f"Train C-index: {best_config['train_c_index']:.4f}")
    print(f"Hyperparameters: {best_config['best_params']}")
    print(f"Outer fold: {int(best_config['outer_fold'])}")
    
    # 7. Hyperparameter heatmaps for most important interactions
    # min_samples_leaf vs n_estimators
    plt.figure(figsize=(10, 8))
    pivot = outer_fold_data.pivot_table(
        values='test_c_index', 
        index='min_samples_leaf', 
        columns='n_estimators', 
        aggfunc='mean'
    )
    sns.heatmap(pivot, annot=True, cmap='viridis', fmt='.3f')
    plt.title('Mean Test C-index: min_samples_leaf vs n_estimators')
    plt.savefig(f"{output_dir}/heatmap_leaf_vs_estimators.png")
    plt.show()
    
    # max_features vs n_estimators
    plt.figure(figsize=(10, 8))
    pivot = outer_fold_data.pivot_table(
        values='test_c_index', 
        index='max_features_std', 
        columns='n_estimators', 
        aggfunc='mean'
    )
    sns.heatmap(pivot, annot=True, cmap='viridis', fmt='.3f')
    plt.title('Mean Test C-index: max_features vs n_estimators')
    plt.savefig(f"{output_dir}/heatmap_maxfeatures_vs_estimators.png")
    plt.show()
    
    # 8. Analysis of performance across folds
    plt.figure(figsize=(14, 8))
    
    plt.plot(outer_fold_data['outer_fold'], outer_fold_data['test_c_index'], 
             marker='o', label='Test C-index')
    plt.plot(outer_fold_data['outer_fold'], outer_fold_data['train_c_index'], 
             marker='s', label='Train C-index')
    
    plt.xlabel('Outer Fold')
    plt.ylabel('C-index')
    plt.title('Performance Across Folds')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/performance_across_folds.png")
    plt.show()
    
    # 9. Summary statistics by hyperparameter value
    print("\n=== Mean Performance by Hyperparameter Value ===")
    
    for param in ['max_features_std', 'min_samples_leaf', 'n_estimators']:
        stats = outer_fold_data.groupby(param).agg({
            'test_c_index': ['mean', 'std', 'count'],
            'train_c_index': ['mean', 'std']
        })
        
        print(f"\n{param}:")
        print(stats)
        
        # Visualization of hyperparameter performance statistics
        plt.figure(figsize=(12, 7))
        
        # Plot mean with error bars
        param_values = stats.index.tolist()
        means = stats[('test_c_index', 'mean')].values
        stds = stats[('test_c_index', 'std')].values
        
        plt.bar(range(len(param_values)), means, yerr=stds, capsize=8, alpha=0.7)
        plt.xticks(range(len(param_values)), param_values)
        plt.xlabel(param)
        plt.ylabel('Mean Test C-index')
        plt.title(f'Performance by {param} with Standard Deviation')
        plt.grid(True, alpha=0.3)
        
        for i, (mean, std) in enumerate(zip(means, stds)):
            plt.annotate(f'{mean:.3f} ± {std:.3f}', 
                        xy=(i, mean), 
                        xytext=(0, 10), 
                        textcoords='offset points',
                        ha='center', 
                        va='bottom')
        
        plt.savefig(f"{output_dir}/stats_by_{param}.png")
        plt.show()
    
    # 10. Correlation analysis between parameters and performance
    # Extract numeric features for correlation analysis
    corr_data = outer_fold_data.copy()
    
    # Convert max_features to numeric if possible
    def convert_max_features(x):
        if x == 'sqrt':
            return -1  # Special code for sqrt
        elif x == 'log2':
            return -2  # Special code for log2
        else:
            try:
                return float(x)
            except:
                return np.nan
    
    corr_data['max_features_num'] = corr_data['max_features'].apply(convert_max_features)
    
    # Select numeric columns for correlation
    numeric_cols = ['test_c_index', 'train_c_index', 'overfit_margin', 
                     'min_samples_leaf', 'n_estimators', 'max_features_num']
    
    corr_matrix = corr_data[numeric_cols].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar=True)
    plt.title('Correlation Matrix of Model Parameters and Performance')
    plt.savefig(f"{output_dir}/correlation_matrix.png")
    plt.show()
    
    return outer_fold_data

if __name__ == "__main__":
    # Example usage with Affymetrix data
    analyze_rsf_hyperparameters('rsf/rsf_results_affy/Affy RS_rsf_all_fold_results_20250628.csv')
    print("\nAnalysis complete. Results saved to 'rsf/rsf_results_affy/rsf_analysis_results' directory.")
    
# 0.2, 70, 750