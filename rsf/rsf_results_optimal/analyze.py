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
    output_dir = 'rsf/rsf_results_optimal/rsf_analysis_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract outer fold data (where hyperparameter optimization results are stored)
    outer_fold_data = data[data['fold_type'] == 'outer'].copy()
    
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
        c=outer_fold_data['percentage'], 
        cmap='viridis',
        alpha=0.7, 
        s=100
    )
    plt.colorbar(scatter, label='Feature Percentage')
    
    # Add diagonal line (perfect generalization)
    min_val = min(outer_fold_data['train_c_index'].min(), outer_fold_data['test_c_index'].min()) - 0.05
    max_val = max(outer_fold_data['train_c_index'].max(), outer_fold_data['test_c_index'].max()) + 0.05
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
    
    plt.title('Train vs Test C-index by Feature Percentage')
    plt.xlabel('Train C-index')
    plt.ylabel('Test C-index')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/train_vs_test_cindex.png")
    plt.show()
    
    # 2b. NEW: 3D Plot of Train vs Test C-index vs Num Features
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(
        outer_fold_data['train_c_index'],
        outer_fold_data['test_c_index'],
        outer_fold_data['num_features'],
        c=outer_fold_data['percentage'],
        cmap='viridis',
        s=80,
        alpha=0.7
    )
    
    ax.set_xlabel('Train C-index')
    ax.set_ylabel('Test C-index')
    ax.set_zlabel('Number of Features')
    ax.set_title('3D Relationship: Train C-index, Test C-index, and Feature Count')
    
    # Add a color bar
    cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label('Feature Percentage')
    
    plt.savefig(f"{output_dir}/3d_train_test_features.png")
    plt.show()
    
    # 3. Calculate overfitting margin
    outer_fold_data['overfit_margin'] = outer_fold_data['train_c_index'] - outer_fold_data['test_c_index']
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='percentage', y='overfit_margin', data=outer_fold_data)
    plt.title('Overfitting Margin by Feature Percentage')
    plt.xlabel('Percentage of Features')
    plt.ylabel('Train C-index - Test C-index')
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/overfit_margin_by_percentage.png")
    plt.show()
    
    # 3b. NEW: 3D plot of overfitting margin
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(
        outer_fold_data['train_c_index'],
        outer_fold_data['test_c_index'],
        outer_fold_data['overfit_margin'],
        c=outer_fold_data['num_features'],
        cmap='coolwarm',
        s=80,
        alpha=0.7
    )
    
    ax.set_xlabel('Train C-index')
    ax.set_ylabel('Test C-index')
    ax.set_zlabel('Overfitting Margin')
    ax.set_title('3D Visualization of Overfitting Margin')
    
    # Add a color bar
    cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label('Number of Features')
    
    plt.savefig(f"{output_dir}/3d_overfitting_margin.png")
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
    
    # 4b. NEW: 3D plot of hyperparameter interactions
    # Create a 3D plot for min_samples_leaf, n_estimators, and test_c_index
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(
        outer_fold_data['min_samples_leaf'],
        outer_fold_data['n_estimators'],
        outer_fold_data['test_c_index'],
        c=outer_fold_data['percentage'],
        cmap='viridis',
        s=80,
        alpha=0.7
    )
    
    ax.set_xlabel('min_samples_leaf')
    ax.set_ylabel('n_estimators')
    ax.set_zlabel('Test C-index')
    ax.set_title('3D Hyperparameter Interaction: min_samples_leaf vs n_estimators vs Performance')
    
    cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label('Feature Percentage')
    
    plt.savefig(f"{output_dir}/3d_hyperparameter_interaction.png")
    plt.show()
    
    # 5. Learning curve analysis
    # Group by percentage to see how performance changes with feature count
    pct_stats = outer_fold_data.groupby('percentage').agg({
        'test_c_index': ['mean', 'std'],
        'train_c_index': ['mean', 'std'],
        'num_features': 'first'
    }).reset_index()
    
    percentages = pct_stats['percentage'].values
    test_means = pct_stats[('test_c_index', 'mean')].values
    test_stds = pct_stats[('test_c_index', 'std')].values
    train_means = pct_stats[('train_c_index', 'mean')].values
    train_stds = pct_stats[('train_c_index', 'std')].values
    num_features = pct_stats[('num_features', 'first')].values
    
    plt.figure(figsize=(12, 8))
    plt.errorbar(num_features, test_means, yerr=test_stds, 
                 marker='o', markersize=8, capsize=6, label='Test C-index')
    plt.errorbar(num_features, train_means, yerr=train_stds, 
                 marker='s', markersize=8, capsize=6, label='Train C-index')
    
    plt.xlabel('Number of Features')
    plt.ylabel('C-index')
    plt.title('Learning Curve: Performance vs Number of Features')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add percentage labels
    for i, pct in enumerate(percentages):
        plt.annotate(f"{pct*100}%", 
                    (num_features[i], test_means[i]),
                    textcoords="offset points",
                    xytext=(0,-15), 
                    ha='center')
    
    plt.savefig(f"{output_dir}/learning_curve.png")
    plt.show()
    
    # 5b. NEW: 3D Learning curve visualization with performance gap
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create x data points (number of features)
    x = num_features
    
    # Create a meshgrid for plotting surfaces
    X, Y = np.meshgrid(x, [0, 1])
    
    # Create Z values - performance scores for train (0) and test (1)
    Z = np.zeros(X.shape)
    Z[0, :] = train_means
    Z[1, :] = test_means
    
    # Plot the surface
    surf = ax.plot_surface(X, Y, Z, cmap='coolwarm', alpha=0.8, edgecolor='k')
    
    # Plot data points for emphasis
    ax.scatter(x, np.zeros_like(x), train_means, c='b', marker='o', s=100, label='Train C-index')
    ax.scatter(x, np.ones_like(x), test_means, c='r', marker='^', s=100, label='Test C-index')
    
    # Connect corresponding train-test points with lines to visualize gap
    for i in range(len(x)):
        ax.plot([x[i], x[i]], [0, 1], [train_means[i], test_means[i]], 'k--', alpha=0.6)
    
    ax.set_xlabel('Number of Features')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Train', 'Test'])
    ax.set_zlabel('C-index')
    ax.set_title('3D Learning Curve: Train-Test Performance Gap')
    
    # Add percentage annotations
    for i, pct in enumerate(percentages):
        ax.text(x[i], 0, train_means[i], f"{pct*100}%", color='black')
    
    plt.colorbar(surf, ax=ax, shrink=0.6, aspect=10)
    ax.legend()
    
    plt.savefig(f"{output_dir}/3d_learning_curve.png")
    plt.show()
    
    # 6. Best configuration analysis
    # Get the best performing configuration overall
    best_config = outer_fold_data.loc[outer_fold_data['test_c_index'].idxmax()]
    
    print("\n=== Best Configuration ===")
    print(f"Test C-index: {best_config['test_c_index']:.4f}")
    print(f"Train C-index: {best_config['train_c_index']:.4f}")
    print(f"Feature percentage: {best_config['percentage']}")
    print(f"Number of features: {int(best_config['num_features'])}")
    print(f"Hyperparameters: {best_config['best_params']}")
    print(f"Outer fold: {int(best_config['outer_fold'])}")
    
    # Best configuration for each percentage
    best_by_pct = outer_fold_data.loc[outer_fold_data.groupby('percentage')['test_c_index'].idxmax()]
    
    print("\n=== Best Configuration by Feature Percentage ===")
    for _, row in best_by_pct.iterrows():
        print(f"\n{row['percentage']*100}% features ({int(row['num_features'])} features):")
        print(f"  Test C-index: {row['test_c_index']:.4f}")
        print(f"  Train C-index: {row['train_c_index']:.4f}")
        print(f"  Hyperparameters: {row['best_params']}")
        print(f"  Outer fold: {int(row['outer_fold'])}")
    
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
    
    # 7b. NEW: 3D heatmap visualization
    # Create interactive 3D surface plots for hyperparameter interactions
    # max_features vs min_samples_leaf vs performance
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get unique values for each parameter
    min_samples_leaf_values = sorted(outer_fold_data['min_samples_leaf'].unique())
    max_features_values = ['log2', 'sqrt', '500']  # Common values in the dataset
    
    # Create a matrix to hold the mean test scores
    Z = np.zeros((len(min_samples_leaf_values), len(max_features_values)))
    
    # Fill the matrix with mean test scores
    for i, leaf in enumerate(min_samples_leaf_values):
        for j, feat in enumerate(max_features_values):
            subset = outer_fold_data[(outer_fold_data['min_samples_leaf'] == leaf) & 
                                    (outer_fold_data['max_features_std'] == feat)]
            if len(subset) > 0:
                Z[i, j] = subset['test_c_index'].mean()
            else:
                Z[i, j] = np.nan
    
    # Create meshgrid
    X, Y = np.meshgrid(range(len(max_features_values)), range(len(min_samples_leaf_values)))
    
    # Plot the surface
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='k', alpha=0.7)
    
    # Set labels
    ax.set_xlabel('max_features')
    ax.set_ylabel('min_samples_leaf')
    ax.set_zlabel('Mean Test C-index')
    ax.set_title('3D Hyperparameter Surface: max_features vs min_samples_leaf')
    
    # Set the tick labels
    ax.set_xticks(range(len(max_features_values)))
    ax.set_xticklabels(max_features_values)
    ax.set_yticks(range(len(min_samples_leaf_values)))
    ax.set_yticklabels(min_samples_leaf_values)
    
    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    plt.savefig(f"{output_dir}/3d_surface_maxfeatures_vs_minsamples.png")
    plt.show()
    
    # 8. Analysis of performance across folds
    plt.figure(figsize=(14, 8))
    
    for pct in sorted(outer_fold_data['percentage'].unique()):
        subset = outer_fold_data[outer_fold_data['percentage'] == pct]
        subset_sorted = subset.sort_values('outer_fold')
        
        plt.plot(subset_sorted['outer_fold'], subset_sorted['test_c_index'], 
                 marker='o', label=f'{pct*100}% features')
    
    plt.xlabel('Outer Fold')
    plt.ylabel('Test C-index')
    plt.title('Test Performance Across Folds by Feature Percentage')
    plt.legend(title='Feature Percentage')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/performance_across_folds.png")
    plt.show()
    
    # 8b. NEW: Enhanced visualization of fold stability
    plt.figure(figsize=(16, 10))
    
    # Create a grouped bar chart for a clearer comparison
    bar_width = 0.15
    percentages = sorted(outer_fold_data['percentage'].unique())
    num_percentages = len(percentages)
    
    for i, pct in enumerate(percentages):
        subset = outer_fold_data[outer_fold_data['percentage'] == pct]
        subset_sorted = subset.sort_values('outer_fold')
        
        positions = np.arange(len(subset_sorted)) + (i - num_percentages/2 + 0.5) * bar_width
        
        plt.bar(positions, subset_sorted['test_c_index'], 
                width=bar_width, label=f'{pct*100}% features',
                alpha=0.7)
    
    plt.xlabel('Outer Fold')
    plt.ylabel('Test C-index')
    plt.title('Fold Stability Comparison by Feature Percentage')
    plt.legend(title='Feature Percentage')
    plt.grid(True, alpha=0.3)
    plt.xticks(np.arange(15), [str(i+1) for i in range(15)])
    
    plt.savefig(f"{output_dir}/fold_stability_bars.png")
    plt.show()
    
    # 9. Summary statistics by hyperparameter value
    # Create tables of mean performance by hyperparameter value
    print("\n=== Mean Performance by Hyperparameter Value ===")
    
    for param in ['max_features_std', 'min_samples_leaf', 'n_estimators']:
        stats = outer_fold_data.groupby(param).agg({
            'test_c_index': ['mean', 'std', 'count'],
            'train_c_index': ['mean', 'std']
        })
        
        print(f"\n{param}:")
        print(stats)
        
        # 9b. NEW: Visualization of hyperparameter performance statistics
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
    
    # Create summary table for overall analysis
    summary = pd.DataFrame({
        'Feature %': [pct * 100 for pct in percentages],
        'Num Features': num_features,
        'Test C-index': [f"{m:.4f} ± {s:.4f}" for m, s in zip(test_means, test_stds)],
        'Train C-index': [f"{m:.4f} ± {s:.4f}" for m, s in zip(train_means, train_stds)],
        'Train-Test Gap': [f"{t-v:.4f}" for t, v in zip(train_means, test_means)]
    })
    
    print("\n=== Summary by Feature Percentage ===")
    print(summary.to_string(index=False))
    
    # 10. NEW: Correlation analysis between parameters and performance
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
                     'num_features', 'min_samples_leaf', 'n_estimators', 
                     'max_features_num']
    
    corr_matrix = corr_data[numeric_cols].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar=True)
    plt.title('Correlation Matrix of Model Parameters and Performance')
    plt.savefig(f"{output_dir}/correlation_matrix.png")
    plt.show()
    
    # Save summary to CSV
    summary.to_csv(f"{output_dir}/rsf_performance_summary.csv", index=False)
    
    return outer_fold_data, pct_stats

if __name__ == "__main__":
    # Example usage
    analyze_rsf_hyperparameters('rsf/rsf_results_optimal/20250331_rsf_all_fold_results.csv')
    print("\nAnalysis complete. Results saved to 'rsf/rsf_results_optimal/rsf_analysis_results' directory.")