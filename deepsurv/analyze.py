import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ast import literal_eval
import os
from mpl_toolkits.mplot3d import Axes3D

def analyze_deepsurv_hyperparameters(csv_file_path, close_plots=True):
    """
    Comprehensive analysis of DeepSurv hyperparameter tuning results.
    
    Parameters:
    -----------
    csv_file_path : str
        Path to the CSV file with hyperparameter tuning results
    close_plots : bool, default=True
        If True, plots are automatically closed after saving.
        If False, plots remain open until manually closed by the user.
    """
    print(f"Loading data from {csv_file_path}")
    # Load the data
    data = pd.read_csv(csv_file_path)
    
    # Create output directory
    output_dir = 'deepsurv/deepsurv_analysis_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Clean data: convert layers from string to proper format
    data['layers'] = data['layers'].apply(literal_eval)
    data['layers_str'] = data['layers'].apply(lambda x: 'x'.join(str(i) for i in x))
    data['num_layers'] = data['layers'].apply(len)
    data['total_neurons'] = data['layers'].apply(sum)
    
    # Remove rows with -inf performance (failed models)
    valid_data = data[data['val_ci'] != -float('inf')].copy()
    print(f"Removed {len(data) - len(valid_data)} failed runs (infinite loss)")
    
    # Basic statistics
    print("\n=== Overall Performance Statistics ===")
    print(valid_data['val_ci'].describe())
    
    # 1. Performance by network architecture
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='layers_str', y='val_ci', data=valid_data)
    plt.title('Validation C-index by Network Architecture')
    plt.xlabel('Network Architecture')
    plt.ylabel('Validation C-index')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/val_ci_by_architecture.png")
    plt.show(block=False) if not close_plots else plt.close()
    
    # 2. Performance by dropout
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='dropout', y='val_ci', data=valid_data)
    plt.title('Validation C-index by Dropout Rate')
    plt.xlabel('Dropout Rate')
    plt.ylabel('Validation C-index')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/val_ci_by_dropout.png")
    plt.show(block=False) if not close_plots else plt.close()
    
    # 3. Performance by learning rate
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='learning_rate', y='val_ci', data=valid_data)
    plt.title('Validation C-index by Learning Rate')
    plt.xlabel('Learning Rate')
    plt.ylabel('Validation C-index')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/val_ci_by_learning_rate.png")
    plt.show(block=False) if not close_plots else plt.close()
    
    # 4. Performance by weight decay
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='weight_decay', y='val_ci', data=valid_data)
    plt.title('Validation C-index by Weight Decay')
    plt.xlabel('Weight Decay (L2 Regularization)')
    plt.ylabel('Validation C-index')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/val_ci_by_weight_decay.png")
    plt.show(block=False) if not close_plots else plt.close()
    
    # 5. Learning rate vs weight decay heatmap
    plt.figure(figsize=(10, 8))
    pivot = valid_data.pivot_table(
        values='val_ci', 
        index='learning_rate', 
        columns='weight_decay', 
        aggfunc='mean'
    )
    sns.heatmap(pivot, annot=True, cmap='viridis', fmt='.3f')
    plt.title('Mean Validation C-index: Learning Rate vs Weight Decay')
    plt.savefig(f"{output_dir}/heatmap_lr_vs_wd.png")
    plt.show(block=False) if not close_plots else plt.close()
    
    # 6. Dropout vs learning rate heatmap
    plt.figure(figsize=(10, 8))
    pivot = valid_data.pivot_table(
        values='val_ci', 
        index='dropout', 
        columns='learning_rate', 
        aggfunc='mean'
    )
    sns.heatmap(pivot, annot=True, cmap='viridis', fmt='.3f')
    plt.title('Mean Validation C-index: Dropout vs Learning Rate')
    plt.savefig(f"{output_dir}/heatmap_dropout_vs_lr.png")
    plt.show(block=False) if not close_plots else plt.close()
    
    # 7. Architecture vs dropout heatmap
    plt.figure(figsize=(12, 8))
    pivot = valid_data.pivot_table(
        values='val_ci', 
        index='layers_str', 
        columns='dropout', 
        aggfunc='mean'
    )
    sns.heatmap(pivot, annot=True, cmap='viridis', fmt='.3f')
    plt.title('Mean Validation C-index: Architecture vs Dropout')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/heatmap_arch_vs_dropout.png")
    plt.show(block=False) if not close_plots else plt.close()
    
    # 8. 3D scatter plot: learning_rate vs weight_decay vs val_ci
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    markers = ['o', 's', '^', 'D']
    layer_types = valid_data['layers_str'].unique()
    
    for i, layer_type in enumerate(layer_types):
        subset = valid_data[valid_data['layers_str'] == layer_type]
        ax.scatter(
            subset['learning_rate'],
            subset['weight_decay'],
            subset['val_ci'],
            label=layer_type,
            marker=markers[i % len(markers)],
            s=80,
            alpha=0.7
        )
    
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Weight Decay')
    ax.set_zlabel('Validation C-index')
    ax.set_title('3D Parameter Space: Learning Rate vs Weight Decay vs Performance')
    ax.legend()
    
    plt.savefig(f"{output_dir}/3d_lr_wd_performance.png")
    plt.show(block=False) if not close_plots else plt.close()
    
    # 9. 3D scatter plot: dropout vs total_neurons vs val_ci
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    sc = ax.scatter(
        valid_data['dropout'],
        valid_data['total_neurons'],
        valid_data['val_ci'],
        c=valid_data['num_layers'],
        cmap='viridis',
        s=80,
        alpha=0.7
    )
    
    ax.set_xlabel('Dropout Rate')
    ax.set_ylabel('Total Neurons')
    ax.set_zlabel('Validation C-index')
    ax.set_title('3D Parameter Space: Dropout vs Network Size vs Performance')
    
    cbar = fig.colorbar(sc, ax=ax, pad=0.1)
    cbar.set_label('Number of Layers')
    
    plt.savefig(f"{output_dir}/3d_dropout_neurons_performance.png")
    plt.show(block=False) if not close_plots else plt.close()
    
    # 10. Best configuration analysis
    best_config = valid_data.loc[valid_data['val_ci'].idxmax()]
    
    print("\n=== Best Configuration ===")
    print(f"Validation C-index: {best_config['val_ci']:.4f}")
    print(f"Network Architecture: {best_config['layers_str']}")
    print(f"Dropout: {best_config['dropout']}")
    print(f"Learning Rate: {best_config['learning_rate']}")
    print(f"Weight Decay: {best_config['weight_decay']}")
    
    # 11. Summary statistics by hyperparameter
    print("\n=== Mean Performance by Hyperparameter Value ===")
    
    for param in ['layers_str', 'dropout', 'learning_rate', 'weight_decay']:
        stats = valid_data.groupby(param).agg({
            'val_ci': ['mean', 'std', 'count', 'max']
        })
        
        print(f"\n{param}:")
        print(stats)
        
        # Visualization of hyperparameter performance statistics
        plt.figure(figsize=(12, 7))
        
        param_values = stats.index.tolist()
        means = stats[('val_ci', 'mean')].values
        stds = stats[('val_ci', 'std')].values
        
        plt.bar(range(len(param_values)), means, yerr=stds, capsize=8, alpha=0.7)
        plt.xticks(range(len(param_values)), param_values, rotation=45 if param == 'layers_str' else 0)
        plt.xlabel(param)
        plt.ylabel('Mean Validation C-index')
        plt.title(f'Performance by {param} with Standard Deviation')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        for i, (mean, std) in enumerate(zip(means, stds)):
            plt.annotate(f'{mean:.3f} ± {std:.3f}', 
                        xy=(i, mean), 
                        xytext=(0, 10), 
                        textcoords='offset points',
                        ha='center', 
                        va='bottom')
        
        plt.savefig(f"{output_dir}/stats_by_{param}.png")
        plt.show(block=False) if not close_plots else plt.close()
    
    # 12. Correlation analysis
    numeric_data = pd.get_dummies(valid_data[['dropout', 'learning_rate', 'weight_decay', 'val_ci', 'num_layers', 'total_neurons']])
    corr_matrix = numeric_data.corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar=True)
    plt.title('Correlation Matrix of Model Parameters and Performance')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/correlation_matrix.png")
    plt.show(block=False) if not close_plots else plt.close()
    
    # 13. Distribution of validation C-index
    plt.figure(figsize=(12, 6))
    sns.histplot(valid_data['val_ci'], kde=True, bins=20)
    plt.axvline(x=best_config['val_ci'], color='r', linestyle='--', 
                label=f'Best: {best_config["val_ci"]:.4f}')
    plt.axvline(x=valid_data['val_ci'].mean(), color='g', linestyle='--', 
                label=f'Mean: {valid_data["val_ci"].mean():.4f}')
    plt.title('Distribution of Validation C-index')
    plt.xlabel('Validation C-index')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/val_ci_distribution.png")
    plt.show(block=False) if not close_plots else plt.close()
    
    # 14. Failure analysis
    failure_analysis = {}
    
    for param in ['layers_str', 'dropout', 'learning_rate', 'weight_decay']:
        param_counts = data.groupby(param).size()
        failed_counts = data[data['val_ci'] == -float('inf')].groupby(param).size()
        
        # Fill in zeros for parameter values with no failures
        for val in param_counts.index:
            if val not in failed_counts:
                failed_counts[val] = 0
        
        failure_rates = failed_counts / param_counts
        
        failure_analysis[param] = pd.DataFrame({
            'total_runs': param_counts,
            'failed_runs': failed_counts,
            'failure_rate': failure_rates
        })
        
        plt.figure(figsize=(12, 7))
        plt.bar(range(len(failure_rates)), failure_rates.values, alpha=0.7)
        plt.xticks(range(len(failure_rates)), failure_rates.index, 
                   rotation=45 if param == 'layers_str' else 0)
        plt.xlabel(param)
        plt.ylabel('Failure Rate')
        plt.title(f'Failure Rate by {param}')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        for i, rate in enumerate(failure_rates.values):
            plt.annotate(f'{rate:.2f}', 
                        xy=(i, rate), 
                        xytext=(0, 5), 
                        textcoords='offset points',
                        ha='center', 
                        va='bottom')
        
        plt.savefig(f"{output_dir}/failure_rate_by_{param}.png")
        plt.show(block=False) if not close_plots else plt.close()
    
    print("\n=== Failure Analysis ===")
    for param, analysis in failure_analysis.items():
        print(f"\n{param} failure rates:")
        print(analysis)
    
    # If plots are kept open, wait for user to close them
    if not close_plots:
        print("\nPlots are now displayed. Close plot windows to continue...")
        plt.show()  # This will block until all plots are closed
    
    return valid_data

if __name__ == "__main__":
    # Example usage with DeepSurv data
    analyze_deepsurv_hyperparameters('deepsurv/20250409_deepsurv_hyperparam_search_results.csv', close_plots=False)
    print("\nAnalysis complete. Results saved to 'deepsurv/deepsurv_analysis_results' directory.")