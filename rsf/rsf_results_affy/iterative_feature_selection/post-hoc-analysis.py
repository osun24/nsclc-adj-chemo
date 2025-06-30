"""
Post-hoc Analysis: Feature vs Performance Visualization with Log Scale

This script creates enhanced visualizations of the iterative feature selection results,
specifically focusing on the relationship between number of features and model performance
using logarithmic scaling for better visualization of the feature reduction process.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
import os

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_iteration_data(csv_path="rsf/rsf_results_affy/iterative_feature_selection/20250628_iteration_summary.csv"):
    """Load the iteration summary data"""
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} iterations from {csv_path}")
        return df
    except FileNotFoundError:
        print(f"File {csv_path} not found. Please ensure the file exists in the current directory.")
        return None

def create_log_scale_performance_plot(df, save_dir="./"):
    """
    Create feature vs performance plots with logarithmic scaling for number of features
    """
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Log scale - Performance vs Features (Primary plot)
    ax1.errorbar(df['N_Features'], df['Mean_Test_C_Index'], 
                yerr=df['Std_Test_C_Index'], marker='o', capsize=5, 
                label='Test C-Index', linewidth=2, markersize=8, color='#1f77b4')
    ax1.errorbar(df['N_Features'], df['Mean_Train_C_Index'], 
                yerr=df['Std_Train_C_Index'], marker='s', capsize=5, 
                label='Train C-Index', linewidth=2, markersize=8, color='#ff7f0e')
    
    # Find and highlight best performance
    best_idx = df['Mean_Test_C_Index'].idxmax()
    best_n_features = df.loc[best_idx, 'N_Features']
    best_score = df.loc[best_idx, 'Mean_Test_C_Index']
    ax1.scatter(best_n_features, best_score, color='red', s=200, marker='*', 
               label=f'Best: {best_score:.4f} ({best_n_features} features)', zorder=5)
    
    ax1.set_xscale('log')
    ax1.set_xlabel('Number of Features (log scale)', fontsize=12)
    ax1.set_ylabel('C-Index', fontsize=12)
    ax1.set_title('Performance vs Number of Features (Log Scale)', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Add annotations for key points
    # Annotate best point
    ax1.annotate(f'Best\nIter {df.loc[best_idx, "Iteration"]}', 
                xy=(best_n_features, best_score),
                xytext=(best_n_features*0.3, best_score + 0.01),
                arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                fontsize=10, ha='center')
    
    # Plot 2: Overfitting Analysis with Log Scale
    overfitting_gap = df['Mean_Train_C_Index'] - df['Mean_Test_C_Index']
    ax2.plot(df['N_Features'], overfitting_gap, 'ro-', linewidth=2, markersize=6, alpha=0.7)
    ax2.set_xscale('log')
    ax2.set_xlabel('Number of Features (log scale)', fontsize=12)
    ax2.set_ylabel('Overfitting Gap (Train - Test C-Index)', fontsize=12)
    ax2.set_title('Overfitting vs Number of Features (Log Scale)', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # Add horizontal line at optimal gap
    optimal_gap = np.median(overfitting_gap)
    ax2.axhline(y=optimal_gap, linestyle='--', color='green', alpha=0.7, 
               label=f'Median Gap: {optimal_gap:.3f}')
    ax2.legend()
    
    # Plot 3: Linear scale for comparison
    ax3.errorbar(df['N_Features'], df['Mean_Test_C_Index'], 
                yerr=df['Std_Test_C_Index'], marker='o', capsize=5, 
                label='Test C-Index', linewidth=2, markersize=6, color='#1f77b4')
    ax3.errorbar(df['N_Features'], df['Mean_Train_C_Index'], 
                yerr=df['Std_Train_C_Index'], marker='s', capsize=5, 
                label='Train C-Index', linewidth=2, markersize=6, color='#ff7f0e')
    ax3.scatter(best_n_features, best_score, color='red', s=150, marker='*', 
               label=f'Best: {best_score:.4f}', zorder=5)
    
    ax3.set_xlabel('Number of Features (linear scale)', fontsize=12)
    ax3.set_ylabel('C-Index', fontsize=12)
    ax3.set_title('Performance vs Number of Features (Linear Scale)', fontsize=14)
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Performance improvement rate
    # Calculate improvement rate (derivative approximation)
    test_scores = df['Mean_Test_C_Index'].values
    n_features = df['N_Features'].values
    
    # Calculate rate of improvement per feature removed
    improvement_rates = []
    for i in range(1, len(test_scores)):
        delta_score = test_scores[i] - test_scores[i-1]
        delta_features = n_features[i-1] - n_features[i]  # Features removed
        if delta_features > 0:
            rate = delta_score / delta_features * 1000  # Per 1000 features removed
            improvement_rates.append(rate)
        else:
            improvement_rates.append(0)
    
    # Plot improvement rate
    if len(improvement_rates) > 0:
        ax4.plot(df['N_Features'][1:], improvement_rates, 'go-', linewidth=2, markersize=6, alpha=0.7)
        ax4.set_xscale('log')
        ax4.set_xlabel('Number of Features (log scale)', fontsize=12)
        ax4.set_ylabel('C-Index Improvement Rate\n(per 1000 features removed)', fontsize=12)
        ax4.set_title('Performance Improvement Rate', fontsize=14)
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0, linestyle='--', color='red', alpha=0.5)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(save_dir, "post_hoc_log_scale_analysis.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Log scale analysis plot saved to: {output_path}")
    plt.show()
    
    return fig

def create_detailed_log_analysis(df, save_dir="./"):
    """
    Create detailed analysis with log-scale focusing on key regions
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Zoomed log-scale plot focusing on the optimal region
    ax1.errorbar(df['N_Features'], df['Mean_Test_C_Index'], 
                yerr=df['Std_Test_C_Index'], marker='o', capsize=5, 
                label='Test C-Index', linewidth=2, markersize=8, color='#1f77b4')
    ax1.errorbar(df['N_Features'], df['Mean_Train_C_Index'], 
                yerr=df['Std_Train_C_Index'], marker='s', capsize=5, 
                label='Train C-Index', linewidth=2, markersize=8, color='#ff7f0e')
    
    # Highlight top 5 performing iterations
    top_5_idx = df.nlargest(5, 'Mean_Test_C_Index').index
    for i, idx in enumerate(top_5_idx):
        color = plt.cm.Reds(0.5 + i*0.1)
        ax1.scatter(df.loc[idx, 'N_Features'], df.loc[idx, 'Mean_Test_C_Index'], 
                   color=color, s=100, marker='*', zorder=5, alpha=0.8)
        ax1.annotate(f'#{i+1}\n({df.loc[idx, "N_Features"]}f)', 
                    xy=(df.loc[idx, 'N_Features'], df.loc[idx, 'Mean_Test_C_Index']),
                    xytext=(5, 5), textcoords='offset points', 
                    fontsize=8, alpha=0.8)
    
    ax1.set_xscale('log')
    ax1.set_xlabel('Number of Features (log scale)', fontsize=12)
    ax1.set_ylabel('C-Index', fontsize=12)
    ax1.set_title('Top 5 Performing Feature Sets (Log Scale)', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Feature reduction efficiency
    # Calculate cumulative performance gain per log-unit of feature reduction
    log_features = np.log10(df['N_Features'])
    baseline_score = df['Mean_Test_C_Index'].iloc[0]
    cumulative_gain = df['Mean_Test_C_Index'] - baseline_score
    
    ax2.plot(df['N_Features'], cumulative_gain, 'bo-', linewidth=2, markersize=6, alpha=0.7)
    ax2.set_xscale('log')
    ax2.set_xlabel('Number of Features (log scale)', fontsize=12)
    ax2.set_ylabel('Cumulative C-Index Gain\n(vs. Initial Performance)', fontsize=12)
    ax2.set_title('Cumulative Performance Gain', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, linestyle='--', color='red', alpha=0.5)
    
    # Highlight maximum gain
    max_gain_idx = cumulative_gain.idxmax()
    max_gain = cumulative_gain.iloc[max_gain_idx]
    max_gain_features = df.loc[max_gain_idx, 'N_Features']
    ax2.scatter(max_gain_features, max_gain, color='red', s=150, marker='*', zorder=5)
    ax2.annotate(f'Max Gain: +{max_gain:.4f}\n({max_gain_features} features)', 
                xy=(max_gain_features, max_gain),
                xytext=(max_gain_features*0.1, max_gain + 0.005),
                arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                fontsize=10, ha='center')
    
    plt.tight_layout()
    
    # Save detailed analysis
    output_path = os.path.join(save_dir, "post_hoc_detailed_log_analysis.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Detailed log analysis plot saved to: {output_path}")
    plt.show()
    
    return fig

def generate_summary_statistics(df):
    """Generate summary statistics for the feature selection process"""
    print("\n" + "="*60)
    print("FEATURE SELECTION SUMMARY STATISTICS")
    print("="*60)
    
    # Basic statistics
    total_features_removed = df['N_Features'].iloc[0] - df['N_Features'].iloc[-1]
    total_iterations = len(df)
    avg_features_per_iteration = total_features_removed / total_iterations
    
    print(f"Total iterations: {total_iterations}")
    print(f"Initial features: {df['N_Features'].iloc[0]:,}")
    print(f"Final features: {df['N_Features'].iloc[-1]:,}")
    print(f"Total features removed: {total_features_removed:,}")
    print(f"Average features removed per iteration: {avg_features_per_iteration:.1f}")
    
    # Performance statistics
    initial_performance = df['Mean_Test_C_Index'].iloc[0]
    final_performance = df['Mean_Test_C_Index'].iloc[-1]
    best_performance = df['Mean_Test_C_Index'].max()
    best_iteration = df.loc[df['Mean_Test_C_Index'].idxmax(), 'Iteration']
    best_n_features = df.loc[df['Mean_Test_C_Index'].idxmax(), 'N_Features']
    
    print(f"\nPerformance Statistics:")
    print(f"Initial C-Index: {initial_performance:.4f}")
    print(f"Final C-Index: {final_performance:.4f}")
    print(f"Best C-Index: {best_performance:.4f} (Iteration {best_iteration}, {best_n_features} features)")
    print(f"Total performance gain: {best_performance - initial_performance:.4f}")
    print(f"Performance gain vs final: {best_performance - final_performance:.4f}")
    
    # Feature reduction efficiency
    features_for_best = df['N_Features'].iloc[0] - best_n_features
    efficiency = (best_performance - initial_performance) / (features_for_best / 1000)
    print(f"Efficiency: {efficiency:.6f} C-Index gain per 1000 features removed")
    
    # Identify key transition points using log scale
    log_features = np.log10(df['N_Features'])
    log_ranges = [
        (4, 5, "10K-100K features"),
        (3, 4, "1K-10K features"), 
        (2, 3, "100-1K features"),
        (1, 2, "10-100 features")
    ]
    
    print(f"\nPerformance by Feature Range (Log Scale):")
    for min_log, max_log, label in log_ranges:
        mask = (log_features >= min_log) & (log_features < max_log)
        if mask.any():
            range_df = df[mask]
            avg_performance = range_df['Mean_Test_C_Index'].mean()
            std_performance = range_df['Mean_Test_C_Index'].std()
            print(f"{label}: {avg_performance:.4f} ± {std_performance:.4f} (n={len(range_df)})")

def main():
    """Main function to run post-hoc analysis"""
    print("="*80)
    print("POST-HOC ANALYSIS: FEATURE VS PERFORMANCE (LOG SCALE)")
    print("="*80)
    
    # Load data
    df = load_iteration_data()
    if df is None:
        return
    
    # Generate summary statistics
    generate_summary_statistics(df)
    
    # Create log-scale visualizations
    print(f"\nCreating log-scale performance visualizations...")
    create_log_scale_performance_plot(df)
    
    print(f"\nCreating detailed log-scale analysis...")
    create_detailed_log_analysis(df)
    
    # Additional analysis: Feature selection phases
    print(f"\n" + "="*60)
    print("FEATURE SELECTION PHASES ANALYSIS")
    print("="*60)
    
    # Identify phases based on performance improvement rate
    test_scores = df['Mean_Test_C_Index'].values
    improvement_rate = np.diff(test_scores)
    
    # Find phases where improvement rate changes significantly
    improvement_rate_change = np.diff(improvement_rate)
    significant_changes = np.where(np.abs(improvement_rate_change) > np.std(improvement_rate_change))[0]
    
    if len(significant_changes) > 0:
        print("Significant performance change points (iterations):")
        for i, change_point in enumerate(significant_changes + 2):  # +2 because of double diff
            if change_point < len(df):
                n_features = df.iloc[change_point]['N_Features']
                c_index = df.iloc[change_point]['Mean_Test_C_Index']
                print(f"  Phase {i+1}: Iteration {change_point+1}, {n_features} features, C-Index {c_index:.4f}")
    
    print(f"\nAnalysis complete! Check the generated plots:")
    print("- post_hoc_log_scale_analysis.png")
    print("- post_hoc_detailed_log_analysis.png")

if __name__ == "__main__":
    main()
