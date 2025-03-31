import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def create_filtered_corr(df, name, threshold=0.7, output_dir='eda'):
    """
    Creates a correlation matrix heatmap showing only correlations above the threshold.
    
    Args:
        df: DataFrame containing the data
        name: Name for the plot title and saved file
        threshold: Absolute correlation threshold to include (default 0.7)
        output_dir: Directory to save output files
    """
    print(f"Computing correlation matrix for {df.shape[1]} features...")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate correlation matrix
    corr_matrix = df.corr()
    
    # Get a boolean mask for high correlations, excluding self-correlations
    mask = (np.abs(corr_matrix) > threshold) & (np.abs(corr_matrix) < 1.0)
    
    # Create DataFrame with high correlations
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                high_corr_pairs.append({
                    'Feature1': corr_matrix.columns[i],
                    'Feature2': corr_matrix.columns[j],
                    'Correlation': corr_matrix.iloc[i, j]
                })
    
    # Create DataFrame from correlation pairs and sort by absolute correlation
    high_corr_df = pd.DataFrame(high_corr_pairs)
    if len(high_corr_df) > 0:
        high_corr_df = high_corr_df.sort_values('Correlation', key=abs, ascending=False)
        
        # Save to CSV
        csv_path = os.path.join(output_dir, f"{name.replace(' ', '-')}-high-correlations.csv")
        high_corr_df.to_csv(csv_path, index=False)
        print(f"Saved {len(high_corr_df)} high correlation pairs to {csv_path}")
        
        # Extract top correlated features for visualization
        if len(high_corr_df) > 100:
            print(f"Too many high correlations ({len(high_corr_df)}), showing top 100")
            top_features = set()
            for _, row in high_corr_df.head(100).iterrows():
                top_features.add(row['Feature1'])
                top_features.add(row['Feature2'])
            top_features = list(top_features)
        else:
            top_features = list(set(high_corr_df['Feature1'].tolist() + high_corr_df['Feature2'].tolist()))
        
        print(f"Creating heatmap with {len(top_features)} highly correlated features")
        
        # Create a smaller correlation matrix with just these features
        small_corr = corr_matrix.loc[top_features, top_features]
        
        # Plot the heatmap
        plt.figure(figsize=(16, 14))
        sns.heatmap(small_corr, annot=False, cmap='coolwarm', center=0, 
                   linewidths=0.5, square=True, xticklabels=True, yticklabels=True)
        plt.title(f'High Correlation Features (|r| > {threshold}) - {name}', fontsize=16)
        plt.xticks(rotation=90, fontsize=8)
        plt.yticks(fontsize=8)
        plt.tight_layout()
        
        # Save the figure
        fig_path = os.path.join(output_dir, f"{name.replace(' ', '-')}-high-correlations.png")
        plt.savefig(fig_path, dpi=300)
        print(f"Saved correlation heatmap to {fig_path}")
        plt.close()
        
        # Print top 10 highest correlations
        print("\nTop 10 highest absolute correlations:")
        print(high_corr_df.head(10))
    else:
        print(f"No correlations above threshold {threshold} found.")

if __name__ == "__main__":
    print("Loading training data...")
    train_df = pd.read_csv("allTrain.csv")
    
    # Remove non-feature columns
    exclude_cols = ['OS_STATUS', 'OS_MONTHS']
    feature_df = train_df.drop(columns=exclude_cols)
    
    # First try with high threshold
    create_filtered_corr(feature_df, "ALL Features", threshold=0.8)
    
    # If needed, try with lower threshold
    create_filtered_corr(feature_df, "ALL Features Medium", threshold=0.7)
    
    print("\nAnalysis complete.")
