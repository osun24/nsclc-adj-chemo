
"""
Script to re-rank features by median importance across cross-validation folds.

This script reads the combined fold permutation importance results and 
calculates the median importance for each feature across all folds,
then re-ranks features based on this median value.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path


def calculate_median_importance(input_file, output_file=None):
    """
    Calculate median importance for each feature across folds and re-rank.
    
    Parameters:
    -----------
    input_file : str
        Path to the input CSV file containing fold-wise importance scores
    output_file : str, optional
        Path to save the output CSV file. If None, creates a filename based on input.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with features ranked by median importance
    """
    # Read the data
    print(f"Reading data from: {input_file}")
    df = pd.read_csv(input_file)
    
    # Check the structure
    print(f"Data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Number of unique features: {df['Feature'].nunique()}")
    print(f"Number of folds: {df['Fold'].nunique()}")
    
    # Calculate median importance for each feature
    print("Calculating median importance across folds...")
    median_importance = df.groupby('Feature').agg({
        'Importance': ['median', 'mean', 'std', 'min', 'max', 'count']
    }).round(6)
    
    # Flatten column names
    median_importance.columns = ['Median_Importance', 'Mean_Importance', 
                                'Std_Importance', 'Min_Importance', 
                                'Max_Importance', 'Fold_Count']
    
    # Reset index to make Feature a column
    median_importance = median_importance.reset_index()
    
    # Sort by median importance in descending order (highest importance first)
    median_importance = median_importance.sort_values('Median_Importance', 
                                                     ascending=False)
    
    # Add rank column
    median_importance['Rank'] = range(1, len(median_importance) + 1)
    
    # Reorder columns for better readability
    median_importance = median_importance[['Rank', 'Feature', 'Median_Importance', 
                                         'Mean_Importance', 'Std_Importance',
                                         'Min_Importance', 'Max_Importance', 
                                         'Fold_Count']]
    
    # Display top and bottom ranked features
    print("\nTop 10 features by median importance:")
    print(median_importance.head(10).to_string(index=False))
    
    print("\nBottom 10 features by median importance:")
    print(median_importance.tail(10).to_string(index=False))
    
    # Save results
    if output_file is None:
        # Create output filename based on input filename
        input_path = Path(input_file)
        output_file = input_path.parent / f"{input_path.stem}_median_ranked.csv"
    
    print(f"\nSaving results to: {output_file}")
    median_importance.to_csv(output_file, index=False)
    
    # Print summary statistics
    print(f"\nSummary Statistics:")
    print(f"Total features: {len(median_importance)}")
    print(f"Features with positive median importance: {(median_importance['Median_Importance'] > 0).sum()}")
    print(f"Features with negative median importance: {(median_importance['Median_Importance'] < 0).sum()}")
    print(f"Median importance range: {median_importance['Median_Importance'].min():.6f} to {median_importance['Median_Importance'].max():.6f}")
    
    return median_importance


def main():
    """Main function to run the re-ranking analysis."""
    # Set the input file path
    input_file = "rsf/rsf_results_affy/Affy RS_combined_fold_permutation_importance.csv"
    
    # Check if file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found!")
        print("Please make sure you're running this script from the correct directory.")
        return
    
    try:
        # Calculate median importance and re-rank
        results = calculate_median_importance(input_file)
        
        # Optional: Create additional analysis files
        
        # Save only top features (e.g., top 100 or features with positive importance)
        top_features = results[results['Median_Importance'] > 0]
        if len(top_features) > 0:
            top_output = "Affy_top_features_median_ranked.csv"
            top_features.to_csv(top_output, index=False)
            print(f"Top {len(top_features)} features (positive importance) saved to: {top_output}")
        
        # Save top N features (e.g., top 50)
        top_n = 50
        top_n_features = results.head(top_n)
        top_n_output = f"Affy_top_{top_n}_features_median_ranked.csv"
        top_n_features.to_csv(top_n_output, index=False)
        print(f"Top {top_n} features saved to: {top_n_output}")
        
        print("\nRe-ranking completed successfully!")
        
    except Exception as e:
        print(f"Error occurred during processing: {str(e)}")
        raise


if __name__ == "__main__":
    main()