#!/usr/bin/env python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # Load the CSV file
    filename = "rsf/rsf_results_GPL570 3-13-25 RS_rsf_hyperparameter_results.csv"
    df = pd.read_csv(filename)
    
    # Print basic information
    print("Data shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("Head of the data:")
    print(df.head())
    print("\nSummary statistics:")
    print(df.describe())
    
    # ---------------------------
    # Visualization 1: Histogram of Mean Test Score
    plt.figure(figsize=(8, 6))
    plt.hist(df['mean_test_score'], bins=20, color='skyblue', edgecolor='black')
    plt.title("Distribution of Mean Test Score")
    plt.xlabel("Mean Test Score")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("hist_mean_test_score.png")
    plt.show()
    
    # ---------------------------
    # Visualization 2: Scatter Plot - n_estimators vs Mean Test Score
    plt.figure(figsize=(8, 6))
    plt.scatter(df['param_n_estimators'], df['mean_test_score'], color='darkgreen', alpha=0.7)
    plt.title("n_estimators vs Mean Test Score")
    plt.xlabel("n_estimators")
    plt.ylabel("Mean Test Score")
    plt.tight_layout()
    plt.savefig("scatter_n_estimators_mean_test_score.png")
    plt.show()
    
    # ---------------------------
    # Visualization 3: Scatter Plot - min_samples_split vs Mean Test Score
    plt.figure(figsize=(8, 6))
    plt.scatter(df['param_min_samples_split'], df['mean_test_score'], color='darkblue', alpha=0.7)
    plt.title("min_samples_split vs Mean Test Score")
    plt.xlabel("min_samples_split")
    plt.ylabel("Mean Test Score")
    plt.tight_layout()
    plt.savefig("scatter_min_samples_split_mean_test_score.png")
    plt.show()
    
    # ---------------------------
    # Visualization 4: Scatter Plot - min_samples_leaf vs Mean Test Score
    plt.figure(figsize=(8, 6))
    plt.scatter(df['param_min_samples_leaf'], df['mean_test_score'], color='maroon', alpha=0.7)
    plt.title("min_samples_leaf vs Mean Test Score")
    plt.xlabel("min_samples_leaf")
    plt.ylabel("Mean Test Score")
    plt.tight_layout()
    plt.savefig("scatter_min_samples_leaf_mean_test_score.png")
    plt.show()
    
    # ---------------------------
    # Visualization 5: Boxplot of Mean Test Score by max_features
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='param_max_features', y='mean_test_score', data=df)
    plt.title("Mean Test Score by max_features")
    plt.xlabel("max_features")
    plt.ylabel("Mean Test Score")
    plt.tight_layout()
    plt.savefig("boxplot_max_features_mean_test_score.png")
    plt.show()
    
    # ---------------------------
    # Visualization 6: Correlation Heatmap for Numerical Features
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.title("Correlation Heatmap of Numerical Features")
    plt.tight_layout()
    plt.savefig("correlation_heatmap.png")
    plt.show()

if __name__ == "__main__":
    main()
