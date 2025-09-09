import os
import time
import pandas as pd
import numpy as np
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored
import joblib
import sys
import datetime
import matplotlib.pyplot as plt

# Redirect console output to both the terminal and a log file
class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
    def flush(self):
        for f in self.files:
            f.flush()

# Define custom concordance metric function
def rsf_concordance_metric(y, y_pred):
    """Custom concordance metric for RSF."""
    # Check if y_pred contains NaNs and handle
    if np.isnan(y_pred).any():
        print("Warning: NaNs found in predictions. Concordance may be unreliable.")
        # Option 1: Return a default low value
        # return 0.5
        # Option 2: Filter out NaNs (might change dataset size)
        valid_indices = ~np.isnan(y_pred)
        if np.sum(valid_indices) == 0:
            return 0.5 # No valid predictions
        y_pred = y_pred[valid_indices]
        y_event = y['OS_STATUS'][valid_indices]
        y_time = y['OS_MONTHS'][valid_indices]
        return concordance_index_censored(y_event, y_time, y_pred)[0]
    
    return concordance_index_censored(y['OS_STATUS'], y['OS_MONTHS'], y_pred)[0]


def clean_data(df):
    """Drops specified columns: batch labels, RFS/PFS, and sample IDs."""
    cols_to_drop = []
    
    # Identify batch labels
    batch_cols = [col for col in df.columns if 'batch' in col.lower()]
    cols_to_drop.extend(batch_cols)

    # Identify RFS or PFS columns
    rfs_pfs_cols = [col for col in df.columns if 'RFS' in col or 'PFS' in col]
    cols_to_drop.extend(rfs_pfs_cols)

    # Identify sample IDs/names and unnamed columns
    sample_id_cols = [col for col in df.columns if ('sample' in col.lower() or 'id' in col.lower() or 'unnamed' in col.lower()) and 'histology' not in col.lower()]
    cols_to_drop.extend(sample_id_cols)
    
    # Drop identified columns
    if cols_to_drop:
        # Ensure columns exist before dropping
        existing_cols_to_drop = [col for col in cols_to_drop if col in df.columns]
        if existing_cols_to_drop:
            print(f"Dropping columns: {existing_cols_to_drop}")
            df = df.drop(columns=existing_cols_to_drop)
        else:
            print("No columns to drop found in this dataframe.")
            
    return df

def forest_analysis(rsf, X_train, y_train, output_dir, current_date):
    """Performs analysis on the trained forest and returns a summary string."""
    # Statistics for leaf nodes per tree
    leaf_nodes_per_tree = [estimator.tree_.n_leaves for estimator in rsf.estimators_]
    
    # Find which leaf node each training sample falls into for all trees
    leaf_ids_all_trees = rsf.apply(X_train.astype(np.float32).values)
    
    all_node_sizes = []
    event_ratios = []

    for tree_idx, leaf_ids_tree in enumerate(leaf_ids_all_trees.T):
        unique_leaf_ids = np.unique(leaf_ids_tree)
        for leaf_id in unique_leaf_ids:
            leaf_samples_mask = leaf_ids_tree == leaf_id
            leaf_size = np.sum(leaf_samples_mask)
            all_node_sizes.append(leaf_size)
            
            leaf_events = y_train['OS_STATUS'][leaf_samples_mask]
            event_ratio = np.sum(leaf_events) / leaf_size if leaf_size > 0 else 0
            event_ratios.append(event_ratio)

    # Calculate statistics
    avg_leaf_nodes = np.mean(leaf_nodes_per_tree)
    std_leaf_nodes = np.std(leaf_nodes_per_tree)
    min_leaf_nodes, max_leaf_nodes = np.min(leaf_nodes_per_tree), np.max(leaf_nodes_per_tree)

    avg_node_size = np.mean(all_node_sizes)
    std_node_size = np.std(all_node_sizes)
    min_node_size, max_node_size = np.min(all_node_sizes), np.max(all_node_sizes)

    avg_event_ratio = np.mean(event_ratios)
    std_event_ratio = np.std(event_ratios)

    # Create visualizations
    # 1. Distribution of leaf nodes per tree
    plt.figure(figsize=(10, 6))
    plt.hist(leaf_nodes_per_tree, bins=20, color='skyblue', edgecolor='black')
    plt.title('Distribution of Leaf Nodes per Tree')
    plt.xlabel('Number of Leaf Nodes')
    plt.ylabel('Frequency')
    plt.grid(alpha=0.3)
    leaf_nodes_hist_path = os.path.join(output_dir, f"{current_date}_leaf_nodes_distribution.png")
    plt.savefig(leaf_nodes_hist_path)
    plt.close()

    # 2. Distribution of node sizes
    plt.figure(figsize=(10, 6))
    plt.hist(all_node_sizes, bins=50, color='lightgreen', edgecolor='black', range=(0, np.percentile(all_node_sizes, 99)))
    plt.title('Distribution of Leaf Node Sizes (up to 99th percentile)')
    plt.xlabel('Node Size (# of samples)')
    plt.ylabel('Frequency')
    plt.grid(alpha=0.3)
    node_sizes_hist_path = os.path.join(output_dir, f"{current_date}_node_sizes_distribution.png")
    plt.savefig(node_sizes_hist_path)
    plt.close()

    # 3. Distribution of event ratios
    plt.figure(figsize=(10, 6))
    plt.hist(event_ratios, bins=20, color='salmon', edgecolor='black')
    plt.title('Distribution of Event Ratios in Leaf Nodes')
    plt.xlabel('Event Ratio (proportion of events)')
    plt.ylabel('Frequency')
    plt.grid(alpha=0.3)
    event_ratios_hist_path = os.path.join(output_dir, f"{current_date}_event_ratios_distribution.png")
    plt.savefig(event_ratios_hist_path)
    plt.close()
    
    analysis_summary = f"""
### Tree Structure Statistics:
- **Number of trees**: {len(rsf.estimators_)}
- **Leaf nodes per tree**: {avg_leaf_nodes:.2f} ± {std_leaf_nodes:.2f} (mean ± std)
- **Range of leaf nodes**: {min_leaf_nodes} to {max_leaf_nodes}
- **Average leaf node size**: {avg_node_size:.2f} ± {std_node_size:.2f} samples
- **Range of node sizes**: {min_node_size} to {max_node_size} samples
- **Event ratio in leaf nodes**: {avg_event_ratio:.4f} ± {std_event_ratio:.4f}

### Visualizations:
![Distribution of Leaf Nodes per Tree]({os.path.basename(leaf_nodes_hist_path)})
![Distribution of Leaf Node Sizes]({os.path.basename(node_sizes_hist_path)})
![Distribution of Event Ratios in Leaf Nodes]({os.path.basename(event_ratios_hist_path)})

### Key Findings:
- The forest consists of {len(rsf.estimators_)} trees with an average of {avg_leaf_nodes:.1f} leaf nodes per tree.
- Most leaf nodes contain between {np.percentile(all_node_sizes, 25):.1f} and {np.percentile(all_node_sizes, 75):.1f} samples (interquartile range).
- The event ratio distribution shows {'significant variation' if std_event_ratio > 0.2 else 'moderate homogeneity'} across leaf nodes.
- {'Some leaf nodes are heavily skewed toward events or censoring' if max(event_ratios) > 0.9 or min(event_ratios) < 0.1 else 'Leaf nodes generally maintain balanced event/censoring proportions'}.
    """
    return analysis_summary

def main():
    """Main function to run the RSF analysis."""
    # --- Setup ---
    output_dir = "rsf/rsf_results_merged"
    os.makedirs(output_dir, exist_ok=True)

    current_date = datetime.datetime.now().strftime("%Y%m%d")
    log_file = open(os.path.join(output_dir, f"{current_date}_LOG-rsf-replication.txt"), "w")
    sys.stdout = Tee(sys.stdout, log_file)
    
    start_time = time.time()

    # --- Data Loading and Cleaning ---
    print("--- Loading and Cleaning Data ---")
    try:
        train = pd.read_csv("train_merged.csv")
        test = pd.read_csv("test_merged.csv")
        print("Original train data shape:", train.shape)
        print("Original test data shape:", test.shape)
    except FileNotFoundError as e:
        print(f"Error: {e}. Make sure 'train_merged.csv' and 'test_merged.csv' are in the correct directory.")
        sys.exit(1)

    train = clean_data(train)
    test = clean_data(test)
    print("Cleaned train data shape:", train.shape)
    print("Cleaned test data shape:", test.shape)

    # --- Preprocessing ---
    print("\n--- Preprocessing Data ---")
    
    # Handle categorical variables before creating survival arrays
    categorical_cols = train.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        print(f"One-hot encoding categorical columns: {list(categorical_cols)}")
        train = pd.get_dummies(train, columns=categorical_cols, dummy_na=False, dtype=float)
        test = pd.get_dummies(test, columns=categorical_cols, dummy_na=False, dtype=float)
    
    # Align columns after one-hot encoding, before creating X/y splits
    train_cols = set(train.columns)
    test_cols = set(test.columns)
    
    missing_in_test = list(train_cols - test_cols)
    if 'OS_STATUS' in missing_in_test: missing_in_test.remove('OS_STATUS')
    if 'OS_MONTHS' in missing_in_test: missing_in_test.remove('OS_MONTHS')
    
    missing_in_train = list(test_cols - train_cols)
    if 'OS_STATUS' in missing_in_train: missing_in_train.remove('OS_STATUS')
    if 'OS_MONTHS' in missing_in_train: missing_in_train.remove('OS_MONTHS')

    for col in missing_in_test:
        test[col] = 0.0
    for col in missing_in_train:
        train[col] = 0.0
        
    # Ensure order is the same
    test = test[train.columns]

    # Create structured arrays for survival analysis
    y_train = Surv.from_dataframe('OS_STATUS', 'OS_MONTHS', train)
    X_train = train.drop(columns=['OS_STATUS', 'OS_MONTHS'])

    y_test = Surv.from_dataframe('OS_STATUS', 'OS_MONTHS', test)
    X_test = test.drop(columns=['OS_STATUS', 'OS_MONTHS'])

    print(f"Training set features: {X_train.shape[1]}")
    print(f"Test set features: {X_test.shape[1]}")

    # --- Model Training ---
    print("\n--- Model Training ---")
    rsf = RandomSurvivalForest(
        n_estimators=750,
        max_depth=5,
        min_samples_leaf=50,
        max_features=0.5,
        random_state=42,
        n_jobs=-1
    )

    print("Fitting Random Survival Forest model...")
    rsf.fit(X_train, y_train)

    # --- Evaluation ---
    print("\n--- Model Evaluation ---")
    train_c_index = rsf_concordance_metric(y_train, rsf.predict(X_train))
    print(f"Training C-index: {train_c_index:.4f}")

    test_c_index = rsf_concordance_metric(y_test, rsf.predict(X_test))
    print(f"Test C-index: {test_c_index:.4f}")

    # --- Save Artifacts ---
    print("\n--- Saving Artifacts ---")
    num_features = X_train.shape[1]
    model_file = os.path.join(output_dir, f"{current_date}_rsf_model-{len(rsf.estimators_)}-trees-maxdepth-{rsf.max_depth}-{num_features}-features.pkl")
    joblib.dump(rsf, model_file)
    print(f"Model saved to {model_file}")

    # --- Analysis and Reporting ---
    print("\n--- Forest Analysis ---")
    analysis_summary = forest_analysis(rsf, X_train, y_train, output_dir, current_date)

    spec_file = os.path.join(output_dir, f"{current_date}_rsf_model_spec-{len(rsf.estimators_)}-trees-maxdepth-{rsf.max_depth}-{num_features}-features.md")
    with open(spec_file, "w") as f:
        f.write(f"# RSF Model Specification:\n")
        f.write(f"Model file: {os.path.basename(model_file)}\n")
        f.write(f"Number of features: {num_features}\n")
        f.write(f"Number of trees: {len(rsf.estimators_)}\n")
        f.write(f"Max depth: {rsf.max_depth}\n")
        f.write(f"min_samples_leaf: {rsf.min_samples_leaf}\n")
        f.write(f"max_features: {rsf.max_features}\n")
        f.write(f"Random state: {rsf.random_state}\n\n")
        
        f.write(f"# Performance Metrics:\n")
        f.write(f"Training C-index: {train_c_index:.4f}\n")
        f.write(f"Test C-index: {test_c_index:.4f}\n\n")
        
        f.write(f"## A Walk through the Forest:\n")
        f.write(analysis_summary)
            
        f.write(f"\n# Date: {current_date}\n")
        f.write(f"# Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"Model specification saved to {os.path.basename(spec_file)}")

    end_time = time.time()
    print(f"\nScript finished in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
