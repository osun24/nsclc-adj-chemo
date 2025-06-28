import os
import time
import pandas as pd
import numpy as np
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.metrics import make_scorer
import joblib
import sys
import datetime
import os
import time
import datetime
import pandas as pd
import numpy as np
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
import joblib
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

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
    return concordance_index_censored(y['OS_STATUS'], y['OS_MONTHS'], y_pred)[0]

output_dir = "rsf/rsf_results_affy"  # Directory to save output files
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

current_date = datetime.datetime.now().strftime("%Y%m%d")  # Added current date for file naming

log_file = open(os.path.join(output_dir, f"{current_date}_LOG-rsf-feature-search.txt"), "w")
sys.stdout = Tee(sys.stdout, log_file)

print("Loading train data from: affyTrain.csv")
train_orig = pd.read_csv("affyTrain.csv")

print(f"Number of events in original training set: {train_orig['OS_STATUS'].sum()} | Censored cases: {train_orig.shape[0] - train_orig['OS_STATUS'].sum()}")
print("Original train data shape:", train_orig.shape)

print("Loading validation data from: affyValidation.csv")
valid_orig = pd.read_csv("affyValidation.csv")

print(f"Number of events in validation set: {valid_orig['OS_STATUS'].sum()} | Censored cases: {valid_orig.shape[0] - valid_orig['OS_STATUS'].sum()}")
print("Validation data shape:", valid_orig.shape)

# Combine train and validation datasets into one training set
print("Combining train and validation datasets...")
train = pd.concat([train_orig, valid_orig], axis=0, ignore_index=True)

print(f"Number of events in combined training set: {train['OS_STATUS'].sum()} | Censored cases: {train.shape[0] - train['OS_STATUS'].sum()}")
print("Combined training data shape:", train.shape)

start = time.time()

# Set Adjuvant Chemo's 'ACT' to 1 and 'OBS' to 0
train['Adjuvant Chemo'] = train['Adjuvant Chemo'].map({'ACT': 1, 'OBS': 0})

# Create structured arrays for survival analysis
y_train = Surv.from_dataframe('OS_STATUS', 'OS_MONTHS', train)

X_train = train.drop(columns=['OS_STATUS', 'OS_MONTHS'])

num_of_cov = 300

# {'max_depth': 10, 'max_features': 500, 'min_samples_leaf': 60, 'n_estimators': 750} 
# num_of_cov = 675; # n_estimators = 750; max_depth = 3; min_samples_leaf = 70; max_features = 0.5 (6-15-25)
rsf = RandomSurvivalForest(
    n_estimators=750,
    max_depth=3,
    min_samples_leaf=80,
    max_features=0.5,  # 0.5 * 13062 = 6531
    random_state=42,
    n_jobs=-1
)


"""
num_of_cov = 675
 n_estimators=750,
    max_depth=3,
    min_samples_leaf=70,
    max_features=0.5,  # 0.5 * 13062 = 6531
    random_state=42,
    n_jobs=-1
    
Training C-index: 0.7970
Validation C-index: 0.7119"""
"""
n_estimators = 500
max_depth = 5
min_samples_leaf = 70
max_features = 0.1  # 0.1 * 13062 = 1306
Training C-index: 0.7802
Validation C-index: 0.6985

max_depth = 4 
Training C-index: 0.7802
Validation C-index: 0.6984

max_depth = 3
Training C-index: 0.7780
Validation C-index: 0.6989

max_depth = 2
Training C-index: 0.7578
Validation C-index: 0.6947

max_depth = 1
Training C-index: 0.7041
Validation C-index: 0.6848

n_estimators=500,
    max_depth=5,
    min_samples_leaf=70,
    max_features=0.2,
    random_state=42,
    n_jobs=-1
Training C-index: 0.7884
Validation C-index: 0.7013

max_features=0.5,
Training C-index: 0.7943
Validation C-index: 0.7061

max_features = None
Training C-index: 0.7955
Validation C-index: 0.7041

maybe there is a relationship between max_features and n_estimators
as n_estimaors increases, max_features can be smaller --> more stable estimates, pushing down of spurious links
"""

""" 
param_grid = {
                "n_estimators": [500, 750, 1000],
                "min_samples_leaf": [50, 60, 70],    
                "max_features": ["sqrt", 500, 0.1], # 0.1 * 13062 = 1306
                "max_depth": [2, 3, 4, 5],
            }"""

# set covariates (Affy RS_rsf_all_fold_results_20250615.csv)
# take the top 50 features from the pre-selection
covariates = pd.read_csv("rsf/rsf_results_affy/Affy_top_features_median_ranked.csv")
# Affy RS_combined_fold_permutation_importance_median_ranked.csv

covariates = covariates['Feature'].tolist()[:num_of_cov]  # Take top 50 features

# Filter X_train to only include the covariates
X_train = X_train[covariates]

# Fit the model
print("Fitting Random Survival Forest model...")
rsf.fit(X_train, y_train)


# Print C-index on combined training data and test data
train_c_index = rsf_concordance_metric(y_train, rsf.predict(X_train))

print(f"Training C-index (train + validation combined): {train_c_index:.4f}")

# TEST 
print("Loading test data from: affyTest.csv")
test = pd.read_csv("affyTest.csv")
test['Adjuvant Chemo'] = test['Adjuvant Chemo'].map({'ACT': 1, 'OBS': 0})

# keep only the covariates used in training
test = test[covariates + ['OS_STATUS', 'OS_MONTHS']]

print(f"Number of events in test set: {test['OS_STATUS'].sum()} | Censored cases: {test.shape[0] - test['OS_STATUS'].sum()}")
print("Test data shape:", test.shape)

# Create structured arrays for test data
y_test = Surv.from_dataframe('OS_STATUS', 'OS_MONTHS', test)
X_test = test.drop(columns=['OS_STATUS', 'OS_MONTHS'])

# Evaluate C-index on test data
test_c_index = rsf_concordance_metric(y_test, rsf.predict(X_test))
print(f"Test C-index: {test_c_index:.4f}")

# Print actual max depth and number of trees
print(f"Actual max depth: {rsf.max_depth}")
print(f"Number of trees in the forest: {len(rsf.estimators_)}")

# Save the fitted model
model_file = os.path.join(output_dir, f"{current_date}_rsf_model-{len(rsf.estimators_)}-trees-maxdepth-{rsf.max_depth}-{num_of_cov}-features.pkl")
joblib.dump(rsf, model_file)
print(f"Model saved to {model_file}")

def forest_analysis(rsf, X_train, y_train):
    # Statistics for leaf nodes per tree
    leaf_nodes_per_tree = []
    all_node_sizes = []
    event_ratios = []

    # Process each tree in the forest
    for tree_idx, estimator in enumerate(rsf.estimators_):
        # Count leaf nodes (nodes with no children)
        tree = estimator.tree_
        is_leaf = tree.children_left == -1
        leaf_count = np.sum(is_leaf)
        leaf_nodes_per_tree.append(leaf_count)
        
        # Find which leaf node each training sample falls into
        leaf_ids = estimator.apply(X_train.astype(np.float32).values)
        
        # Get unique leaf node IDs
        unique_leaf_ids = np.unique(leaf_ids)
        
        for leaf_id in unique_leaf_ids:
            # Samples in this leaf
            leaf_samples = leaf_ids == leaf_id
            leaf_size = np.sum(leaf_samples)
            all_node_sizes.append(leaf_size)
            
            # Get event status for samples in this leaf
            # Need to extract the OS_STATUS from structured array
            leaf_events = y_train['OS_STATUS'][leaf_samples]
            event_ratio = np.sum(leaf_events) / leaf_size if leaf_size > 0 else 0
            event_ratios.append(event_ratio)

    # Calculate statistics
    avg_leaf_nodes = np.mean(leaf_nodes_per_tree)
    std_leaf_nodes = np.std(leaf_nodes_per_tree)
    min_leaf_nodes = min(leaf_nodes_per_tree)
    max_leaf_nodes = max(leaf_nodes_per_tree)

    avg_node_size = np.mean(all_node_sizes)
    std_node_size = np.std(all_node_sizes)
    min_node_size = min(all_node_sizes)
    max_node_size = max(all_node_sizes)

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
    plt.hist(all_node_sizes, bins=20, color='lightgreen', edgecolor='black')
    plt.title('Distribution of Leaf Node Sizes')
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
    
    forest_analysis = f"""
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
- The forest consists of {len(rsf.estimators_)} trees with an average of {avg_leaf_nodes:.1f} leaf nodes per tree
- Most leaf nodes contain between {np.percentile(all_node_sizes, 25):.1f} and {np.percentile(all_node_sizes, 75):.1f} samples (interquartile range)
- The event ratio distribution shows {'significant variation' if std_event_ratio > 0.2 else 'moderate homogeneity'} across leaf nodes
- {'Some leaf nodes are heavily skewed toward events or censoring' if max(event_ratios) > 0.9 or min(event_ratios) < 0.1 else 'Leaf nodes generally maintain balanced event/censoring proportions'}
    """
    return forest_analysis


# Save model spec, performance to md
with open(os.path.join(output_dir, f"{current_date}_rsf_model_spec-{len(rsf.estimators_)}-trees-maxdepth-{rsf.max_depth}-{num_of_cov}-features.md"), "w") as f:
    f.write(f"# RSF Model Specification:\n")
    f.write(f"Model file: {model_file}\n")
    f.write(f"Number of covariates: {num_of_cov}\n")
    f.write(f"Number of trees: {len(rsf.estimators_)}\n")
    f.write(f"Max depth: {rsf.max_depth}\n")
    f.write(f"min_samples_leaf: {rsf.min_samples_leaf}\n")
    f.write(f"max_features: {rsf.max_features}\n")
    f.write(f"min_weight_fraction_leaf: {rsf.min_weight_fraction_leaf}\n")
    f.write(f"Bootstrap: {rsf.bootstrap}\n")
    f.write(f"min_samples_split: {rsf.min_samples_split}\n")
    f.write(f"max_leaf_nodes: {rsf.max_leaf_nodes}\n")
    f.write(f"oob_score: {rsf.oob_score}\n")
    f.write(f"warm_start: {rsf.warm_start}\n")
    f.write(f"max_samples: {rsf.max_samples}\n")
    f.write(f"Random state: {rsf.random_state}\n")
    
    
    f.write(f"# Performance Metrics:\n")
    f.write(f"Training C-index (train + validation combined): {train_c_index:.4f}\n")
    f.write(f"Test C-index: {test_c_index:.4f}\n")
    
    f.write(f"Covariates \n")
    for cov in covariates:
        f.write(f"- {cov}\n")
        
    f.write(f"\n ## A Walk through the Forest:\n")
    f.write(forest_analysis(rsf, X_train, y_train))
        
    f.write(f"# Date: {current_date}\n")
    f.write(f"# Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
print(f"Model specification saved to {os.path.join(output_dir, f'{current_date}_rsf_model_spec-{len(rsf.estimators_)}-trees-maxdepth-{rsf.max_depth}.md')}")
    

""""
# Run permutation importance
perm_result = permutation_importance(rsf, X_train, y_train,
                                       scoring=lambda est, X, y: rsf_concordance_metric(y, est.predict(X)),
                                       n_repeats=5, random_state=42, n_jobs=-1)
importances = perm_result.importances_mean
importance_df = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": importances,
    "Std": perm_result.importances_std
}).sort_values(by="Importance", ascending=False)

preselect_csv = os.path.join(output_dir, f"{current_date}_rsf_preselection_importances.csv")
importance_df.to_csv(preselect_csv, index=False)
print(f"RSF pre-selection importances saved to {preselect_csv}")

top_preselect = importance_df.head(50)
plt.figure(figsize=(12, 8))
plt.barh(top_preselect["Feature"][::-1], top_preselect["Importance"][::-1],
            xerr=top_preselect["Std"][::-1], color=(9/255, 117/255, 181/255))
plt.xlabel("Permutation Importance")
plt.title("RSF Pre-Selection (Top 50 Features)")
plt.tight_layout()
preselect_plot = os.path.join(output_dir, f"{current_date}_rsf_preselection_importances_1SE.png")
plt.savefig(preselect_plot)
plt.close()
print(f"RSF pre-selection plot saved to {preselect_plot}")
"""