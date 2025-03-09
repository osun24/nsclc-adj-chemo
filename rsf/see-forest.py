import numpy as np
import joblib
import matplotlib.colors as mcolors
import networkx as nx
from pyvis.network import Network
from matplotlib import cm
import pandas as pd

# Import RSF model
rsf = joblib.load('rsf_model-300-c0.641.pkl')

def calculate_survival_stats(survival_function):
    # Extract times and corresponding survival probabilities
    times = [t for t, s in survival_function]
    probs = [s for t, s in survival_function]
    
    # Median survival time: the first time the probability drops to 0.5 or below
    median_survival_time = next((time for time, prob in survival_function if prob <= 0.5), None)
    
    # If median survival time is still None, check for a sharp drop
    if median_survival_time is None:
        # Check if there's an immediate drop from initial probability to 0
        if probs and probs[0] == 1 and probs[-1] == 0:
            median_survival_time = times[-1]  # Use last time as the drop point
        elif times:
            median_survival_time = max(times)  # Fallback to max time
    
    # Minimum and maximum survival times (based on time points where survival is > 0)
    min_survival_time = min(times) if times else "N/A"
    max_survival_time = max(times) if times else "N/A"
    
    return min_survival_time, median_survival_time, max_survival_time

def build_graph(tree_, node_id, G, feature_names, root_fixed=False):
    node_id = int(node_id)  # Ensure node_id is a native Python int

    left_child = tree_.children_left[node_id]
    right_child = tree_.children_right[node_id]
    threshold = tree_.threshold[node_id]
    feature = tree_.feature[node_id]

    # Root node color and fixed position option
    if node_id == 0:
        color = 'green'
        feature_name = feature_names[feature] if feature != -2 else "No Feature"
        label = f"Root Node: {feature_name} ≤ {threshold:.2f}" if feature != -2 else "Root Node"
        G.add_node(node_id, label=label, color=color, physics=not root_fixed)

    # Check if it's a leaf node
    elif left_child == -1 and right_child == -1:
        # Calculate global min and max values across all leaf nodes
        all_values = [tree_.value[n] for n in range(tree_.node_count) 
                      if tree_.children_left[n] == -1 and tree_.children_right[n] == -1]
        global_min = np.min([np.min(value) for value in all_values])
        global_max = np.max([np.max(value) for value in all_values])

        # Create a custom colormap from red to black
        red_to_black = mcolors.LinearSegmentedColormap.from_list("RedBlack", ["#FF0000", "black"])

        # It's a leaf node
        value = tree_.value[node_id]

        # Calculate mean and median of leaf node values
        value_mean = np.mean(value)
        value_median = np.median(value)
        value_count = np.sum(tree_.n_node_samples[node_id])
        
        # Use the custom colormap to assign colors based on the mean value
        cmap = red_to_black
        norm_value = (value_mean - global_min) / (global_max - global_min)  # Normalize
        color = cm.colors.to_hex(cmap(norm_value))  # Convert color to hex for pyvis

        # Set the label with mean and median values
        min_surv, median_surv, max_surv = calculate_survival_stats(value)
        label = f"Leaf node\nMin: {min_surv:.2f}\nMedian: {median_surv:.2f}\nMax: {max_surv:.2f}\nCount: {value_count}"
        title = f"Full Value List: {value}"

        # Add the node with the calculated color
        G.add_node(node_id, label=label, title=title, color=color)
    else:
        # Decision node
        feature_name = feature_names[feature]
        label = f"{feature_name} ≤ {threshold:.2f}"

        color = 'blue'
        G.add_node(node_id, label=label, color=color)

    # Add edges with labels indicating the condition direction
    if left_child != -1:
        # Left child corresponds to the condition being True (≤ threshold)
        G.add_edge(node_id, left_child, label=f"True", color='green')
        build_graph(tree_, left_child, G, feature_names, root_fixed)
    if right_child != -1:
        # Right child corresponds to the condition being False (> threshold)
        G.add_edge(node_id, right_child, label=f"False", color='red')
        build_graph(tree_, right_child, G, feature_names, root_fixed)

# Select a tree to visualize (e.g., the first tree)
tree = rsf.estimators_[0]

# Build the graph
G = nx.DiGraph()

# Data preprocessing (unchanged)
surv = pd.read_csv('GPL570merged.csv')
surv = pd.get_dummies(surv, columns=["Stage", "Histology", "Race"])
surv = surv.drop(columns=['PFS_MONTHS','RFS_MONTHS'])
print(surv.columns[surv.isna().any()].tolist())
print(surv['Smoked?'].isna().sum())  # 121
surv = surv.dropna()  # left with 457 samples

top_100 =  [
    "TP53",
    "KRAS",
    "EGFR",
    "ERCC1",
    "BRCA1",
    "RRM1",
    "BRAF",
    "MET",
    "ALK",
    "STK11",
    "RB1",
    "CCNB1",
    "CCND1",
    "CDKN2A",
    "CDK4",
    "CDK6",
    "MYC",
    "BCL2",
    "BAX",
    "MLH1",
    "MSH2",
    "MSH6",
    "ATM",
    "ATR",
    "CHEK1",
    "CHEK2",
    "FANCA",
    "FANCD2",
    "XRCC1",
    "XRCC2",
    "XRCC3",
    "RAD51",
    "TYMS",
    "TUBB3",
    "ABCC1",
    "ABCB1",
    "KEAP1",
    "NFE2L2",
    "PTEN",
    "PIK3CA",
    "AKT1",
    "ERBB2",
    "FGFR1",
    "CUL3",
    "GSTM1",
    "GSTP1",
    "SOD2",
    "CASP3",
    "CASP9",
    "MDM2",
    "CDKN1A",
    "CDKN1B",
    "PARP1",
    "MTHFR",
    "DUT",
    "SLFN11",
    "PDK1",
    "MCL1",
    "CCNE1",
    "PKM",
    "HIF1A",
    "VEGFA",
    "E2F1",
    "BRCC3",
    "MRE11",
    "NBN",
    "RAD50",
    "RAD17",     # Alternative to CHEK1 duplication
    "APAF1",
    "ATG5",
    "ATG7",
    "SIRT1",
    "MTHFD2",
    "DNMT1",
    "DNMT3A",
    "TLE1",
    "SOX2",
    "NKX2-1",
    "GTF2I",
    "PRC1",
    "KDM5B",
    "SMARCA4",
    "ARID1A",
    "BRIP1",
    "POLD1",
    "POLE",
    "MCM2",
    "MCM4",
    "CDC20",
    "CDH1",
    "VIM",
    "SPARC",
    "SNAI1",
    "TWIST1",
    "ERBB3",
    "HERPUD1",
    "GAPDH",
    "ACTB",
    "CD8A",
    "CD274"
]
top_100 = list(set(top_100[:77]))
    
clinical_vars = ["Adjuvant Chemo", "Age", "Stage", "Sex", "Histology", "Race", "Smoked?"] + [col for col in surv.columns if col.startswith('Race')] + [col for col in surv.columns if col.startswith('Histology')] 
selected_covariates = list(set(top_100[:77]).union(set(clinical_vars)))
selected_covariates = [col for col in selected_covariates if col in surv.columns]
df = surv[['OS_STATUS', 'OS_MONTHS'] + selected_covariates]
        
# Check the feature index mapping
for i, estimator in enumerate(rsf.estimators_):
    feature_indices = np.unique(estimator.tree_.feature)
    print(f"Tree {i} uses features: {feature_indices}")

# MUST HAVE SAME ORDER AS TRAINING!!!!!
covariates = df.columns[2:]

#assert list(surv.columns) == covariates, "Mismatch between dataframe columns and covariates"

# Start building the graph from the root (node_id = 0) with the option to fix the root node at the top
build_graph(tree.tree_, 0, G, covariates, root_fixed=True)

# Check if the graph has nodes
if len(G.nodes) == 0:
    print("The graph is empty. Please check the tree structure.")
else:
    # Create an interactive network visualization
    net = Network(height='1600px', width='100%', directed=True)

    # Enable physics so nodes move dynamically with others when dragged
    net.barnes_hut()

    # Add nodes and edges to the network
    for node_id, data in G.nodes(data=True):
        # Safely get 'label', 'color', and 'title', providing defaults if missing
        label = data.get('label', f"Node {node_id}")
        color = data.get('color', 'gray')  # Default color if none is specified
        title = data.get('title', '')  # Empty title if none is specified

        # Add the node to the pyvis network
        net.add_node(int(node_id), label=label, color=color, title=title, physics=True)

    # Add edges to the network with labels
    for source_id, target_id, edge_data in G.edges(data=True):
        label = edge_data.get('label', '')
        color = edge_data.get('color', 'gray')
        net.add_edge(int(source_id), int(target_id), label=label, color=color)

    # Customize the network visualization to show edge labels
    net.set_options("""
    var options = {
      "physics": {
        "enabled": true,
        "stabilization": {
          "enabled": true,
          "iterations": 200,
          "updateInterval": 25
        },
        "barnesHut": {
          "gravitationalConstant": -6000,
          "centralGravity": 0.1,
          "springLength": 40,
          "springConstant": 0.04
        }
      },
      "edges": {
        "font": {
          "size": 12,
          "align": "middle"
        },
        "arrows": {
          "to": { "enabled": true, "scaleFactor": 0.5 }
        },
        "smooth": {
          "enabled": true,
          "type": "dynamic"
        }
      },
      "nodes": {
        "font": {
          "size": 14,
          "face": "Tahoma"
        }
      }
    }
    """)

    # Save the network visualization to an HTML file
    net.save_graph('rsf/rsf_tree_visualization.html')