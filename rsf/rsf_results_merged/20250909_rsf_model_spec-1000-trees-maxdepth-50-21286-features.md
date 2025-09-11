# RSF Model Specification:
Model file: 20250909_rsf_model-1000-trees-maxdepth-50-21286-features.pkl
Number of features: 21286
Number of trees: 1000
Max depth: 50
min_samples_leaf: 6
max_features: 37
Random state: 42

# Performance Metrics:
Training C-index: 0.9513
Test C-index: 0.6381

## A Walk through the Forest:

### Tree Structure Statistics:
- **Number of trees**: 1000
- **Leaf nodes per tree**: 9.92 ± 0.90 (mean ± std)
- **Range of leaf nodes**: 7 to 13
- **Average leaf node size**: 12.30 ± 4.41 samples
- **Range of node sizes**: 6 to 42 samples
- **Event ratio in leaf nodes**: 0.6156 ± 0.2396

### Visualizations:
![Distribution of Leaf Nodes per Tree](20250909_leaf_nodes_distribution.png)
![Distribution of Leaf Node Sizes](20250909_node_sizes_distribution.png)
![Distribution of Event Ratios in Leaf Nodes](20250909_event_ratios_distribution.png)

### Key Findings:
- The forest consists of 1000 trees with an average of 9.9 leaf nodes per tree.
- Most leaf nodes contain between 9.0 and 14.0 samples (interquartile range).
- The event ratio distribution shows significant variation across leaf nodes.
- Some leaf nodes are heavily skewed toward events or censoring.
    
# Date: 20250909
# Time: 2025-09-09 20:49:22
