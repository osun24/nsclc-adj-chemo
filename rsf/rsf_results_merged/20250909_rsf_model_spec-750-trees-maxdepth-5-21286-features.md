# RSF Model Specification:
Model file: 20250909_rsf_model-750-trees-maxdepth-5-21286-features.pkl
Number of features: 21286
Number of trees: 750
Max depth: 5
min_samples_leaf: 50
max_features: 0.5
Random state: 42

# Performance Metrics:
Training C-index: 0.5000
Test C-index: 0.5000

## A Walk through the Forest:

### Tree Structure Statistics:
- **Number of trees**: 750
- **Leaf nodes per tree**: 1.00 ± 0.00 (mean ± std)
- **Range of leaf nodes**: 1 to 1
- **Average leaf node size**: 122.00 ± 0.00 samples
- **Range of node sizes**: 122 to 122 samples
- **Event ratio in leaf nodes**: 0.5902 ± 0.0000

### Visualizations:
![Distribution of Leaf Nodes per Tree](20250909_leaf_nodes_distribution.png)
![Distribution of Leaf Node Sizes](20250909_node_sizes_distribution.png)
![Distribution of Event Ratios in Leaf Nodes](20250909_event_ratios_distribution.png)

### Key Findings:
- The forest consists of 750 trees with an average of 1.0 leaf nodes per tree.
- Most leaf nodes contain between 122.0 and 122.0 samples (interquartile range).
- The event ratio distribution shows moderate homogeneity across leaf nodes.
- Leaf nodes generally maintain balanced event/censoring proportions.
    
# Date: 20250909
# Time: 2025-09-09 20:47:37
