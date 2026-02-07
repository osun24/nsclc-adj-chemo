import os
import joblib
import numpy as np

MODEL_PATH = "rsf_interactions_iptw_bounded-11-22-1SE/rsf_final.joblib"
FEATURES_PATH = "rsf_interactions_iptw_bounded-11-22-1SE/features_used.txt"
FEATURE_NAME = "Adjuvant Chemo"


def main():
    rsf = joblib.load(MODEL_PATH)

    if not hasattr(rsf, "estimators_"):
        raise ValueError("Loaded model does not have estimators_.")

    if hasattr(rsf, "feature_names_in_"):
        feature_names = list(rsf.feature_names_in_)
    else:
        if not os.path.exists(FEATURES_PATH):
            raise ValueError(
                "Model does not have feature_names_in_ and features_used.txt was not found."
            )
        with open(FEATURES_PATH, "r") as f:
            feature_names = [line.strip() for line in f.readlines() if line.strip()]

    if FEATURE_NAME not in feature_names:
        print(f"Feature '{FEATURE_NAME}' not in model features.")
        print(f"Available features: {feature_names}")
        return

    target_idx = feature_names.index(FEATURE_NAME)

    trees_with_feature = 0
    total_trees = len(rsf.estimators_)
    tree_indices = []

    for i, est in enumerate(rsf.estimators_):
        tree = est.tree_
        features = tree.feature  # ndarray of feature indices
        if np.any(features == target_idx):
            trees_with_feature += 1
            tree_indices.append(i)

    print(f"Total trees: {total_trees}")
    print(f"Trees with '{FEATURE_NAME}' in any node: {trees_with_feature}")
    print(f"Proportion: {trees_with_feature / total_trees:.3f}")
    print(f"Tree indices (0-based): {tree_indices}")


if __name__ == "__main__":
    main()
