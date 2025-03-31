import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def compute_cv(series):
    """Compute coefficient of variation for a pandas Series."""
    mean = series.mean()
    std = series.std()
    return std / mean if mean != 0 else np.nan

def compute_geNorm_M(df, candidate_genes):
    """
    Compute geNorm stability measure M for each candidate gene.
    For each gene, M is defined as the average standard deviation of the log2 expression ratios
    between that gene and all other candidate genes.
    """
    M_values = {}
    for gene in candidate_genes:
        sd_list = []
        for other_gene in candidate_genes:
            if gene != other_gene:
                # Calculate log2 ratio for each sample
                log_ratio = np.log2(df[gene] / df[other_gene])
                sd_list.append(log_ratio.std())
        M_values[gene] = np.mean(sd_list)
    return M_values

def main():
    # Replace 'gene_expression.csv' with your file path.
    # The CSV should have sample IDs as rows and gene symbols as columns.
    data = pd.read_csv('all.merged.csv', index_col=0)
    
    # List of candidate housekeeping genes (common examples)
    candidate_genes = ["ACTB", "GAPDH", "B2M", "UBC", "TBP"]
    # Filter candidate genes that are available in the dataset
    candidate_genes = [gene for gene in candidate_genes if gene in data.columns]
    if not candidate_genes:
        print("None of the candidate housekeeping genes were found in the dataset!")
        return

    # Compute CV for each candidate gene
    cv_values = {gene: compute_cv(data[gene]) for gene in candidate_genes}
    # Compute geNorm M values for the candidate genes
    M_values = compute_geNorm_M(data, candidate_genes)
    
    # Print stability metrics
    print("Housekeeping Gene Stability Evaluation:")
    print("Gene\t\tCV\t\tgeNorm M")
    for gene in candidate_genes:
        print(f"{gene:8s}\t{cv_values[gene]:.4f}\t\t{M_values[gene]:.4f}")
    
    # Plot the results for a visual comparison
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    ax[0].bar(cv_values.keys(), cv_values.values(), color='skyblue')
    ax[0].set_title("Coefficient of Variation (CV)")
    ax[0].set_ylabel("CV")
    
    ax[1].bar(M_values.keys(), M_values.values(), color='salmon')
    ax[1].set_title("geNorm Stability Measure (M)")
    ax[1].set_ylabel("M value")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()