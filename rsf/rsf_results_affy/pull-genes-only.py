import pandas as pd

imp = pd.read_csv("/Users/owensun/Downloads/code/nsclc-adj-chemo/rsf/rsf_results_affy/Affy RS_rsf_preselection_importances_1SE.csv")

# get first column only
genes = imp.iloc[:, 0].tolist()

print("Number of genes:", len(genes))
print("Genes:", genes)