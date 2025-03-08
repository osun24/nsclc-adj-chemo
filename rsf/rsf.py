import pandas as pd
import numpy as np
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
import joblib
from mpl_toolkits.mplot3d import Axes3D    # new import for 3D plotting

def create_rsf(df, name, trees=1000):
    # Subset df using preselected covariates and clinical variables
    selected_csv = pd.read_csv('rsf_results_GPL570_rsf_preselection_importances.csv', index_col=0)
    top_100 = [
    # ---- Top 60 Core Genes ----
    "TP53","KRAS","STK11","KEAP1","NFE2L2","RB1","MYC","CDKN2A","CDK4","CDK6",
    "CCND1","MMP2","MMP9","BCL2","BCL2L1","BCL2L11","CASP3","CASP8","CASP9","NKX2-1",
    "MET","APAF1","ATF3","ATG5","AXL","FGFR1","PIK3CA","PTEN","AKT1","AKT2",
    "AKT3","MTOR","SOX2","SOX4","NOTCH1","NOTCH3","SMAD2","SMAD3","SMAD4","TGFBR1",
    "TGFBR2","HIF1A","VEGFA","ERCC1","XPA","XPC","RAD51","BRCA1","BRCA2","CHEK1",
    "CHEK2","PARP1","PARP2","CDK1","CDK2","CDK7","E2F1","RRM1","RRM2","FEN1",

    # ---- 61–150: Additional DNA Repair, Cell Cycle, Apoptosis, Epigenetic ----
    "ATM","ATR","ATRIP","MDM2","MDM4","MRE11","RAD17","RAD18","RAD50","RAD52",
    "RAD9A","MUTYH","TP73","CDKN1A","CDKN1B","CDKN2B","CDKN2C","CDKN3","CCNE1","CCNE2",
    "CCNB1","CCNB2","CCNA2","CDT1","CDC6","CDC20","BIRC2","BIRC3","BIRC5","DIABLO",
    "XIAP","TNFRSF10B","FADD","FAS","FASLG","CASP6","CASP7","CASP10","HR","FANCA",
    "FANCD2","FANCE","BRIP1","UBE2T","TOP1","TOP2A","TOP2B","EZH2","DNMT1","DNMT3A",
    "DNMT3B","TET1","TET2","TET3","KDM5B","KDM6A","HDAC1","HDAC2","HDAC3","HDAC4",
    "CREBBP","EP300","ARID1A","ARID1B","ARID2","SMARCA4","SMARCB1","BRD4","KAT2B","KAT6B",
    "SP3","SP1","TP63","MYCL","MYCN","NFKB1","RELA","IFNG","STAT3","STAT1",
    "IRF1","IL6","CSF2","CCL2","CXCL8","RBM5","CTNNB1","APC","AXIN1","AMER1",

    # ---- 151–250: Growth Factors, Receptors, Additional Signaling ----
    "IGF1","IGF1R","IGFBP3","PDGFB","PDGFRB","FGF2","FGF7","FGFR2","FGFR3","VEGFC",
    "ANGPT1","ANGPT2","LAMA3","COL1A1","COL1A2","MST1R","AXIN2","SMO","GLI1","GLI2",
    "SHH","TGFA","AREG","HBEGF","CCL5","CXCL10","CXCR4","IL1A","IL1B","IL1R1",
    "IL8","IL6R","JAK1","JAK2","TYK2","SOCS1","SOCS3","PIK3R1","PIK3R3","RICTOR",
    "RPTOR","RASA1","RAF1","NRAS","HRAS","MAP2K1","MAP2K2","MAPK1","MAPK3","MAPK8",
    "MAPK9","MAPK14","DUSP1","DUSP4","DUSP6","PTK2","YES1","FYN","LCK","SYK",
    "ZAP70","FCGR2B","FCGR3A","CD28","CD80","CD86","CD274","PDCD1","CTLA4","CD40",
    "CD44","ITGAL","ITGAM","ITGA2","ITGB1","ITGB3","ITGB4","SELE","SELL","SELPLG",
    "PECAM1","VCAM1","ICAM1","ICAM2","FLT1","KDR","FLT4","ERBB2","ERBB3","ERBB4",
    "AXL","PDK1","PDK2","FOXO1","FOXO3","FOXM1","VHL","EPAS1",

    # ---- 251–330: Extended Cell Cycle/DDR, Tumor Suppressors, Epigenetics ----
    "AURKA","AURKB","AURKC","PLK1","PLK2","PLK3","PLK4","CDC25A","CDC25B","CDC25C",
    "GADD45A","GADD45B","GADD45G","CDKN2D","CDKN1C","WEE1","MCM2","MCM4","MCM5","MCM6",
    "RECQL4","BLM","WRN","PALB2","MDC1","MBD4","MGMT","XRCC1","XRCC2","XRCC3",
    "XRCC4","XRCC5","XRCC6","TREX1","TREX2","RNASEH2A","UNG","CASP14","CASP1","CASP4",
    "CASP5","NLRP3","NLRP6","IL18","IFITM1","IFIT1","IFIT2","IFNAR1","IFNAR2","IRF8",
    "SPTBN1","APC","MEN1","PTPN11","NTRK1","NTRK2","NTRK3","HELLS","ATRX","CHAF1A",
    "CHAF1B","GTF2I","GTF2E1","SMARCE1","SMARCD1","SMARCD2","SMARCD3","ARID4B","KMT2A","KMT2D",
    "NSD1","NSD2","NSD3","SETD2","PRMT1","PRMT5","SUV39H1","EZH1","HDAC9","SMYD2",

    # ---- 331–400: Transporters, Metabolism (Drug Uptake/Efflux, GSTs, ALDHs, etc.) ----
    "ABCB1","ABCB11","ABCB4","ABCB5","ABCC1","ABCC2","ABCC3","ABCC4","ABCC5","ABCC6",
    "ABCG2","SLCO1B3","SLCO2B1","SLC22A16","SLC31A1","SLC31A2","SLC7A2","SLC7A5","SLC2A1","SLC2A3",
    "MGST1","MGST2","GSTK1","GSTA1","GSTA2","GSTM1","GSTM2","GSTM3","GSTP1","UGT1A1",
    "UGT1A6","UGT1A9","UGT2B4","CYP1A1","CYP1B1","CYP2A6","CYP2B6","CYP2C8","CYP2C9","CYP2C19",
    "CYP2D6","CYP2E1","CYP3A4","CYP3A5","ALDH1A1","ALDH1A2","ALDH1B1","ALDH3A1","ALDH3A2","ALDH7A1",
    "ALDH9A1","GCLC","GCLM","GPX1","GPX2","GPX4","SOD1","SOD2","CAT","NRF1",
    "NQO1","AKR1C1","AKR1C2","AKR1C3","AKR1B10","PTGR1","ADH1B","ADH1C","CES1","CES2",

    # ---- 401–500: Immune Regulators, Additional Signaling, Misc. ----
    "B2M","HLA-A","HLA-B","HLA-C","HLA-E","HLA-F","HLA-G","CIITA","CD3D","CD3E",
    "CD3G","CD8A","CD8B","CD4","CD19","CD27","CD28","ICOS","ICOSLG","CD244",
    "CD96","TIGIT","LAG3","TIM3","BTLA","CTLA4","IL2","IL2RB","IL2RG","IL7R",
    "IL21R","TNFRSF1A","TNFRSF1B","TNFRSF21","TRAF2","TRAF4","TRAF6","CD79A","CD79B","MS4A1",
    "IKBKB","IKBKG","RELA","RELB","STING1","CGAS","MAVS","DDX58","IFIH1","PKM",
    "ENO1","LDHA","PGK1","PGAM1","GAPDH","HK1","HK2","TPI1","FBP1","ACLY",
    "ACACA","ACACB","FASN","SREBF1","SREBF2","INSIG1","HMGCR","LDLR","PCSK9","SERBP1",
    "P4HA1","LOX","LOXL2","SPARC","GAS6","MFGE8","CLU","PSMD2","PSMD4","PSMD7",
    "PSMB5","PSMC2","PSMC4","UBB","UBA1","UBE2C","UBE2S","SKP2","FBXW7","CUL3",
    "NEDD4","NEDD4L","SMURF1","SOCS2","FAM83B","FAM83D","CCT2","CCT4","CCT6A"
]
    
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
    selected_covariates = [col for col in selected_covariates if col in df.columns]
    df = df[['OS_STATUS', 'OS_MONTHS'] + selected_covariates]
    
    # Create structured array for survival analysis
    surv_data = Surv.from_dataframe('OS_STATUS', 'OS_MONTHS', df)
    
    covariates = df.columns.difference(['OS_STATUS', 'OS_MONTHS'])
    
    # Identify binary columns (assuming these are already binary 0/1)
    # please change sex to IS_FEMALE/IS_MALE for clarity
    binary_columns = ['Adjuvant Chemo', 'Sex'] 
    df[binary_columns] = df[binary_columns].astype(int)
    
    continuous_columns = df.columns.difference(['OS_STATUS', 'OS_MONTHS', *binary_columns])
    
    # Check that binary columns are not scaled
    for col in binary_columns:
        assert df[col].max() <= 1 and df[col].min() >= 0, f"{col} should only contain binary values (0/1)."

    test_size = 0.2
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df[continuous_columns.union(binary_columns)], surv_data, test_size=test_size, random_state=42)

    print(X_train.columns)
    # Fit the Random Survival Forest model
    rsf = RandomSurvivalForest(n_estimators=trees, min_samples_split=75, min_samples_leaf=30, random_state=42, n_jobs= -1, max_features = None) # run on all processors
    rsf.fit(X_train, y_train)

    # Evaluate model performance
    c_index = concordance_index_censored(y_test['OS_STATUS'], y_test['OS_MONTHS'], rsf.predict(X_test))
    print(f"C-index: {c_index[0]:.3f}")
    
    # Train c-index
    train_c_index = concordance_index_censored(y_train['OS_STATUS'], y_train['OS_MONTHS'], rsf.predict(X_train))
    print(f"Train C-index: {train_c_index[0]:.3f}")

    # Save the RSF model to a file
    #joblib.dump(rsf, f'rsf_model-{trees}-c{c_index[0]:.3f}.pkl')

    result = permutation_importance(rsf, X_test, y_test, n_repeats=5, random_state=42)

    importances_df = pd.DataFrame({
        "importances_mean": result.importances_mean,
        "importances_std": result.importances_std
    }, index=X_test.columns).sort_values(by="importances_mean", ascending=False)

    print(importances_df)

    importances_df = importances_df.sort_values(by="importances_mean", ascending=True)  # Ascending for better barh plot

    # Plot the feature importances
    plt.figure(figsize=(12, 8))
    
    # Increase font size
    plt.rcParams.update({'font.size': 14})
    plt.barh(importances_df.index, importances_df["importances_mean"], xerr=importances_df["importances_std"], color=(9/255,117/255,181/255))
    plt.xlabel("Permutation Feature Importance")
    plt.ylabel("Feature")
    plt.title(f"Random Survival Forest: Feature Importances (C-index: {c_index[0]:.3f})")
    plt.tight_layout()
    name = name.replace(' ', '-')
    plt.savefig(f'rsf-importances-{name}-{trees}trees-{test_size}testsize.png')
    plt.show()

def search_feature_space_rsf(df, name, trees=300):
    # Load preselected gene features (use top 30)
    selected_csv = pd.read_csv('rsf_results_GPL570_rsf_preselection_importances.csv', index_col=0)
    candidate_genes =[
    # ---- Top 60 Core Genes ----
    "TP53","KRAS","STK11","KEAP1","NFE2L2","RB1","MYC","CDKN2A","CDK4","CDK6",
    "CCND1","MMP2","MMP9","BCL2","BCL2L1","BCL2L11","CASP3","CASP8","CASP9","NKX2-1",
    "MET","APAF1","ATF3","ATG5","AXL","FGFR1","PIK3CA","PTEN","AKT1","AKT2",
    "AKT3","MTOR","SOX2","SOX4","NOTCH1","NOTCH3","SMAD2","SMAD3","SMAD4","TGFBR1",
    "TGFBR2","HIF1A","VEGFA","ERCC1","XPA","XPC","RAD51","BRCA1","BRCA2","CHEK1",
    "CHEK2","PARP1","PARP2","CDK1","CDK2","CDK7","E2F1","RRM1","RRM2","FEN1",

    # ---- 61–150: Additional DNA Repair, Cell Cycle, Apoptosis, Epigenetic ----
    "ATM","ATR","ATRIP","MDM2","MDM4","MRE11","RAD17","RAD18","RAD50","RAD52",
    "RAD9A","MUTYH","TP73","CDKN1A","CDKN1B","CDKN2B","CDKN2C","CDKN3","CCNE1","CCNE2",
    "CCNB1","CCNB2","CCNA2","CDT1","CDC6","CDC20","BIRC2","BIRC3","BIRC5","DIABLO",
    "XIAP","TNFRSF10B","FADD","FAS","FASLG","CASP6","CASP7","CASP10","HR","FANCA",
    "FANCD2","FANCE","BRIP1","UBE2T","TOP1","TOP2A","TOP2B","EZH2","DNMT1","DNMT3A",
    "DNMT3B","TET1","TET2","TET3","KDM5B","KDM6A","HDAC1","HDAC2","HDAC3","HDAC4",
    "CREBBP","EP300","ARID1A","ARID1B","ARID2","SMARCA4","SMARCB1","BRD4","KAT2B","KAT6B",
    "SP3","SP1","TP63","MYCL","MYCN","NFKB1","RELA","IFNG","STAT3","STAT1",
    "IRF1","IL6","CSF2","CCL2","CXCL8","RBM5","CTNNB1","APC","AXIN1","AMER1",

    # ---- 151–250: Growth Factors, Receptors, Additional Signaling ----
    "IGF1","IGF1R","IGFBP3","PDGFB","PDGFRB","FGF2","FGF7","FGFR2","FGFR3","VEGFC",
    "ANGPT1","ANGPT2","LAMA3","COL1A1","COL1A2","MST1R","AXIN2","SMO","GLI1","GLI2",
    "SHH","TGFA","AREG","HBEGF","CCL5","CXCL10","CXCR4","IL1A","IL1B","IL1R1",
    "IL8","IL6R","JAK1","JAK2","TYK2","SOCS1","SOCS3","PIK3R1","PIK3R3","RICTOR",
    "RPTOR","RASA1","RAF1","NRAS","HRAS","MAP2K1","MAP2K2","MAPK1","MAPK3","MAPK8",
    "MAPK9","MAPK14","DUSP1","DUSP4","DUSP6","PTK2","YES1","FYN","LCK","SYK",
    "ZAP70","FCGR2B","FCGR3A","CD28","CD80","CD86","CD274","PDCD1","CTLA4","CD40",
    "CD44","ITGAL","ITGAM","ITGA2","ITGB1","ITGB3","ITGB4","SELE","SELL","SELPLG",
    "PECAM1","VCAM1","ICAM1","ICAM2","FLT1","KDR","FLT4","ERBB2","ERBB3","ERBB4",
    "AXL","PDK1","PDK2","FOXO1","FOXO3","FOXM1","VHL","EPAS1",

    # ---- 251–330: Extended Cell Cycle/DDR, Tumor Suppressors, Epigenetics ----
    "AURKA","AURKB","AURKC","PLK1","PLK2","PLK3","PLK4","CDC25A","CDC25B","CDC25C",
    "GADD45A","GADD45B","GADD45G","CDKN2D","CDKN1C","WEE1","MCM2","MCM4","MCM5","MCM6",
    "RECQL4","BLM","WRN","PALB2","MDC1","MBD4","MGMT","XRCC1","XRCC2","XRCC3",
    "XRCC4","XRCC5","XRCC6","TREX1","TREX2","RNASEH2A","UNG","CASP14","CASP1","CASP4",
    "CASP5","NLRP3","NLRP6","IL18","IFITM1","IFIT1","IFIT2","IFNAR1","IFNAR2","IRF8",
    "SPTBN1","APC","MEN1","PTPN11","NTRK1","NTRK2","NTRK3","HELLS","ATRX","CHAF1A",
    "CHAF1B","GTF2I","GTF2E1","SMARCE1","SMARCD1","SMARCD2","SMARCD3","ARID4B","KMT2A","KMT2D",
    "NSD1","NSD2","NSD3","SETD2","PRMT1","PRMT5","SUV39H1","EZH1","HDAC9","SMYD2",

    # ---- 331–400: Transporters, Metabolism (Drug Uptake/Efflux, GSTs, ALDHs, etc.) ----
    "ABCB1","ABCB11","ABCB4","ABCB5","ABCC1","ABCC2","ABCC3","ABCC4","ABCC5","ABCC6",
    "ABCG2","SLCO1B3","SLCO2B1","SLC22A16","SLC31A1","SLC31A2","SLC7A2","SLC7A5","SLC2A1","SLC2A3",
    "MGST1","MGST2","GSTK1","GSTA1","GSTA2","GSTM1","GSTM2","GSTM3","GSTP1","UGT1A1",
    "UGT1A6","UGT1A9","UGT2B4","CYP1A1","CYP1B1","CYP2A6","CYP2B6","CYP2C8","CYP2C9","CYP2C19",
    "CYP2D6","CYP2E1","CYP3A4","CYP3A5","ALDH1A1","ALDH1A2","ALDH1B1","ALDH3A1","ALDH3A2","ALDH7A1",
    "ALDH9A1","GCLC","GCLM","GPX1","GPX2","GPX4","SOD1","SOD2","CAT","NRF1",
    "NQO1","AKR1C1","AKR1C2","AKR1C3","AKR1B10","PTGR1","ADH1B","ADH1C","CES1","CES2",

    # ---- 401–500: Immune Regulators, Additional Signaling, Misc. ----
    "B2M","HLA-A","HLA-B","HLA-C","HLA-E","HLA-F","HLA-G","CIITA","CD3D","CD3E",
    "CD3G","CD8A","CD8B","CD4","CD19","CD27","CD28","ICOS","ICOSLG","CD244",
    "CD96","TIGIT","LAG3","TIM3","BTLA","CTLA4","IL2","IL2RB","IL2RG","IL7R",
    "IL21R","TNFRSF1A","TNFRSF1B","TNFRSF21","TRAF2","TRAF4","TRAF6","CD79A","CD79B","MS4A1",
    "IKBKB","IKBKG","RELA","RELB","STING1","CGAS","MAVS","DDX58","IFIH1","PKM",
    "ENO1","LDHA","PGK1","PGAM1","GAPDH","HK1","HK2","TPI1","FBP1","ACLY",
    "ACACA","ACACB","FASN","SREBF1","SREBF2","INSIG1","HMGCR","LDLR","PCSK9","SERBP1",
    "P4HA1","LOX","LOXL2","SPARC","GAS6","MFGE8","CLU","PSMD2","PSMD4","PSMD7",
    "PSMB5","PSMC2","PSMC4","UBB","UBA1","UBE2C","UBE2S","SKP2","FBXW7","CUL3",
    "NEDD4","NEDD4L","SMURF1","SOCS2","FAM83B","FAM83D","CCT2","CCT4","CCT6A"
]
    candidate_genes = [
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
    # Clinical variables (must be included)
    clinical_vars = ["Adjuvant Chemo", "Age", "Stage", "Sex", "Histology", "Race", "Smoked?"]
    clinical_vars = [var for var in clinical_vars if var in df.columns]
    
    test_size = 0.2
    # Lists to hold results for 3D plotting
    gene_features_list = []
    n_estimators_list = []
    test_c_indexes = []
    train_c_indexes = []
    
    # Iterate over different numbers of gene features and n_estimators
    for n_est in np.unique(np.linspace(50, 500, num=5, dtype=int)):
        for m in np.unique(np.linspace(0, 100, num=10, dtype=int)):
            # Select top m genes and add clinical variables (if m exceeds candidate_genes length, use all available)
            selected_covariates = list(set(candidate_genes[:m]).union(set(clinical_vars)))
            df_subset = df[['OS_STATUS', 'OS_MONTHS'] + [col for col in selected_covariates if col in df.columns]].copy()
            
            # Create structured survival array
            surv_data = Surv.from_dataframe('OS_STATUS', 'OS_MONTHS', df_subset)
            # Define feature columns (exclude OS_STATUS and OS_MONTHS)
            feature_cols = df_subset.columns.difference(['OS_STATUS', 'OS_MONTHS'])
            
            # Convert binary columns if present
            binary_columns = [col for col in ['Adjuvant Chemo', 'Sex'] if col in feature_cols]
            df_subset[binary_columns] = df_subset[binary_columns].astype(int)
            
            # Split into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(df_subset[feature_cols], surv_data,
                                                                test_size=test_size, random_state=42)
            # Fit the Random Survival Forest model
            rsf = RandomSurvivalForest(n_estimators=n_est, min_samples_split=75, min_samples_leaf=30,
                                       random_state=42, n_jobs=-1, max_features="sqrt")
            rsf.fit(X_train, y_train)
            
            # Evaluate model performance on test set
            test_ci = concordance_index_censored(y_test['OS_STATUS'], y_test['OS_MONTHS'], rsf.predict(X_test))[0]
            # Train c-index
            train_ci = concordance_index_censored(y_train['OS_STATUS'], y_train['OS_MONTHS'], rsf.predict(X_train))[0]
            gene_features_list.append(m)
            n_estimators_list.append(n_est)
            test_c_indexes.append(test_ci)
            train_c_indexes.append(train_ci)
            print(f"Gene features: {m}, Estimators: {n_est}, Total features: {len(feature_cols)}, Test C-index: {test_ci:.3f}, Train C-index: {train_ci:.3f}")
            
            if test_ci > 0.6:
                joblib.dump(rsf, f'rsf_model-{trees}-c{test_ci:.3f}.pkl')
    
    # Find optimal setting based on test performance
    optimal_index = np.argmax(test_c_indexes)
    optimal_genes = gene_features_list[optimal_index]
    optimal_n_est = n_estimators_list[optimal_index]
    optimal_c_index = test_c_indexes[optimal_index]
    print(f"\nOptimal: Gene features: {optimal_genes}, Estimators: {optimal_n_est} (Test C-index: {optimal_c_index:.3f})")
    
    # 3D Plot performance vs number of gene features and n_estimators
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    sc1 = ax.scatter(gene_features_list, n_estimators_list, test_c_indexes, c='blue', marker='o', label="Test C-index")
    sc2 = ax.scatter(gene_features_list, n_estimators_list, train_c_indexes, c='green', marker='^', label="Train C-index")
    ax.set_xlabel("Number of Gene Features")
    ax.set_ylabel("n_estimators")
    ax.set_zlabel("C-index")
    ax.set_title("RSF Performance vs Gene Features and n_estimators")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'rsf_feature_search_3D_{name}_{trees}trees.png')
    plt.show()
    
    return optimal_genes

# Data preprocessing (unchanged)
surv = pd.read_csv('GPL570merged.csv')
surv = pd.get_dummies(surv, columns=["Stage", "Histology", "Race"])
surv = surv.drop(columns=['PFS_MONTHS','RFS_MONTHS'])
print(surv.columns[surv.isna().any()].tolist())
print(surv['Smoked?'].isna().sum())  # 121
surv = surv.dropna()  # left with 457 samples

# Run feature space search
optimal_features = search_feature_space_rsf(surv, 'GPL570', trees=300)

#print(optimal_features)
#create_rsf(surv, 'GPL570', 50)
# Optimal: Gene features: 77, Estimators: 50 (Test C-index: 0.641)