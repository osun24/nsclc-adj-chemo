import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

# Set the default font to Arial
#matplotlib.rcParams['font.family'] = 'Arial'

def extract_clinical_variables(file_path):
    # Read the CSV file using pandas
    df_raw = pd.read_csv(file_path, dtype=str, on_bad_lines='skip')
    
    print(df_raw.columns)
    
    # Clean column names and drop completely empty rows
    df_raw.columns = [col.strip() for col in df_raw.columns]
    df_raw.dropna(how='all', inplace=True)
    
    study_variables = {}
    study_n_mapping = {}
    study_sample_sizes = {}
    
    for index, row in df_raw.iterrows():
        if isinstance(row.iloc[0], str) and row.iloc[0].startswith("GSE"):
            study_id = row.iloc[0].strip()
            n_value_raw = row.iloc[1] if len(row) > 1 else "Unknown"
            try:
                n_value_int = int(n_value_raw)
            except ValueError:
                continue  # Skip studies with non-numeric sample sizes
            
            study_sample_sizes[study_id] = n_value_int
            study_n_mapping[study_id] = f"{study_id} ({n_value_int})"
            
            # Get clinical variables from the notes column
            notes = row.iloc[2].split("\n")
            clinical_vars = []
            for note in notes:
                if note.startswith("Clinical: "):
                    clinical_vars.extend(note.replace("Clinical: ", "").split(", "))
            clinical_vars = [var.strip() for var in clinical_vars if var.strip()]
            # Combine "Smoking History" and "Smoking History (binary)" into "Smoked?"
            if "Smoking History" in clinical_vars or "Smoking History (binary)" in clinical_vars:
                clinical_vars = [var for var in clinical_vars if var not in ("Smoking History", "Smoking History (binary)")]
                clinical_vars.append("Smoked?")
            study_variables[study_id] = set(clinical_vars)
    
    # Get unique clinical variables across all studies
    if not study_variables:
        return pd.DataFrame()  # return empty DataFrame if no valid studies
    all_variables = sorted(set().union(*study_variables.values()))
    
    # Create a binary presence matrix (variables as rows, studies as columns)
    df = pd.DataFrame(0, index=all_variables, columns=study_variables.keys())
    for study, variables in study_variables.items():
        df.loc[list(variables), study] = 1

    # Compute total sample size (weighted denominator)
    total_sample_size = sum(study_sample_sizes[study] for study in study_variables)
    
    # Drop clinical variables missing in >30% of the overall (weighted) sample
    keep_vars = []
    for var in df.index:
        weighted_presence = sum(study_sample_sizes[study] for study in study_variables if df.loc[var, study] == 1)
        if weighted_presence / total_sample_size >= 0.3:
            keep_vars.append(var)
    df = df.loc[keep_vars]
    
    # Append the count of studies for each variable to the variable labels
    variable_counts = df.sum(axis=1)
    df.index = [f"{var} ({int(count)})" for var, count in zip(df.index, variable_counts)]
    
    # Rename columns with appended (n) values
    df.rename(columns=study_n_mapping, inplace=True)
    
    return df

def plot_heatmap(df):
    plt.figure(figsize=(12, 8))
    ax = sns.heatmap(df, annot=False, cmap="Blues", linewidths=0.5, linecolor="gray", cbar=True)
    plt.xlabel("Studies")
    plt.ylabel("Clinical Variables")
    plt.title("Shared Clinical Variables Across Studies (Present in >30% of the Overall Sample)")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    
    # Bold ytick labels for variables present in a majority of studies
    majority_threshold = df.shape[1] / 2
    ytick_text = [label.get_text() for label in ax.get_yticklabels()]
    # Adjust for potential reversal of tick order in the heatmap
    index_order = df.index if ytick_text[0] == df.index[0] else df.index[::-1]
    for tick, var in zip(ax.get_yticklabels(), index_order):
        if df.loc[var].sum() > majority_threshold:
            tick.set_fontweight("bold")
    
    plt.tight_layout()
    plt.savefig("clinical_variables_heatmap.png", dpi = 300)
    plt.show()

if __name__ == "__main__":
    file_path = "Clinical Variables - RAW - Feb. 5.csv"  # Modify with actual file path
    df = extract_clinical_variables(file_path)
    print(df)  # Print dataframe for review
    plot_heatmap(df)
