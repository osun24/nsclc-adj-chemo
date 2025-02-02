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
    
    # Extract study names and sample sizes
    df_raw.columns = [col.strip() for col in df_raw.columns]
    df_raw.dropna(how='all', inplace=True)  # Remove empty rows
    
    study_variables = {}
    study_n_mapping = {}
    
    for index, row in df_raw.iterrows():
        if isinstance(row.iloc[0], str) and row.iloc[0].startswith("GSE"):
            study_id = row.iloc[0].strip()
            n_value = row.iloc[1] if len(row) > 1 else "Unknown"
            study_n_mapping[study_id] = f"{study_id} ({n_value})"
            
            # get the clinical variables (in the notes column)
            notes = row.iloc[2].split("\n")
            
            # go through notes and look for line that starts with "Clinical: "
            
            clinical_vars = []
            for note in notes:
                if note.startswith("Clinical: "):
                    clinical_vars.extend(note.replace("Clinical: ", "").split(", "))
            
            # if clinical vars contains an empty string or just a space, remove it
            clinical_vars = [var.strip() for var in clinical_vars if var.strip()]
                
            study_variables[study_id] = set(clinical_vars)
    
    # Get the unique clinical variables across all studies
    all_variables = sorted(set().union(*study_variables.values()))
    
    # Create a binary presence matrix
    df = pd.DataFrame(0, index=all_variables, columns=study_variables.keys())
    for study, variables in study_variables.items():
        df.loc[list(variables), study] = 1

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
    plt.title("Shared Clinical Variables Across Studies")
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
    file_path = "Included Raw Clinical Variables.csv"  # Modify with actual file path
    df = extract_clinical_variables(file_path)
    print(df)  # Print dataframe for review
    plot_heatmap(df)
