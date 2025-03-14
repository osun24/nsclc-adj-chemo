import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Load CSV file
data_file = "GPL96merged.csv"  # adjust path if needed
df = pd.read_csv(data_file)

# Print unique values for non-numeric columns before any processing
non_numeric_cols = df.select_dtypes(include=['object']).columns
for col in non_numeric_cols:
    print(f"Unique values in '{col}' before processing:", df[col].unique())

# Print the unique items in column 7
column_7_name = df.columns[6]  # Column index is 0-based, so column 7 is index 6
unique_items = df[column_7_name].unique()
print(f"Unique items in column '{column_7_name}':")
print(unique_items)

# One-hot encode categorical variables (smoked has unknown)
df = pd.get_dummies(df, columns=["Stage", "Histology", "Race", "Smoked?"])

# Drop PFS/RFS 
# No RFS for GPL96, so only drop PFS
df = df.drop(columns=['PFS_MONTHS', 'RFS_MONTHS'])

print("Columns with NA values:", df.columns[df.isna().any()].tolist())

print("Shape before dropping NA", df.shape)
df = df.dropna()  # Retain 457 samples
print("Data shape:", df.shape)

# Separate events and censored cases
events = df[df["OS_STATUS"] == 1]
censored = df[df["OS_STATUS"] == 0]

# Split events: 60% train, 40% temp (split later into 20/20)
train_events, temp_events = train_test_split(events, test_size=0.4, random_state=42)
test_events, valid_events = train_test_split(temp_events, test_size=0.5, random_state=42)

# Split censored cases similarly
train_censored, temp_censored = train_test_split(censored, test_size=0.4, random_state=42)
test_censored, valid_censored = train_test_split(temp_censored, test_size=0.5, random_state=42)

# Combine events and censored splits
train = pd.concat([train_events, train_censored]).sample(frac=1, random_state=42).reset_index(drop=True)
test = pd.concat([test_events, test_censored]).sample(frac=1, random_state=42).reset_index(drop=True)
validation = pd.concat([valid_events, valid_censored]).sample(frac=1, random_state=42).reset_index(drop=True)

# print number of columns that have SMOKED in the name
smoked_columns = df.columns[df.columns.str.contains("Smoked")]
print(f"Columns with 'Smoked' in the name: {len(smoked_columns)}")

# Save outputs to CSV files
train.to_csv("GPL96train.csv", index=False)
test.to_csv("GPL96test.csv", index=False)
validation.to_csv("GPL96validation.csv", index=False)

print(f"Training set: {len(train)} samples")
print(f"Testing set: {len(test)} samples")
print(f"Validation set: {len(validation)} samples")

# print number of events/censored in each set
print(f"Training set: {len(train[train['OS_STATUS'] == 1])} events, {len(train[train['OS_STATUS'] == 0])} censored")
print(f"Testing set: {len(test[test['OS_STATUS'] == 1])} events, {len(test[test['OS_STATUS'] == 0])} censored")
print(f"Validation set: {len(validation[validation['OS_STATUS'] == 1])} events, {len(validation[validation['OS_STATUS'] == 0])} censored")


# Check for non-numeric columns
non_numeric_columns = df.columns[df.apply(lambda col: pd.to_numeric(col, errors='coerce').isna().any())]
print("Non-numeric columns at the end of preprocessing:", non_numeric_columns.tolist())

"""GPL570:
Unique values in 'Stage' before processing: ['II' 'IB' 'III' 'IA' 'IV']
Unique values in 'Histology' before processing: ['Squamous Cell Carcinoma' 'Adenocarcinoma' 'Large Cell Carcinoma'
 'Adenosquamous Carcinoma']
Unique values in 'Race' before processing: ['Caucasian' 'African American' 'Unknown' 'Asian']
Unique values in 'Smoked?' before processing: ['Yes' 'No' 'Unknown']
Unique items in column 'Race':
['Caucasian' 'African American' 'Unknown' 'Asian']
Columns with NA values: []
Shape before dropping NA (578, 21379)
Data shape: (578, 21379)
Columns with 'Smoked' in the name: 3
Training set: 346 samples
Testing set: 116 samples
Validation set: 116 samples
Training set: 122 events, 224 censored
Testing set: 41 events, 75 censored
Validation set: 41 events, 75 censored
Non-numeric columns at the end of preprocessing: []"""

"""GPL96:
Unique values in 'Stage' before processing: ['IA' 'IB' 'II' 'III' 'Unknown' 'I']
Unique values in 'Histology' before processing: ['Adenocarcinoma' 'Squamous Cell Carcinoma'
 'Large Cell Undifferentiated Carcinoma']
Unique values in 'Race' before processing: ['White' 'Unknown' 'Asian' 'Black or African American'
 'Native Hawaiian or Other Pacific Islander']
Unique values in 'Smoked?' before processing: ['Yes' 'No' 'Unknown']
Unique items in column 'Race':
['White' 'Unknown' 'Asian' 'Black or African American'
 'Native Hawaiian or Other Pacific Islander']
Columns with NA values: ['OS_MONTHS']
Shape before dropping NA (576, 13061)
Data shape: (575, 13061)
Columns with 'Smoked' in the name: 3
Training set: 344 samples
Testing set: 115 samples
Validation set: 116 samples
Training set: 177 events, 167 censored
Testing set: 59 events, 56 censored
Validation set: 60 events, 56 censored
Non-numeric columns at the end of preprocessing: []"""