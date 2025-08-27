import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Load CSV file
data_file = "affymetrix.merged.filtered.csv"  # adjust path if needed
df = pd.read_csv(data_file)

# Drop batch
df = df.drop(columns=['Batch'])

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

# print number of na in OS_MONTHS
print("Number of NA values in OS_MONTHS:", df["OS_MONTHS"].isna().sum())

print("Columns with NA values:", df.columns[df.isna().any()].tolist())

print("Shape before dropping NA", df.shape)
df = df.dropna()  # Retain 457 samples
print("Data shape:", df.shape)

# Create combined stratification variable
df['strata'] = df['OS_STATUS'].astype(str) + '_' + df['Adjuvant Chemo']
print("Stratifying by OS_STATUS and Adjuvant Chemo:", df['strata'].value_counts().to_dict())

# Split using stratification
train, temp = train_test_split(df, test_size=0.4, random_state=42, stratify=df['strata'])
test, validation = train_test_split(temp, test_size=0.5, random_state=42, stratify=temp['strata'])

# Remove temporary column
train = train.drop('strata', axis=1)
test = test.drop('strata', axis=1)
validation = validation.drop('strata', axis=1)

# print number of columns that have SMOKED in the name
smoked_columns = df.columns[df.columns.str.contains("Smoked")]
print(f"Columns with 'Smoked' in the name: {len(smoked_columns)}")

# Save outputs to CSV files
train.to_csv("affyFilteredTrain.csv", index=False)
test.to_csv("affyFilteredTest.csv", index=False)
validation.to_csv("affyFilteredValidation.csv", index=False)

print(f"Training set: {len(train)} samples (OS_STATUS: {train['OS_STATUS'].value_counts().to_dict()} | Adjuvant Chemo: {train['Adjuvant Chemo'].value_counts().to_dict()})")
print(f"Testing set: {len(test)} samples (OS_STATUS: {test['OS_STATUS'].value_counts().to_dict()} | Adjuvant Chemo: {test['Adjuvant Chemo'].value_counts().to_dict()})")
print(f"Validation set: {len(validation)} samples (OS_STATUS: {validation['OS_STATUS'].value_counts().to_dict()} | Adjuvant Chemo: {validation['Adjuvant Chemo'].value_counts().to_dict()})")

# Check for non-numeric columns in train/test/validation sets
train_non_numeric = train.columns[train.apply(lambda col: pd.to_numeric(col, errors='coerce').isna().any())]
test_non_numeric = test.columns[test.apply(lambda col: pd.to_numeric(col, errors='coerce').isna().any())]
validation_non_numeric = validation.columns[validation.apply(lambda col: pd.to_numeric(col, errors='coerce').isna().any())]

print("Non-numeric columns in train set:", train_non_numeric.tolist())
print("Non-numeric columns in test set:", test_non_numeric.tolist())
print("Non-numeric columns in validation set:", validation_non_numeric.tolist())

# PRINT UNIQUE VALUES FOR Adjuvant Chemo
if 'Adjuvant Chemo' in df.columns:
    print("Unique values in 'Adjuvant Chemo' before processing:", df['Adjuvant Chemo'].unique())  

""" 
Unique values in 'Adjuvant Chemo' before processing: ['ACT' 'OBS']
Unique values in 'Stage' before processing: ['II' 'IB' 'III' 'IA']
Unique values in 'Histology' before processing: ['Squamous Cell Carcinoma' 'Adenocarcinoma' 'Large Cell Carcinoma'
 'Adenosquamous Carcinoma']
Unique values in 'Race' before processing: ['Caucasian' 'African American' 'Unknown' 'Asian'
 'Native Hawaiian or Other Pacific Islander']
Unique values in 'Smoked?' before processing: ['Yes' 'No' 'Unknown']
Unique items in column 'Smoked?':
['Yes' 'No' 'Unknown']
Number of NA values in OS_MONTHS: 0
Columns with NA values: []
Shape before dropping NA (1353, 13060)
Data shape: (1353, 13060)
Stratifying by OS_STATUS and Adjuvant Chemo: {'0_OBS': 654, '1_OBS': 476, '1_ACT': 128, '0_ACT': 95}
Columns with 'Smoked' in the name: 3
Training set: 811 samples (OS_STATUS: {0: 449, 1: 362} | Adjuvant Chemo: {'OBS': 677, 'ACT': 134})
Testing set: 271 samples (OS_STATUS: {0: 150, 1: 121} | Adjuvant Chemo: {'OBS': 227, 'ACT': 44})
Validation set: 271 samples (OS_STATUS: {0: 150, 1: 121} | Adjuvant Chemo: {'OBS': 226, 'ACT': 45})
Non-numeric columns in train set: ['Adjuvant Chemo']
Non-numeric columns in test set: ['Adjuvant Chemo']
Non-numeric columns in validation set: ['Adjuvant Chemo']
Unique values in 'Adjuvant Chemo' before processing: ['ACT' 'OBS']
"""