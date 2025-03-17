import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Load CSV file
data_file = "all.merged.csv"  # adjust path if needed
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

# print number of na in OS_MONTHS
print("Number of NA values in OS_MONTHS:", df["OS_MONTHS"].isna().sum())

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
train.to_csv("allTrain.csv", index=False)
test.to_csv("allTest.csv", index=False)
validation.to_csv("allValidation.csv", index=False)

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
Non-numeric columns at the end of preprocessing: []

ALL: 
Unique values in 'Histology' before processing: ['Squamous Cell Carcinoma' 'Adenocarcinoma' 'Large Cell Carcinoma'
 'Adenosquamous Carcinoma' 'Unknown']
Unique values in 'Stage' before processing: ['II' 'IA' 'IB' 'III' 'IV' 'Unknown']
Unique values in 'Smoked?' before processing: ['Unknown' 'Yes' 'No']
Unique values in 'Race' before processing: ['Unknown' 'Caucasian' 'African American' 'Asian'
 'Native Hawaiian or Other Pacific Islander']
Unique items in column 'RFS_MONTHS':
[ 86.63 105.88   5.06  15.44   2.99   7.29   9.17    nan   8.05   3.12
  16.26  14.36   2.76   2.73   8.74  11.79   7.85   1.77  36.76 111.14
 132.75  83.15  13.17  63.8   16.52  97.04   8.11 145.3    4.57   3.81
  84.72 138.21   6.24   7.95  95.43  68.53  56.08  31.73 100.    56.77
   1.22  56.7   42.87  94.71  11.04   0.2    4.37   3.38  89.88 139.82
  10.68  78.19 163.53   2.33 161.43  55.52  90.34  10.05 141.72 123.52
  68.46 102.83   6.08   6.96   5.88   4.53   2.1   93.66   8.38   0.23
  18.53  12.88 109.99  74.47 176.91  74.74  20.43 104.11  72.54  78.98
  13.37   5.32  12.29 165.97  77.83 133.77 165.74  28.55   3.88 178.15
  17.44  19.78 149.38   8.43   1.9   86.7   39.1   78.97  62.63  20.2
   4.77  63.97  61.17  60.93  45.3   19.67  14.6   28.7   17.93  48.73
  49.17  61.87 102.23  38.07  60.83  53.67  67.13  60.87  73.03 111.4
  44.27  97.5   96.    57.47  54.6   10.5    3.3    3.33   5.43   6.53
  24.1   22.53   8.33  42.7    9.9   17.07  21.17  27.1   46.23  38.13
   8.6    9.07  35.07  26.    20.73   9.57  45.43   6.6   33.23  21.87
   8.63  76.3   63.87  68.27  51.3   21.2   10.6   66.83  65.5   80.83
  52.2   33.    24.3   78.47  27.27  66.97  84.97 113.7   60.2   64.57
  50.67  45.83  58.53  53.23  57.13  54.57  55.1   39.87  56.73  68.33
  41.87  38.47 101.93  64.7   53.2   81.67  62.7   67.8   71.13  15.37
  10.43  18.    11.37  21.03  13.07  16.    26.33  27.5   57.5   12.8
  19.57  76.07  27.73  37.53  25.73  16.5   41.4   32.9   27.57  34.9
  56.87  11.87  15.43  18.23  41.73  18.9   82.87  24.67  34.73  42.23
  33.9   34.7   61.97  57.23  49.8   25.13  13.43  57.77  25.8   97.1
  60.1   63.73  49.37  41.03  65.47 111.73  71.93 115.53 108.77  60.77
  81.63  66.6   61.73  63.13  62.03  55.63 106.3   60.73  60.3   85.07
  98.93 102.2   47.07  43.9   39.33  56.27  47.23  47.53  49.63  61.6
  59.33  69.47  37.13  93.53  55.43  57.53  54.3   72.63  63.9   48.2
  38.73  31.37  54.1   61.83  46.8   74.03  28.3   27.7  107.3   86.17
  81.4   63.67  61.13  97.27  76.17  71.63  85.37  64.9   61.37  65.7
  80.3   54.13  81.    50.53  60.63  80.43  61.63  72.7   72.27  71.8
  69.1   66.47  71.23  63.6   55.33]
Number of NA values in OS_MONTHS: 1
Columns with NA values: ['OS_MONTHS']
Shape before dropping NA (1376, 12363)
Data shape: (1375, 12363)
Columns with 'Smoked' in the name: 3
Training set: 824 samples
Testing set: 275 samples
Validation set: 276 samples
Training set: 338 events, 486 censored
Testing set: 113 events, 162 censored
Validation set: 113 events, 163 censored
Non-numeric columns at the end of preprocessing: []"""