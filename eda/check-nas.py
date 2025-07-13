import pandas as pd

train_orig = pd.read_csv("affyTrain.csv")
valid_orig = pd.read_csv("affyValidation.csv")

# Combine train and validation
train = pd.concat([train_orig, valid_orig], axis=0, ignore_index=True)

print(f"Combined training data shape: {train.shape}")
print(f"Number of events: {train['OS_STATUS'].sum()}")
print(f"Number of censored: {train.shape[0] - train['OS_STATUS'].sum()}")

# Preprocess data
train['Adjuvant Chemo'] = train['Adjuvant Chemo'].replace({'OBS': 0, 'ACT': 1})
valid_orig['Adjuvant Chemo'] = valid_orig['Adjuvant Chemo'].replace({'OBS': 0, 'ACT': 1})

# Check for missing values
missing_values = train.isnull().sum()
print("Missing values in each column:")
print(missing_values[missing_values > 0])

# count number of different datatypes 
data_types_count = train.dtypes.value_counts()
print("\nCount of different data types:")
print(data_types_count)

# print int64's
int64_columns = train.select_dtypes(include=['int64']).columns
print("\nColumns with int64 data type:")
print(int64_columns)

# check for NaN values in float64 columns
float64_columns = train.select_dtypes(include=['float64']).columns
nan_in_float64 = train[float64_columns].isnull().any()
print("\nFloat64 columns with NaN values:")
print(nan_in_float64[nan_in_float64].index.tolist())

# Use pd.isnull() to check for NaN values in the entire DataFrame
nan_in_dataframe = pd.isnull(train).any()
print("\nColumns with NaN values in the DataFrame:")
print(nan_in_dataframe[nan_in_dataframe].index.tolist())

nan_in_dataframe = pd.isnull(valid_orig).sum()
print("\nColumns with NaN values in the Validation DataFrame:")
# get indices of columns with NaN values
print(nan_in_dataframe[nan_in_dataframe > 0].index.tolist())
