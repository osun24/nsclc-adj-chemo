import os
import time
import pandas as pd
import numpy as np
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.metrics import make_scorer
import joblib
import sys
import datetime
import os
import time
import datetime
import pandas as pd
import numpy as np
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
import joblib
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

# Redirect console output to both the terminal and a log file
class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
    def flush(self):
        for f in self.files:
            f.flush()

# Define custom concordance metric function
def rsf_concordance_metric(y, y_pred):
    return concordance_index_censored(y['OS_STATUS'], y['OS_MONTHS'], y_pred)[0]

output_dir = "rsf/rsf_results_affy"  # Directory to save output files
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

current_date = datetime.datetime.now().strftime("%Y%m%d")  # Added current date for file naming

log_file = open(os.path.join(output_dir, f"{current_date}_LOG-rsf-feature-search.txt"), "w")
sys.stdout = Tee(sys.stdout, log_file)

print("Loading train data from: affyTrain.csv")
train = pd.read_csv("affyTrain.csv")

print(f"Number of events in training set: {train['OS_STATUS'].sum()} | Censored cases: {train.shape[0] - train['OS_STATUS'].sum()}")
print("Train data shape:", train.shape)

print("Loading validation data from: affyValidation.csv")
valid = pd.read_csv("affyValidation.csv")

print(f"Number of events in validation set: {valid['OS_STATUS'].sum()} | Censored cases: {valid.shape[0] - valid['OS_STATUS'].sum()}")
print("Validation data shape:", valid.shape)

start = time.time()

# Set Adjuvant Chemo's 'ACT' to 1 and 'OBS' to 0
train['Adjuvant Chemo'] = train['Adjuvant Chemo'].map({'ACT': 1, 'OBS': 0})
valid['Adjuvant Chemo'] = valid['Adjuvant Chemo'].map({'ACT': 1, 'OBS': 0})

# Create structured arrays for survival analysis
y_train = Surv.from_dataframe('OS_STATUS', 'OS_MONTHS', train)
y_valid = Surv.from_dataframe('OS_STATUS', 'OS_MONTHS', valid)

X_train = train.drop(columns=['OS_STATUS', 'OS_MONTHS'])
X_valid = valid.drop(columns=['OS_STATUS', 'OS_MONTHS'])

# Make RSF 
# {'max_depth': 10, 'max_features': 500, 'min_samples_leaf': 60, 'n_estimators': 750}
rsf = RandomSurvivalForest(
    n_estimators=500,
    max_depth=5,
    min_samples_leaf=70,
    max_features=0.2,
    random_state=42,
    n_jobs=-1
)

"""
n_estimators = 500
max_depth = 5
min_samples_leaf = 70
max_features = 0.1  # 0.1 * 13062 = 1306
Training C-index: 0.7802
Validation C-index: 0.6985

max_depth = 4 
Training C-index: 0.7802
Validation C-index: 0.6984

max_depth = 3
Training C-index: 0.7780
Validation C-index: 0.6989

max_depth = 2
Training C-index: 0.7578
Validation C-index: 0.6947

max_depth = 1
Training C-index: 0.7041
Validation C-index: 0.6848

n_estimators=500,
    max_depth=5,
    min_samples_leaf=70,
    max_features=0.2,
    random_state=42,
    n_jobs=-1
Training C-index: 0.7884
Validation C-index: 0.7013

max_features=0.5,
Training C-index: 0.7943
Validation C-index: 0.7061

max_features = None
Training C-index: 0.7955
Validation C-index: 0.7041

maybe there is a relationship between max_features and n_estimators
as n_estimaors increases, max_features can be smaller --> more stable estimates, pushing down of spurious links
"""

""" 
param_grid = {
                "n_estimators": [500, 750, 1000],
                "min_samples_leaf": [50, 60, 70],    
                "max_features": ["sqrt", 500, 0.1], # 0.1 * 13062 = 1306
                "max_depth": [2, 3, 4, 5],
            }"""

# Fit the model
print("Fitting Random Survival Forest model...")
rsf.fit(X_train, y_train)


# Print C-index on training data, testing data, and validation data
train_c_index = rsf_concordance_metric(y_train, rsf.predict(X_train))
valid_c_index = rsf_concordance_metric(y_valid, rsf.predict(X_valid))

print(f"Training C-index: {train_c_index:.4f}")
print(f"Validation C-index: {valid_c_index:.4f}")

# TEST 
print("Loading test data from: affyTest.csv")
test = pd.read_csv("affyTest.csv")
test['Adjuvant Chemo'] = test['Adjuvant Chemo'].map({'ACT': 1, 'OBS': 0})

print(f"Number of events in test set: {test['OS_STATUS'].sum()} | Censored cases: {test.shape[0] - test['OS_STATUS'].sum()}")
print("Test data shape:", test.shape)

# Create structured arrays for test data
y_test = Surv.from_dataframe('OS_STATUS', 'OS_MONTHS', test)
X_test = test.drop(columns=['OS_STATUS', 'OS_MONTHS'])

# Evaluate C-index on test data
test_c_index = rsf_concordance_metric(y_test, rsf.predict(X_test))
print(f"Test C-index: {test_c_index:.4f}")

# Print actual max depth and number of trees
print(f"Actual max depth: {rsf.max_depth}")
print(f"Number of trees in the forest: {len(rsf.estimators_)}")
""""
# Run permutation importance
perm_result = permutation_importance(rsf, X_train, y_train,
                                       scoring=lambda est, X, y: rsf_concordance_metric(y, est.predict(X)),
                                       n_repeats=5, random_state=42, n_jobs=-1)
importances = perm_result.importances_mean
importance_df = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": importances,
    "Std": perm_result.importances_std
}).sort_values(by="Importance", ascending=False)

preselect_csv = os.path.join(output_dir, f"{current_date}_rsf_preselection_importances.csv")
importance_df.to_csv(preselect_csv, index=False)
print(f"RSF pre-selection importances saved to {preselect_csv}")

top_preselect = importance_df.head(50)
plt.figure(figsize=(12, 8))
plt.barh(top_preselect["Feature"][::-1], top_preselect["Importance"][::-1],
            xerr=top_preselect["Std"][::-1], color=(9/255, 117/255, 181/255))
plt.xlabel("Permutation Importance")
plt.title("RSF Pre-Selection (Top 50 Features)")
plt.tight_layout()
preselect_plot = os.path.join(output_dir, f"{current_date}_rsf_preselection_importances_1SE.png")
plt.savefig(preselect_plot)
plt.close()
print(f"RSF pre-selection plot saved to {preselect_plot}")
"""