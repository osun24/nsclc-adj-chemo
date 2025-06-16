import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# --- load the two runs --------------------------------------------------------
file_a = Path('rsf/rsf_results_affy/Affy RS_rsf_preselection_importances_1SE 6-15-25.csv')
file_b = Path('rsf/rsf_results_affy/Affy RS_rsf_preselection_importances_1SE.csv')

df_a = pd.read_csv(file_a)[['Feature', 'Importance']]
df_b = pd.read_csv(file_b)[['Feature', 'Importance']]

# --- align, fill any missing features with 0 ----------------------------------
merged = (df_a.rename(columns={'Importance': 'Importance_A'})
            .merge(df_b.rename(columns={'Importance': 'Importance_B'}),
                   on='Feature', how='outer')
            .fillna(0))

# --- global comparison metrics ------------------------------------------------
n_features = len(merged)
mean_a = merged['Importance_A'].mean()
mean_b = merged['Importance_B'].mean()
r, _ = pearsonr(merged['Importance_A'], merged['Importance_B'])
rmse = np.sqrt(np.mean((merged['Importance_A'] - merged['Importance_B'])**2))

print(f'# features          : {n_features:,}')
print(f'Mean importance  A : {mean_a:.2e}')
print(f'Mean importance  B : {mean_b:.2e}')
print(f'Pearson r (A vs B) : {r:.3f}')
print(f'RMSE (A vs B)      : {rmse:.2e}\n')

# --- biggest movers -----------------------------------------------------------
merged['Delta'] = merged['Importance_A'] - merged['Importance_B']

top_inc = merged.nlargest(5, 'Delta')
top_dec = merged.nsmallest(5, 'Delta')

print('Largest increases (A - B)')
print(top_inc[['Feature', 'Importance_B', 'Importance_A', 'Delta']]
      .to_string(index=False))

print('\nLargest decreases (A - B)')
print(top_dec[['Feature', 'Importance_B', 'Importance_A', 'Delta']]
      .to_string(index=False))

# --------------------------------------------------------------------------
# 2. Helper: split into positive and negative importances,
#    return log10‐scaled magnitudes (zeros dropped)
# --------------------------------------------------------------------------
def split_signed(csv_path):
    s = pd.read_csv(csv_path)['Importance']
    pos = np.log10(s[s > 0])          # log10 of positives
    neg = np.log10((-s[s < 0]))       # log10 of |negatives|
    return pos, neg

pos_a, neg_a = split_signed(file_a)
pos_b, neg_b = split_signed(file_b)

# --------------------------------------------------------------------------
# 3. Plot four overlaid histograms:
#    Run A positives / negatives, Run B positives / negatives
# --------------------------------------------------------------------------
plt.figure(figsize=(9, 6))

plt.hist(pos_a, bins=100, alpha=0.5, label='Run A  +', density=True)
plt.hist(neg_a, bins=100, alpha=0.5, label='Run A  -', density=True)
plt.hist(pos_b, bins=100, alpha=0.5, label='Run B  +', density=True)
plt.hist(neg_b, bins=100, alpha=0.5, label='Run B  -', density=True)

plt.xlabel('log10(|feature importance|)')
plt.ylabel('Density')
plt.title('Signed distribution of feature importances')
plt.legend()
plt.tight_layout()
plt.show()