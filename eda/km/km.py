import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
import matplotlib.pyplot as plt

data = pd.read_csv('GPL570merged.csv')
data = pd.get_dummies(data, columns=["Stage", "Histology", "Race"])
data = data.drop(columns=['PFS_MONTHS','RFS_MONTHS'])

# Extract the relevant columns for Kaplan-Meier estimation
T = data['OS_MONTHS']
E = data['OS_STATUS']

# Replace the values in the 'OS_STATUS' column
# 0: Censored
# 1: Event
print(E)

# Create a KaplanMeierFitter instance
kmf = KaplanMeierFitter(label = 'GPL570')

# Fit the data into the model
kmf.fit(T, event_observed=E)

print(kmf.event_table)

plt.rcParams.update({'font.size': 14})
# Plot the Kaplan-Meier estimate
plt.figure(figsize=(10, 6))
kmf.plot_survival_function(ci_show=True)
plt.title('Kaplan-Meier Survival Curve (95% CI)')
plt.xlabel('Overall Survival Time (Months)')
plt.ylabel('Overall Survival Probability')
plt.grid(True)
add_at_risk_counts(kmf)
plt.tight_layout()
plt.savefig('GPL570-Kaplan-Meier.png')
plt.show()