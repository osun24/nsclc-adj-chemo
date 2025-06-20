import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
from lifelines.statistics import logrank_test
import matplotlib.pyplot as plt

data = pd.read_csv('affymetrix.merged.csv')
data = pd.get_dummies(data, columns=["Stage", "Histology", "Race"])
data = data.drop(columns=['PFS_MONTHS','RFS_MONTHS'])

data['Adjuvant Chemo'] = data['Adjuvant Chemo'].replace({'OBS': 0, 'ACT': 1})

# Filter to stage IA and IB based on the dummy columns created by get_dummies
stage_IA = data[data['Stage_IA'] == 1]
stage_IB = data[data['Stage_IB'] == 1]

plt.rcParams.update({'font.size': 14})
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

for ax, (stage_label, stage_df) in zip(axes, [("Stage IA", stage_IA), ("Stage IB", stage_IB)]):
    # Split by Adjuvant Chemo status
    chemo_yes = stage_df[stage_df["Adjuvant Chemo"] == 1]
    chemo_no = stage_df[stage_df["Adjuvant Chemo"] == 0]
    
    kmf_yes = KaplanMeierFitter(label=f"{stage_label} - Adjuvant Chemo")
    kmf_no = KaplanMeierFitter(label=f"{stage_label} - No Adjuvant Chemo")
    
    kmf_yes.fit(chemo_yes['OS_MONTHS'], event_observed=chemo_yes['OS_STATUS'])
    kmf_no.fit(chemo_no['OS_MONTHS'], event_observed=chemo_no['OS_STATUS'])
    
    kmf_yes.plot_survival_function(ax=ax, ci_show=True)
    kmf_no.plot_survival_function(ax=ax, ci_show=True)
    ax.set_title(f"Kaplan-Meier Curve for {stage_label}")
    ax.set_xlabel('Overall Survival Time (Months)')
    ax.set_ylabel('Overall Survival Probability')
    ax.grid(True)
    
    # Display at-risk counts below the curves using fitter objects
    add_at_risk_counts(kmf_yes, kmf_no, ax=ax)
    
    # Compute logrank test and annotate p-value on the plot
    results = logrank_test(chemo_yes['OS_MONTHS'], chemo_no['OS_MONTHS'],
                           event_observed_A=chemo_yes['OS_STATUS'],
                           event_observed_B=chemo_no['OS_STATUS'])
    ax.text(0.1, 0.1, f"p = {results.p_value:.3f}", transform=ax.transAxes)
    print(f"{stage_label} logrank test p-value: {results.p_value}")

plt.tight_layout()
plt.savefig('eda/km/Affy-Kaplan-Meier-by-Adjuvant.png')
plt.show()

# New combined survival analysis for Stage IA and IB together
combined = pd.concat([stage_IA, stage_IB])
chemo_yes_combined = combined[combined["Adjuvant Chemo"] == 1]
chemo_no_combined = combined[combined["Adjuvant Chemo"] == 0]

fig_combined, ax_combined = plt.subplots(figsize=(16,8))
kmf_yes_combined = KaplanMeierFitter(label="Combined - Adjuvant Chemo")
kmf_no_combined = KaplanMeierFitter(label="Combined - No Adjuvant Chemo")

kmf_yes_combined.fit(chemo_yes_combined['OS_MONTHS'], event_observed=chemo_yes_combined['OS_STATUS'])
kmf_no_combined.fit(chemo_no_combined['OS_MONTHS'], event_observed=chemo_no_combined['OS_STATUS'])

kmf_yes_combined.plot_survival_function(ax=ax_combined, ci_show=True)
kmf_no_combined.plot_survival_function(ax=ax_combined, ci_show=True)
ax_combined.set_title("Kaplan-Meier Curve for Combined Stage IA & IB")
ax_combined.set_xlabel('Overall Survival Time (Months)')
ax_combined.set_ylabel('Overall Survival Probability')
ax_combined.grid(True)

# Add at-risk counts below the curves
add_at_risk_counts(kmf_yes_combined, kmf_no_combined, ax=ax_combined)

# Compute logrank test and annotate p-value on the plot
results_combined = logrank_test(chemo_yes_combined['OS_MONTHS'], chemo_no_combined['OS_MONTHS'],
                                event_observed_A=chemo_yes_combined['OS_STATUS'],
                                event_observed_B=chemo_no_combined['OS_STATUS'])
ax_combined.text(0.1, 0.1, f"p = {results_combined.p_value:.3f}", transform=ax_combined.transAxes)
print(f"Combined logrank test p-value: {results_combined.p_value}")

plt.tight_layout()
plt.savefig('eda/km/Affy-Kaplan-Meier-by-Adjuvant-2.png')
plt.show()

# New analysis: Kaplan-Meier survival for the entire dataset
chemo_yes_all = data[data["Adjuvant Chemo"] == 1]
chemo_no_all = data[data["Adjuvant Chemo"] == 0]

fig_all, ax_all = plt.subplots(figsize=(16,8))
kmf_yes_all = KaplanMeierFitter(label="Entire Dataset - Adjuvant Chemo")
kmf_no_all = KaplanMeierFitter(label="Entire Dataset - No Adjuvant Chemo")

kmf_yes_all.fit(chemo_yes_all['OS_MONTHS'], event_observed=chemo_yes_all['OS_STATUS'])
kmf_no_all.fit(chemo_no_all['OS_MONTHS'], event_observed=chemo_no_all['OS_STATUS'])

kmf_yes_all.plot_survival_function(ax=ax_all, ci_show=True)
kmf_no_all.plot_survival_function(ax=ax_all, ci_show=True)
ax_all.set_title("Kaplan-Meier Curve for Entire Dataset")
ax_all.set_xlabel('Overall Survival Time (Months)')
ax_all.set_ylabel('Overall Survival Probability')
ax_all.grid(True)

# Add at-risk counts
add_at_risk_counts(kmf_yes_all, kmf_no_all, ax=ax_all)

# Perform logrank test and annotate p-value on the plot
results_all = logrank_test(chemo_yes_all['OS_MONTHS'], chemo_no_all['OS_MONTHS'],
                           event_observed_A=chemo_yes_all['OS_STATUS'],
                           event_observed_B=chemo_no_all['OS_STATUS'])
ax_all.text(0.1, 0.1, f"p = {results_all.p_value:.3f}", transform=ax_all.transAxes)
print(f"Entire dataset logrank test p-value: {results_all.p_value}")

plt.tight_layout()
plt.savefig('eda/km/Affy-Kaplan-Meier-Entire-Dataset.png')
plt.show()