import matplotlib.pyplot as plt

# Parsed data from the output
percentages = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
mean_c_indices = [0.710, 0.698, 0.693, 0.686, 0.671, 0.671, 0.662, 0.657, 0.654, 0.641]
se_values = [0.017, 0.014, 0.015, 0.017, 0.018, 0.019, 0.012, 0.015, 0.016, 0.013]

# Plotting
plt.figure(figsize=(10, 6))
plt.errorbar(percentages, mean_c_indices, yerr=se_values, fmt='-o', capsize=5)
plt.title("Model Performance by Percentage of Features Included")
plt.xlabel("Percentage of Features Included")
plt.ylabel("Mean CV C-index")
plt.grid(True)
plt.xticks(percentages)
plt.tight_layout()
plt.savefig("rsf/features-vs-fold.png", dpi=300)
plt.show()