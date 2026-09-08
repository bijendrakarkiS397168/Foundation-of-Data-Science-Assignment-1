import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Read Excel sheets
file_name = "dataset.xlsx"

data_1h = pd.read_excel(
    file_name,
    sheet_name="Estadisticas_1H"
)

data_2h = pd.read_excel(
    file_name,
    sheet_name="Estadisticas_2H"
)

# Select required first-half columns
first_half = data_1h[
    [
        "id_partido",
        "equipo_local",
        "equipo_visitante",
        "corners_local_1h",
        "corners_visitante_1h"
    ]
].copy()

# Select required second-half columns
second_half = data_2h[
    [
        "id_partido",
        "corners_local_2h",
        "corners_visitante_2h"
    ]
].copy()

# Match the two sheets by match ID
data = pd.merge(
    first_half,
    second_half,
    on="id_partido",
    how="inner"
)

# Calculate total corners in each half
data["First_Half_Corners"] = (
    data["corners_local_1h"]
    + data["corners_visitante_1h"]
)

data["Second_Half_Corners"] = (
    data["corners_local_2h"]
    + data["corners_visitante_2h"]
)

# Difference = second half - first half
data["Difference"] = (
    data["Second_Half_Corners"]
    - data["First_Half_Corners"]
)

# Descriptive statistics
n = len(data)

mean_first = data["First_Half_Corners"].mean()
mean_second = data["Second_Half_Corners"].mean()
mean_difference = data["Difference"].mean()

sd_difference = data["Difference"].std(ddof=1)
standard_error = sd_difference / np.sqrt(n)

# 95% confidence interval
critical_t = stats.t.ppf(
    0.975,
    df=n - 1
)

ci_lower = (
    mean_difference
    - critical_t * standard_error
)

ci_upper = (
    mean_difference
    + critical_t * standard_error
)

# One-sample t-test on paired differences
t_statistic, p_value = stats.ttest_1samp(
    data["Difference"],
    popmean=0
)

# Print statistical results
print("--- FINAL STATISTICAL RESULTS ---")
print("Sample size:", n)
print("Average first-half corners:", mean_first)
print("Average second-half corners:", mean_second)
print("Mean difference:", mean_difference)
print("95% CI:", (ci_lower, ci_upper))
print("t-statistic:", t_statistic)
print("Degrees of freedom:", n - 1)
print("p-value:", p_value)

if p_value < 0.05:
    print("Decision: Reject H0")
    print("Conclusion: Statistically significant difference.")
else:
    print("Decision: Fail to reject H0")
    print("Conclusion: No statistically significant difference.")

# Create presentation figure
means = [
    mean_first,
    mean_second
]

labels = [
    "First Half",
    "Second Half"
]

plt.figure(figsize=(8, 5))

plt.bar(labels, means)

plt.ylabel("Average Corners per Match")
plt.title("Average Corners: First Half vs Second Half")

for i, value in enumerate(means):
    plt.text(
        i,
        value + 0.05,
        f"{value:.2f}",
        ha="center"
    )

plt.tight_layout()

plt.savefig(
    "figures/task1.png",
    dpi=300
)

plt.close()

print("\nFigure saved successfully:")
print("figures/task1.png")