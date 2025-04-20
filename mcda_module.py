import numpy as np
import pandas as pd

# --- Configuration ---
criteria = [
    "Customer Footfall Potential",
    "Non-Interest Income Potential",
    "Real Estate Efficiency",
    "Digital Infrastructure Readiness",
    "Demographic Alignment",
    "Regulatory Friction"
]

# Example AHP-derived weights (should sum to 1)
weights = np.array([0.25, 0.20, 0.15, 0.15, 0.15, 0.10])

# --- Load Data ---
try:
    df = pd.read_csv("mcda_scores.csv")  # 18 rows, 6 columns + 'Model Name'
except FileNotFoundError:
    raise FileNotFoundError("Please provide a valid path to 'mcda_scores.csv'")

# --- Normalize scores (scale 0–5 to 0–1) ---
score_matrix = df[criteria].values / 5

# --- Compute Weighted Scores ---
weighted = score_matrix * weights
df["Total Score"] = weighted.sum(axis=1)
df["Rank"] = df["Total Score"].rank(ascending=False, method="min").astype(int)
df.sort_values("Rank", inplace=True)

# --- Save and Print Results ---
df.to_csv("mcda_ranked_models.csv", index=False)
print(df[["Model Name", "Total Score", "Rank"] + criteria])
