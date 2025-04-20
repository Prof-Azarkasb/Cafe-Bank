import pandas as pd

# Load datasets
wb = pd.read_csv("worldbank_eodb.csv")
itu = pd.read_csv("itu_digital_inclusion.csv")
unhabitat = pd.read_csv("unhabitat_urban_index.csv")

# Merge on country or city identifiers
merged = wb.merge(itu, on="Country").merge(unhabitat, on="City")

# Handle missing values
merged.fillna(method='ffill', inplace=True)

# Normalize key indicators (0–1)
columns_to_norm = ["Digital Access", "Urban Footfall Index", "Ease of Doing Business"]
for col in columns_to_norm:
    min_val = merged[col].min()
    max_val = merged[col].max()
    merged[col + "_Norm"] = (merged[col] - min_val) / (max_val - min_val)

# Export for MCDA use
merged.to_csv("merged_indicators_normalized.csv", index=False)
