import pandas as pd
import numpy as np

# -------------------------------
# Step 1: Load Score Matrix (CSV)
# -------------------------------
def load_mcda_scores(filepath):
    try:
        df = pd.read_csv(filepath)
        expected_columns = [
            'Model Name',
            'Customer Footfall Potential',
            'Non-Interest Income Potential',
            'Real Estate Efficiency',
            'Digital Infrastructure Readiness',
            'Demographic Alignment',
            'Regulatory Friction'
        ]
        assert all(col in df.columns for col in expected_columns), "Missing one or more required columns."
        return df
    except Exception as e:
        raise RuntimeError(f"Error loading or validating the CSV file: {e}")

# --------------------------------------------
# Step 2: AHP-Based Normalized Weight Vector
# --------------------------------------------
def get_ahp_weights():
    # AHP-derived weights from expert judgment (normalized, sum = 1)
    return {
        'Customer Footfall Potential': 0.23,
        'Non-Interest Income Potential': 0.21,
        'Real Estate Efficiency': 0.16,
        'Digital Infrastructure Readiness': 0.17,
        'Demographic Alignment': 0.13,
        'Regulatory Friction': 0.10
    }

# ----------------------------------------------------
# Step 3: Compute Weighted Scores and Model Rankings
# ----------------------------------------------------
def compute_mcda_rankings(df, weights):
    score_cols = list(weights.keys())
    for col in score_cols:
        if df[col].isnull().any():
            raise ValueError(f"Missing score detected in column: {col}")

    # Normalize scores between 0 and 1 for each criterion
    for col in score_cols:
        min_val, max_val = df[col].min(), df[col].max()
        df[col + '_norm'] = (df[col] - min_val) / (max_val - min_val)

    # Calculate weighted score
    df['Total Score'] = sum(df[col + '_norm'] * weight for col, weight in weights.items())

    # Sort by total score descending
    df_sorted = df.sort_values(by='Total Score', ascending=False).reset_index(drop=True)
    df_sorted['Rank'] = df_sorted.index + 1

    return df_sorted[['Rank', 'Model Name', 'Total Score'] + score_cols]

# --------------------------------
# Step 4: Run Full MCDA Pipeline
# --------------------------------
def main():
    filepath = "mcda_scores.csv"  # Replace with your actual path
    print("[INFO] Loading score matrix...")
    df_scores = load_mcda_scores(filepath)

    print("[INFO] Retrieving AHP weights...")
    ahp_weights = get_ahp_weights()

    print("[INFO] Performing MCDA scoring and ranking...")
    df_ranked = compute_mcda_rankings(df_scores, ahp_weights)

    print("\n[RESULT] MCDA Ranking Table:\n")
    print(df_ranked.to_string(index=False))

    # Optional: Export to Excel/CSV
    df_ranked.to_csv("mcda_results_ranked.csv", index=False)
    print("\n[INFO] Results exported to 'mcda_results_ranked.csv'.")

if __name__ == "__main__":
    main()
