import pandas as pd
import numpy as np

def simulate_metrics(income, cost, non_interest_income):
    if income == 0:
        raise ValueError("Total income must be non-zero")

    cir = cost / income
    roi = (income - cost) / cost
    niir = non_interest_income / income
    oer = (cost - non_interest_income) / income

    return round(cir, 3), round(roi, 3), round(niir, 3), round(oer, 3)

# --- Example Scenario ---
income = 75000
cost = 50000
non_interest_income = 60000

results = simulate_metrics(income, cost, non_interest_income)
labels = ["CIR", "ROI", "NIIR", "OER"]

print("Simulated Financial Metrics:")
for metric, val in zip(labels, results):
    print(f"{metric}: {val}")
