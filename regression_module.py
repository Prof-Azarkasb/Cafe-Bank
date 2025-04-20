import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# --- Load dataset ---
df = pd.read_csv("regression_data.csv")  # Columns: NII, OC, NP

# --- Linear Regression (OLS) ---
X = df[["NII", "OC"]]
y = df["NP"]
X_linear = sm.add_constant(X)

model_linear = sm.OLS(y, X_linear).fit()
print(model_linear.summary())

# --- Nonlinear Regression (Polynomial Degree 2) ---
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

model_nonlin = LinearRegression().fit(X_poly, y)
y_pred_poly = model_nonlin.predict(X_poly)

# --- Visualization ---
plt.figure(figsize=(10, 6))
plt.scatter(y, y_pred_poly, c='green', edgecolor='k')
plt.plot([y.min(), y.max()], [y.min(), y.max()], '--r')
plt.xlabel("Observed Net Profit")
plt.ylabel("Predicted (Polynomial Regression)")
plt.title("Nonlinear Fit: Net Profit vs Features")
plt.grid(True)
plt.tight_layout()
plt.savefig("nonlinear_fit.png")
plt.show()
