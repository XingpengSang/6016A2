
from pathlib import Path
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

# 1. Load the daily returns
input_file = Path("data") / "sp500_returns_2021_2025.csv"

df = pd.read_csv(input_file, parse_dates=["Date"])
df = df.set_index("Date")
returns = df["Return_Percent"].dropna()

print("Observations used:", len(returns))

# 2. Prepare candidate AR, MA, and ARMA models
model_results = []

for p in range(4):       # AR order: 0, 1, 2, 3
    for q in range(4):   # MA order: 0, 1, 2, 3

        # Skip ARIMA(0, 0, 0), which is the white-noise baseline
        if p == 0 and q == 0:
            continue

        if q == 0:
            model_name = f"AR({p})"
        elif p == 0:
            model_name = f"MA({q})"
        else:
            model_name = f"ARMA({p},{q})"

        try:
            model = ARIMA(returns, order=(p, 0, q))
            fitted_model = model.fit()

            model_results.append({
                "Model": model_name,
                "AIC": fitted_model.aic,
                "SIC_BIC": fitted_model.bic,
                "Converged": fitted_model.mle_retvals.get(
                    "converged", True
                )
            })

            print(f"Fitted: {model_name}")

        except Exception as error:
            print(f"Failed: {model_name} | {error}")

# 3. Compare models
if not model_results:
    raise RuntimeError("No candidate models were fitted successfully.")

comparison = pd.DataFrame(model_results)

# Save the comparison table
results_folder = Path("results")
results_folder.mkdir(exist_ok=True)

output_file = results_folder / "model_comparison.csv"
comparison.sort_values("AIC").to_csv(output_file, index=False)

# Display results ranked by AIC
print("\nModel comparison (sorted by AIC):")
print(comparison.sort_values("AIC").round(4).to_string(index=False))

# 4. Identify the minimum AIC and minimum SIC/BIC
best_aic = comparison.loc[comparison["AIC"].idxmin()]
best_bic = comparison.loc[comparison["SIC_BIC"].idxmin()]

print("\nBest model according to AIC:")
print(best_aic.to_string())

print("\nBest model according to SIC/BIC:")
print(best_bic.to_string())

print("\nComparison table saved to:", output_file.resolve())