
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox

# 1. Load the daily returns
input_file = Path("data") / "sp500_returns_2021_2025.csv"

df = pd.read_csv(input_file)
returns = df["Return_Percent"].dropna().to_numpy()

print("Observations:", len(returns))

# 2. Fit the two selected candidate models
models = {
    "ARMA(1,1)": (1, 0, 1),
    "MA(1)": (0, 0, 1),
}

results_folder = Path("results")
results_folder.mkdir(exist_ok=True)

diagnostic_results = []

fig, axes = plt.subplots(2, 1, figsize=(10, 7))

for ax, (name, order) in zip(axes, models.items()):

    print("\n" + "=" * 45)
    print("Model:", name)

    fitted = ARIMA(returns, order=order).fit()
    residuals = fitted.resid

    # 3. Plot the residual ACF
    plot_acf(
        residuals,
        lags=30,
        ax=ax,
        zero=False,
        title=f"Residual ACF: {name}"
    )

    # 4. Ljung-Box test at lags 10 and 20
    p_plus_q = order[0] + order[2]

    lb = acorr_ljungbox(
        residuals,
        lags=[10, 20],
        model_df=p_plus_q,
        return_df=True
    )

    print("\nLjung-Box test:")
    print(lb.round(4))

    for lag, row in lb.iterrows():
        diagnostic_results.append({
            "Model": name,
            "Lag": lag,
            "Ljung_Box_Q": row["lb_stat"],
            "p_value": row["lb_pvalue"]
        })

plt.tight_layout()

figure_file = results_folder / "residual_acf_comparison.png"
plt.savefig(figure_file, dpi=300)
plt.show()

# 5. Save the diagnostic results
results_file = results_folder / "residual_diagnostics.csv"
pd.DataFrame(diagnostic_results).to_csv(results_file, index=False)

print("\nACF figure saved to:", figure_file.resolve())
print("Diagnostic results saved to:", results_file.resolve())