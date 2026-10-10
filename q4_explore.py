
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# 1. Read the daily returns file
input_file = Path("data") / "sp500_returns_2021_2025.csv"

if not input_file.exists():
    raise FileNotFoundError(
        f"Cannot find data file: {input_file.resolve()}"
    )

df = pd.read_csv(input_file, parse_dates=["Date"])
df = df.set_index("Date")

returns = df["Return_Percent"].dropna()

# 2. Display basic information
print("Number of daily returns:", len(returns))
print("Start date:", returns.index.min().date())
print("End date:", returns.index.max().date())

print("\nSummary statistics:")
print(returns.describe())

# 3. Plot daily returns
figures_folder = Path("figures")
figures_folder.mkdir(exist_ok=True)

plt.figure(figsize=(12, 5))
plt.plot(returns.index, returns)
plt.axhline(y=0, linestyle="--")
plt.title("S&P 500 Daily Returns (2021-2025)")
plt.xlabel("Date")
plt.ylabel("Daily Return (%)")
plt.tight_layout()

# 4. Save the figure
figure_file = figures_folder / "sp500_daily_returns.png"
plt.savefig(figure_file, dpi=300)
plt.show()

print("\nFigure saved to:", figure_file.resolve())